bl_info = {
    "name": "Glacier 2 — 007 First Light Toolkit",
    "description": (
        "Full modding toolkit for 007 First Light (Glacier / KNT engine). "
        "Import .prim models and .borg skeletons with skin weights and shape keys; "
        "reshape meshes and export back to game-valid .prim (+meta). Load and edit "
        "materials (.MATI/.MATB), swap or repoint textures, and one-click build real "
        "Blender render materials from the game's shaders. The render-material "
        "engine detects the material family (skin, eye, hair, fabric, generic) and "
        "builds a family-aware Principled graph (basecolor, SRM, normal + detail "
        "normal, translucency, AO, emission, alpha) with parameters wired in. "
        "Native pure-Python texture codec: "
        "decode and encode .TEXT/.TEXD (BC1/BC3/BC4/BC5/BC7) to and from PNG/TGA, "
        "auto-detect formats, generate a missing .TEXT/.TEXD from an image, and package "
        "everything with correct DISTINCT TEXT/TEXD hashes and metas. Plus LOD tools and "
        "material-name resolution from IOI paths."),
    "author": "Glacier modding community",
    "version": (1, 35, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > 007 Mesh Tools  •  File > Import/Export > Glacier 2 007",
    "category": "Import-Export",
}

import os
import struct
import math
import re

import bpy
import mathutils
from mathutils import Vector, Quaternion, Matrix
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import (StringProperty, BoolProperty, CollectionProperty,
                       IntProperty, FloatProperty, FloatVectorProperty,
                       EnumProperty)
from bpy.types import Operator


# =============================================================================
# Binary reader (little-endian, read-only)
# =============================================================================
class Reader:
    def __init__(self, stream):
        self.f = stream

    def close(self):
        self.f.close()

    def seek(self, pos):
        self.f.seek(pos)

    def tell(self):
        return self.f.tell()

    def size(self):
        cur = self.f.tell()
        self.f.seek(0, 2)
        end = self.f.tell()
        self.f.seek(cur)
        return end

    def u8(self):
        return self.f.read(1)[0]

    def i16(self):
        return struct.unpack("<h", self.f.read(2))[0]

    def u16(self):
        return struct.unpack("<H", self.f.read(2))[0]

    def i32(self):
        return struct.unpack("<i", self.f.read(4))[0]

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def f32(self):
        return struct.unpack("<f", self.f.read(4))[0]

    def fvec(self, n):
        return [self.f32() for _ in range(n)]

    def u8vec(self, n):
        return list(self.f.read(n))

    def align16(self):
        pos = self.f.tell()
        pad = (16 - (pos % 16)) % 16
        if pad:
            self.f.seek(pos + pad)

    def fixed_str(self, n):
        raw = self.f.read(n)
        z = raw.find(b"\x00")
        if z >= 0:
            raw = raw[:z]
        return raw.decode("utf-8", "replace")


# =============================================================================
# PRIM — 007 First Light RenderPrimitive (read-only)
# =============================================================================
def decode_unit_vec4ub(b):
    # 007FL packed unit vector: xyz = (byte-128)/127.5, 4th byte = handedness.
    return [(b[0] - 128) / 127.5, (b[1] - 128) / 127.5, (b[2] - 128) / 127.5, b[3]]


class HeaderFlags:
    def __init__(self, value):
        self.bitfield = value

    def hasBones(self):           return self.bitfield & 0b1 == 1
    def isWeightedObject(self):   return self.bitfield & 0b1000 == 8
    def isLinkedObject(self):     return self.bitfield & 0b100 == 4


class Vertex:
    __slots__ = ("position", "normal", "uv", "color", "weight", "joint")

    def __init__(self):
        self.position = [0.0, 0.0, 0.0, 0]
        self.normal = [0.0, 0.0, 1.0, 128]
        self.uv = [[0.0, 0.0]]
        self.color = [0xFF, 0xFF, 0xFF, 0xFF]
        self.weight = [0.0, 0.0, 0.0, 0.0]
        self.joint = [0, 0, 0, 0]


class VertexBuffer:
    def __init__(self):
        self.vertices = []

    def read(self, br, count, mesh, weighted=False):
        self.vertices = [Vertex() for _ in range(count)]

        # Positions: int16x4, 8 B/vert. XYZ quantized; W is the 4th bone index
        # on weighted meshes.
        ps, pb = mesh.pos_scale, mesh.pos_bias
        for v in self.vertices:
            x = br.i16(); y = br.i16(); z = br.i16(); w = br.i16()
            v.position[0] = (x / 32767.0) * ps[0] + pb[0]
            v.position[1] = (y / 32767.0) * ps[1] + pb[1]
            v.position[2] = (z / 32767.0) * ps[2] + pb[2]
            v.position[3] = w

        sub = mesh.sub_type
        tsb = mesh.tex_scale_bias
        # uvChannelCount from PRIM_OBJECT_SUBTYPE: 3/4/5 -> 2/3/4 UV sets.
        uvc = {3: 2, 4: 3, 5: 4}.get(sub, 1)

        # 007FL vertex streams are TIGHTLY PACKED - verified on real files: each
        # object is exactly (8 pos + [8 skin] + (12 NTB + 4*uvc UV) + [4 colour])
        # bytes/vertex with NO inter-stream 16-byte padding. The old align16()
        # calls skipped 8 real bytes whenever a stream had an odd vertex count,
        # which shifted and scrambled the UVs on exactly those objects.

        def read_ntb_uv(uv_n):
            # Normal(4) + Tangent(4) + Bitangent(4) + uv_n x UV(4). Channel 0
            # becomes the Blender UVMap; extra channels are stepped over.
            for v in self.vertices:
                v.normal = decode_unit_vec4ub(br.u8vec(4))
                br.u8vec(4)  # tangent
                br.u8vec(4)  # bitangent
                u = br.i16(); vv = br.i16()
                v.uv[0][0] = (u / 32767.0) * tsb[0] + tsb[2]
                v.uv[0][1] = (vv / 32767.0) * tsb[1] + tsb[3]
                for _ in range(uv_n - 1):
                    br.i16(); br.i16()

        # WEIGHTED is decided by the header flag, NOT the subtype - a weighted
        # mesh can still be STANDARD_UV_2/3/4.
        if weighted:
            for v in self.vertices:
                br.u8vec(8)  # skinning-adjacency block (weights come from skin_off)

            # The NTB+UV record is 12 + 4*uvc bytes, but the UV channel sits in
            # one of two slots depending on the vertex declaration:
            #   face/body : Normal Tangent Bitangent UV  -> UV at +12 (+colour after)
            #   eyelash/hair: Normal Tangent UV Bitangent -> UV at +8 (no colour)
            # Both are subtype-2 36 B/vert, so the only reliable discriminator is
            # which slot yields a coherent UV map. Decode both and score each
            # against the index buffer (a real UV map has short triangle edges).
            rec = 12 + 4 * uvc
            block = bytes(br.u8vec(rec * count))

            def _decode_uv(off):
                out = []
                for k in range(count):
                    b = k * rec + off
                    u = struct.unpack_from("<h", block, b)[0]
                    w = struct.unpack_from("<h", block, b + 2)[0]
                    out.append(((u / 32767.0) * tsb[0] + tsb[2],
                                (w / 32767.0) * tsb[1] + tsb[3]))
                return out

            def _score(uvs):
                idx = getattr(mesh, "indices", None) or []
                if len(idx) < 3:
                    return 0.0
                step = max(3, ((len(idx) // 3) // 2000) * 3 or 3)
                tot = 0.0; n = 0
                for t in range(0, len(idx) - 2, step):
                    a = idx[t]; b = idx[t + 1]
                    if a < count and b < count:
                        ua = uvs[a]; ub = uvs[b]
                        dx = ua[0] - ub[0]; dy = ua[1] - ub[1]
                        tot += (dx * dx + dy * dy) ** 0.5; n += 1
                return tot / n if n else 0.0

            uv_std = _decode_uv(12)
            uv_alt = _decode_uv(8)
            # default to the standard +12 slot; only switch to +8 when it is
            # CLEARLY cleaner, so coarse low-poly LODs don't flip on noise.
            use_alt = _score(uv_alt) < _score(uv_std) * 0.6
            chosen = uv_alt if use_alt else uv_std
            for v, uv in zip(self.vertices, chosen):
                v.uv[0][0] = uv[0]; v.uv[0][1] = uv[1]
            for k, v in enumerate(self.vertices):
                v.normal = decode_unit_vec4ub(block[k * rec:k * rec + 4])

            # Records that keep the UV in the last slot are followed by a 4 B/vert
            # vertex-colour stream; the +8 variant uses that slot for bitangent.
            if not use_alt:
                for v in self.vertices:
                    v.color = br.u8vec(4)
        elif sub in (0, 1, 2, 3, 4, 5):
            # Unweighted STANDARD / LINKED / STANDARD_UV_2..4: NTB + UVs packed
            # straight after positions, no skinning block and no colour stream.
            read_ntb_uv(uvc)
        else:
            br.u8vec(16 * count)  # unverified subtype, skip


class PrimMesh:
    def __init__(self):
        self.sub_type = 0
        self.material_id = 0
        self.min = [0.0, 0.0, 0.0]
        self.max = [0.0, 0.0, 0.0]
        self.pos_scale = [1.0] * 4
        self.pos_bias = [0.0] * 4
        self.tex_scale_bias = [1.0, 1.0, 0.0, 0.0]
        self.num_vertices = 0
        self.num_indices = 0
        self.indices = []
        self.vertexBuffer = VertexBuffer()
        self.aux_offset = 0
        self.cloth_data_offset = 0        # field +0x18; nonzero = cloth/hair

    def read(self, br, weighted):
        # PRIM_OBJECT (44 bytes)
        br.u8(); br.u8(); br.u16()                 # PRIM_HEADER
        self.sub_type = br.u8()
        br.u8()                                    # properties
        br.u8(); br.u8(); br.u8(); br.u8()         # lodmask, variant, zbias, zoffset
        self.material_id = br.u16()
        br.u32()                                   # wire colour
        br.u8vec(4)                                # color1
        self.min = br.fvec(3)
        self.max = br.fvec(3)

        # Flattened submesh fields (7 uint32)
        self.num_vertices = br.u32()
        vbo = br.u32()
        self.num_indices = br.u32()
        br.u32()                                   # unknown_0C
        ibo = br.u32()
        self.aux_offset = br.u32()                 # aux/collision stream offset
        self.cloth_data_offset = br.u32()          # cloth sim blob offset (0 = none)

        self.pos_scale = br.fvec(4)
        self.pos_bias = br.fvec(4)
        self.tex_scale_bias = br.fvec(4)
        br.u32()                                   # cloth id

        resume = br.tell()                         # start of weighted trailer

        if self.num_indices > 0:
            br.seek(ibo)
            self.indices = [br.u16() for _ in range(self.num_indices)]

        if self.num_vertices > 0:
            br.seek(vbo)
            self.vertexBuffer.read(br, self.num_vertices, self, weighted)

        br.seek(resume)


class PrimMeshWeighted(PrimMesh):
    def read(self, br, weighted):
        super().read(br, weighted)

        br.u32()                                   # bone indices offset (runtime - unused)
        br.u32()                                   # bone info offset    (runtime - unused)
        br.u32()                                   # copy bones count
        br.u32()                                   # copy bones offset
        skin_off = br.u32()                        # per-vertex skinning

        resume = br.tell()
        if skin_off:
            br.seek(skin_off)
            for v in self.vertexBuffer.vertices:
                w0 = br.u8(); w1 = br.u8(); w2 = br.u8(); w3 = br.u8()
                packed = br.u32()
                b0 = packed & 0x3FF
                b1 = (packed >> 10) & 0x3FF
                b2 = (packed >> 20) & 0x3FF
                b3 = int(v.position[3])            # 4th bone rides in the position W lane
                v.weight = [w0 / 255.0, w1 / 255.0, w2 / 255.0, w3 / 255.0]
                v.joint = [b0, b1, b2, b3]
        br.seek(resume)


class PrimHeaderObj:
    def __init__(self):
        self.property_flags = HeaderFlags(0)
        self.bone_rig_resource_index = 0xFFFFFFFF
        self.object_table = []

    def read(self, br):
        br.u8(); br.u8(); br.u16()                 # PRIM_HEADER
        self.property_flags = HeaderFlags(br.u32())
        br.u32()                                   # unknownPadding (007FL)
        self.bone_rig_resource_index = br.u32()
        count = br.u32()
        table_off = br.u32()
        br.fvec(3); br.fvec(3)                     # total bounds

        weighted = self.property_flags.isWeightedObject()

        br.seek(table_off)
        offsets = [br.u32() for _ in range(count)]

        self.object_table = []
        for off in offsets:
            br.seek(off)
            mesh = PrimMeshWeighted() if weighted else PrimMesh()
            mesh.read(br, weighted)
            self.object_table.append(mesh)


class RenderPrimitive:
    def __init__(self):
        self.header = PrimHeaderObj()

    def read(self, br):
        br.seek(0)
        offset = br.u64()
        br.seek(offset)
        self.header.read(br)

    def num_objects(self):
        return len(self.header.object_table)


def read_prim(filepath):
    f = open(os.fsencode(filepath), "rb")
    br = Reader(f)
    try:
        prim = RenderPrimitive()
        prim.read(br)
        return prim
    finally:
        br.close()


def read_prim_bytes(data):
    import io
    prim = RenderPrimitive()
    prim.read(Reader(io.BytesIO(bytes(data))))
    return prim


def prim_is_weighted(filepath):
    """True when the prim's meshes are skinned (and can take a rig)."""
    f = open(os.fsencode(filepath), "rb")
    br = Reader(f)
    try:
        br.seek(0)
        br.seek(br.u64())
        br.u8(); br.u8(); br.u16()
        return HeaderFlags(br.u32()).isWeightedObject()
    finally:
        br.close()


# =============================================================================
# BORG — 007 First Light BoneRig / skeleton (read-only; bones + bind pose only)
# =============================================================================
class BoneDef:
    __slots__ = ("name", "parent")


class SVQ:
    __slots__ = ("rotation", "position")


class BoneRig:
    def __init__(self):
        self.bone_definitions = []
        self.bind_poses = []

    def read(self, br):
        header_offset = br.u64()
        br.seek(header_offset)
        n = br.u32()
        br.u32()                                   # animated bones
        defs_off = br.u32()
        bind_off = br.u32()
        # remaining offsets (inv-global mats, constraints, poses) are not needed
        # to build the armature, so they are skipped.

        br.seek(defs_off)
        for _ in range(n):
            b = BoneDef()
            br.fvec(3)                             # center
            b.parent = br.i32()
            br.fvec(3)                             # size
            b.name = br.fixed_str(34)
            br.i16()                               # body part
            self.bone_definitions.append(b)

        br.seek(bind_off)
        for _ in range(n):
            s = SVQ()
            s.rotation = br.fvec(4)                # quaternion x,y,z,w
            s.position = br.fvec(4)                # x,y,z,w
            self.bind_poses.append(s)


def read_borg(filepath):
    f = open(os.fsencode(filepath), "rb")
    br = Reader(f)
    try:
        borg = BoneRig()
        borg.read(br)
        return borg
    finally:
        br.close()


# =============================================================================
# Armature builder
#
# Adapted from the glTF Blender IO addon (Apache 2.0), as used by the original
# io_scene_glacier addon. https://github.com/KhronosGroup/glTF-Blender-IO
# =============================================================================
class _Bone:
    def __init__(self):
        self.name = None
        self.children = []
        self.parent = None
        self.base_trs = (Vector((0, 0, 0)), Quaternion((1, 0, 0, 0)), Vector((1, 1, 1)))
        self.rotation_after = Quaternion((1, 0, 0, 0))
        self.rotation_before = Quaternion((1, 0, 0, 0))

    def trs(self):
        t, r, s = self.base_trs
        m = _scale_rot_swap_matrix(self.rotation_before)
        return (self.rotation_after @ t,
                self.rotation_after @ r @ self.rotation_before,
                m @ s)


def _nearby_signed_perm_matrix(rot):
    m = rot.to_matrix()
    x, y, z = m[0], m[1], m[2]
    a, b, c = abs(x[0]), abs(x[1]), abs(x[2])
    i = 0 if a >= b and a >= c else 1 if b >= c else 2
    x[i] = 1 if x[i] > 0 else -1
    x[(i + 1) % 3] = 0
    x[(i + 2) % 3] = 0
    a, b = abs(y[(i + 1) % 3]), abs(y[(i + 2) % 3])
    j = (i + 1) % 3 if a >= b else (i + 2) % 3
    y[j] = 1 if y[j] > 0 else -1
    y[(j + 1) % 3] = 0
    y[(j + 2) % 3] = 0
    k = (0 + 1 + 2) - i - j
    z[k] = 1 if z[k] > 0 else -1
    z[(k + 1) % 3] = 0
    z[(k + 2) % 3] = 0
    return m


def _scale_rot_swap_matrix(rot):
    m = _nearby_signed_perm_matrix(rot)
    m.transpose()
    for i in range(3):
        for j in range(3):
            m[i][j] = abs(m[i][j])
    return m


def _pick_bone_length(bones, bone_id):
    bone = bones[bone_id]
    child_locs = [bones[c].editbone_trans for c in bone.children]
    if child_locs:
        return min(loc.length for loc in child_locs)
    return bones[bone.parent].bone_length


def _pick_bone_rotation(bones, bone_id, parent_rot):
    bone = bones[bone_id]
    child_locs = [bones[c].editbone_trans for c in bone.children]
    if child_locs:
        centroid = sum(child_locs, Vector((0, 0, 0)))
        rot = Vector((0, 1, 0)).rotation_difference(centroid)
        return _nearby_signed_perm_matrix(rot).to_quaternion()
    return parent_rot


def _local_rotation(bones, bone_id, rot):
    bones[bone_id].rotation_before @= rot
    rot_inv = rot.conjugated()
    for child in bones[bone_id].children:
        bones[child].rotation_after = rot_inv @ bones[child].rotation_after


def _rotate_edit_bone(bones, bone_id, rot):
    bones[bone_id].editbone_rot @= rot
    rot_inv = rot.conjugated()
    for child_id in bones[bone_id].children:
        child = bones[child_id]
        child.editbone_trans = rot_inv @ child.editbone_trans
        child.editbone_rot = rot_inv @ child.editbone_rot
    _local_rotation(bones, bone_id, rot)


def _prettify_bones(bones):
    def visit(bone_id, parent_rot=None):
        bone = bones[bone_id]
        bone.bone_length = _pick_bone_length(bones, bone_id)
        if bone.bone_length < 0.0001:
            bone.bone_length = 0.001
        rot = _pick_bone_rotation(bones, bone_id, parent_rot)
        if rot is not None:
            _rotate_edit_bone(bones, bone_id, rot)
        for child in bone.children:
            visit(child, parent_rot=rot)
    visit(0)


def _calc_bone_matrices(bones):
    def visit(bone_id):
        bone = bones[bone_id]
        parent_bind = Matrix.Identity(4)
        parent_edit = Matrix.Identity(4)
        if bone.parent >= 0:
            parent_bind = bones[bone.parent].bind_arma_mat
            parent_edit = bones[bone.parent].editbone_arma_mat
        t, r = bone.bind_trans, bone.bind_rot
        ltp = Matrix.Translation(t) @ Quaternion(r).to_matrix().to_4x4()
        bone.bind_arma_mat = parent_bind @ ltp
        t, r = bone.editbone_trans, bone.editbone_rot
        ltp = Matrix.Translation(t) @ Quaternion(r).to_matrix().to_4x4()
        bone.editbone_arma_mat = parent_edit @ ltp
        for child in bone.children:
            visit(child)
    visit(0)


def _get_bone_trs(svq):
    # Glacier -> Blender axis convention (proven to keep world bone positions
    # identical to the raw mesh frame, so mesh and armature line up).
    t = Vector([svq.position[0], -svq.position[2], svq.position[1]])
    r = Quaternion([svq.rotation[3], -svq.rotation[0], svq.rotation[2], -svq.rotation[1]])
    s = Vector([1, 1, 1])
    return t, r, s


def _init_bones(borg, bones):
    for i, bone in enumerate(borg.bone_definitions):
        bl = _Bone()
        bones[i] = bl
        bl.name = bone.name
        bl.base_trs = _get_bone_trs(borg.bind_poses[i])
        if i == 0:
            rot = mathutils.Euler((0.0, 0.0, 0.0), "XYZ")
            rot.rotate_axis("X", math.radians(-90.0))
            bl.base_trs[1].rotate(rot)
        bl.bind_trans = Vector(bl.base_trs[0])
        bl.bind_rot = Quaternion(bl.base_trs[1])
        bl.editbone_trans = Vector(bl.bind_trans)
        bl.editbone_rot = Quaternion(bl.bind_rot)
        bl.parent = bone.parent

    for i, bone in enumerate(borg.bone_definitions):
        if len(borg.bone_definitions) >= bone.parent >= 0:
            bones[bone.parent].children.append(i)


def _compute_bones(borg):
    bones = {}
    _init_bones(borg, bones)
    _prettify_bones(bones)
    _calc_bone_matrices(bones)
    return bones


def build_armature_object(context, collection, borg, name, reorient=False):
    """Create a Blender armature object from a parsed BoneRig, link it to the
    collection and return it (kept in the scene). When reorient is True, each
    bone is pointed at its child(ren) for a clean, poseable rig; the pose is
    recomputed from the new orientation so the skinned result is unchanged."""
    amt = bpy.data.armatures.new(name)
    bones = _compute_bones(borg)

    arma_obj = bpy.data.objects.new(name, amt)
    collection.objects.link(arma_obj)

    order = []

    def visit(i):
        order.append(i)
        for c in bones[i].children:
            visit(c)
    visit(0)

    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    context.view_layer.objects.active = arma_obj
    bpy.ops.object.mode_set(mode="EDIT")

    for i in order:
        bone = bones[i]
        eb = amt.edit_bones.new(bone.name)
        bone.bl_name = eb.name
        eb.use_connect = False
        m = bone.editbone_arma_mat
        eb.head = m @ Vector((0, 0, 0))
        eb.tail = m @ Vector((0, 1, 0))
        eb.length = bone.bone_length
        eb.align_roll(m @ Vector((0, 0, 1)) - eb.head)

    for i in order:
        bone = bones[i]
        if bone.parent >= 0:
            amt.edit_bones[bone.bl_name].parent = amt.edit_bones[bones[bone.parent].bl_name]

    # optional: point each bone at its child(ren) for a clean, poseable rig. Heads
    # are never moved (they carry the bind pose); only tail direction/roll change,
    # and the pose below is recomputed from the new orientation so the skinned
    # mesh deforms identically (verified: P' R'^-1 == P R^-1).
    er_new = {}
    if reorient:
        for i in order:
            bone = bones[i]
            eb = amt.edit_bones[bone.bl_name]
            kids = bone.children
            if len(kids) == 1:
                ch = amt.edit_bones[bones[kids[0]].bl_name]
                if (ch.head - eb.head).length > 1e-4:
                    eb.tail = ch.head
            elif len(kids) > 1:
                avg = Vector((0, 0, 0))
                for c in kids:
                    avg = avg + amt.edit_bones[bones[c].bl_name].head
                avg = avg / len(kids)
                if (avg - eb.head).length > 1e-4:
                    eb.tail = avg
            elif bone.parent >= 0:
                peb = amt.edit_bones[bones[bone.parent].bl_name]
                d = eb.head - peb.head
                if d.length > 1e-4:
                    eb.tail = eb.head + d.normalized() * max(d.length * 0.5, 0.01)
            if (eb.tail - eb.head).length < 1e-5:
                eb.tail = eb.head + Vector((0, 0, 0.02))
        for i in order:
            er_new[i] = amt.edit_bones[bones[i].bl_name].matrix.to_quaternion()
    else:
        for i in order:
            er_new[i] = bones[i].editbone_rot

    bpy.ops.object.mode_set(mode="OBJECT")

    for i in order:
        bone = bones[i]
        pb = arma_obj.pose.bones[bone.bl_name]
        t, r, s = bone.trs()
        et, er = bone.editbone_trans, bone.editbone_rot
        ern = er_new[i]
        pb.location = ern.conjugated() @ (t - et)
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = ern.conjugated() @ r @ er.conjugated() @ ern
        pb.scale = s

    return arma_obj


# =============================================================================
# Mesh builder + skinning
# =============================================================================
def build_mesh(prim, name, index):
    sub = prim.header.object_table[index]
    verts = [(v.position[0], v.position[1], v.position[2]) for v in sub.vertexBuffer.vertices]
    idx = sub.indices
    faces = [(idx[i], idx[i + 1], idx[i + 2]) for i in range(0, len(idx), 3)]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    flat_uv = []
    flat_col = []
    loop_normals = []
    for i in idx:
        v = sub.vertexBuffer.vertices[i]
        flat_uv += [v.uv[0][0], 1.0 - v.uv[0][1]]
        flat_col += [v.color[0] / 255.0, v.color[1] / 255.0,
                     v.color[2] / 255.0, v.color[3] / 255.0]
        loop_normals.append((v.normal[0], v.normal[1], v.normal[2]))

    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_layer.data.foreach_set("uv", flat_uv)

    col = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="CORNER")
    col.data.foreach_set("color", flat_col)

    for poly in mesh.polygons:
        poly.use_smooth = True

    mesh.validate(clean_customdata=False)
    mesh.update()

    try:
        mesh.normals_split_custom_set(loop_normals)
    except Exception as e:
        print("[007 import] custom normals skipped for %s: %s" % (name, e))

    # The imported custom split normals already describe the shading, so any
    # "sharp" edge flags are redundant and just litter the mesh with hard edges
    # that look wrong in the viewport. Clear them for clean smooth shading.
    try:
        for e in mesh.edges:
            e.use_edge_sharp = False
    except Exception:
        pass

    return mesh


def apply_skinning(obj, sub, borg):
    """One vertex group per bone (in bone order, so the group index equals the
    global BORG bone index the prim stores), then assign the 4 influences."""
    for bone in borg.bone_definitions:
        obj.vertex_groups.new(name=bone.name)
    vgs = list(obj.vertex_groups)
    n = len(vgs)
    for vi, v in enumerate(sub.vertexBuffer.vertices):
        for k in range(4):
            w = v.weight[k]
            if w:
                j = int(v.joint[k])
                if 0 <= j < n:
                    vgs[j].add((vi,), w, "REPLACE")


# =============================================================================
# Operators
# =============================================================================
class IMPORT_SCENE_OT_glacier2_borg(Operator, ImportHelper):
    """Import a 007 First Light BoneRig (.borg) as an armature"""
    bl_idname = "import_scene.glacier2_007_borg"
    bl_label = "Import 007 Skeleton (.borg)"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".borg"
    filter_glob: StringProperty(default="*.borg;*.BORG", options={"HIDDEN"})

    reorient_bones: BoolProperty(
        name="Reorient Bones",
        description="Point each bone at its child for a clean, poseable rig. Bone "
                    "heads (the bind pose) are not moved and skinning is unchanged - "
                    "only the visual bone direction/roll. Turn off to keep the raw "
                    "game orientation",
        default=False,
    )

    def draw(self, context):
        self.layout.prop(self, "reorient_bones")

    def execute(self, context):
        try:
            borg = read_borg(self.filepath)
        except Exception as e:
            self.report({"ERROR"}, "Failed to read .borg: %s" % e)
            return {"CANCELLED"}

        name = bpy.path.display_name_from_filepath(self.filepath)
        collection = bpy.data.collections.new(name)
        context.scene.collection.children.link(collection)
        arma = build_armature_object(context, collection, borg, name,
                                     reorient=self.reorient_bones)
        context.view_layer.objects.active = arma
        self.report({"INFO"}, "Imported skeleton: %d bones" % len(borg.bone_definitions))
        return {"FINISHED"}


def _mesh_vertex_coords(me):
    """Per-vertex positions accounting for the current shape-key mix (relative
    keys), WITHOUT modifiers (so the armature pose is not baked in). Falls back
    to plain vertex coords when there are no shape keys."""
    sk = getattr(me, "shape_keys", None)
    if not sk or not sk.key_blocks:
        return [(v.co.x, v.co.y, v.co.z) for v in me.vertices]
    blocks = sk.key_blocks
    basis = blocks[0].data
    n = len(me.vertices)
    coords = [[basis[i].co[0], basis[i].co[1], basis[i].co[2]] for i in range(n)]
    if getattr(sk, "use_relative", True):
        for kb in blocks[1:]:
            val = kb.value
            if val == 0.0:
                continue
            rel = kb.relative_key.data if kb.relative_key else basis
            for i in range(n):
                a = kb.data[i].co
                b = rel[i].co
                coords[i][0] += val * (a[0] - b[0])
                coords[i][1] += val * (a[1] - b[1])
                coords[i][2] += val * (a[2] - b[2])
    return [(c[0], c[1], c[2]) for c in coords]


class IMPORT_SCENE_OT_glacier2_prim(Operator, ImportHelper):
    """Import a 007 First Light RenderPrimitive (.prim), optionally with a rig"""
    bl_idname = "import_scene.glacier2_007_prim"
    bl_label = "Import 007 Model (.prim)"
    bl_options = {"UNDO", "PRESET"}

    filename_ext = ".prim"
    filter_glob: StringProperty(default="*.prim;*.PRIM", options={"HIDDEN"})

    use_rig: BoolProperty(
        name="Import Rig (.borg)",
        description="Also import a skeleton and bind the mesh to it with weights",
        default=False,
    )
    rig_filepath: StringProperty(
        name="Rig",
        description="Path to the .borg skeleton",
        subtype="FILE_PATH",
    )
    import_shapekeys: BoolProperty(
        name="Import Shape Keys",
        description="Give each imported mesh a 'Basis' shape key so you can sculpt "
                    "blend shapes / facial morphs on top (the .prim itself stores no "
                    "morph targets, so this sets up shape-key editing)",
        default=False,
    )
    reorient_bones: BoolProperty(
        name="Reorient Bones",
        description="Point each bone at its child for a clean, poseable rig. Bone "
                    "heads (the bind pose) are not moved and skinning is unchanged - "
                    "only the visual bone direction/roll. Turn off to keep the raw "
                    "game orientation",
        default=False,
    )

    auto_materials: BoolProperty(
        name="Auto-Apply Materials & Textures",
        description="After import, automatically find the model's .MATI materials and "
                    ".TEXT/.TEXD textures sitting in the same folder as the .prim, "
                    "decode them, build the render materials and assign them to the "
                    "imported meshes - so the model comes in already textured. Turn off "
                    "to import the bare mesh and set materials up yourself",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="Import options:")
        layout.prop(self, "import_shapekeys")
        layout.prop(self, "auto_materials")
        weighted = False
        if self.filepath and os.path.exists(self.filepath) and self.filepath.lower().endswith(".prim"):
            try:
                weighted = prim_is_weighted(self.filepath)
            except Exception:
                weighted = False
        if not weighted:
            layout.label(text="This model is not skinned (no rig)", icon="INFO")
            return
        layout.prop(self, "use_rig")
        row = layout.row()
        row.enabled = self.use_rig
        row.prop(self, "rig_filepath")
        if self.use_rig and self.rig_filepath and not os.path.exists(
                self.rig_filepath.replace(os.sep, "/")):
            layout.label(text="Rig path not found", icon="ERROR")
        row2 = layout.row()
        row2.enabled = self.use_rig
        row2.prop(self, "reorient_bones")

    def execute(self, context):
        try:
            prim = read_prim(self.filepath)
        except Exception as e:
            self.report({"ERROR"}, "Failed to read .prim: %s" % e)
            return {"CANCELLED"}

        name = bpy.path.display_name_from_filepath(self.filepath)
        collection = bpy.data.collections.new(name)
        context.scene.collection.children.link(collection)

        weighted = prim.header.property_flags.isWeightedObject()

        borg = None
        arma = None
        if self.use_rig and weighted and self.rig_filepath:
            rig_path = self.rig_filepath.replace(os.sep, "/")
            if os.path.exists(rig_path):
                try:
                    borg = read_borg(rig_path)
                except Exception as e:
                    self.report({"WARNING"}, "Rig failed to load (%s); importing mesh only" % e)
                    borg = None
                if borg is not None:
                    rig_name = bpy.path.display_name_from_filepath(rig_path)
                    arma = build_armature_object(context, collection, borg, rig_name,
                                                 reorient=self.reorient_bones)
            else:
                self.report({"WARNING"}, "Rig path not found; importing mesh only")

        src = self.filepath.replace(os.sep, "/")
        rig_used = self.rig_filepath.replace(os.sep, "/") if (self.use_rig and self.rig_filepath) else ""
        lod_counter = {}

        for i in range(prim.num_objects()):
            mesh = build_mesh(prim, "%s_%d" % (name, i), i)
            obj = bpy.data.objects.new(mesh.name, mesh)
            collection.objects.link(obj)

            # bookkeeping for export (template patching needs the original file
            # + which object this Blender mesh corresponds to)
            obj["glacier_source_prim"] = src
            obj["glacier_prim_index"] = i
            if rig_used:
                obj["glacier_rig_path"] = rig_used

            # LOD grouping: objects share a material id; table order within a
            # material group is the LOD chain (0 = highest detail)
            mat = prim.header.object_table[i].material_id
            obj["glacier_material_id"] = mat
            obj["glacier_lod_index"] = lod_counter.get(mat, 0)
            lod_counter[mat] = lod_counter.get(mat, 0) + 1

            if self.import_shapekeys and not obj.data.shape_keys:
                obj.shape_key_add(name="Basis", from_mix=False)

            if arma is not None:
                apply_skinning(obj, prim.header.object_table[i], borg)
                modifier = obj.modifiers.new(name="Armature", type="ARMATURE")
                modifier.object = arma
                obj.parent = arma

        context.view_layer.update()

        applied = ""
        if self.auto_materials:
            applied = self._auto_apply(context, src, collection)

        self.report({"INFO"}, "Imported %d object(s)%s%s" % (
            prim.num_objects(), " + rig" if arma is not None else "", applied))
        return {"FINISHED"}

    def _auto_apply(self, context, src, collection):
        """Find the .MATI / .TEXT / .TEXD next to the imported .prim, load them,
        build the render materials and assign them to the imported meshes."""
        sc = context.scene
        folder = os.path.dirname(src)
        if not folder or not os.path.isdir(folder):
            return ""
        sc.glacier_scan_folder = folder
        if not (sc.glacier_tex_folder or "").strip():
            sc.glacier_tex_folder = folder
        sc.glacier_scan_model_only = True
        # select the freshly imported meshes (helps any selection-based context)
        try:
            for o in context.selected_objects:
                o.select_set(False)
            for o in collection.objects:
                if o.type == "MESH":
                    o.select_set(True)
        except Exception:
            pass
        try:
            res = bpy.ops.glacier.scan_folder("EXEC_DEFAULT")
            if "FINISHED" not in res:
                return "  (no materials found next to the .prim)"
            bpy.ops.glacier.build_materials("EXEC_DEFAULT", apply_to="MODEL")
            try:
                bpy.ops.glacier.set_shading("EXEC_DEFAULT")
            except Exception:
                pass
            return "  + materials & textures applied"
        except Exception as e:
            self.report({"WARNING"}, "Auto-apply skipped (%s). Load materials from the "
                        "007 Mesh Tools > Materials section." % e)
            return ""


# =============================================================================
# Export — template patching
#
# The original .prim is kept byte-for-byte and only the data the user can change
# by reshaping (vertex positions, optionally normals) is overwritten in place.
# Runtime skinning structures (BoneInfo / BoneIndices), collision and the index
# buffer are never touched, so they stay identical to the file the game already
# loads without crashing. This requires the vertex COUNT to be unchanged.
# =============================================================================
def _enc_unit_byte(c):
    return max(0, min(255, int(round(c * 127.5 + 128.0))))


def quantize_positions(coords):
    """coords: list of (x,y,z). Returns (scale3, bias3, list of (ix,iy,iz))."""
    scale = [1.0, 1.0, 1.0]
    bias = [0.0, 0.0, 0.0]
    for axis in range(3):
        vals = [c[axis] for c in coords]
        lo, hi = min(vals), max(vals)
        bias[axis] = (lo + hi) / 2.0
        s = (hi - lo) / 2.0
        scale[axis] = s if s > 1e-6 else 1e-6
    iq = []
    for c in coords:
        comp = []
        for axis in range(3):
            q = (c[axis] - bias[axis]) / scale[axis]
            q = max(-1.0, min(1.0, q))
            comp.append(int(round(q * 32767.0)))
        iq.append(comp)
    return scale, bias, iq


def walk_prim_objects(data):
    """Walk the on-disk PRIM and return (header_off, weighted, [obj_meta...])."""
    def u32(o): return struct.unpack_from("<I", data, o)[0]
    def u64(o): return struct.unpack_from("<Q", data, o)[0]

    header_off = u64(0)
    flags = u32(header_off + 4)
    weighted = (flags & 0b1000) == 8
    count = u32(header_off + 16)
    table_off = u32(header_off + 20)

    objs = []
    for k in range(count):
        off = u32(table_off + k * 4)
        objs.append({
            "off": off,
            "sub_type": data[off + 4],
            "num_vertices": u32(off + 44),
            "vbo": u32(off + 48),
            "num_indices": u32(off + 52),
            "pos_scale_off": off + 72,
            "pos_bias_off": off + 88,
        })
    return header_off, weighted, objs


def patch_object(data, meta, coords, normals):
    """Overwrite positions (and normals if given) for one object, in place."""
    vbo = meta["vbo"]
    n = meta["num_vertices"]
    scale, bias, iq = quantize_positions(coords)

    for i in range(n):
        base = vbo + i * 8
        # only x,y,z (6 bytes); the W lane (4th bone index) is left untouched
        struct.pack_into("<hhh", data, base, iq[i][0], iq[i][1], iq[i][2])

    # pos_scale[0:3] / pos_bias[0:3]; 4th float of each is left untouched
    struct.pack_into("<fff", data, meta["pos_scale_off"], scale[0], scale[1], scale[2])
    struct.pack_into("<fff", data, meta["pos_bias_off"], bias[0], bias[1], bias[2])

    if normals is not None:
        def _a16(x): return (x + 15) & ~15
        sub = meta["sub_type"]
        if sub == 2:            # weighted: positions(8N) [align] subA(8N) [align] NTB+UV(16)
            suba_start = _a16(vbo + n * 8)
            nrm_base = _a16(suba_start + n * 8)
            stride = 16
        elif sub in (0, 1):     # linked/standard: positions(8N) [align] NTB+UV(16)
            nrm_base = _a16(vbo + n * 8)
            stride = 16
        else:
            return
        for i in range(n):
            o = nrm_base + i * stride
            w = data[o + 3]     # preserve handedness byte
            data[o + 0] = _enc_unit_byte(normals[i][0])
            data[o + 1] = _enc_unit_byte(normals[i][1])
            data[o + 2] = _enc_unit_byte(normals[i][2])
            data[o + 3] = w


# ---- meta (.meta binary + .meta.json) -------------------------------------
def derive_meta_path(prim_path):
    base, ext = os.path.splitext(prim_path)
    candidates = [
        base + "_" + ext[1:].upper() + ".meta",   # 01C7..._PRIM.meta (RPKG-Tool)
        prim_path + ".meta",
        base + ".meta",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_meta(data):
    if len(data) < 40:
        return None
    m = {
        "resource_id": struct.unpack_from("<Q", data, 0)[0],
        "data_offset": struct.unpack_from("<Q", data, 8)[0],
        "size_raw": struct.unpack_from("<I", data, 16)[0],
        "ext_raw": bytes(data[20:24]),
        "refs_table_size": struct.unpack_from("<I", data, 24)[0],
        "size_uncompressed": struct.unpack_from("<I", data, 28)[0],
        "size_memory": struct.unpack_from("<I", data, 32)[0],
        "size_video": struct.unpack_from("<I", data, 36)[0],
        "dummy": 0,
        "refs": [],
    }
    if m["refs_table_size"] > 0 and len(data) >= 44:
        cnt = struct.unpack_from("<H", data, 40)[0]
        m["dummy"] = struct.unpack_from("<H", data, 42)[0]
        hbase = 44 + cnt
        # bounds: need hbase + cnt*8 <= len(data)
        if hbase + cnt * 8 > len(data):
            cnt = max(0, (len(data) - hbase) // 8)
        flags = data[44:44 + cnt]
        for i in range(cnt):
            h = struct.unpack_from("<Q", data, hbase + i * 8)[0]
            m["refs"].append((h, flags[i] if i < len(flags) else 0))
    return m


def build_meta_binary(m, extra_refs):
    refs = list(m["refs"]) + list(extra_refs)
    if refs:
        table = bytearray()
        table += struct.pack("<H", len(refs))
        table += struct.pack("<H", m["dummy"])
        table += bytes(f & 0xFF for (_, f) in refs)
        for (h, _) in refs:
            table += struct.pack("<Q", h)
        refs_size = len(table)
    else:
        table = b""
        refs_size = 0

    out = bytearray()
    out += struct.pack("<Q", m["resource_id"])
    out += struct.pack("<Q", m["data_offset"])
    out += struct.pack("<I", m["size_raw"])
    out += m["ext_raw"]
    out += struct.pack("<I", refs_size)
    out += struct.pack("<I", m["size_uncompressed"])
    out += struct.pack("<I", m["size_memory"])
    out += struct.pack("<I", m["size_video"])
    out += table
    return bytes(out)


def build_meta_json(m, extra_refs):
    refs = list(m["refs"]) + list(extra_refs)
    return {
        "hash_value": "%016X" % m["resource_id"],
        "hash_offset": m["data_offset"],
        "hash_size": m["size_raw"],
        "hash_resource_type": m["ext_raw"].decode("ascii", "replace").strip("\x00"),
        "hash_reference_table_size": (4 + len(refs) + len(refs) * 8) if refs else 0,
        "hash_reference_table_dummy": m["dummy"],
        "hash_size_final": m["size_uncompressed"],
        "hash_size_in_memory": m["size_memory"],
        "hash_size_in_video_memory": m["size_video"],
        "hash_reference_data": [
            {"hash": "%016X" % h, "flag": "%02X" % f} for (h, f) in refs
        ],
    }


# =============================================================================
# RPKG v2 archive reader  (007 First Light raw chunk reader)
# -----------------------------------------------------------------------------
# Reads the game's packed .rpkg / chunkNN.rpkg directly so TEXT/TEXD (and any
# other resource) can be mass-extracted without RPKG-Tool. Format from the 010
# template (RPKG.txt): header -> ResourceDataTable -> ResourceMetadataTable ->
# resource data. Magic "GKPR" (RPKG) or "2KPR" (RPK2). Resources may be XOR
# scrambled and/or LZ4 compressed; we descramble + decompress on extract.
#
# The XOR scramble key. First Light (verified empirically on chunk data) uses
# the long-standing Glacier resource scramble. We auto-detect on extract by
# validating the result, and the key is user-overridable in the UI, so a wrong
# guess here is recoverable without a code change.
# =============================================================================
RPKG_XOR_KEY = bytes((0xDC, 0x45, 0xA6, 0x9C, 0xD3, 0x72, 0x4C, 0xAB))


def rpkg_xor(data, key=RPKG_XOR_KEY):
    """In-place style XOR descramble/scramble (symmetric). Returns bytes."""
    if not key:
        return bytes(data)
    out = bytearray(data)
    klen = len(key)
    for i in range(len(out)):
        out[i] ^= key[i % klen]
    return bytes(out)


def _rpkg_ext_from_archive(raw4):
    """Resource extension is stored reversed in the archive (e.g. TXET->TEXT)."""
    return raw4[::-1].decode("ascii", "replace").rstrip("\x00")


class RpkgArchive:
    """Parsed index of an RPKG v2 archive. Does NOT load resource data; extract
    seeks the file per-resource so multi-GB chunks stay memory-light."""

    def __init__(self, path):
        self.path = path
        self.is_rpk2 = False
        self.entries = []          # list of dicts (see _parse)
        self.by_hash = {}          # "%016X" -> entry
        self._parse()

    def _parse(self):
        with open(os.fsencode(self.path), "rb") as f:
            head = f.read(64)
            magic = head[0:4]
            if magic not in (b"GKPR", b"2KPR"):
                raise ValueError("Not an RPKG archive (magic %r). Point at a "
                                 ".rpkg or chunk file." % magic)
            off = 4
            patch_id = 0
            if magic == b"2KPR":
                self.is_rpk2 = True
                # uint Unknown, ubyte chunkID, ubyte chunkType,
                # ubyte chunkPatchID, char langCode[2]
                off += 4
                off += 1                                   # chunkID
                off += 1                                   # chunkType
                patch_id = head[off]; off += 1             # chunkPatchID
                off += 2                                   # languageCode[2]
            res_count = struct.unpack_from("<I", head, off)[0]; off += 4
            hash_table_size = struct.unpack_from("<I", head, off)[0]; off += 4
            meta_table_size = struct.unpack_from("<I", head, off)[0]; off += 4
            if patch_id > 0:
                del_count = struct.unpack_from("<I", head, off)[0]; off += 4
                off += 8 * del_count                       # deletion hashes

            data_table_off = off
            meta_table_off = data_table_off + hash_table_size

            f.seek(0, 2)
            file_size = f.tell()
            if meta_table_off + meta_table_size > file_size:
                raise ValueError(
                    "This file only contains the chunk's hash table, not the "
                    "resource data (it's %d bytes but the index needs %d). Point "
                    "at the full chunkNN.rpkg in the game's Runtime folder, not a "
                    "stripped .meta hash list." % (file_size,
                                                   meta_table_off + meta_table_size))

            f.seek(data_table_off)
            data_tbl = f.read(hash_table_size)
            f.seek(meta_table_off)
            meta_tbl = f.read(meta_table_size)

        # ResourceDataEntry: u64 id, u64 offset, u32 rawSize  (20 bytes)
        data = []
        for i in range(res_count):
            rid, roff, rsize = struct.unpack_from("<QQI", data_tbl, i * 20)
            data.append((rid, roff, rsize))

        # ResourceMetadataEntry (variable). First Light has NO states table.
        # The reference layout in-archive uses the same GROUPED format as
        # standalone .meta files: [count, dummy, flags[N], hashes[N]].
        mo = 0
        mtbl_len = len(meta_tbl)
        entries = []
        for i in range(res_count):
            if mo + 20 > mtbl_len:
                break                                      # truncated table
            rid, roff, rsize = data[i]
            ext = _rpkg_ext_from_archive(meta_tbl[mo:mo + 4]); mo += 4
            refs_size = struct.unpack_from("<I", meta_tbl, mo)[0]; mo += 4
            size_unc = struct.unpack_from("<I", meta_tbl, mo)[0]; mo += 4
            size_mem = struct.unpack_from("<I", meta_tbl, mo)[0]; mo += 4
            size_vid = struct.unpack_from("<I", meta_tbl, mo)[0]; mo += 4
            refs = []
            dummy = 0
            if refs_size > 0 and mo + 4 <= mtbl_len:
                ref_count = struct.unpack_from("<H", meta_tbl, mo)[0]; mo += 2
                dummy = struct.unpack_from("<H", meta_tbl, mo)[0]; mo += 2
                # grouped layout: flags[N] then hashes[N]
                fbase = mo
                hbase = fbase + ref_count
                if hbase + ref_count * 8 <= mtbl_len:
                    for k in range(ref_count):
                        flag = meta_tbl[fbase + k]
                        rh = struct.unpack_from("<Q", meta_tbl, hbase + k * 8)[0]
                        refs.append((rh, flag))
                mo = hbase + ref_count * 8
            actual = rsize & 0x3FFFFFFF
            entry = {
                "hash": "%016X" % rid,
                "rid": rid,
                "ext": ext,
                "offset": roff,
                "actual_size": actual,
                "size_raw": rsize,
                "compressed": bool(rsize & 0x40000000),
                "scrambled": bool(rsize & 0x80000000),
                "size_uncompressed": size_unc,
                "size_memory": size_mem,
                "size_video": size_vid,
                "refs": refs,
                "dummy": dummy,
            }
            entries.append(entry)
            self.by_hash[entry["hash"]] = entry
        self.entries = entries

    def filter(self, exts=None, search=""):
        """Return entries whose ext is in `exts` (set/None=all) and whose hash
        contains `search` (case-insensitive)."""
        s = (search or "").upper().strip()
        out = []
        for e in self.entries:
            if exts is not None and e["ext"] not in exts:
                continue
            if s and s not in e["hash"]:
                continue
            out.append(e)
        return out

    def read_raw(self, entry):
        """Read a resource's stored bytes (still scrambled/compressed)."""
        with open(os.fsencode(self.path), "rb") as f:
            f.seek(entry["offset"])
            return f.read(entry["actual_size"])

    def extract(self, entry, key=RPKG_XOR_KEY):
        """Return the fully decoded resource bytes (descrambled + decompressed)."""
        raw = self.read_raw(entry)
        if entry["scrambled"]:
            raw = rpkg_xor(raw, key)
        if entry["compressed"]:
            raw = lz4_decompress(raw, entry["size_uncompressed"])
        return raw

    def standalone_meta(self, entry):
        """Synthesise the standalone <hash>_<EXT>.meta binary for this resource,
        so the rest of the toolkit (pairing TEXT->TEXD, decoding) just works."""
        m = {
            "resource_id": entry["rid"],
            "data_offset": 0,
            "size_raw": entry["actual_size"],
            "ext_raw": (entry["ext"] + "\x00" * 4)[:4].encode("ascii", "replace"),
            "refs_table_size": 0,
            "size_uncompressed": entry["size_uncompressed"],
            "size_memory": entry["size_memory"],
            "size_video": entry["size_video"],
            "dummy": entry["dummy"],
            "refs": list(entry["refs"]),
        }
        return build_meta_binary(m, [])


# module-level cache so scan + extract don't re-parse a huge chunk twice
_RPKG_CACHE = {}


def rpkg_open_cached(path):
    """Return a parsed RpkgArchive, cached by (path, mtime)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    key = os.path.abspath(path)
    hit = _RPKG_CACHE.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    arc = RpkgArchive(path)
    _RPKG_CACHE[key] = (mtime, arc)
    return arc


def hash_from_filename(path):
    """Extract a 16-hex Glacier hash from a filename stem, or None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split("_")[0].split(".")[0]
    if len(stem) == 16:
        try:
            return int(stem, 16)
        except ValueError:
            return None
    return None


def parse_ref_overrides(text):
    """Parse 'OLDHASH=NEWHASH' lines into {old_int: new_int}. Ignores bad lines."""
    remap = {}
    for line in text.replace(",", "\n").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        a, b = line.split("=", 1)
        a, b = a.strip(), b.strip()
        try:
            remap[int(a, 16)] = int(b, 16)
        except ValueError:
            continue
    return remap


# -----------------------------------------------------------------------------
# MATI (material instance) parser  [reverse-engineered, 007 First Light]
#
# Layout: fixed header, a null-terminated string table of property names, then
# a flat list of property records. Each record is:
#     u16 nameOffset (relative to stringtable base = first_string - 1)
#     u16 type        (0x01 float, 0x02 texture, 0x03 RGB, 0x08 transform)
#     u32 sub          (type sub-descriptor, preserved verbatim)
#     <data>           length by type: 4 / 8 / 12 / 32 bytes
# For a texture record the 8 data bytes are u32 flags + u32 textureIndex, and
# textureIndex indexes the MATI meta's reference list -> the actual texture hash.
# -----------------------------------------------------------------------------
_MATI_TYPELEN = {0x01: 4, 0x02: 8, 0x03: 12, 0x08: 32}


def parse_mati(data):
    def printable(b):
        return 32 <= b < 127

    n = len(data)
    i = 8
    while i < n - 8 and not all(printable(data[i + k]) for k in range(8)):
        i += 1
    while i > 0 and data[i - 1] != 0:
        i -= 1
    base = i - 1

    strings = {}
    j = i
    while j < n:
        if data[j] == 0:
            j += 1
            continue
        if not printable(data[j]):
            break
        z = data.find(b"\x00", j)
        if z < 0:
            break
        strings[j - base] = data[j:z].decode("ascii", "replace")
        j = z + 1

    def walk(s):
        o = s
        r = []
        while o < n:
            if o + 8 > n:
                return None
            nm, t, sub = struct.unpack_from("<HHI", data, o)
            if t not in _MATI_TYPELEN:
                return None
            dl = _MATI_TYPELEN[t]
            if o + 8 + dl > n:
                return None
            r.append((o, nm, t, sub, o + 8, dl))
            o += 8 + dl
        return r if o == n else None

    records = []
    for s in range(i, n):
        r = walk(s)
        if r and len(r) >= 2:
            records = r
            break

    textures, params = [], []
    for (o, nm, t, sub, doff, dl) in records:
        name = strings.get(nm, "@0x%X" % nm)
        if t == 0x02:
            flags, idx = struct.unpack_from("<II", data, doff)
            textures.append({"name": name, "index": idx, "data_off": doff})
        elif t in (0x01, 0x03):
            vals = list(struct.unpack_from("<%df" % (dl // 4), data, doff))
            params.append({"name": name, "type": t, "data_off": doff, "values": vals})
    return {"strings": strings, "records": records,
            "textures": textures, "params": params}


def derive_sibling_resource(prim_path, hash_int, ext):
    """Path to <hash>.<EXT> next to the PRIM (RPKG-Tool naming)."""
    folder = os.path.dirname(prim_path)
    return os.path.join(folder, "%016X.%s" % (hash_int, ext.upper()))


def meta_path_candidates(res_path, ext):
    """The two .meta naming conventions seen in the wild for a resource file:
      <hash>.<EXT>.meta   (dot style, e.g. RPKG-Tool first-light)
      <hash>_<EXT>.meta   (underscore style)
    Returns both, dot style first (it preserves the resource's own filename)."""
    base, _ = os.path.splitext(res_path)
    return [res_path + ".meta", base + "_" + ext.upper() + ".meta"]


def meta_uses_dot_style(res_path, ext):
    """True if a resource's meta is <hash>.<EXT>.meta (dot), False if it is
    <hash>_<EXT>.meta (underscore). Defaults to dot when neither exists (the
    RPKG-Tool first-light convention)."""
    dot, und = meta_path_candidates(res_path, ext)
    if res_path and os.path.exists(dot):
        return True
    if res_path and os.path.exists(und):
        return False
    return True


def meta_out_name(hash_int, ext, dot_style):
    """Output .meta filename for a freshly written resource, in the chosen style."""
    if dot_style:
        return "%016X.%s.meta" % (hash_int, ext.upper())
    return "%016X_%s.meta" % (hash_int, ext.upper())


def derive_resource_meta(res_path, ext):
    """Best .meta path for a resource: the dot-style sibling if it exists, else
    the underscore form. Used when reading; the dot form mirrors the input."""
    cands = meta_path_candidates(res_path, ext)
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def find_resource_meta(res_path, ext, search_dirs):
    """Locate a resource's .meta next to the file, then by scanning the given
    directories recursively. Accepts both .<EXT>.meta and _<EXT>.meta. Returns
    path or ''."""
    cands = meta_path_candidates(res_path, ext)
    for c in cands:
        if os.path.exists(c):
            return c
    wanted = {os.path.basename(c).lower() for c in cands}
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for dirpath, _dirs, files in os.walk(d, onerror=lambda e: None):
                for fn in files:
                    if fn.lower() in wanted:
                        return os.path.join(dirpath, fn)
        except (OSError, PermissionError):
            continue
    return ""


# -----------------------------------------------------------------------------
# MATB (material blueprint) parser  [reverse-engineered, 007 First Light]
# A flat list of property *definitions* (the schema the MATI instantiates):
#     u8 type, u32 nameLen, char name[nameLen] (null-terminated)
# type: 1 texture, 2 color, 4 float, 8 int/enum. No values, no texture refs.
# -----------------------------------------------------------------------------
_MATB_KIND = {1: "texture", 2: "color", 4: "float", 8: "int"}


def parse_matb(data):
    props = []
    o, n = 0, len(data)
    while o + 5 <= n:
        t = data[o]
        ln = struct.unpack_from("<I", data, o + 1)[0]
        if ln <= 0 or o + 5 + ln > n:
            break
        name = data[o + 5:o + 5 + ln].split(b"\x00")[0].decode("ascii", "replace")
        props.append({"name": name, "type": t, "kind": _MATB_KIND.get(t, "?")})
        o += 5 + ln
    return props


def hash_from_path(path):
    """16-hex resource hash from a filename stem, else None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = stem.split("_")[0]
    if len(stem) == 16:
        try:
            int(stem, 16)
            return stem.upper()
        except ValueError:
            pass
    return None


# -----------------------------------------------------------------------------
# Glacier TEXT / TEXD texture header  (reverse-engineered from 007 First Light
# samples; validated against the file's own mip tables and its _TEXT.meta).
#
# Layout of a .TEXT body:
#   0x00 u32   type/version          (1)
#   0x04 u32   total (compressed) data size of the full texture
#   0x08 u32   flags
#   0x0C u16   width                 (the FULL texture size; the TEXT itself only
#   0x0E u16   height                 stores the small streaming mips)
#   0x10 u16   format                (RenderFormat enum, see _RENDER_FORMAT)
#   0x12 u16   mip_count
#   0x14 u32   (interpret/dimension flags)
#   0x18 u32[14] mip end-offsets, UNCOMPRESSED   (clean per-format mip sizes)
#   0x50 u32[14] mip end-offsets, COMPRESSED      (per-mip Oodle/Kraken sizes)
#   0x88 u32   size of the float aux block that follows the header
#   0x8C u32   ...
#   then a float aux block, then the (compressed) streaming mip payload.
#
# IMPORTANT: the pixel payload is per-mip Oodle/Kraken compressed (table at 0x50
# holds the compressed sizes; their ratio to the 0x18 uncompressed sizes climbs
# toward 1.0 for small mips - the signature of block compression on top of BCn).
# Oodle is a proprietary codec with no open implementation, so the RAW PIXELS
# cannot be decoded in pure Python here. This parser reads the metadata only.
# -----------------------------------------------------------------------------
_RENDER_FORMAT = {
    0x1C: "R8G8B8A8", 0x2C: "R32", 0x37: "R8G8", 0x45: "A8",
    0x4C: "BC1", 0x4F: "BC2", 0x52: "BC3",
    0x55: "BC4", 0x58: "BC5", 0x5E: "BC7",
}
_FORMAT_BLOCK_BYTES = {0x4C: 8, 0x4F: 16, 0x52: 16, 0x55: 8, 0x58: 16, 0x5E: 16}
_FMT_NAME_TO_CODE = {"BC1": 0x4C, "BC3": 0x52, "BC4": 0x55, "BC5": 0x58, "BC7": 0x5E}


def _bc_code_for(name):
    """Map a glacier_bc_format choice to a fmt_code, or None for 'Auto'."""
    if name == "AUTO":
        return None
    return _FMT_NAME_TO_CODE.get(name, 0x4C)


def parse_text_header(data):
    """Parse a .TEXT body and return its metadata. Pixels are NOT decoded
    (they are Oodle-compressed). Returns None if it doesn't look like a TEXT."""
    if len(data) < 0x90:
        return None
    try:
        ttype = struct.unpack_from("<I", data, 0x00)[0]
        total = struct.unpack_from("<I", data, 0x04)[0]
        flags = struct.unpack_from("<I", data, 0x08)[0]
        width = struct.unpack_from("<H", data, 0x0C)[0]
        height = struct.unpack_from("<H", data, 0x0E)[0]
        fmt = struct.unpack_from("<H", data, 0x10)[0]
        mips = struct.unpack_from("<H", data, 0x12)[0]
        t_unc = [struct.unpack_from("<I", data, 0x18 + i * 4)[0] for i in range(14)]
        t_cmp = [struct.unpack_from("<I", data, 0x50 + i * 4)[0] for i in range(14)]
    except struct.error:
        return None
    if not (0 < width <= 16384 and 0 < height <= 16384 and 0 < mips <= 14):
        return None
    return {
        "type": ttype, "total": total, "flags": flags,
        "width": width, "height": height, "format": fmt,
        "format_name": _RENDER_FORMAT.get(fmt, "0x%02X" % fmt),
        "mips": mips, "mip_offsets_uncompressed": t_unc,
        "mip_offsets_compressed": t_cmp,
    }


# -----------------------------------------------------------------------------
# Native texture converter (no external tool).
#
# 007 First Light textures (GlacierGame "KNT"/Bond, TextureMapHeaderV4) store
# each BCn mip LZ4-block-compressed. Both LZ4 and BC1 are implemented here in
# pure Python, so .tga/.png can be turned into a real .TEXT + .TEXD pair, and a
# .TEXT+.TEXD pair can be decoded back to pixels, entirely inside Blender.
# Validated against real game files (round-trip ~0.1/255). Output format is BC1.
# -----------------------------------------------------------------------------
def lz4_decompress(src, out_size):
    out = bytearray(); i = 0; n = len(src)
    try:
        while i < n:
            tok = src[i]; i += 1; ll = tok >> 4
            if ll == 15:
                while i < n:
                    b = src[i]; i += 1; ll += b
                    if b != 255:
                        break
            out += src[i:i+ll]; i += ll
            if i >= n or len(out) >= out_size:
                break
            off = src[i] | (src[i+1] << 8); i += 2; ml = (tok & 15) + 4
            if ml - 4 == 15:
                while i < n:
                    b = src[i]; i += 1; ml += b
                    if b != 255:
                        break
            s = len(out) - off
            if off == 0 or s < 0:           # malformed (e.g. raw, not LZ4)
                break
            for k in range(ml):
                out.append(out[s+k])
    except IndexError:
        pass
    return bytes(out[:out_size])


def _lz4_emit_len(out, n):
    while n >= 255:
        out.append(255); n -= 255
    out.append(n)


def lz4_compress(src):
    n = len(src); out = bytearray()
    MIN_MATCH = 4; MFLIMIT = 12; LAST = 5
    if n < MFLIMIT:
        ll = n
        out.append((15 << 4) if ll >= 15 else (ll << 4))
        if ll >= 15:
            _lz4_emit_len(out, ll - 15)
        out += src
        return bytes(out)
    ht = {}; i = 0; anchor = 0; limit = n - LAST; mflimit = n - MFLIMIT
    while i < mflimit:
        seq = src[i:i+4]; h = hash(seq)
        cand = ht.get(h, -1); ht[h] = i
        if 0 <= cand and i - cand <= 0xFFFF and src[cand:cand+4] == seq:
            m = 4
            while i + m < limit and src[cand+m] == src[i+m]:
                m += 1
            off = i - cand; litlen = i - anchor
            out.append((min(litlen, 15) << 4) | min(m - MIN_MATCH, 15))
            if litlen >= 15:
                _lz4_emit_len(out, litlen - 15)
            out += src[anchor:i]
            out.append(off & 0xFF); out.append((off >> 8) & 0xFF)
            if m - MIN_MATCH >= 15:
                _lz4_emit_len(out, m - MIN_MATCH - 15)
            i += m; anchor = i
        else:
            i += 1
    litlen = n - anchor
    out.append((15 << 4) if litlen >= 15 else (litlen << 4))
    if litlen >= 15:
        _lz4_emit_len(out, litlen - 15)
    out += src[anchor:n]
    return bytes(out)


def _rgb565(c):
    return (((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31)


def _to565(r, g, b):
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def bc1_decode(data, w, h):
    need = max(1, (w+3)//4) * max(1, (h+3)//4) * 8
    if len(data) < need:
        data = bytes(data) + b"\x00" * (need - len(data))
    out = bytearray(w*h*4); p = 0
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            c0, c1 = struct.unpack_from('<HH', data, p)
            idx = struct.unpack_from('<I', data, p+4)[0]; p += 8
            r0, g0, b0 = _rgb565(c0); r1, g1, b1 = _rgb565(c1)
            cl = [(r0, g0, b0), (r1, g1, b1)]
            if c0 > c1:
                cl.append(((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3))
                cl.append(((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3))
            else:
                cl.append(((r0+r1)//2, (g0+g1)//2, (b0+b1)//2)); cl.append((0, 0, 0))
            for py in range(4):
                for px in range(4):
                    ci = (idx >> (2*(py*4+px))) & 3; x = bx+px; y = by+py
                    if x < w and y < h:
                        o = (y*w+x)*4; out[o:o+3] = bytes(cl[ci]); out[o+3] = 255
    return out


def bc1_encode(rgba, w, h):
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            tex = []
            for py in range(4):
                for px in range(4):
                    x = min(bx+px, w-1); y = min(by+py, h-1); o = (y*w+x)*4
                    tex.append((rgba[o], rgba[o+1], rgba[o+2]))
            mn = [255, 255, 255]; mx = [0, 0, 0]
            for (r, g, b) in tex:
                if r < mn[0]: mn[0] = r
                if g < mn[1]: mn[1] = g
                if b < mn[2]: mn[2] = b
                if r > mx[0]: mx[0] = r
                if g > mx[1]: mx[1] = g
                if b > mx[2]: mx[2] = b
            c0 = _to565(*mx); c1 = _to565(*mn)
            if c0 < c1:
                c0, c1 = c1, c0
            if c0 == c1:
                out += struct.pack('<HHI', c0, c1, 0); continue
            p0 = _rgb565(c0); p1 = _rgb565(c1)
            pal = [p0, p1,
                   ((2*p0[0]+p1[0])//3, (2*p0[1]+p1[1])//3, (2*p0[2]+p1[2])//3),
                   ((p0[0]+2*p1[0])//3, (p0[1]+2*p1[1])//3, (p0[2]+2*p1[2])//3)]
            idx = 0
            for i, t in enumerate(tex):
                best = 0; bd = 1 << 30
                for k in range(4):
                    dr = t[0]-pal[k][0]; dg = t[1]-pal[k][1]; db = t[2]-pal[k][2]
                    d = dr*dr+dg*dg+db*db
                    if d < bd:
                        bd = d; best = k
                idx |= best << (2*i)
            out += struct.pack('<HHI', c0, c1, idx)
    return bytes(out)


def _gen_mips(rgba, w, h, num):
    mips = [(rgba, w, h)]; cw, ch, cur = w, h, rgba
    for _ in range(num-1):
        nw = max(1, cw//2); nh = max(1, ch//2); nxt = bytearray(nw*nh*4)
        for y in range(nh):
            for x in range(nw):
                sx = x*2; sy = y*2
                for c in range(4):
                    s = 0
                    for dy in range(2):
                        for dx in range(2):
                            xx = min(sx+dx, cw-1); yy = min(sy+dy, ch-1)
                            s += cur[(yy*cw+xx)*4+c]
                    nxt[(y*nw+x)*4+c] = s // 4
        mips.append((bytes(nxt), nw, nh)); cur, cw, ch = nxt, nw, nh
    return mips


def _bc1_block_bytes(w, h):
    return max(1, (w+3)//4) * max(1, (h+3)//4) * 8


# ---- BC4 / BC5 / BC3 (alpha block is the BC4 algorithm) ---------------------
def _bc4_encode_block(vals):
    mn = min(vals); mx = max(vals)
    out = bytearray()
    out.append(mx); out.append(mn)
    if mx == mn:
        out += b"\x00" * 6
        return out
    palette = [mx, mn] + [((6-k)*mx + (k+1)*mn)//7 for k in range(6)]
    bits = 0
    for i, v in enumerate(vals):
        best = 0; bd = 1 << 30
        for k in range(8):
            d = abs(v - palette[k])
            if d < bd:
                bd = d; best = k
        bits |= best << (3*i)
    for k in range(6):
        out.append((bits >> (8*k)) & 0xFF)
    return out


def _bc4_decode_block(data, p):
    r0 = data[p]; r1 = data[p+1]
    bits = int.from_bytes(data[p+2:p+8], "little")
    if r0 > r1:
        pal = [r0, r1] + [((6-k)*r0 + (k+1)*r1)//7 for k in range(6)]
    else:
        pal = [r0, r1] + [((4-k)*r0 + (k+1)*r1)//5 for k in range(4)] + [0, 255]
    return [pal[(bits >> (3*i)) & 7] for i in range(16)]


def _block_texels(rgba, w, h, bx, by, ch):
    out = []
    for py in range(4):
        for px in range(4):
            x = min(bx+px, w-1); y = min(by+py, h-1)
            out.append(rgba[(y*w+x)*4+ch])
    return out


def bc4_encode(rgba, w, h, ch=0):
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            out += _bc4_encode_block(_block_texels(rgba, w, h, bx, by, ch))
    return bytes(out)


def bc5_encode(rgba, w, h):
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            out += _bc4_encode_block(_block_texels(rgba, w, h, bx, by, 0))
            out += _bc4_encode_block(_block_texels(rgba, w, h, bx, by, 1))
    return bytes(out)


def bc3_encode(rgba, w, h):
    out = bytearray()
    # alpha block (BC4 on A) + colour block (BC1 on RGB), per 4x4
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            out += _bc4_encode_block(_block_texels(rgba, w, h, bx, by, 3))
            # reuse bc1_encode on a single 4x4 by slicing — simpler: inline
            tex = [(rgba[(min(by+py, h-1)*w+min(bx+px, w-1))*4+0],
                    rgba[(min(by+py, h-1)*w+min(bx+px, w-1))*4+1],
                    rgba[(min(by+py, h-1)*w+min(bx+px, w-1))*4+2])
                   for py in range(4) for px in range(4)]
            mn = [255, 255, 255]; mx = [0, 0, 0]
            for (r, g, b) in tex:
                mn[0] = min(mn[0], r); mn[1] = min(mn[1], g); mn[2] = min(mn[2], b)
                mx[0] = max(mx[0], r); mx[1] = max(mx[1], g); mx[2] = max(mx[2], b)
            c0 = _to565(*mx); c1 = _to565(*mn)
            if c0 < c1:
                c0, c1 = c1, c0
            if c0 == c1:
                out += struct.pack("<HHI", c0, c1, 0); continue
            p0 = _rgb565(c0); p1 = _rgb565(c1)
            pal = [p0, p1,
                   ((2*p0[0]+p1[0])//3, (2*p0[1]+p1[1])//3, (2*p0[2]+p1[2])//3),
                   ((p0[0]+2*p1[0])//3, (p0[1]+2*p1[1])//3, (p0[2]+2*p1[2])//3)]
            idx = 0
            for i, t in enumerate(tex):
                best = 0; bd = 1 << 30
                for k in range(4):
                    dr = t[0]-pal[k][0]; dg = t[1]-pal[k][1]; db = t[2]-pal[k][2]
                    d = dr*dr+dg*dg+db*db
                    if d < bd:
                        bd = d; best = k
                idx |= best << (2*i)
            out += struct.pack("<HHI", c0, c1, idx)
    return bytes(out)


def bc3_decode(data, w, h):
    need = max(1, (w+3)//4) * max(1, (h+3)//4) * 16
    if len(data) < need:
        data = bytes(data) + b"\x00" * (need - len(data))
    out = bytearray(w*h*4); p = 0
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            alpha = _bc4_decode_block(data, p); p += 8
            rgb = bc1_decode(data[p:p+8], 4, 4); p += 8
            for py in range(4):
                for px in range(4):
                    x = bx+px; y = by+py
                    if x < w and y < h:
                        o = (y*w+x)*4; s = (py*4+px)*4
                        out[o] = rgb[s]; out[o+1] = rgb[s+1]; out[o+2] = rgb[s+2]
                        out[o+3] = alpha[py*4+px]
    return out


def bc4_decode(data, w, h):
    need = max(1, (w+3)//4) * max(1, (h+3)//4) * 8
    if len(data) < need:
        data = bytes(data) + b"\x00" * (need - len(data))
    out = bytearray(w*h*4); p = 0
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            red = _bc4_decode_block(data, p); p += 8
            for py in range(4):
                for px in range(4):
                    x = bx+px; y = by+py
                    if x < w and y < h:
                        o = (y*w+x)*4; v = red[py*4+px]
                        out[o] = v; out[o+1] = v; out[o+2] = v; out[o+3] = 255
    return out


def bc5_decode(data, w, h):
    need = max(1, (w+3)//4) * max(1, (h+3)//4) * 16
    if len(data) < need:
        data = bytes(data) + b"\x00" * (need - len(data))
    out = bytearray(w*h*4); p = 0
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            red = _bc4_decode_block(data, p); p += 8
            grn = _bc4_decode_block(data, p); p += 8
            for py in range(4):
                for px in range(4):
                    x = bx+px; y = by+py
                    if x < w and y < h:
                        o = (y*w+x)*4
                        out[o] = red[py*4+px]; out[o+1] = grn[py*4+px]
                        out[o+2] = 0; out[o+3] = 255
    return out



_W2 = [0, 21, 43, 64]
_W3 = [0, 9, 18, 27, 37, 46, 55, 64]
_W4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64]
_WEIGHTS = {2: _W2, 3: _W3, 4: _W4}

# mode params: (ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2)
_MODES = {
    0: (3, 4, 0, 0, 4, 0, 1, 0, 3, 0), 1: (2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    2: (3, 6, 0, 0, 5, 0, 0, 0, 2, 0), 3: (2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    4: (1, 0, 2, 1, 5, 6, 0, 0, 2, 3), 5: (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    6: (1, 0, 0, 0, 7, 7, 1, 0, 4, 0), 7: (2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
}


def _unq(v, bits):
    if bits >= 8:
        return v & 0xFF
    return ((v << (8 - bits)) | (v >> (2 * bits - 8))) & 0xFF


def _interp(e0, e1, w):
    return ((64 - w) * e0 + w * e1 + 32) >> 6


# --- BC7 subset partition tables (canonical, from the BC7 spec) --------------
# Per-texel subset index for 2-subset and 3-subset partitioned modes.
_BC7_P2 = (
    (0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1),(0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1),
    (0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1),(0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1),
    (0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1),(0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1),
    (0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1),(0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1),
    (0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1),(0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1),
    (0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1),(0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1),
    (0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1),(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1),
    (0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1),(0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1),
    (0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1),(0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0),
    (0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0),(0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0),
    (0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0),(0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0),
    (0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0),(0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1),
    (0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0),(0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0),
    (0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0),(0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0),
    (0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0),(0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0),
    (0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0),(0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0),
    (0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1),(0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1),
    (0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0),(0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0),
    (0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0),(0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0),
    (0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1),(0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1),
    (0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0),(0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0),
    (0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0),(0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0),
    (0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0),(0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1),
    (0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1),(0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0),
    (0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0),(0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0),
    (0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0),(0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0),
    (0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1),(0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1),
    (0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0),(0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0),
    (0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1),(0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1),
    (0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1),(0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1),
    (0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1),(0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0),
    (0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0),(0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1),
)
_BC7_P3 = (
    (0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2),(0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1),
    (0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1),(0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1),
    (0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2),(0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2),
    (0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1),(0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1),
    (0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2),(0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2),
    (0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2),(0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2),
    (0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2),(0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2),
    (0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2),(0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0),
    (0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2),(0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0),
    (0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2),(0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1),
    (0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2),(0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1),
    (0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2),(0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0),
    (0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0),(0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2),
    (0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0),(0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1),
    (0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2),(0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2),
    (0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1),(0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1),
    (0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2),(0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1),
    (0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2),(0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0),
    (0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0),(0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0),
    (0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0),(0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1),
    (0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1),(0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2),
    (0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1),(0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2),
    (0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1),(0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1),
    (0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1),(0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1),
    (0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2),(0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1),
    (0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2),(0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2),
    (0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2),(0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2),
    (0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2),(0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2),
    (0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2),(0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2),
    (0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2),(0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2),
    (0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1),(0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2),
    (0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2),(0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0),
)
# fix-up (anchor) index per partition: where each subset's index drops its MSB
_BC7_A2 = (
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,15,15,15,15,15, 2, 2,15)
_BC7_A3a = (
     3, 3,15,15, 8, 3,15,15, 8, 8, 6, 6, 6, 5, 3, 3,
     3, 3, 8,15, 3, 3, 6,10, 5, 8, 8, 6, 8, 5,15,15,
     8,15, 3, 5, 6,10, 8,15,15, 3,15, 5,15,15,15,15,
     3,15, 5, 5, 5, 8, 5,10, 5,10, 8,13,15,12, 3, 3)
_BC7_A3b = (
    15, 8, 8, 3,15,15, 3, 8,15,15,15,15,15,15,15, 8,
    15, 8,15, 3,15, 8,15, 8, 3,15, 6,10,15,15,10, 8,
    15, 3,15,10,10, 8, 9,10, 6,15, 8,15, 3, 6, 6, 8,
    15, 3,15,15,15,15,15,15,15,15,15,15, 3,15,15, 8)


def bc7_decode(data, w, h):
    out = bytearray(w * h * 4)
    nbx = (w + 3) // 4
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            off = ((by // 4) * nbx + bx // 4) * 16
            blk = data[off:off + 16]
            if len(blk) < 16:
                blk = bytes(blk) + b"\x00" * (16 - len(blk))
            texels = _decode_block(blk)
            for py in range(4):
                for px in range(4):
                    x = bx + px; y = by + py
                    if x < w and y < h:
                        out[(y * w + x) * 4:(y * w + x) * 4 + 4] = bytes(texels[py * 4 + px])
    return out


def _decode_block(blk):
    val = int.from_bytes(blk, "little")
    mode = 0
    while mode < 8 and not (val >> mode) & 1:
        mode += 1
    if mode == 8:
        return [[0, 0, 0, 255]] * 16
    pos = mode + 1

    def get(n):
        nonlocal pos
        r = (val >> pos) & ((1 << n) - 1)
        pos += n
        return r

    ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 = _MODES[mode]
    part = get(pb) if pb else 0
    rot = get(rb) if rb else 0
    idxmode = get(isb) if isb else 0
    ne = ns * 2
    r = [get(cb) for _ in range(ne)]
    g = [get(cb) for _ in range(ne)]
    b = [get(cb) for _ in range(ne)]
    a = [get(ab) for _ in range(ne)] if ab else [0] * ne
    pbits = []
    if epb:
        pbits = [get(1) for _ in range(ne)]
    elif spb:
        sp = [get(1) for _ in range(ns)]
        pbits = [sp[i // 2] for i in range(ne)]
    ep = []
    for i in range(ne):
        if pbits:
            cr = _unq((r[i] << 1) | pbits[i], cb + 1)
            cg = _unq((g[i] << 1) | pbits[i], cb + 1)
            cbv = _unq((b[i] << 1) | pbits[i], cb + 1)
            ca = _unq((a[i] << 1) | pbits[i], ab + 1) if ab else 255
        else:
            cr = _unq(r[i], cb); cg = _unq(g[i], cb); cbv = _unq(b[i], cb)
            ca = _unq(a[i], ab) if ab else 255
        ep.append([cr, cg, cbv, ca])

    # Partitioned modes (0/1/2/3/7): each texel belongs to a subset chosen by
    # the partition table, and is interpolated between that subset's two
    # endpoints. Each subset's anchor texel stores one fewer index bit.
    if ns > 1:
        if ns == 2:
            psel = _BC7_P2[part]
            anchors = (0, _BC7_A2[part])
        else:
            psel = _BC7_P3[part]
            anchors = (0, _BC7_A3a[part], _BC7_A3b[part])
        wt = _WEIGHTS[ib]
        idx = [0] * 16
        for p in range(16):
            idx[p] = get(ib - (1 if p in anchors else 0))
        texels = []
        for p in range(16):
            s = psel[p]
            e0 = ep[2 * s]; e1 = ep[2 * s + 1]
            w = wt[idx[p]]
            texels.append([_interp(e0[0], e1[0], w), _interp(e0[1], e1[1], w),
                           _interp(e0[2], e1[2], w),
                           _interp(e0[3], e1[3], w) if ab else 255])
        return texels

    wt1 = _WEIGHTS[ib]
    wt2 = _WEIGHTS[ib2] if ib2 else None
    idx1 = [0] * 16; idx2 = [0] * 16
    for p in range(16):
        idx1[p] = get(ib - (1 if p == 0 else 0))
    if ib2:
        for p in range(16):
            idx2[p] = get(ib2 - (1 if p == 0 else 0))

    e0, e1 = ep[0], ep[1]
    texels = []
    for p in range(16):
        if ib2:
            if idxmode == 0:
                ci, ai, cw, aw = idx1[p], idx2[p], wt1, wt2
            else:
                ci, ai, cw, aw = idx2[p], idx1[p], wt2, wt1
            px = [_interp(e0[0], e1[0], cw[ci]), _interp(e0[1], e1[1], cw[ci]),
                  _interp(e0[2], e1[2], cw[ci]), _interp(e0[3], e1[3], aw[ai])]
        else:
            wv = wt1[idx1[p]]
            px = [_interp(e0[0], e1[0], wv), _interp(e0[1], e1[1], wv),
                  _interp(e0[2], e1[2], wv), _interp(e0[3], e1[3], wv) if ab else 255]
        if rot == 1:
            px[0], px[3] = px[3], px[0]
        elif rot == 2:
            px[1], px[3] = px[3], px[1]
        elif rot == 3:
            px[2], px[3] = px[3], px[2]
        texels.append(px)
    return texels


def bc7_encode(rgba, w, h):
    """Mode 6: single subset, 7-bit+pbit RGBA endpoints, 4-bit indices."""
    out = bytearray()
    for by in range(0, h, 4):
        for bx in range(0, w, 4):
            tex = []
            for py in range(4):
                for px in range(4):
                    x = min(bx + px, w - 1); y = min(by + py, h - 1)
                    o = (y * w + x) * 4
                    tex.append((rgba[o], rgba[o+1], rgba[o+2], rgba[o+3]))
            out += _encode_block_mode6(tex)
    return bytes(out)


def _encode_block_mode6(tex):
    e0 = [min(t[c] for t in tex) for c in range(4)]
    e1 = [max(t[c] for t in tex) for c in range(4)]

    def quant(ep):
        best = None
        for pbit in (0, 1):
            q = [max(0, min(127, (v - pbit + 1) >> 1)) for v in ep]
            rec = [((qi << 1) | pbit) for qi in q]
            err = sum((rec[c] - ep[c]) ** 2 for c in range(4))
            if best is None or err < best[0]:
                best = (err, q, pbit, rec)
        return best[1], best[2], best[3]

    q0, p0, rec0 = quant(e0)
    q1, p1, rec1 = quant(e1)
    d = [rec1[c] - rec0[c] for c in range(4)]
    dd = sum(x * x for x in d) or 1
    idx = []
    for t in tex:
        num = sum((t[c] - rec0[c]) * d[c] for c in range(4))
        k = int(round(num / dd * 15))
        idx.append(0 if k < 0 else (15 if k > 15 else k))
    if idx[0] >= 8:
        q0, q1 = q1, q0
        p0, p1 = p1, p0
        idx = [15 - k for k in idx]
    bits = 0; pos = 0

    def put(v, n):
        nonlocal bits, pos
        bits |= (v & ((1 << n) - 1)) << pos
        pos += n

    put(1 << 6, 7)
    for c in range(4):
        put(q0[c], 7); put(q1[c], 7)
    put(p0, 1); put(p1, 1)
    put(idx[0], 3)
    for i in range(1, 16):
        put(idx[i], 4)
    return bits.to_bytes(16, "little")


_BC_DECODE = {0x4C: bc1_decode, 0x52: bc3_decode, 0x55: bc4_decode,
              0x58: bc5_decode, 0x5E: bc7_decode}
_BC_ENCODE = {0x4C: bc1_encode, 0x52: bc3_encode, 0x55: bc4_encode,
              0x58: bc5_encode, 0x5E: bc7_encode}


def write_png(path, w, h, rgba):
    import zlib as _z
    raw = bytearray()
    for y in range(h):
        raw.append(0); raw += rgba[y*w*4:(y*w+w)*4]

    def ch(t, da):
        c = t+da
        return struct.pack(">I", len(da))+c+struct.pack(">I", _z.crc32(c) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n"
                + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + ch(b"IDAT", _z.compress(bytes(raw), 9)) + ch(b"IEND", b""))


def write_tga(path, w, h, rgba):
    hdr = bytearray(18)
    hdr[2] = 2; hdr[12] = w & 0xFF; hdr[13] = (w >> 8) & 0xFF
    hdr[14] = h & 0xFF; hdr[15] = (h >> 8) & 0xFF; hdr[16] = 32; hdr[17] = 0x20
    body = bytearray(w*h*4)
    for i in range(w*h):
        body[i*4] = rgba[i*4+2]; body[i*4+1] = rgba[i*4+1]
        body[i*4+2] = rgba[i*4]; body[i*4+3] = rgba[i*4+3]
    with open(path, "wb") as f:
        f.write(bytes(hdr) + bytes(body))


def _pad(buf, n):
    return buf if len(buf) >= n else (bytes(buf) + bytes(n - len(buf)))


def _mip_bytes(chunk, unc_size):
    """A mip's pixels: stored raw when LZ4 couldn't shrink it (compressed size >=
    uncompressed), otherwise LZ4-decompressed. Always padded to unc_size."""
    if len(chunk) >= unc_size:
        return _pad(bytes(chunk[:unc_size]), unc_size)
    return _pad(lz4_decompress(chunk, unc_size), unc_size)


def _smoothness(rgba, w, h):
    """Mean absolute difference between horizontally adjacent pixels. A texture
    decoded with the WRONG block format looks like noise (high value); the right
    format is smooth (low value)."""
    s = 0; cnt = 0
    step = max(1, (w * h) // 4096)        # sample for speed on big mips
    for y in range(0, h):
        row = y * w * 4
        for x in range(1, w, step):
            o = row + x * 4; p = row + (x - 1) * 4
            s += abs(rgba[o]-rgba[p]) + abs(rgba[o+1]-rgba[p+1]) + abs(rgba[o+2]-rgba[p+2])
            cnt += 3
    return s / cnt if cnt else 0


def _smallest_text_mip(text, info, fmt):
    """Decode the smallest mip stored in the TEXT with the given format -> rgba,
    used cheaply for format auto-detection."""
    dec = _BC_DECODE.get(fmt)
    if dec is None:
        return None
    blk = 8 if fmt in (0x4C, 0x55) else 16
    ts = text[0x91]
    atlas = struct.unpack_from("<I", text, 0x88)[0]
    start = 0x98 + atlas
    n = info["mips"]
    last = n - 1                          # smallest mip index
    w = max(1, info["width"] >> last); h = max(1, info["height"] >> last)
    prev = info["mip_offsets_compressed"][last-1] if last >= 1 else 0
    size_c = info["mip_offsets_compressed"][last] - prev
    sb = max(1, (w+3)//4) * max(1, (h+3)//4) * blk
    # offset of this mip within the TEXT payload
    text_first = ts                       # first mip stored in TEXT
    off = start + (info["mip_offsets_compressed"][last-1] -
                   (info["mip_offsets_compressed"][text_first-1] if text_first >= 1 else 0)) \
        if last >= 1 else start
    try:
        return w, h, dec(_mip_bytes(text[off:off+size_c], sb), w, h)
    except Exception:
        return None


def detect_texture_format(text):
    """Guess the block format from the header's mip sizes + image smoothness, for
    files whose format code we don't recognise. Returns a fmt_code or None."""
    info = parse_text_header(text)
    if info is None:
        return None
    w, h = info["width"], info["height"]
    blocks = max(1, (w+3)//4) * max(1, (h+3)//4)
    mip0 = info["mip_offsets_uncompressed"][0]
    bpb = mip0 / blocks if blocks else 0
    if abs(bpb - 8) < 0.5:
        cands = [0x4C, 0x55]              # BC1, BC4
    elif abs(bpb - 16) < 0.5:
        cands = [0x58, 0x52, 0x5E]        # BC5, BC3, BC7
    else:
        return None
    best = None
    for fmt in cands:
        r = _smallest_text_mip(text, info, fmt)
        if r is None:
            continue
        sm = _smoothness(r[2], r[0], r[1])
        if best is None or sm < best[0]:
            best = (sm, fmt)
    return best[1] if best else None


def decode_texture_file(text_path, texd_path=None, force_fmt=None):
    """Decode a .TEXT (+ optional .TEXD for full-res) to (w, h, rgba). Picks the
    largest mip available. Supports BC1/BC3/BC4/BC5/BC7. If the header's format
    code is unrecognised, the format is auto-detected from the data."""
    text = bytearray(open(text_path, "rb").read())
    info = parse_text_header(text)
    if info is None:
        raise ValueError("not a valid 007 .TEXT")
    fmt = force_fmt if force_fmt else info["format"]
    if fmt not in _BC_DECODE:
        guess = detect_texture_format(text)
        if guess is not None:
            fmt = guess
    dec = _BC_DECODE.get(fmt)
    if dec is None:
        raise ValueError("format 0x%X not recognised and couldn't be auto-detected "
                         "(supported: BC1/BC3/BC4/BC5/BC7)" % info["format"])
    w, h = info["width"], info["height"]
    blk = 8 if fmt in (0x4C, 0x55) else 16
    full_bytes = max(1, (w+3)//4) * max(1, (h+3)//4) * blk
    # the full-res mip0 lives in the .TEXD; fall back to the largest TEXT mip
    if texd_path and os.path.exists(texd_path):
        td = open(texd_path, "rb").read()
        end = info["mip_offsets_compressed"][0] or len(td)
        return w, h, dec(_mip_bytes(td[:end], full_bytes), w, h)
    # no TEXD: decode the first (largest) mip stored in the TEXT itself
    ts = text[0x91]
    atlas = struct.unpack_from("<I", text, 0x88)[0]
    start = 0x98 + atlas
    prev = info["mip_offsets_compressed"][ts-1] if ts >= 1 else 0
    size_c = info["mip_offsets_compressed"][ts] - prev
    sw = max(1, w >> ts); sh = max(1, h >> ts)
    sb = max(1, (sw+3)//4) * max(1, (sh+3)//4) * blk
    return sw, sh, dec(_mip_bytes(text[start:start+size_c], sb), sw, sh)


def lz4_decompress_consumed(data, out_size):
    """Like lz4_decompress but also reports how many input bytes were consumed, so
    we can walk the concatenated LZ4 mips inside a headerless .TEXD. Returns
    (bytes, consumed) or (None, consumed) on malformed input."""
    out = bytearray(); i = 0; n = len(data)
    try:
        while len(out) < out_size and i < n:
            tok = data[i]; i += 1; lit = tok >> 4
            if lit == 15:
                while i < n:
                    b = data[i]; i += 1; lit += b
                    if b != 255:
                        break
            out += data[i:i+lit]; i += lit
            if len(out) >= out_size or i + 2 > n:
                break
            off = data[i] | (data[i+1] << 8); i += 2; ml = tok & 15
            if ml == 15:
                while i < n:
                    b = data[i]; i += 1; ml += b
                    if b != 255:
                        break
            ml += 4; s = len(out) - off
            if s < 0 or off == 0:
                return None, i
            for k in range(ml):
                out.append(out[s+k])
    except Exception:
        return None, i
    return bytes(out[:out_size]), i


def _texd_walk_consumes(data, bpb, W, H):
    """True if a full mip chain of (W,H) at bpb bytes/block decompresses to consume
    the whole .TEXD exactly."""
    pos = 0; w = W; h = H
    while True:
        blk = max(1, (w+3)//4) * max(1, (h+3)//4) * bpb
        if pos > len(data):
            return False
        r, c = lz4_decompress_consumed(data[pos:], blk)
        if r is None or len(r) < blk:
            return False
        pos += c
        if w <= 1 and h <= 1:
            break
        w = max(1, w >> 1); h = max(1, h >> 1)
    return pos == len(data)


def detect_texd_geometry(data):
    """Headerless .TEXD -> (bytes_per_block, W, H) whose mip chain consumes the file
    exactly, or None. 8 bytes/block = BC1/BC4, 16 = BC3/BC5/BC7."""
    found = []
    for bpb in (8, 16):
        for sw in range(2, 14):
            for sh in range(2, 14):
                if abs(sw - sh) > 3:
                    continue
                W = 1 << sw; H = 1 << sh
                if _texd_walk_consumes(data, bpb, W, H):
                    found.append((bpb, W, H))
    if not found:
        return None
    found.sort(key=lambda t: (abs(t[1] - t[2]), -(t[1] * t[2])))
    return found[0]


def decode_texd_standalone(texd_path):
    """Decode a .TEXD that has NO matching .TEXT (so no header). Auto-detects the
    dimensions and block format from the data. Returns (w, h, rgba, fmt_code)."""
    data = open(texd_path, "rb").read()
    geo = detect_texd_geometry(data)
    if geo is None:
        raise ValueError("no .TEXT header and couldn't auto-detect .TEXD geometry")
    bpb, W, H = geo
    cands = [0x4C, 0x55] if bpb == 8 else [0x58, 0x52, 0x5E]
    # walk to a small mip and pick the format that decodes smoothest
    pos = 0; w = W; h = H
    while w > 64 and h > 64:
        blk = max(1, (w+3)//4) * max(1, (h+3)//4) * bpb
        _, c = lz4_decompress_consumed(data[pos:], blk); pos += c
        w = max(1, w >> 1); h = max(1, h >> 1)
    blk = max(1, (w+3)//4) * max(1, (h+3)//4) * bpb
    smip, _ = lz4_decompress_consumed(data[pos:], blk)
    best = None
    if smip is not None:
        for fmt in cands:
            dec = _BC_DECODE.get(fmt)
            if dec is None:
                continue
            try:
                sm = _smoothness(dec(_pad(smip, blk), w, h), w, h)
            except Exception:
                continue
            if best is None or sm < best[0]:
                best = (sm, fmt)
    fmt = best[1] if best else cands[0]
    dec = _BC_DECODE[fmt]
    blk0 = max(1, (W+3)//4) * max(1, (H+3)//4) * bpb
    mip0, _ = lz4_decompress_consumed(data, blk0)
    return W, H, dec(_pad(mip0 or b"", blk0), W, H), fmt


def index_textures_by_hash(search_dirs):
    """Walk search_dirs once and map hash -> path for every .TEXT and .TEXD."""
    text_by, texd_by = {}, {}
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for dirpath, _dirs, files in os.walk(d, onerror=lambda e: None,
                                                 followlinks=False):
                for fn in files:
                    low = fn.lower()
                    full = os.path.join(dirpath, fn)
                    h = hash_from_path(full)
                    if not h:
                        continue
                    if low.endswith(".text"):
                        text_by.setdefault(h.upper(), full)
                    elif low.endswith(".texd"):
                        texd_by.setdefault(h.upper(), full)
        except Exception:
            continue
    return text_by, texd_by


def pair_text_to_texd(search_dirs):
    """Scan every .meta under search_dirs and, for each .TEXT meta, map the TEXT
    hash -> its .TEXD hash (the 0x9F reference), or -> None when the texture has no
    .TEXD. Reads the pairing straight from the meta's bytes, so it works no matter
    where the .TEXT file lives or how its meta is named (dot or underscore)."""
    pairs = {}
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for dirpath, _dirs, files in os.walk(d, onerror=lambda e: None,
                                                 followlinks=False):
                for fn in files:
                    if not fn.lower().endswith(".meta"):
                        continue
                    try:
                        mm = parse_meta(bytearray(open(os.path.join(dirpath, fn),
                                                       "rb").read()))
                    except Exception:
                        continue
                    if mm.get("ext_raw") != b"TEXT":
                        continue
                    thash = "%016X" % mm["resource_id"]
                    refs = mm.get("refs", [])
                    texd = None
                    for (rh, fl) in refs:
                        if fl == 0x9F and rh != mm["resource_id"]:
                            texd = rh
                            break
                    if texd is None:
                        for (rh, fl) in refs:
                            if rh != mm["resource_id"]:   # never pair a TEXT to itself
                                texd = rh
                                break
                    pairs.setdefault(thash, texd)
        except Exception:
            continue
    return pairs


def organize_texture_dest(base, filename, organize):
    """Output path for a texture file. When `organize`, nest it in TYPE/<hash>/ so
    the .TEXT and .TEXD (with DIFFERENT hashes) land in their own separate folders;
    otherwise write flat into `base`."""
    if not organize:
        return os.path.join(base, filename)
    core = filename
    for suf in (".meta.json", ".meta"):
        if core.endswith(suf):
            core = core[:-len(suf)]
            break
    if "." in core and not core.endswith("."):
        ext = core.rsplit(".", 1)[1]
    elif "_" in core:
        ext = core.rsplit("_", 1)[1]
    else:
        ext = "MISC"
    cuts = [i for i in (filename.find("."), filename.find("_")) if i >= 0]
    hashname = filename[:min(cuts)] if cuts else os.path.splitext(filename)[0]
    folder = os.path.join(base, ext.upper(), hashname)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


def write_texture_pair(out_dir, thash, tdata, texd_data, meta_tmpl=None,
                       dest_fn=None, texd_hash=None):
    """Write <hash>.TEXT + <hash>.TEXD and their game-valid .meta (+ .json). The
    TEXT meta references the TEXD (flag 0x9F) and carries size_video, or the game
    crashes. Returns (text_path, texd_path)."""
    import json as _json
    dest = dest_fn or (lambda fn: os.path.join(out_dir, fn))
    info = parse_text_header(bytearray(tdata))
    tm = None
    if meta_tmpl:
        cm = derive_resource_meta(meta_tmpl, "TEXT")
        if cm and os.path.exists(cm):
            try:
                tm = parse_meta(bytearray(open(cm, "rb").read()))
            except Exception:
                tm = None
    if tm is None:
        tm = {"resource_id": thash, "data_offset": 0xFFFFFFFFFFFFFFFF,
              "size_raw": 0x80000000, "ext_raw": b"TEXT",
              "size_uncompressed": len(tdata), "size_memory": 0xFFFFFFFF,
              "size_video": 0, "dummy": 0xC000, "refs": []}
    tm["resource_id"] = thash
    tm["size_uncompressed"] = len(tdata)
    if not tm.get("dummy"):
        tm["dummy"] = 0xC000
    # The .TEXD must be written under ITS OWN hash (different from the .TEXT), and
    # the .TEXT meta must reference it. Priority: explicit texd_hash from the caller,
    # then the template meta's 0x9F ref, then (last resort) the TEXT hash.
    if texd_hash is None:
        texd_hash = tm["refs"][0][0] if tm.get("refs") else thash
    flag = tm["refs"][0][1] if tm.get("refs") else 0x9F
    tm["refs"] = [(texd_hash, flag)] + (list(tm["refs"][1:]) if tm.get("refs") else [])
    if info is not None:
        sc_ts = bytearray(tdata)[0x91]
        mu = info["mip_offsets_uncompressed"]
        tm["size_video"] = mu[sc_ts-1] if sc_ts >= 1 else mu[0]
    dot = meta_uses_dot_style(meta_tmpl, "TEXT") if meta_tmpl else True
    out_tex = dest("%016X.TEXT" % thash)
    with open(out_tex, "wb") as f:
        f.write(tdata)
    tmeta = dest(meta_out_name(thash, "TEXT", dot))
    with open(tmeta, "wb") as f:
        f.write(build_meta_binary(tm, []))
    with open(tmeta + ".json", "w") as f:
        _json.dump(build_meta_json(tm, []), f, indent=2)
    dm = {"resource_id": texd_hash, "data_offset": 0xFFFFFFFFFFFFFFFF,
          "size_raw": 0, "ext_raw": b"TEXD", "size_uncompressed": len(texd_data),
          "size_memory": 0xFFFFFFFF, "size_video": tm["size_video"],
          "dummy": 0, "refs": []}
    out_texd = dest("%016X.TEXD" % texd_hash)
    with open(out_texd, "wb") as f:
        f.write(texd_data)
    dmeta = dest(meta_out_name(texd_hash, "TEXD", dot))
    with open(dmeta, "wb") as f:
        f.write(build_meta_binary(dm, []))
    with open(dmeta + ".json", "w") as f:
        _json.dump(build_meta_json(dm, []), f, indent=2)
    return out_tex, out_texd


def write_texture_only(out_dir, thash, tdata, meta_tmpl=None, dest_fn=None):
    """Write just a .TEXT (+ meta) with NO .TEXD - for tiny single-mip textures that
    never had a high-res half. The meta drops any TEXD (0x9F) ref. Returns the
    .TEXT path."""
    import json as _json
    dest = dest_fn or (lambda fn: os.path.join(out_dir, fn))
    tm = None
    if meta_tmpl:
        cm = derive_resource_meta(meta_tmpl, "TEXT")
        if cm and os.path.exists(cm):
            try:
                tm = parse_meta(bytearray(open(cm, "rb").read()))
            except Exception:
                tm = None
    if tm is None:
        tm = {"resource_id": thash, "data_offset": 0xFFFFFFFFFFFFFFFF,
              "size_raw": 0x80000000, "ext_raw": b"TEXT",
              "size_uncompressed": len(tdata), "size_memory": 0xFFFFFFFF,
              "size_video": 0, "dummy": 0xC000, "refs": []}
    tm["resource_id"] = thash
    tm["size_uncompressed"] = len(tdata)
    if not tm.get("dummy"):
        tm["dummy"] = 0xC000
    tm["refs"] = [(h, f) for (h, f) in tm.get("refs", []) if f != 0x9F]
    tm["size_video"] = 0
    dot = meta_uses_dot_style(meta_tmpl, "TEXT") if meta_tmpl else True
    out_tex = dest("%016X.TEXT" % thash)
    with open(out_tex, "wb") as f:
        f.write(tdata)
    tmeta = dest(meta_out_name(thash, "TEXT", dot))
    with open(tmeta, "wb") as f:
        f.write(build_meta_binary(tm, []))
    with open(tmeta + ".json", "w") as f:
        _json.dump(build_meta_json(tm, []), f, indent=2)
    return out_tex


def read_png(path):
    import zlib as _z
    d = open(path, "rb").read()
    if d[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    i = 8; w = h = colt = bitd = 0; idat = bytearray(); pal = None; trns = None
    while i < len(d):
        ln = struct.unpack_from(">I", d, i)[0]; typ = d[i+4:i+8]
        ch = d[i+8:i+8+ln]; i += 12+ln
        if typ == b"IHDR":
            w, h, bitd, colt = struct.unpack_from(">IIBB", ch, 0)[:4]
        elif typ == b"PLTE":
            pal = ch
        elif typ == b"tRNS":
            trns = ch
        elif typ == b"IDAT":
            idat += ch
        elif typ == b"IEND":
            break
    if bitd != 8:
        raise ValueError("only 8-bit PNG supported")
    raw = _z.decompress(bytes(idat))
    ch_n = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colt]
    bpp = max(1, ch_n); stride = w*ch_n
    out = bytearray(w*h*4); prev = bytearray(stride); pos = 0

    def pae(a, b, c):
        p = a+b-c; pa_ = abs(p-a); pb = abs(p-b); pc = abs(p-c)
        return a if (pa_ <= pb and pa_ <= pc) else (b if pb <= pc else c)
    for y in range(h):
        f = raw[pos]; pos += 1; line = bytearray(raw[pos:pos+stride]); pos += stride
        for x in range(stride):
            a = line[x-bpp] if x >= bpp else 0; b = prev[x]
            c = prev[x-bpp] if x >= bpp else 0
            if f == 1:
                line[x] = (line[x]+a) & 255
            elif f == 2:
                line[x] = (line[x]+b) & 255
            elif f == 3:
                line[x] = (line[x]+((a+b) >> 1)) & 255
            elif f == 4:
                line[x] = (line[x]+pae(a, b, c)) & 255
        prev = line
        for x in range(w):
            o = (y*w+x)*4
            if colt == 6:
                out[o:o+4] = line[x*4:x*4+4]
            elif colt == 2:
                out[o] = line[x*3]; out[o+1] = line[x*3+1]; out[o+2] = line[x*3+2]; out[o+3] = 255
            elif colt == 0:
                g = line[x]; out[o] = g; out[o+1] = g; out[o+2] = g; out[o+3] = 255
            elif colt == 4:
                g = line[x*2]; out[o] = g; out[o+1] = g; out[o+2] = g; out[o+3] = line[x*2+1]
            elif colt == 3:
                ix = line[x]; out[o] = pal[ix*3]; out[o+1] = pal[ix*3+1]; out[o+2] = pal[ix*3+2]
                out[o+3] = trns[ix] if (trns and ix < len(trns)) else 255
    return w, h, bytes(out)


def read_tga(path):
    d = open(path, "rb").read()
    idlen = d[0]; imgtype = d[2]
    w, h = struct.unpack_from("<HH", d, 12); bpp = d[16]; desc = d[17]
    off = 18+idlen; topdown = bool(desc & 0x20); nb = bpp//8
    out = bytearray(w*h*4); px_list = []
    if imgtype in (2, 3):
        for k in range(w*h):
            px_list.append(d[off+k*nb:off+k*nb+nb])
    elif imgtype in (10, 11):
        i = off
        while len(px_list) < w*h:
            p = d[i]; i += 1; cnt = (p & 0x7F)+1
            if p & 0x80:
                px = d[i:i+nb]; i += nb; px_list.extend([px]*cnt)
            else:
                for _ in range(cnt):
                    px_list.append(d[i:i+nb]); i += nb
    else:
        raise ValueError("unsupported TGA type %d" % imgtype)
    for k, px in enumerate(px_list):
        x = k % w; y = k // w; yy = y if topdown else (h-1-y)
        o = (yy*w+x)*4
        if nb >= 3:
            out[o] = px[2]; out[o+1] = px[1]; out[o+2] = px[0]
            out[o+3] = px[3] if nb == 4 else 255
        else:
            g = px[0]; out[o] = g; out[o+1] = g; out[o+2] = g; out[o+3] = 255
    return w, h, bytes(out)


def read_image_rgba(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return read_png(path)
    if ext == ".tga":
        return read_tga(path)
    raise ValueError("only .png and .tga are supported natively")


def _resize_nn(rgba, w, h, nw, nh):
    if (w, h) == (nw, nh):
        return rgba
    out = bytearray(nw*nh*4)
    for y in range(nh):
        sy = y*h//nh
        for x in range(nw):
            sx = x*w//nw; s = (sy*w+sx)*4; o = (y*nw+x)*4
            out[o:o+4] = rgba[s:s+4]
    return bytes(out)


def build_texture_v4(rgba, w, h, template, fmt_code=0x4C):
    """Build (text_bytes, texd_bytes) from RGBA. `template` supplies the header
    fields that must stay game-valid (type/flags/interpret/dims/atlas/scale).
    fmt_code selects BC1/BC3/BC4/BC5."""
    num = template["num_mips"]
    rgba = _resize_nn(rgba, w, h, template["width"], template["height"])
    w, h = template["width"], template["height"]
    mips = _gen_mips(rgba, w, h, num)
    encoder = _BC_ENCODE.get(fmt_code, bc1_encode)
    enc = [encoder(m, mw, mh) for (m, mw, mh) in mips]
    comp = [lz4_compress(e) for e in enc]
    mip_unc = []; acc = 0
    for e in enc:
        acc += len(e); mip_unc.append(acc)
    mip_cmp = []; acc = 0
    for c in comp:
        acc += len(c); mip_cmp.append(acc)
    while len(mip_unc) < 14:
        mip_unc.append(0)
    while len(mip_cmp) < 14:
        mip_cmp.append(0)
    texd = b"".join(comp)
    ts = template["text_scale"]; atlas_size = template["atlas_size"]
    atlas = template.get("atlas_bytes", b"")
    hdr = bytearray(152)
    struct.pack_into("<H", hdr, 0x00, 1)
    struct.pack_into("<H", hdr, 0x02, template["type_"])
    struct.pack_into("<I", hdr, 0x04, 152 + atlas_size + len(texd))
    struct.pack_into("<I", hdr, 0x08, template["flags"])
    struct.pack_into("<H", hdr, 0x0C, w)
    struct.pack_into("<H", hdr, 0x0E, h)
    struct.pack_into("<H", hdr, 0x10, fmt_code)
    hdr[0x12] = num; hdr[0x13] = template["default_mip"]
    hdr[0x14] = template["interpret"]; hdr[0x15] = template["dims"]
    for i in range(14):
        struct.pack_into("<I", hdr, 0x18+i*4, mip_unc[i])
    for i in range(14):
        struct.pack_into("<I", hdr, 0x50+i*4, mip_cmp[i])
    struct.pack_into("<I", hdr, 0x88, atlas_size)
    struct.pack_into("<I", hdr, 0x8C, 0x98)
    hdr[0x90] = 0xFF; hdr[0x91] = ts; hdr[0x92] = ts; hdr[0x93] = num - ts
    text = bytes(hdr) + atlas + b"".join(comp[ts:])
    return text, texd


def _text_scale_for(w, h, num_mips):
    import math
    if num_mips <= 1:
        return 0
    if w*h == 16:        # tiny BC1
        return 1
    return max(0, int(math.floor(math.log2(w*h)*0.5 - 6.5)))


def template_from_text(text_bytes):
    """Parse an existing .TEXT into a build template (preserves atlas + fields)."""
    h = parse_text_header(text_bytes)
    if h is None:
        return None
    asz = struct.unpack_from("<I", text_bytes, 0x88)[0]
    return {
        "width": h["width"], "height": h["height"], "num_mips": h["mips"],
        "type_": struct.unpack_from("<H", text_bytes, 0x02)[0],
        "flags": struct.unpack_from("<I", text_bytes, 0x08)[0],
        "default_mip": text_bytes[0x13], "interpret": text_bytes[0x14],
        "dims": text_bytes[0x15], "atlas_size": asz,
        "atlas_bytes": bytes(text_bytes[0x98:0x98+asz]),
        "text_scale": text_bytes[0x91],
    }


def template_from_scratch(w, h):
    import math
    num = int(math.log2(max(w, h))) + 1
    return {"width": w, "height": h, "num_mips": num, "type_": 0, "flags": 0,
            "default_mip": 0, "interpret": 0, "dims": 0, "atlas_size": 0,
            "atlas_bytes": b"", "text_scale": _text_scale_for(w, h, num)}


def convert_image_native(img_path, out_text, out_texd, template_text=None,
                         fmt_code=0x4C):
    """Turn a .png/.tga into a .TEXT (+ .TEXD), pure Python. Returns
    (ok, message). If template_text (an existing .TEXT) is given, its header and
    atlas are preserved and the image is fitted to its size.

    fmt_code=None means 'match the original': use the template's own format when
    it is one we can encode (BC1/BC3/BC4/BC5), else fall back to BC1."""
    try:
        w, h, rgba = read_image_rgba(img_path)
    except Exception as e:
        return False, "image read failed: %s" % e
    warn = ""
    if template_text:
        tmpl = template_from_text(template_text)
        if tmpl is None:
            return False, "template .TEXT is not a valid 007 header"
        if fmt_code is None:
            orig = struct.unpack_from("<H", template_text, 0x10)[0]
            if orig in _BC_ENCODE:
                fmt_code = orig
            else:
                det = detect_texture_format(bytearray(template_text))
                if det in _BC_ENCODE:
                    fmt_code = det
                else:
                    fmt_code = 0x4C
                    warn = (" (original format 0x%X can't be re-encoded yet; used BC1 - "
                            "the game may not accept this, pick a supported format)" % orig)
    else:
        if fmt_code is None:
            fmt_code = 0x4C
        if (w & (w-1)) or (h & (h-1)):
            return False, "image must be power-of-two (e.g. 256x256, 1024x512)"
        tmpl = template_from_scratch(w, h)
    try:
        text, texd = build_texture_v4(rgba, w, h, tmpl, fmt_code)
    except Exception as e:
        return False, "encode failed: %s" % e
    with open(out_text, "wb") as f:
        f.write(text)
    with open(out_texd, "wb") as f:
        f.write(texd)
    fname = {0x4C: "BC1", 0x52: "BC3", 0x55: "BC4", 0x58: "BC5", 0x5E: "BC7"}.get(fmt_code, "BC?")
    return True, "%dx%d %s%s" % (tmpl["width"], tmpl["height"], fname, warn)


# -----------------------------------------------------------------------------
# Experimental custom-topology serializer
#
# Rebuilds the whole .prim from per-object vertex data, so the vertex count and
# topology can change. This is LOSSY relative to template-patching: tangents are
# written as neutral, collision is dropped, and the skin partition (BoneInfo /
# BoneIndices) is regenerated as a SINGLE batch. The single batch is structurally
# plausible (bone_remap is 0xFF -> global bone indices, no palette limit) but is
# NOT verified in-game for skinned meshes. Same vertex count = use the safe path.
# -----------------------------------------------------------------------------
def _align16(b):
    while len(b) % 16:
        b.append(0)


# Max distinct bones referenced by one runtime skin batch. Originals stay <=12;
# the GPU palette overflows above this, so edited objects are packed to this cap.
_SKIN_PALETTE = 12


def _q_i16(val, bias, scale):
    q = (val - bias) / scale if scale else 0.0
    q = max(-1.0, min(1.0, q))
    return int(round(q * 32767.0))


def _enc_normal(n):
    return bytes([_enc_unit_byte(n[0]), _enc_unit_byte(n[1]), _enc_unit_byte(n[2]), 128])


def _weights_to_255(ws):
    iv = [max(0, min(255, int(round(w * 255.0)))) for w in ws]
    k = max(range(4), key=lambda i: ws[i]) if any(ws) else 0
    iv[k] = max(0, min(255, iv[k] + (255 - sum(iv))))
    return iv


def _compute_tangents(positions, normals, uvs, indices):
    """Per-vertex tangent/bitangent from UV gradients (Lengyel), Gram-Schmidt
    orthonormalised against the normal. Without these the skin shader's normal
    mapping degenerates and the mesh renders blank/black."""
    n = len(positions)
    acc = [[0.0, 0.0, 0.0] for _ in range(n)]
    for t in range(0, len(indices) - 2, 3):
        i0, i1, i2 = indices[t], indices[t + 1], indices[t + 2]
        if i0 >= n or i1 >= n or i2 >= n:
            continue
        p0, p1, p2 = positions[i0], positions[i1], positions[i2]
        u0, u1, u2 = uvs[i0], uvs[i1], uvs[i2]
        e1 = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        e2 = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
        du1, dv1 = u1[0] - u0[0], u1[1] - u0[1]
        du2, dv2 = u2[0] - u0[0], u2[1] - u0[1]
        r = du1 * dv2 - du2 * dv1
        f = 1.0 / r if abs(r) > 1e-12 else 0.0
        s = ((e1[0] * dv2 - e2[0] * dv1) * f,
             (e1[1] * dv2 - e2[1] * dv1) * f,
             (e1[2] * dv2 - e2[2] * dv1) * f)
        for i in (i0, i1, i2):
            acc[i][0] += s[0]; acc[i][1] += s[1]; acc[i][2] += s[2]
    tans, bitans = [], []
    for i in range(n):
        nx, ny, nz = normals[i]
        tx, ty, tz = acc[i]
        d = nx * tx + ny * ty + nz * tz
        tx -= nx * d; ty -= ny * d; tz -= nz * d
        l = (tx * tx + ty * ty + tz * tz) ** 0.5
        if l > 1e-8:
            tx /= l; ty /= l; tz /= l
        else:
            tx, ty, tz = (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0)
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        tans.append((tx, ty, tz)); bitans.append((bx, by, bz))
    return tans, bitans


# =============================================================================
# LOD propagation: transfer LOD 0 edits to lower LODs
# =============================================================================
def _nearest_vertex_map(src, dst):
    """For each vertex in *dst*, return the index of the nearest vertex in
    *src*. Uses a spatial grid for O(N) average-case instead of brute O(N²)."""
    if not src or not dst:
        return [0] * len(dst)
    xs = [p[0] for p in src]
    ys = [p[1] for p in src]
    zs = [p[2] for p in src]
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    cell = ext / max(1, int(len(src) ** 0.33))
    if cell < 1e-8:
        cell = 1e-8
    inv = 1.0 / cell
    grid = {}
    for i, p in enumerate(src):
        key = (int(p[0] * inv), int(p[1] * inv), int(p[2] * inv))
        grid.setdefault(key, []).append(i)
    result = []
    for p in dst:
        cx = int(p[0] * inv); cy = int(p[1] * inv); cz = int(p[2] * inv)
        best_d, best_i = float("inf"), 0
        for r in range(1, 4):                      # expand search radius if needed
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        for j in grid.get((cx + dx, cy + dy, cz + dz), []):
                            q = src[j]
                            d = (p[0]-q[0])**2 + (p[1]-q[1])**2 + (p[2]-q[2])**2
                            if d < best_d:
                                best_d = d; best_i = j
            if best_d < float("inf"):
                break                              # found at least one
        result.append(best_i)
    return result


def _propagate_lod_edits(template_data, blender_objs):
    """Compare each LOD 0 object's current Blender positions to its original
    positions in the template .prim. For every lower LOD in the same material
    group, find the nearest LOD 0 vertex and apply the same displacement.

    Returns ``{prim_index: [(x,y,z), ...]}`` for every lower LOD that was
    modified. LOD 0 objects are NOT included (the caller already has their
    current positions from Blender)."""
    orig = read_prim_bytes(template_data)
    # group objects by material; within a group, sort by lod_index
    groups = {}
    for o in blender_objs:
        mat = o.get("glacier_material_id", -1)
        lod = o.get("glacier_lod_index", 0)
        idx = int(o["glacier_prim_index"])
        groups.setdefault(mat, []).append((lod, idx, o))
    for g in groups.values():
        g.sort()

    propagated = {}
    for mat, items in groups.items():
        if len(items) < 2 or items[0][0] != 0:
            continue                               # no LOD 0 or only one LOD
        _, lod0_idx, lod0_obj = items[0]
        if lod0_idx >= len(orig.header.object_table):
            continue
        lod0_sm = orig.header.object_table[lod0_idx]
        lod0_orig = [(v.position[0], v.position[1], v.position[2])
                     for v in lod0_sm.vertexBuffer.vertices]
        lod0_cur = _mesh_vertex_coords(lod0_obj.data)
        n0 = min(len(lod0_orig), len(lod0_cur))
        if n0 == 0:
            continue
        deltas = [(lod0_cur[i][0] - lod0_orig[i][0],
                   lod0_cur[i][1] - lod0_orig[i][1],
                   lod0_cur[i][2] - lod0_orig[i][2]) for i in range(n0)]
        # check if anything actually changed
        if max(abs(d[0]) + abs(d[1]) + abs(d[2]) for d in deltas) < 1e-5:
            continue

        for lod_level, lod_idx, lod_obj in items[1:]:
            if lod_idx >= len(orig.header.object_table):
                continue
            lod_sm = orig.header.object_table[lod_idx]
            lod_orig = [(v.position[0], v.position[1], v.position[2])
                        for v in lod_sm.vertexBuffer.vertices]
            mapping = _nearest_vertex_map(lod0_orig[:n0], lod_orig)
            new_pos = []
            for k, p in enumerate(lod_orig):
                j = mapping[k]
                dd = deltas[j]
                new_pos.append((p[0] + dd[0], p[1] + dd[1], p[2] + dd[2]))
            propagated[lod_idx] = new_pos
    return propagated


def build_custom_prim(template, objects, weighted):
    """Rebuild a .prim with new topology. Unchanged objects keep their original
    vertex buffer + skin partition verbatim (byte-perfect, exactly like the safe
    path); edited objects are regenerated with REAL tangents and the ORIGINAL
    quantisation/bounding box so they decode and shade correctly in-game.
    Each object dict carries: orig_off, sub_type, changed, positions, normals,
    uvs, colors, indices, joints, weights."""
    b = bytearray()
    b += struct.pack("<Q", 0)          # header offset (patched at the end)
    b += struct.pack("<Q", 0)          # padding
    obj_offsets = []
    g_min = [float("inf")] * 3         # grow the global bounds to fit edits
    g_max = [float("-inf")] * 3

    for od in objects:
        n = len(od["positions"])
        orig_off = od["orig_off"]
        off = len(b)
        obj_offsets.append(off)
        sub = od["sub_type"]
        changed = od.get("changed", True)

        # original per-object quantisation, bbox and stream layout
        o_nv = struct.unpack_from("<I", template, orig_off + 44)[0]
        o_vbo = struct.unpack_from("<I", template, orig_off + 48)[0]
        o_ps = struct.unpack_from("<4f", template, orig_off + 72)
        o_pb = struct.unpack_from("<4f", template, orig_off + 88)
        o_tsb = struct.unpack_from("<4f", template, orig_off + 104)
        o_cloth = struct.unpack_from("<I", template, orig_off + 120)[0]

        # ALWAYS derive fresh scale/bias from the ACTUAL positions + UVs -
        # even when the vertex count is unchanged.  The user may have reshaped
        # the mesh (moved vertices) without adding or removing geometry, and
        # re-using the original scale/bias would CLAMP any vertex that left
        # the original bounding box.  Unchanged objects that weren't edited
        # will re-quantise from their own decoded floats, producing a near-
        # exact round-trip (sub-millimetre noise).
        if n:
            psc, pbi, _iq = quantize_positions(od["positions"])
            use_ps = (psc[0], psc[1], psc[2], o_ps[3])
            use_pb = (pbi[0], pbi[1], pbi[2], o_pb[3])
            us = [uv[0] for uv in od["uvs"]] or [0.0]
            vs = [uv[1] for uv in od["uvs"]] or [0.0]
            ulo, uhi = min(us), max(us)
            vlo, vhi = min(vs), max(vs)
            usx = ((uhi - ulo) / 2.0) or 1e-6
            vsx = ((vhi - vlo) / 2.0) or 1e-6
            use_tsb = (usx, vsx, (ulo + uhi) / 2.0, (vlo + vhi) / 2.0)
        else:
            use_ps, use_pb, use_tsb = o_ps, o_pb, o_tsb

        for p in od["positions"]:
            for a in range(3):
                if p[a] < g_min[a]:
                    g_min[a] = p[a]
                if p[a] > g_max[a]:
                    g_max[a] = p[a]

        # PRIM_OBJECT (44 B) copied verbatim; ALWAYS patch bbox from actual
        # positions so the engine's culling box encompasses the current shape.
        po = bytearray(template[orig_off:orig_off + 44])
        if n:
            bbmin = [min(p[a] for p in od["positions"]) for a in range(3)]
            bbmax = [max(p[a] for p in od["positions"]) for a in range(3)]
            obmin = struct.unpack_from("<3f", template, orig_off + 20)
            obmax = struct.unpack_from("<3f", template, orig_off + 32)
            bbmin = [min(bbmin[a], obmin[a]) for a in range(3)]   # never shrink
            bbmax = [max(bbmax[a], obmax[a]) for a in range(3)]   # (avoid culling)
            struct.pack_into("<3f", po, 20, *bbmin)
            struct.pack_into("<3f", po, 32, *bbmax)
        b += po

        fields_off = len(b)
        b += struct.pack("<I", n)                       # vertexCount
        b += struct.pack("<I", 0)                       # vbo (patched)
        b += struct.pack("<I", len(od["indices"]))      # num_indices
        b += struct.pack("<I", 0)                       # additional indices
        b += struct.pack("<I", 0)                       # ibo (patched)
        b += struct.pack("<I", 0)                       # aux/collision (dropped)
        b += struct.pack("<I", 0)                       # unknown_18
        b += struct.pack("<4f", *use_ps)                # pos scale (fresh if edited)
        b += struct.pack("<4f", *use_pb)                # pos bias
        b += struct.pack("<4f", *use_tsb)               # tex scale/bias
        b += struct.pack("<I", o_cloth)                 # preserve cloth id
        if weighted:
            b += b"\x00" * 20                           # +124 weighted trailer

        # --- index buffer (always re-emitted; identical bytes for unchanged) ---
        _align16(b)
        ibo = len(b)
        for idx in od["indices"]:
            b += struct.pack("<H", idx & 0xFFFF)

        # --- vertex buffer (ALWAYS re-emitted from current positions) ---
        # Glacier aligns each vertex stream to a 16-byte boundary. An odd
        # vertex count leaves an 8-byte gap after positions (n×8 mod 16 = 8)
        # and similarly after sub-A. The export must mirror this or the game
        # reads UVs/normals from the wrong offset.
        _align16(b)
        vbo = len(b)
        ts = (use_tsb[0], use_tsb[1]); tb = (use_tsb[2], use_tsb[3])
        for i, p in enumerate(od["positions"]):
            wlane = int(od["joints"][i][3]) if (weighted and od["joints"]) else 0
            b += struct.pack("<hhh", _q_i16(p[0], use_pb[0], use_ps[0]),
                                     _q_i16(p[1], use_pb[1], use_ps[1]),
                                     _q_i16(p[2], use_pb[2], use_ps[2]))
            b += struct.pack("<h", wlane)
        _align16(b)                                    # pad after positions
        tans, bitans = _compute_tangents(od["positions"], od["normals"],
                                         od["uvs"], od["indices"])
        if weighted and sub == 2:
            if n == o_nv:                            # keep original Sub-A bytes
                # source sub-A also sits at the aligned offset in the template
                src_suba = ((o_vbo + o_nv * 8) + 15) & ~15
                b += template[src_suba:src_suba + n * 8]
            else:                                    # synth Sub-A from UVs
                for i in range(n):
                    u = _q_i16(od["uvs"][i][0], tb[0], ts[0])
                    v = _q_i16(od["uvs"][i][1], tb[1], ts[1])
                    b += struct.pack("<hhhh", u, v, u, v)
            _align16(b)                                # pad after sub-A
            for i in range(n):
                b += _enc_normal(od["normals"][i])
                b += _enc_normal(tans[i])
                b += _enc_normal(bitans[i])
                b += struct.pack("<hh", _q_i16(od["uvs"][i][0], tb[0], ts[0]),
                                        _q_i16(od["uvs"][i][1], tb[1], ts[1]))
            _align16(b)                                # pad after NTB+UV
            for i in range(n):
                c = od["colors"][i]
                b += bytes([c[0], c[1], c[2], c[3]])
        else:
            for i in range(n):
                b += _enc_normal(od["normals"][i])
                b += _enc_normal(tans[i])
                b += _enc_normal(bitans[i])
                b += struct.pack("<hh", _q_i16(od["uvs"][i][0], tb[0], ts[0]),
                                        _q_i16(od["uvs"][i][1], tb[1], ts[1]))

        # --- skin partition (weighted only) ---
        # When the vertex count matches the original, copy the existing skin
        # partition verbatim (byte-exact BoneInfo / BoneIndices / weights) so
        # the runtime batching stays valid. Only regenerate when the count
        # changed and the original partition no longer fits.
        _align16(b)
        if weighted:
            same_count = (n == o_nv)
            if same_count:
                o_bi, o_binfo, _cc, _co, o_skin = struct.unpack_from("<5I", template, orig_off + 124)
                bone_info_off = len(b)
                tsize = struct.unpack_from("<H", template, o_binfo)[0]
                b += template[o_binfo:o_binfo + tsize]
                _align16(b)
                bone_indices_off = len(b)
                icount = struct.unpack_from("<I", template, o_bi)[0]
                b += template[o_bi:o_bi + icount * 2]
                _align16(b)
                skin_off = len(b)
                b += template[o_skin:o_skin + o_nv * 8]
                _align16(b)
            else:
                # Regenerate the skin partition. The runtime skins in batches
                # whose bone PALETTE is capped (originals stay <=12 bones/batch);
                # a single batch of all bones overflows the palette and the mesh
                # explodes. Greedily pack vertices so every batch references at
                # most _SKIN_PALETTE distinct bones and each vertex (with all 4 of
                # its influences) lives in exactly one batch.
                w255 = [_weights_to_255(list(od["weights"][i])) for i in range(n)]
                vbones = []
                for i in range(n):
                    j = od["joints"][i]
                    s = set(int(j[k]) for k in range(4) if w255[i][k] > 0)
                    vbones.append(s if s else {int(j[0]) if od["joints"] else 0})
                batches, pal, cur = [], set(), []
                for i in range(n):
                    if cur and len(pal | vbones[i]) > _SKIN_PALETTE:
                        batches.append(cur); pal, cur = set(), []
                    pal |= vbones[i]; cur.append(i)
                if cur:
                    batches.append(cur)

                bone_info_off = len(b)
                nacc = len(batches)
                total_size = 4 + 255 + 1 + nacc * 8
                b += struct.pack("<H", total_size)
                b += struct.pack("<H", nacc)
                b += bytes([0xFF] * 255)                 # bone_remap (shared rig)
                b += bytes([0])
                cursor = 2
                for batch in batches:
                    b += struct.pack("<II", cursor, len(batch))
                    cursor += len(batch)
                _align16(b)
                bone_indices_off = len(b)
                index_count = 2 + sum(len(x) for x in batches)   # == n + 2
                b += struct.pack("<I", index_count)
                for batch in batches:
                    for vi in batch:
                        b += struct.pack("<H", vi & 0xFFFF)
                _align16(b)
                skin_off = len(b)
                for i in range(n):
                    b += bytes(w255[i])
                    j = od["joints"][i]
                    packed = (int(j[0]) & 0x3FF) | ((int(j[1]) & 0x3FF) << 10) | ((int(j[2]) & 0x3FF) << 20)
                    b += struct.pack("<I", packed)
                _align16(b)
            struct.pack_into("<IIIII", b, off + 124,
                             bone_indices_off, bone_info_off, 0, 0, skin_off)

        struct.pack_into("<I", b, fields_off + 4, vbo)
        struct.pack_into("<I", b, fields_off + 16, ibo)

    _align16(b)
    obj_table_off = len(b)
    for o in obj_offsets:
        b += struct.pack("<I", o)

    _align16(b)
    header_off = len(b)
    th = struct.unpack_from("<Q", template, 0)[0]
    b += template[th:th + 16]          # prims, property_flags, unknownPadding, bone_rig
    b += struct.pack("<I", len(obj_offsets))
    b += struct.pack("<I", obj_table_off)
    # global bounds: original, grown to fit any edited geometry (avoids culling
    # when the mesh now extends past the original box)
    o_gmin = struct.unpack_from("<3f", template, th + 24)
    o_gmax = struct.unpack_from("<3f", template, th + 36)
    if g_min[0] <= g_max[0]:
        gmin = [min(o_gmin[a], g_min[a]) for a in range(3)]
        gmax = [max(o_gmax[a], g_max[a]) for a in range(3)]
    else:
        gmin, gmax = list(o_gmin), list(o_gmax)
    b += struct.pack("<3f", *gmin)
    b += struct.pack("<3f", *gmax)

    struct.pack_into("<Q", b, 0, header_off)
    return bytes(b)


class EXPORT_SCENE_OT_glacier2_prim(Operator, ExportHelper):
    """Export the reshaped model back to a 007 First Light .prim (+ meta)"""
    bl_idname = "export_scene.glacier2_007_prim"
    bl_label = "Export 007 Model (.prim)"
    bl_options = {"PRESET"}

    filename_ext = ".prim"
    filter_glob: StringProperty(default="*.prim;*.PRIM", options={"HIDDEN"})

    export_mode: EnumProperty(
        name="Export Mode",
        description="What this export writes. Full = the whole model with your edits; "
                    "the texture-focused modes skip the mesh for quick texture mods",
        items=[
            ("FULL", "Full Model + Edits",
             "Export the .prim mesh plus your edited materials and textures - the "
             "complete model"),
            ("REPLACEMENT", "Texture Replacement",
             "Only what a texture swap needs: the .TEXT/.TEXD (+metas) and, if you "
             "gave the texture a new hash, the repointed .MATI. No mesh"),
            ("TEXTURES", "Textures Only",
             "Only the .TEXT/.TEXD textures (+metas) for your edited slots. No mesh, "
             "no materials"),
            ("MESH", "Mesh Only",
             "Only the reshaped .prim mesh (+meta). No materials or textures"),
        ],
        default="FULL")

    recompute_normals: BoolProperty(
        name="Recompute Normals",
        description="Re-encode smooth vertex normals from the reshaped mesh "
                    "(tangents are left as-is)",
        default=True,
    )
    selected_only: BoolProperty(
        name="Selected Objects Only",
        description="Patch only selected imported objects (unselected objects "
                    "keep their original shape)",
        default=False,
    )
    write_json_meta: BoolProperty(
        name="Write .meta + .meta.json",
        description="Emit the resource metadata next to the .prim so the textures "
                    "and materials resolve when repacked",
        default=True,
    )
    reference_skeleton: BoolProperty(
        name="Add Skeleton Reference",
        description="Add the .borg as an extra dependency in the meta. NOTE: the "
                    "original PRIM does NOT reference its rig (binding is done at "
                    "the entity level); this only bundles the borg. Turn off if it "
                    "causes problems",
        default=True,
    )
    custom_topology: BoolProperty(
        name="Experimental: Custom Mesh",
        description="Rebuild the .prim so the vertex count / topology can change. "
                    "Unedited objects are byte-preserved; edited objects are "
                    "regenerated with real tangents and the original quantisation. "
                    "Collision is dropped and the skin partition of EDITED objects "
                    "is a single batch (test in-game). Leave OFF for safe reshaping",
        default=False,
    )
    propagate_lod: BoolProperty(
        name="Propagate LOD 0 Edits",
        description="Automatically transfer your LOD 0 edits to every lower LOD "
                    "in the same material group. Each lower-LOD vertex finds its "
                    "nearest LOD 0 vertex and receives the same displacement, so "
                    "you only need to sculpt/edit LOD 0",
        default=True,
    )
    export_materials: BoolProperty(
        name="Export .MATI and .MATB",
        description="Also write every material (.MATI/.MATB) and its .meta from the "
                    "model's folder into the output folder, applying any texture/"
                    "parameter overrides. Turn off to export only the .prim",
        default=True,
    )
    only_changed_materials: BoolProperty(
        name="Only Changed Materials",
        description="Write ONLY the materials you actually edited (changed texture, "
                    "image, hash or parameter). Every other material is left out, so "
                    "the rest of the model keeps its original in-game materials. Turn "
                    "off to write the model's whole material set",
        default=True,
    )
    export_textures: BoolProperty(
        name="Export Textures (.TEXT/.TEXD)",
        description="Write the .TEXT + .TEXD (and their .meta) for every texture slot "
                    "set to Custom Texture, Image or TEXT Override into the output "
                    "folder. Turn off to skip textures",
        default=True,
    )
    organize_by_type: BoolProperty(
        name="Sort Into Type Folders",
        description="Put each file in TYPE/<hash>/ - e.g. "
                    "PRIM/01C75259EEAD9C5B/01C75259EEAD9C5B.prim - so every resource "
                    "and its .meta/.meta.json sit together in their own folder, "
                    "grouped by type (PRIM, MATI, MATB, TEXT, TEXD)",
        default=False,
    )
    generate_missing: BoolProperty(
        name="Generate Missing Textures",
        description="For any texture slot that has a decoded image but is missing its "
                    "original .TEXT, build a brand-new .TEXT + .TEXD from the image so "
                    "the game can load it. Handy when you only have a texture's .TEXD "
                    "(the image) and not its .TEXT half - e.g. a basecolor",
        default=True,
    )
    texture_search_dir: StringProperty(
        name="Texture Search Folder",
        description="Extra folder to dig through (sub-folders included) for the .TEXT / "
                    ".TEXD files that your slots point at but don't have a full path for. "
                    "Aim it at your extracted textures - e.g. your WorkingFile folder - "
                    "so the exporter can find each texture's other half and write both "
                    "under their correct, separate game hashes",
        subtype="DIR_PATH", default="",
    )
    textures_only: BoolProperty(
        name="Textures Only (.TEXT/.TEXD)",
        description="Export ONLY the .TEXT + .TEXD (and their metas) for your edited "
                    "texture slots - no .prim, .MATI or .MATB. Use after setting up "
                    "and swapping textures in the panel",
        default=False,
    )
    replacement_only: BoolProperty(
        name="Texture Replacement Only",
        description="Export only what is needed to put your custom texture in the "
                    "game: the .TEXT/.TEXD (+metas) AND, if you gave the texture a new "
                    "hash, the repointed .MATI (+meta). No .prim, no unchanged "
                    "materials. The minimal mod for a texture swap",
        default=False,
    )

    def _gather(self, context):
        pool = context.selected_objects if self.selected_only else context.scene.objects
        objs = [o for o in pool
                if o.type == "MESH" and "glacier_prim_index" in o and "glacier_source_prim" in o]
        return objs

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Mode", icon="EXPORT")
        box.prop(self, "export_mode", text="")
        mode = self.export_mode

        if mode in ("FULL", "MESH"):
            box = layout.box()
            box.label(text="Mesh", icon="MESH_DATA")
            box.prop(self, "recompute_normals")
            box.prop(self, "selected_only")
            box.prop(self, "custom_topology")
            if self.custom_topology:
                note = box.column(align=True)
                note.label(text="Safe for unskinned meshes.", icon="ERROR")
                note.label(text="Skinned parts still explode in-game.")
            box.prop(self, "propagate_lod")
            if self.propagate_lod:
                note = box.column(align=True)
                note.enabled = False
                note.label(text="Copies your LOD 0 edits to all lower LODs")

        if mode == "FULL":
            box = layout.box()
            box.label(text="Materials & Textures", icon="MATERIAL")
            box.prop(self, "export_materials")
            sub = box.column()
            sub.enabled = self.export_materials
            sub.prop(self, "only_changed_materials")
            box.prop(self, "export_textures")

        if mode in ("FULL", "REPLACEMENT", "TEXTURES"):
            box = layout.box()
            box.label(text="Textures", icon="TEXTURE")
            box.prop(self, "generate_missing")
            box.prop(self, "texture_search_dir", text="Search Folder")

        box = layout.box()
        box.label(text="Output", icon="FILE_FOLDER")
        box.prop(self, "write_json_meta")
        sub = box.column()
        sub.enabled = self.write_json_meta
        sub.prop(self, "reference_skeleton")
        box.prop(self, "organize_by_type")

    def _apply_export_mode(self):
        """Map the Export Mode dropdown onto the underlying flags execute() reads."""
        m = getattr(self, "export_mode", "FULL")
        self.textures_only = (m == "TEXTURES")
        self.replacement_only = (m == "REPLACEMENT")
        if m == "MESH":
            self.export_materials = False
            self.export_textures = False

    def _dest(self, base, filename):
        """Return the full output path for `filename`. With 'Sort Into Type
        Folders' on, files go to TYPE/<hash>/<filename> - e.g.
        PRIM/01C75259EEAD9C5B/01C75259EEAD9C5B.prim - so each resource and its
        .meta / .meta.json sit together in their own folder."""
        if not getattr(self, "organize_by_type", False):
            return os.path.join(base, filename)
        core = filename
        if core.endswith(".meta.json"):
            core = core[:-len(".meta.json")]
        elif core.endswith(".meta"):
            core = core[:-len(".meta")]
        if "." in core and not core.endswith("."):
            ext = core.rsplit(".", 1)[1]
        elif "_" in core:
            ext = core.rsplit("_", 1)[1]
        else:
            ext = "MISC"
        # the resource hash = leading name before the first '.' or '_'
        cuts = [i for i in (filename.find("."), filename.find("_")) if i >= 0]
        hashname = filename[:min(cuts)] if cuts else os.path.splitext(filename)[0]
        folder = os.path.join(base, ext.upper(), hashname)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, filename)

    def execute(self, context):
        self._apply_export_mode()
        objs = self._gather(context)
        if not objs:
            self.report({"ERROR"}, "No imported 007 objects found "
                                   "(import a .prim with this addon first)")
            return {"CANCELLED"}

        sources = {o["glacier_source_prim"] for o in objs}
        if len(sources) != 1:
            self.report({"ERROR"}, "Objects come from %d different .prim files; "
                                   "export one model at a time" % len(sources))
            return {"CANCELLED"}
        source = sources.pop()
        if not os.path.exists(source):
            self.report({"ERROR"}, "Original .prim not found at %s "
                                   "(needed as a template)" % source)
            return {"CANCELLED"}

        with open(source, "rb") as f:
            data = bytearray(f.read())

        # Textures-only / replacement-only: skip the mesh work and just package
        # the .TEXT/.TEXD (and, for replacement-only, the repointed .MATI).
        if self.textures_only or self.replacement_only:
            base_dir = os.path.dirname(self.filepath)
            note = self._write_materials(context, source, self.filepath, base_dir)
            note += self._run_generate_missing(context, base_dir)
            mode = "replacement" if self.replacement_only else "textures"
            self.report({"INFO"}, "Exported %s only%s" %
                        (mode, note or " (nothing to write)"))
            return {"FINISHED"}

        try:
            _, weighted, obj_metas = walk_prim_objects(data)
        except Exception as e:
            self.report({"ERROR"}, "Could not parse original .prim: %s" % e)
            return {"CANCELLED"}

        # LOD propagation: transfer LOD 0 edits to lower LODs so the user
        # only has to sculpt the highest-detail mesh.
        lod_overrides = {}              # prim_index -> [(x,y,z), ...]
        lod_note = ""
        if getattr(self, "propagate_lod", False):
            try:
                lod_overrides = _propagate_lod_edits(data, objs)
                if lod_overrides:
                    lod_note = ", propagated edits to %d lower LOD(s)" % len(lod_overrides)
            except Exception as e:
                self.report({"WARNING"}, "LOD propagation skipped: %s" % e)

        if self.custom_topology:
            out_data, patched, note = self._rebuild_custom(
                data, objs, obj_metas, weighted, lod_overrides)
            if out_data is None:
                return {"CANCELLED"}
            data = out_data
        else:
            note = ""
            patched = 0
            for o in objs:
                idx = int(o["glacier_prim_index"])
                if idx >= len(obj_metas):
                    continue
                meta = obj_metas[idx]
                mesh = o.data
                if idx in lod_overrides:
                    coords = lod_overrides[idx]
                else:
                    if len(mesh.vertices) != meta["num_vertices"]:
                        self.report({"ERROR"},
                                    "%s has %d verts but original object %d has %d. "
                                    "Vertex count must be unchanged - or enable "
                                    "'Experimental: Custom Mesh'." %
                                    (o.name, len(mesh.vertices), idx, meta["num_vertices"]))
                        return {"CANCELLED"}
                    coords = _mesh_vertex_coords(mesh)
                normals = None
                if self.recompute_normals:
                    mesh.update()
                    normals = [(v.normal.x, v.normal.y, v.normal.z) for v in mesh.vertices]
                patch_object(data, meta, coords, normals)
                patched += 1

        base_dir = os.path.dirname(self.filepath)
        try:
            out_prim = self._dest(base_dir, os.path.basename(self.filepath))
            with open(out_prim, "wb") as f:
                f.write(data)
        except (OSError, PermissionError) as e:
            self.report({"ERROR"}, "Access denied writing to '%s' (%s). Export to a "
                        "normal folder like your Desktop, not the game install."
                        % (base_dir, type(e).__name__))
            return {"CANCELLED"}

        wrote_meta = ""
        if self.write_json_meta:
            wrote_meta = self._write_meta(context, objs, source, out_prim, base_dir)

        mat_note = ""
        if self.export_materials or self.export_textures:
            mat_note = self._write_materials(context, source, out_prim, base_dir)
        mat_note += self._run_generate_missing(context, base_dir)

        self.report({"INFO"}, "Exported %d object(s) to %s%s%s%s%s" %
                    (patched, os.path.basename(out_prim), note, lod_note, wrote_meta, mat_note))
        return {"FINISHED"}

    def _run_generate_missing(self, context, base_dir):
        """Optionally synthesise .TEXT/.TEXD for slots that only have an image."""
        if not getattr(self, "generate_missing", False):
            return ""
        sc = context.scene
        fmt = _bc_code_for(getattr(sc, "glacier_bc_format", "AUTO"))
        organize = getattr(self, "organize_by_type", False)
        try:
            gen, _warns = _generate_missing_textures(context, base_dir, organize, fmt)
        except Exception:
            return ""
        return (", generated %d missing texture%s" % (gen, "s" if gen != 1 else "")
                if gen else "")

    def _extract_mesh(self, obj, orig_off, sub_type, weighted):
        me = obj.data
        me.calc_loop_triangles()
        n = len(me.vertices)
        positions = _mesh_vertex_coords(me)
        normals = [(v.normal.x, v.normal.y, v.normal.z) for v in me.vertices]
        uvs = [(0.0, 0.0)] * n
        colors = [(255, 255, 255, 255)] * n

        uvl = me.uv_layers.active
        if uvl:
            for loop in me.loops:
                uv = uvl.data[loop.index].uv
                uvs[loop.vertex_index] = (uv.x, 1.0 - uv.y)

        ca = me.color_attributes.active_color if len(me.color_attributes) else None
        if ca:
            def to255(c):
                return (max(0, min(255, int(round(c[0] * 255)))),
                        max(0, min(255, int(round(c[1] * 255)))),
                        max(0, min(255, int(round(c[2] * 255)))),
                        max(0, min(255, int(round(c[3] * 255)))))
            if ca.domain == "POINT":
                for i in range(min(n, len(ca.data))):
                    colors[i] = to255(ca.data[i].color)
            else:  # CORNER
                for loop in me.loops:
                    colors[loop.vertex_index] = to255(ca.data[loop.index].color)

        indices = []
        for tri in me.loop_triangles:
            indices += [tri.vertices[0], tri.vertices[1], tri.vertices[2]]

        joints = weights = None
        if weighted:
            joints = [(0, 0, 0, 0)] * n
            weights = [(1.0, 0.0, 0.0, 0.0)] * n
            for v in me.vertices:
                gs = sorted(v.groups, key=lambda g: -g.weight)[:4]
                js = [g.group for g in gs]
                ws = [g.weight for g in gs]
                while len(js) < 4:
                    js.append(0)
                    ws.append(0.0)
                s = sum(ws) or 1.0
                weights[v.index] = tuple(w / s for w in ws)
                joints[v.index] = tuple(js)

        return {"orig_off": orig_off, "sub_type": sub_type,
                "positions": positions, "normals": normals, "uvs": uvs,
                "colors": colors, "indices": indices,
                "joints": joints, "weights": weights}

    def _rebuild_custom(self, template, objs, obj_metas, weighted,
                        lod_overrides=None):
        lod_overrides = lod_overrides or {}
        by_index = {int(o["glacier_prim_index"]): o for o in objs}
        orig_prim = read_prim_bytes(template)
        rebuilt = []
        changed = 0
        for idx, meta in enumerate(obj_metas):
            if idx in by_index:
                o = by_index[idx]
                d = self._extract_mesh(o, meta["off"], meta["sub_type"], weighted)
                # Apply LOD propagation: override positions for lower LODs
                if idx in lod_overrides:
                    d["positions"] = lod_overrides[idx]
                d["changed"] = (len(d["positions"]) != meta["num_vertices"])
                if d["changed"]:
                    changed += 1
                rebuilt.append(d)
            else:
                d = self._extract_original(orig_prim, idx, meta, weighted)
                if idx in lod_overrides:
                    d["positions"] = lod_overrides[idx]
                d["changed"] = False
                rebuilt.append(d)
        try:
            out = bytearray(build_custom_prim(template, rebuilt, weighted))
        except Exception as e:
            self.report({"ERROR"}, "Custom rebuild failed: %s" % e)
            return None, 0, ""
        note = " [CUSTOM rebuild, %d object(s) changed topology]" % changed
        if weighted and changed:
            self.report({"WARNING"}, "Custom skinned export: edited objects use a "
                                     "regenerated single-batch skin partition - "
                                     "test in-game and keep a backup. Unedited "
                                     "objects are byte-preserved.")
        return out, len(rebuilt), note

    def _extract_original(self, orig_prim, idx, meta, weighted):
        """One untouched object's data taken straight from the original parse."""
        sm = orig_prim.header.object_table[idx]
        vs = sm.vertexBuffer.vertices
        positions = [(v.position[0], v.position[1], v.position[2]) for v in vs]
        normals = [(v.normal[0], v.normal[1], v.normal[2]) for v in vs]
        uvs = [(v.uv[0][0], v.uv[0][1]) for v in vs]
        colors = [tuple(v.color) for v in vs]
        joints = [tuple(v.joint) for v in vs] if weighted else None
        weights = [tuple(v.weight) for v in vs] if weighted else None
        return {"orig_off": meta["off"], "sub_type": meta["sub_type"],
                "positions": positions, "normals": normals, "uvs": uvs,
                "colors": colors, "indices": list(sm.indices),
                "joints": joints, "weights": weights}

    def _write_materials(self, context, source, out_prim, base_dir=None):
        """Export materials and/or textures into the output folder, applying
        texture-slot and parameter overrides. Materials are gated by
        'Export .MATI and .MATB'; textures by 'Export Textures'."""
        import json as _json
        sc = context.scene
        base = base_dir if base_dir is not None else os.path.dirname(out_prim)
        out_dir = base
        src_dir = os.path.dirname(source)
        to = getattr(self, "textures_only", False)
        ro = getattr(self, "replacement_only", False)
        do_mats = ro or (getattr(self, "export_materials", True) and not to)
        do_texs = True if (to or ro) else getattr(self, "export_textures", True)

        # where to look for a resource's .meta if it isn't next to the resource
        meta_search_dirs = [src_dir]
        for attr in ("glacier_scan_folder", "glacier_tex_folder", "glacier_work_dir"):
            p = bpy.path.abspath(getattr(sc, attr, "") or "")
            if p and os.path.isdir(p) and p not in meta_search_dirs:
                meta_search_dirs.append(p)
        for mt in getattr(sc, "glacier_materials", []):
            d = os.path.dirname(mt.path) if mt.path else ""
            if d and d not in meta_search_dirs:
                meta_search_dirs.append(d)
        _esf = bpy.path.abspath(getattr(self, "texture_search_dir", "") or "")
        if _esf and os.path.isdir(_esf) and _esf not in meta_search_dirs:
            meta_search_dirs.append(_esf)

        # auto-fill any blank original hashes from the .MATI metas so Custom Texture
        # slots can be packaged without the user typing a hash
        if do_texs:
            _resolve_old_hashes(sc, meta_search_dirs, force=False)

        # index every *.meta under those folders ONCE (basename -> path). Walking
        # once avoids re-scanning the tree per file and swallows any access-denied
        # on individual files/folders so the export never aborts.
        meta_index = {}
        for d in meta_search_dirs:
            if not d or not os.path.isdir(d):
                continue
            try:
                for dirpath, _dirs, files in os.walk(d, onerror=lambda e: None,
                                                     followlinks=False):
                    for fn in files:
                        if fn.lower().endswith(".meta"):
                            meta_index.setdefault(fn.lower(), os.path.join(dirpath, fn))
            except Exception:
                continue

        def lookup_meta(res_path, ext):
            for c in meta_path_candidates(res_path, ext):
                if os.path.exists(c):
                    return c
            for c in meta_path_candidates(res_path, ext):
                hit = meta_index.get(os.path.basename(c).lower())
                if hit:
                    return hit
            return ""

        tex_changes = {}
        tex_packages = []      # (target_hash_int, ext, text_path, texd_path|None)
        tmp_convert = []       # temp files to clean up after writing
        seen_targets = {}      # target_hash_int -> slot name (collision guard)
        # Index every .TEXT/.TEXD under the search folders and work out each TEXT's
        # paired TEXD hash (from the TEXT meta's 0x9F ref), so the exporter always
        # writes the TEXD under its OWN game hash - never the same as the TEXT.
        # Search widely: model/scan/work folders, the Names folder, and the folder
        # of every slot's .TEXT/.TEXD/image - the user's files often live there.
        texd_dirs = list(meta_search_dirs)
        try:
            for d in _all_texture_dirs(context):
                if d not in texd_dirs:
                    texd_dirs.append(d)
        except Exception:
            pass
        nf = bpy.path.abspath(getattr(sc, "glacier_names_file", "") or "")
        if nf:
            nd = nf if os.path.isdir(nf) else os.path.dirname(nf)
            if nd and os.path.isdir(nd) and nd not in texd_dirs:
                texd_dirs.append(nd)
        # the export dialog's own 'Texture Search Folder'
        esf = bpy.path.abspath(getattr(self, "texture_search_dir", "") or "")
        if esf and os.path.isdir(esf) and esf not in texd_dirs:
            texd_dirs.append(esf)
        for it in getattr(sc, "glacier_tex_slots", []):
            for attr in ("file_path", "file_path_texd", "image_path"):
                p = getattr(it, attr, "")
                if p:
                    d = os.path.dirname(bpy.path.abspath(p))
                    if d and os.path.isdir(d) and d not in texd_dirs:
                        texd_dirs.append(d)
        _text_idx, _texd_idx = index_textures_by_hash(texd_dirs)
        try:
            _texd_pairs = pair_text_to_texd(texd_dirs)
        except Exception:
            _texd_pairs = {}
        try:
            _fill_slot_texd_hashes(sc, texd_dirs)
        except Exception:
            pass

        def _resolve_texd_hash(it, target):
            """Best-effort distinct TEXD hash for a slot: stamped value, then the
            TEXT->TEXD pairing, then reading the slot's own .TEXT meta directly.
            Any value equal to the TEXT's own hash is rejected (a TEXT and its TEXD
            can never share a hash)."""
            if getattr(it, "texd_hash", ""):
                try:
                    v = int(it.texd_hash, 16)
                    if v != target:
                        return v
                except ValueError:
                    pass
            for key in (("%016X" % target) if target else "",
                        (it.new_hash or "").upper(), (it.old_hash or "").upper()):
                k = (key or "").upper()
                if k and k in _texd_pairs and _texd_pairs[k] is not None \
                        and _texd_pairs[k] != target:
                    return _texd_pairs[k]
            # last resort: read the slot's TEXT meta straight off disk
            tpath = ""
            if getattr(it, "file_path", "") and it.tex_source == "CUSTOM":
                tpath = bpy.path.abspath(it.file_path)
            if not (tpath and os.path.exists(tpath)) and it.old_hash:
                tpath = _text_idx.get(it.old_hash.upper(), "")
            if tpath and os.path.exists(tpath):
                cm = derive_resource_meta(tpath, "TEXT")
                if cm and os.path.exists(cm):
                    try:
                        mm = parse_meta(bytearray(open(cm, "rb").read()))
                        for (rh, fl) in mm.get("refs", []):
                            if fl == 0x9F and rh != target:
                                return rh
                        for (rh, fl) in mm.get("refs", []):
                            if rh != target:
                                return rh
                    except Exception:
                        pass
            return None

        for it in getattr(sc, "glacier_tex_slots", []):
            if it.tex_index < 0:           # blueprint schema slot - not swappable
                continue
            if not do_texs:
                continue
            src_mode = it.tex_source
            target = None
            nh = it.new_hash.strip()
            if nh:
                try:
                    target = int(nh, 16)
                except ValueError:
                    target = None

            text_fp = None
            texd_fp = None
            meta_tmpl = None        # original .TEXT whose meta/structure to inherit
            if src_mode == "CUSTOM" and it.file_path:
                text_fp = bpy.path.abspath(it.file_path).replace(os.sep, "/")
                meta_tmpl = text_fp
                if it.file_path_texd:
                    texd_fp = bpy.path.abspath(it.file_path_texd).replace(os.sep, "/")
            elif src_mode == "IMAGE" and it.image_path:
                # convert .tga/.png -> game .TEXT/.TEXD
                img = bpy.path.abspath(it.image_path).replace(os.sep, "/")
                if target is None:
                    target = 0  # placeholder; resolved below to a stable hash
                conv_base = os.path.join(out_dir, "_convert_%s" %
                                         it.slot_name.replace(" ", "_"))
                ct, cd = conv_base + ".TEXT", conv_base + ".TEXD"
                tmpl_path = bpy.path.abspath(it.file_path or "").replace(os.sep, "/")
                if not (tmpl_path and os.path.exists(tmpl_path)) and it.old_hash:
                    cand = os.path.join(src_dir, "%s.TEXT" % it.old_hash.upper())
                    if os.path.exists(cand):
                        tmpl_path = cand
                if tmpl_path and os.path.exists(tmpl_path):
                    meta_tmpl = tmpl_path
                # native pure-Python converter (BC1). Use the slot's own .TEXT as
                # a template if we have one, so size/atlas stay game-valid.
                tmpl_bytes = None
                if meta_tmpl:
                    try:
                        tmpl_bytes = bytearray(open(meta_tmpl, "rb").read())
                    except OSError:
                        tmpl_bytes = None
                ok, msg = convert_image_native(img, ct, cd, tmpl_bytes,
                                               _bc_code_for(sc.glacier_bc_format))
                if not ok:
                    self.report({"WARNING"}, "Slot '%s': %s" % (it.slot_name, msg))
                    continue
                text_fp = ct
                texd_fp = cd if os.path.exists(cd) else None
                tmp_convert.extend([ct, cd])

            if text_fp:
                if target is None or target == 0:        # fall back to filename hash
                    hp = hash_from_path(text_fp)
                    target = int(hp, 16) if hp else None
                if target is None and it.old_hash:        # else reuse this texture's hash
                    try:
                        target = int(it.old_hash, 16)
                    except ValueError:
                        target = None
                if target is None:
                    self.report({"WARNING"}, "Slot '%s': set a 16-hex 'hash' (or name "
                                "the file <hash>.TEXT) so it can be packaged"
                                % it.slot_name)
                    continue
                if target in seen_targets:
                    self.report({"WARNING"}, "Slot '%s' targets the same hash %016X as "
                                "slot '%s' - skipping the duplicate. Give each texture a "
                                "DIFFERENT hash, or replace each in place by leaving the "
                                "hash blank." % (it.slot_name, target, seen_targets[target]))
                    continue
                seen_targets[target] = it.slot_name
                # the distinct game hash of this texture's TEXD (NOT the TEXT hash)
                texd_hash_x = _resolve_texd_hash(it, target)
                # if no explicit .TEXD file was given, find it by its real hash
                if texd_fp is None and texd_hash_x is not None:
                    cand = _texd_idx.get("%016X" % texd_hash_x)
                    if cand and os.path.exists(cand):
                        texd_fp = cand
                tex_packages.append((target, "TEXT", text_fp, texd_fp, meta_tmpl,
                                     texd_hash_x))
            if target is not None:
                # only repoint the material (and thus rewrite the .MATI) when the
                # texture hash actually changes. An in-place replacement reuses the
                # original hash, so the base-game .MATI stays valid and untouched.
                orig = None
                if it.old_hash:
                    try:
                        orig = int(it.old_hash, 16)
                    except ValueError:
                        orig = None
                if target != orig:
                    tex_changes.setdefault(it.mati_hash.upper(), []).append(
                        (it.tex_index, target))

        param_changes = {}
        for it in getattr(sc, "glacier_params", []):
            if it.changed and it.data_off >= 0:
                param_changes.setdefault(it.mati_hash.upper(), []).append(it)

        # every material in the model folder + any loaded from elsewhere
        mati_files, matb_files = [], []
        try:
            for f in os.listdir(src_dir):
                fl = f.lower()
                if fl.endswith(".mati"):
                    mati_files.append(os.path.join(src_dir, f))
                elif fl.endswith(".matb"):
                    matb_files.append(os.path.join(src_dir, f))
        except OSError:
            pass
        for mt in getattr(sc, "glacier_materials", []):
            if mt.path and os.path.exists(mt.path) and os.path.dirname(mt.path) != src_dir:
                (matb_files if mt.is_blueprint else mati_files).append(mt.path)

        def _same(a, b):
            return os.path.abspath(a) == os.path.abspath(b)

        only_changed = getattr(self, "only_changed_materials", True) or ro
        n_mati = n_matb = 0
        for mati_path in (mati_files if do_mats else []):
            key = (hash_from_path(mati_path) or "").upper()
            if only_changed and key not in tex_changes and key not in param_changes:
                continue            # leave this material as the original in-game one
            try:
                data = bytearray(open(mati_path, "rb").read())
            except OSError:
                continue
            for it in param_changes.get(key, []):
                try:
                    if it.type == 0x03:
                        struct.pack_into("<3f", data, it.data_off,
                                         it.color[0], it.color[1], it.color[2])
                    else:
                        struct.pack_into("<f", data, it.data_off, it.fval)
                except struct.error:
                    pass
            out_mati = self._dest(base, os.path.basename(mati_path))
            try:
                with open(out_mati, "wb") as f:
                    f.write(data)
            except (OSError, PermissionError) as e:
                self.report({"WARNING"}, "Could not write %s (%s)"
                            % (os.path.basename(out_mati), type(e).__name__))
                continue
            meta_in = lookup_meta(mati_path, "MATI")
            if meta_in:
                mm = parse_meta(bytearray(open(meta_in, "rb").read()))
                for (idx, new_h) in tex_changes.get(key, []):
                    if 0 <= idx < len(mm["refs"]):
                        mm["refs"][idx] = (new_h, mm["refs"][idx][1])
                out_meta = self._dest(base, os.path.basename(meta_in))
                with open(out_meta, "wb") as f:
                    f.write(build_meta_binary(mm, []))
                with open(out_meta + ".json", "w") as f:
                    _json.dump(build_meta_json(mm, []), f, indent=2)
            else:
                self.report({"WARNING"}, "Wrote %s but found no _MATI.meta for it - "
                            "put the matching _MATI.meta in the same folder (or in "
                            "your Scan folder) so the game can load it"
                            % os.path.basename(mati_path))
            n_mati += 1

        for matb_path in (matb_files if do_mats else []):
            if only_changed:
                continue            # blueprints have no per-model edits to write
            out_matb = self._dest(base, os.path.basename(matb_path))
            try:
                with open(out_matb, "wb") as f:
                    f.write(open(matb_path, "rb").read())
            except (OSError, PermissionError) as e:
                self.report({"WARNING"}, "Could not write %s (%s)"
                            % (os.path.basename(out_matb), type(e).__name__))
                continue
            meta_in = lookup_meta(matb_path, "MATB")
            if meta_in:
                mm = parse_meta(bytearray(open(meta_in, "rb").read()))
                out_meta = self._dest(base, os.path.basename(meta_in))
                with open(out_meta, "wb") as f:
                    f.write(build_meta_binary(mm, []))
                with open(out_meta + ".json", "w") as f:
                    _json.dump(build_meta_json(mm, []), f, indent=2)
            else:
                self.report({"WARNING"}, "Wrote %s but found no _MATB.meta for it - "
                            "put the matching _MATB.meta in the same folder (or in "
                            "your Scan folder) so the game can load it"
                            % os.path.basename(matb_path))
            n_matb += 1

        # bundle custom textures. Writes the .TEXT + paired .TEXD and rebuilds
        # their .meta so the game can load them: the TEXT meta MUST reference the
        # TEXD (flag 0x9F) and carry size_video (the full-res mip size), or the
        # game crashes on load.
        n_tex = 0
        for (thash, ext, fp, texd_path, meta_tmpl, texd_hash_x) in tex_packages:
            try:
                tdata = open(fp, "rb").read()
            except OSError:
                self.report({"WARNING"}, "Texture file not found: %s" % fp)
                continue
            info = parse_text_header(bytearray(tdata))
            if info is None:
                self.report({"WARNING"}, "'%s' is not a valid 007 .TEXT - packaging "
                            "it anyway, but the game may reject it"
                            % os.path.basename(fp))

            # base the TEXT meta on a real one (the slot's original .TEXT) so the
            # flag, dummy and structure are correct; else fall back to the file's
            # own sibling meta; else synthesize.
            src_meta = ""
            for cand_src in (meta_tmpl, fp):
                if cand_src:
                    cm = derive_resource_meta(cand_src, "TEXT")
                    if cm and os.path.exists(cm):
                        src_meta = cm
                        break
            if src_meta:
                tm = parse_meta(bytearray(open(src_meta, "rb").read()))
            else:
                tm = {"resource_id": thash, "data_offset": 0xFFFFFFFFFFFFFFFF,
                      "size_raw": 0x80000000, "ext_raw": b"TEXT",
                      "size_uncompressed": len(tdata), "size_memory": 0xFFFFFFFF,
                      "size_video": 0, "dummy": 0xC000, "refs": []}
            tm["resource_id"] = thash
            tm["size_uncompressed"] = len(tdata)
            if not tm.get("dummy"):
                tm["dummy"] = 0xC000

            # the distinct TEXD hash: explicit pairing wins, then an existing .TEXD
            # path, then the TEXT meta's referenced TEXD, and only as a LAST resort
            # the TEXT hash (which we warn about, since same-name files crash).
            texd_data = None
            texd_hash = texd_hash_x
            if texd_path and os.path.exists(texd_path):
                texd_data = open(texd_path, "rb").read()
                if texd_hash is None:
                    hp = hash_from_path(texd_path)
                    texd_hash = int(hp, 16) if hp else None
            if texd_hash is None and tm.get("refs"):
                texd_hash = tm["refs"][0][0]
            if texd_data is None and texd_hash is not None:
                # find the real TEXD file anywhere we indexed, then beside the .TEXT
                cand = _texd_idx.get("%016X" % texd_hash)
                if not (cand and os.path.exists(cand)):
                    cand = os.path.join(os.path.dirname(meta_tmpl or fp),
                                        "%016X.TEXD" % texd_hash)
                if cand and os.path.exists(cand):
                    texd_data = open(cand, "rb").read()
            if texd_hash is None:
                texd_hash = thash
            collide = (texd_hash == thash and texd_data is not None)
            if collide:
                self.report({"WARNING"}, "Slot for %016X: couldn't work out a separate "
                            "TEXD hash, so the .TEXD was NOT written (it would collide with "
                            "the .TEXT and crash the game). Fix: in Edit Material, set this "
                            "slot's '.TEXD hash' to the real value, or point its '.TEXD' "
                            "field at the original .TEXD file." % thash)
                texd_data = None          # do not write a colliding pair

            # the TEXT meta MUST reference the TEXD (flag 0x9F)
            if tm.get("refs"):
                tm["refs"] = [(texd_hash, tm["refs"][0][1])] + list(tm["refs"][1:])
            else:
                tm["refs"] = [(texd_hash, 0x9F)]

            # size_video = cumulative uncompressed size of the TEXD-only (big)
            # mips = mip_unc[text_scale-1]; this is what the game allocates
            if info is not None:
                ts = bytearray(tdata)[0x91]
                mu = info["mip_offsets_uncompressed"]
                tm["size_video"] = mu[ts-1] if ts >= 1 else mu[0]
            dot = meta_uses_dot_style(meta_tmpl or fp, "TEXT")
            out_tex = self._dest(base, "%016X.TEXT" % thash)
            with open(out_tex, "wb") as f:
                f.write(tdata)
            out_meta = self._dest(base, meta_out_name(thash, "TEXT", dot))
            with open(out_meta, "wb") as f:
                f.write(build_meta_binary(tm, []))
            with open(out_meta + ".json", "w") as f:
                _json.dump(build_meta_json(tm, []), f, indent=2)
            n_tex += 1

            # write the paired TEXD (+ its meta)
            if texd_data is not None:
                src_dmeta = ""
                for cand_src in (texd_path, meta_tmpl):
                    if cand_src:
                        cm = derive_resource_meta(cand_src, "TEXD")
                        if cm and os.path.exists(cm):
                            src_dmeta = cm
                            break
                if src_dmeta:
                    dm = parse_meta(bytearray(open(src_dmeta, "rb").read()))
                else:
                    dm = {"resource_id": texd_hash,
                          "data_offset": 0xFFFFFFFFFFFFFFFF,
                          "size_raw": 0, "ext_raw": b"TEXD",
                          "size_uncompressed": len(texd_data),
                          "size_memory": 0xFFFFFFFF, "size_video": 0,
                          "dummy": 0, "refs": []}
                dm["resource_id"] = texd_hash
                dm["size_uncompressed"] = len(texd_data)
                dm["size_memory"] = 0xFFFFFFFF
                dm["size_video"] = tm["size_video"]
                out_texd = self._dest(base, "%016X.TEXD" % texd_hash)
                with open(out_texd, "wb") as f:
                    f.write(texd_data)
                out_dmeta = self._dest(base, meta_out_name(texd_hash, "TEXD", dot))
                with open(out_dmeta, "wb") as f:
                    f.write(build_meta_binary(dm, []))
                with open(out_dmeta + ".json", "w") as f:
                    _json.dump(build_meta_json(dm, []), f, indent=2)
                n_tex += 1

        if not (n_mati or n_matb or n_tex):
            return ""
        # remove intermediate files produced by image conversion
        for f in tmp_convert:
            for p in (f, f + ".meta", f + ".meta.json"):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
        tail = " (+ %d MATI, %d MATB" % (n_mati, n_matb)
        if n_tex:
            tail += ", %d packaged texture%s" % (n_tex, "s" if n_tex != 1 else "")
        return tail + ")"

    def _write_meta(self, context, objs, source, out_prim, base_dir=None):
        meta_path = derive_meta_path(source)
        if not meta_path:
            self.report({"WARNING"}, "Original meta not found next to the source "
                                     ".prim; skipping meta output")
            return ""
        with open(meta_path, "rb") as f:
            m = parse_meta(bytearray(f.read()))

        # retexture / material swap: remap reference hashes from the
        # "007 Mesh Tools" side panel (Material Overrides)
        remap = {}
        for it in getattr(context.scene, "glacier_overrides", []):
            nh = it.new_hash.strip()
            if nh:
                try:
                    remap[int(it.old_hash, 16)] = int(nh, 16)
                except ValueError:
                    pass
        if remap:
            m["refs"] = [(remap.get(h, h), flag) for (h, flag) in m["refs"]]

        extra = []
        if self.reference_skeleton:
            rig = ""
            for o in objs:
                if o.get("glacier_rig_path"):
                    rig = o["glacier_rig_path"]
                    break
            bh = hash_from_filename(rig) if rig else None
            if bh is not None and all(bh != h for (h, _) in m["refs"]):
                extra.append((bh, 0x5F))  # Normal-type dependency, global language

        out_meta = out_prim + ".meta"
        out_json = out_prim + ".meta.json"
        with open(out_meta, "wb") as f:
            f.write(build_meta_binary(m, extra))
        import json as _json
        with open(out_json, "w") as f:
            _json.dump(build_meta_json(m, extra), f, indent=2)

        tags = []
        if remap:
            tags.append("%d retex" % len(remap))
        if extra:
            tags.append("+borg ref")
        return " (+ meta%s)" % (": " + ", ".join(tags) if tags else "")


# =============================================================================
# "007 Mesh Tools" side panel (View3D > N panel)
# =============================================================================
def _glacier_lod_groups(context):
    """Group imported objects by material id (each group is a LOD chain)."""
    groups = {}
    for o in context.scene.objects:
        if o.type == "MESH" and "glacier_lod_index" in o and "glacier_material_id" in o:
            groups.setdefault(o["glacier_material_id"], []).append(o)
    for objs in groups.values():
        objs.sort(key=lambda o: o["glacier_lod_index"])
    return groups


def _apply_lod_level(context, level):
    for objs in _glacier_lod_groups(context).values():
        maxlod = objs[-1]["glacier_lod_index"]
        target = min(level, maxlod)
        for o in objs:
            o.hide_viewport = (o["glacier_lod_index"] != target)


def _update_lod(self, context):
    _apply_lod_level(context, context.scene.glacier_lod_level)


class GlacierRefOverride(bpy.types.PropertyGroup):
    label: StringProperty(name="Slot", default="")
    old_hash: StringProperty(name="Original", default="")
    new_hash: StringProperty(name="Replace With", default="",
                             description="16-hex hash to swap in, or blank for none")


def _mark_param_changed(self, context):
    self.changed = True


class GlacierTexSlot(bpy.types.PropertyGroup):
    mati_hash: StringProperty(default="")
    slot_name: StringProperty(default="")
    tex_index: IntProperty(default=0)
    old_hash: StringProperty(default="")
    new_hash: StringProperty(name="Replace With", default="",
                             description="16-hex texture hash to point this slot at")
    tex_source: EnumProperty(
        name="Source",
        description="Where this texture slot gets its data when exporting",
        items=[
            ("HASH", "Hash",
             "Point the slot at an existing in-game texture by its 16-hex hash"),
            ("IMAGE", "Custom Texture",
             "Use your own .tga / .png. It is converted to a game .TEXT+.TEXD on "
             "export with the built-in BC1 converter and packaged"),
            ("CUSTOM", "TEXT Override",
             "Use a ready-made game .TEXT (+ .TEXD) file and package it"),
        ],
        default="HASH")
    file_path: StringProperty(
        name="TEXT File", subtype="FILE_PATH", default="",
        description="A game-format .TEXT (low-res streaming + metadata). Make it "
                    "with HMTextureTools / GlacierKit. Its _TEXT.meta sibling is "
                    "copied if present; otherwise a minimal one is generated")
    file_path_texd: StringProperty(
        name="TEXD File", subtype="FILE_PATH", default="",
        description="Optional matching .TEXD (full-res). If left blank the addon "
                    "looks for the .TEXD referenced by the .TEXT next to it")
    texd_hash: StringProperty(
        name="TEXD Hash", default="",
        description="The distinct 16-hex hash of this texture's .TEXD half, read from "
                    "the original .TEXT meta. Filled in automatically")
    image_path: StringProperty(
        name="Image", subtype="FILE_PATH", default="",
        description="A .tga or .png to convert into this texture on export")
    # kept for backward compatibility; mirrors tex_source != 'HASH'
    use_file: BoolProperty(default=False)


_RENDER_ROLE_ITEMS = [
    ("AUTO", "Auto", "Decide from the texture's name and format (BC5 = normal, etc.)"),
    ("BASE", "Base Color", "sRGB albedo / diffuse -> Base Color"),
    ("SRM", "SRM (Rough/Spec/Metal)",
     "Packed map: green -> Roughness, red -> Specular, blue -> Metallic"),
    ("NORMAL", "Normal", "Tangent-space normal map -> Normal"),
    ("DETAIL_NORMAL", "Detail Normal", "Micro / detail normal map"),
    ("TRANSLUCENCY", "Translucency", "Translucency / subsurface -> Subsurface"),
    ("EMISSION", "Emission", "Emissive -> Emission"),
    ("AO", "Ambient Occlusion", "Ambient occlusion map"),
    ("ALPHA", "Alpha", "Opacity / alpha -> Alpha"),
    ("SKIP", "Don't Load", "Ignore this texture when building the render material"),
]


class GlacierRenderSlot(bpy.types.PropertyGroup):
    """One texture the render-material builder may pull from the reference
    (MATI/MATB + TEXT/TEXD). The creator chooses whether to load it and what
    it drives, independently of the export texture overrides."""
    mati_hash: StringProperty(default="")
    slot_name: StringProperty(default="")
    tex_hash: StringProperty(default="")
    texd_hash: StringProperty(default="")
    fmt: StringProperty(default="")
    res: StringProperty(default="")
    enabled: BoolProperty(
        name="Load", default=True,
        description="Load this texture into the render material")
    role: EnumProperty(
        name="Role", items=_RENDER_ROLE_ITEMS, default="AUTO",
        description="What this texture drives in the render material")


class GlacierMatParam(bpy.types.PropertyGroup):
    mati_hash: StringProperty(default="")
    name: StringProperty(default="")
    type: IntProperty(default=1)
    data_off: IntProperty(default=0)
    changed: BoolProperty(default=False)
    fval: FloatProperty(name="Value", update=_mark_param_changed)
    color: FloatVectorProperty(name="Color", subtype="COLOR", size=3,
                               min=0.0, max=1.0, default=(1.0, 1.0, 1.0),
                               update=_mark_param_changed)


class GlacierMaterial(bpy.types.PropertyGroup):
    key: StringProperty(default="")          # 16-hex hash (output filename / lookup)
    label: StringProperty(default="")        # friendly name (shader template etc.)
    path: StringProperty(default="")         # full path to the .MATI on disk
    is_blueprint: BoolProperty(default=False)


def _clear_material(sc, key):
    """Remove any existing material/tex-slot/param entries for a key so loading
    the same material twice does not duplicate its rows."""
    key = (key or "").upper()
    for coll in (sc.glacier_tex_slots, sc.glacier_params):
        for i in range(len(coll) - 1, -1, -1):
            if coll[i].mati_hash.upper() == key:
                coll.remove(i)
    for i in range(len(sc.glacier_materials) - 1, -1, -1):
        if sc.glacier_materials[i].key.upper() == key:
            sc.glacier_materials.remove(i)


def _dedup_materials(sc):
    """Belt-and-braces: collapse any duplicate rows that slipped through, keeping
    the first occurrence of each material / texture slot / parameter."""
    def dd(coll, keyfn):
        seen, rm = set(), []
        for i in range(len(coll)):
            k = keyfn(coll[i])
            if k in seen:
                rm.append(i)
            else:
                seen.add(k)
        for i in reversed(rm):
            coll.remove(i)
    dd(sc.glacier_materials, lambda m: m.key.upper())
    dd(sc.glacier_tex_slots, lambda t: (t.mati_hash.upper(), t.slot_name, t.tex_index))
    dd(sc.glacier_params, lambda p: (p.mati_hash.upper(), p.name))


# EnumProperty items must be kept alive in Python to avoid string corruption.
_MAT_ENUM_CACHE = []


def _material_enum_items(self, context):
    _MAT_ENUM_CACHE.clear()
    mats = getattr(context.scene, "glacier_materials", [])
    if not mats:
        _MAT_ENUM_CACHE.append(("", "(no material loaded)", ""))
    else:
        for mt in mats:
            tag = " (blueprint)" if mt.is_blueprint else ""
            name = "%s  [%s]%s" % (mt.label or mt.key, mt.key[:8], tag)
            _MAT_ENUM_CACHE.append((mt.key, name, mt.path or mt.key))
    return _MAT_ENUM_CACHE


def _sync_active_material(self, context):
    """Keep the edit dropdown in step with the material list selection."""
    sc = context.scene
    i = sc.glacier_materials_index
    if 0 <= i < len(sc.glacier_materials):
        try:
            sc.glacier_active_material = sc.glacier_materials[i].key
        except (TypeError, ValueError):
            pass


_MI_NAME_RE = re.compile(
    r'([0-9A-Fa-f]{16})[^\n\]]{0,600}?[\[/]([A-Za-z0-9_\- ]+)\.(?:mi|mat|material|entitytemplate)\b',
    re.I)


def _pretty_material_name(name):
    """head_bond_v1 -> Head_Bond_V1."""
    parts = [p for p in re.split(r'[ _]+', name.strip()) if p]
    return "_".join(p[:1].upper() + p[1:] for p in parts) if parts else name


def build_resource_name_map(dirs, explicit=""):
    """Build {RESOURCE_HASH_UPPER: name} by scanning text/json/hash-list files for
    lines that map a 16-hex hash to an IOI source path like
    '... [assembly:/.../head_bond_v1.mi] ...'. The material name lives in that path
    (RPKG-Tool's hash list / dependency dump), not in the .MATI itself."""
    names = {}
    files = []
    if explicit and os.path.isfile(explicit):
        files.append(explicit)
    exts = (".txt", ".json", ".csv", ".list", ".hashlist", ".hash_list", ".tsv",
            ".meta.json", ".meta", ".log", ".md")
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for dp, _dd, fs in os.walk(d, onerror=lambda e: None, followlinks=False):
                for fn in fs:
                    low = fn.lower()
                    if low.endswith(exts) or "hash" in low or "depend" in low:
                        full = os.path.join(dp, fn)
                        try:
                            if os.path.getsize(full) <= 64 * 1024 * 1024:
                                files.append(full)
                        except OSError:
                            pass
        except Exception:
            continue
    for f in files:
        low = f.lower()
        # RPKG writes binary .meta (no readable path); try it as JSON first in
        # case it is actually a JSON meta, then fall through to the text scan
        # which harvests any embedded [assembly:/.../name.mi] path.
        if low.endswith(".meta") or low.endswith(".json"):
            try:
                import json as _json
                obj = _json.loads(open(f, "r", encoding="utf-8",
                                       errors="ignore").read())
                _harvest_json_names(obj, names)
            except Exception:
                pass
        try:
            txt = open(f, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for m in _MI_NAME_RE.finditer(txt):
            names.setdefault(m.group(1).upper(), m.group(2).strip())
    return names


def _harvest_json_names(obj, names):
    """Walk a parsed JSON object for {hash, path} pairs and record material names."""
    if isinstance(obj, dict):
        h = obj.get("hash") or obj.get("hash_value") or obj.get("id")
        p = (obj.get("path") or obj.get("ioi_path") or obj.get("resource_path")
             or obj.get("hash_path"))
        if h and p:
            hs = re.sub(r"[^0-9A-Fa-f]", "", str(h))[:16]
            m = re.search(r"/([A-Za-z0-9_\- ]+)\.(?:mi|mat|material|entitytemplate)\b",
                          str(p), re.I)
            if len(hs) == 16 and m:
                names.setdefault(hs.upper(), m.group(1).strip())
        for v in obj.values():
            _harvest_json_names(v, names)
    elif isinstance(obj, list):
        for v in obj:
            _harvest_json_names(v, names)


def _resolve_material_label(mati, key, name_map):
    """Prefer the readable IOI name (Head_Bond_V1) from the name map; else fall back
    to the shader-template / first-string heuristic."""
    if name_map and key and key.upper() in name_map:
        return _pretty_material_name(name_map[key.upper()])
    return _mati_display_name(mati, key)


def _mati_display_name(mati, key):
    """Best human name for a material: the shader template, else any gm_ string,
    else the first name-like string, else the short hash."""
    strings = list(mati["strings"].values())
    record_names = {t["name"] for t in mati["textures"]}
    record_names |= {p["name"] for p in mati["params"]}
    for s in strings:
        if s.startswith("gm_") and "ShaderTemplate" in s:
            return s
    for s in strings:
        if s.startswith("gm_"):
            return s
    for s in strings:
        ss = s.strip()
        if (ss and ss not in record_names and len(ss) > 2
                and not ss.lower().startswith("map")
                and all(32 <= ord(c) < 127 for c in ss)):
            return ss
    return key[:8]


def _load_mati_into_scene(sc, mati_path, name_map=None):
    """Parse a .MATI (+ its sibling meta, searched in the scan folder if not next
    to the file) and add a material with its texture slots and parameters."""
    data = bytearray(open(mati_path, "rb").read())
    mati = parse_mati(data)

    # search dirs for the meta: next to the file, then every folder the user has
    # pointed us at (scan / texture / work folders), each walked recursively
    search_dirs = [os.path.dirname(mati_path)]
    for attr in ("glacier_scan_folder", "glacier_tex_folder", "glacier_work_dir"):
        p = bpy.path.abspath(getattr(sc, attr, "") or "")
        if p and os.path.isdir(p) and p not in search_dirs:
            search_dirs.append(p)

    if name_map is None:
        name_map = build_resource_name_map(
            search_dirs, bpy.path.abspath(getattr(sc, "glacier_names_file", "") or ""))

    matrefs = []
    mmp = find_resource_meta(mati_path, "MATI", search_dirs)
    key = hash_from_path(mati_path)
    if mmp and os.path.exists(mmp):
        mm = parse_meta(bytearray(open(mmp, "rb").read()))
        matrefs = [rh for (rh, _) in mm["refs"]]
        if not key:
            key = "%016X" % mm["resource_id"]
    if not key:
        key = os.path.splitext(os.path.basename(mati_path))[0].upper()

    label = _resolve_material_label(mati, key, name_map)

    _clear_material(sc, key)             # replace, don't duplicate
    mt = sc.glacier_materials.add()
    mt.key = key
    mt.label = label
    mt.path = mati_path
    mt.is_blueprint = False

    for tex in mati["textures"]:
        it = sc.glacier_tex_slots.add()
        it.mati_hash = key
        it.slot_name = tex["name"]
        it.tex_index = tex["index"]
        it.old_hash = ("%016X" % matrefs[tex["index"]]) if tex["index"] < len(matrefs) else ""
        it.new_hash = ""
    for p in mati["params"]:
        it = sc.glacier_params.add()
        it.mati_hash = key
        it.name = p["name"]
        it.type = p["type"]
        it.data_off = p["data_off"]
        if p["type"] == 0x03:
            it.color = (p["values"][0], p["values"][1], p["values"][2])
        else:
            it.fval = p["values"][0]
        it.changed = False
    return key


def _resolve_old_hashes(sc, search_dirs, force=False):
    """Fill texture slots' original hash from each material's .MATI meta. With
    force=False only empty hashes are filled (used at export); force=True refills
    all (the Fill Hashes button). Returns (filled_count, materials_resolved)."""
    extra = [d for d in search_dirs if d and os.path.isdir(d)]
    filled = mats = 0
    for mt in getattr(sc, "glacier_materials", []):
        if mt.is_blueprint or not mt.path:
            continue
        slots = [ts for ts in sc.glacier_tex_slots
                 if ts.mati_hash == mt.key and ts.tex_index >= 0
                 and (force or not ts.old_hash)]
        if not slots:
            continue
        mmp = find_resource_meta(mt.path, "MATI", [os.path.dirname(mt.path)] + extra)
        if not (mmp and os.path.exists(mmp)):
            continue
        try:
            mm = parse_meta(bytearray(open(mmp, "rb").read()))
        except Exception:
            continue
        refs = [rh for (rh, _f) in mm["refs"]]
        mats += 1
        for ts in slots:
            if 0 <= ts.tex_index < len(refs):
                nh = "%016X" % refs[ts.tex_index]
                if ts.old_hash != nh:
                    ts.old_hash = nh
                    filled += 1
    return filled, mats


class GLACIER_UL_overrides(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        changed = bool(item.new_hash.strip()) and item.new_hash.strip().upper() != item.old_hash.upper()
        row = layout.row(align=True)
        row.label(text=item.label or item.old_hash[:10],
                  icon="CHECKMARK" if changed else "DOT")
        if changed:
            row.label(text="\u2192 " + item.new_hash[:10].upper())


class GLACIER_UL_materials(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        # how many slots on this material were edited
        edits = 0
        for ts in context.scene.glacier_tex_slots:
            if ts.mati_hash == item.key and (ts.new_hash.strip() or ts.tex_source != "HASH"):
                edits += 1
        ic = "NODE_MATERIAL" if item.is_blueprint else ("MATERIAL")
        row.label(text=item.label or "(unnamed)", icon=ic)
        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        sub.label(text=item.key[:8])
        if item.is_blueprint:
            sub.label(text="", icon="MOD_BUILD")
        elif edits:
            sub.label(text="", icon="CHECKMARK")


class GLACIER_OT_decode_texture(bpy.types.Operator):
    bl_idname = "glacier.decode_texture"
    bl_label = "Decode to image"
    bl_description = ("Decode a game .TEXT (+ optional .TEXD for full resolution) to "
                      "a .png/.tga in the Work Folder and load it into the .blend's "
                      "image cache (Image / Shader editors)")

    def execute(self, context):
        sc = context.scene
        tp = bpy.path.abspath(sc.glacier_decode_text or "").replace(os.sep, "/")
        if not tp or not os.path.isfile(tp):
            self.report({"WARNING"}, "Pick a .TEXT file to decode")
            return {"CANCELLED"}
        dp = bpy.path.abspath(sc.glacier_decode_texd or "").replace(os.sep, "/")
        if not (dp and os.path.isfile(dp)):
            dp = None
            meta = derive_resource_meta(tp, "TEXT")
            if os.path.exists(meta):
                try:
                    mm = parse_meta(bytearray(open(meta, "rb").read()))
                    for (h, _f) in mm["refs"]:
                        cand = os.path.join(os.path.dirname(tp), "%016X.TEXD" % h)
                        if os.path.exists(cand):
                            dp = cand
                            break
                except Exception:
                    dp = None
        try:
            w, h, rgba = decode_texture_file(tp, dp)
        except Exception as e:
            self.report({"ERROR"}, "Decode failed: %s" % e)
            return {"CANCELLED"}
        ext = ".png" if sc.glacier_decode_fmt == "PNG" else ".tga"
        wd = _glacier_work_dir(context)
        try:
            os.makedirs(wd, exist_ok=True)
            out = os.path.join(wd, os.path.splitext(os.path.basename(tp))[0] + ext)
            if ext == ".png":
                write_png(out, w, h, rgba)
            else:
                write_tga(out, w, h, rgba)
        except (OSError, PermissionError) as e:
            self.report({"ERROR"}, "Could not write to work folder (%s). Set a "
                        "writable Work Folder." % type(e).__name__)
            return {"CANCELLED"}
        img = _load_image_into_blend(out)
        self.report({"INFO"}, "Decoded %dx%d -> %s%s" % (
            w, h, os.path.basename(out),
            " (in image cache)" if img else " (saved; cache load failed)"))
        return {"FINISHED"}


def _glacier_work_dir(context):
    """Folder where decoded/re-encoded textures live. Uses the panel's Work
    Folder, else a folder next to the .blend, else a temp folder."""
    import tempfile
    sc = context.scene
    wd = bpy.path.abspath(getattr(sc, "glacier_work_dir", "") or "")
    if not wd:
        if bpy.data.filepath:
            wd = os.path.join(os.path.dirname(bpy.data.filepath), "007_textures")
        else:
            wd = os.path.join(tempfile.gettempdir(), "007_textures")
    return wd


def _glacier_search_dirs(context):
    sc = context.scene
    dirs = []
    scanf = bpy.path.abspath(getattr(sc, "glacier_scan_folder", "") or "")
    if scanf:
        dirs.append(scanf)
    for o in list(context.selected_objects) + list(context.scene.objects):
        s = o.get("glacier_source_prim")
        if s:
            d = os.path.dirname(s)
            if d not in dirs:
                dirs.append(d)
    for mt in getattr(sc, "glacier_materials", []):
        d = os.path.dirname(mt.path) if mt.path else ""
        if d and d not in dirs:
            dirs.append(d)
    return dirs


def _all_texture_dirs(context):
    """Every folder that might hold textures/metas: Work + Texture folders, the
    material search dirs, plus the folder of each slot's recorded .TEXT file."""
    sc = context.scene
    dirs = [_glacier_work_dir(context),
            bpy.path.abspath(getattr(sc, "glacier_tex_folder", "") or "")]
    dirs += _glacier_search_dirs(context)
    for ts in getattr(sc, "glacier_tex_slots", []):
        for attr in ("file_path", "file_path_texd"):
            p = getattr(ts, attr, "")
            if p:
                d = os.path.dirname(bpy.path.abspath(p))
                if d:
                    dirs.append(d)
    seen = set()
    return [d for d in dirs if d and os.path.isdir(d) and not (d in seen or seen.add(d))]


def _fill_slot_texd_hashes(sc, dirs):
    """Read every .TEXT meta under `dirs` and stamp each texture slot with its
    distinct .TEXD hash (the 0x9F ref of the slot's own .TEXT). Runs at decode, at
    Fill Hashes and at re-encode, so the .TEXD hash is known even when the original
    .TEXD file itself isn't present. Returns the pairing map."""
    pairs = pair_text_to_texd(dirs)
    for ts in getattr(sc, "glacier_tex_slots", []):
        if ts.texd_hash:
            continue
        oh = (ts.old_hash or "").upper()
        if oh and oh in pairs and pairs[oh] is not None:
            ts.texd_hash = "%016X" % pairs[oh]
    return pairs


def _load_image_into_blend(path):
    """Load a .png/.tga into Blender's image cache and pack it into the .blend so
    it shows up in the Image/Shader editors. Returns the image datablock or None."""
    try:
        img = bpy.data.images.load(path, check_existing=True)
    except Exception:
        return None
    try:
        img.reload()
    except Exception:
        pass
    try:
        img.pack()
    except Exception:
        pass
    return img


class GLACIER_OT_fill_hashes(bpy.types.Operator):
    bl_idname = "glacier.fill_hashes"
    bl_label = "Fill Original Hashes"
    bl_description = ("Find each loaded material's .MATI meta (searching the Scan / "
                      "Texture / Work folders and the model's folder) and fill in every "
                      "texture slot's original hash. Custom Texture exports then reuse "
                      "that hash automatically - no typing needed")

    def execute(self, context):
        sc = context.scene
        mats = [mt for mt in sc.glacier_materials if not mt.is_blueprint and mt.path]
        if not mats:
            self.report({"WARNING"}, "No materials loaded yet. In the Materials section "
                        "click 'Load From Imported Model' or 'Scan Folder' first, then "
                        "Fill Hashes (with this version, loading already fills them).")
            return {"CANCELLED"}
        dirs = []
        for attr in ("glacier_scan_folder", "glacier_tex_folder", "glacier_work_dir"):
            p = bpy.path.abspath(getattr(sc, attr, "") or "")
            if p and os.path.isdir(p):
                dirs.append(p)
        dirs += _glacier_search_dirs(context)
        filled, mats_done = _resolve_old_hashes(sc, dirs, force=True)
        # also stamp each slot's distinct .TEXD hash from the .TEXT metas
        td_pairs = _fill_slot_texd_hashes(sc, _all_texture_dirs(context))
        n_td = sum(1 for ts in sc.glacier_tex_slots if ts.texd_hash)
        if mats_done == 0:
            m = mats[0]
            cands = meta_path_candidates(m.path, "MATI")
            self.report({"WARNING"}, "Couldn't find a .MATI meta for '%s'. Looked for "
                        "'%s' or '%s' next to it and in your folders. Make sure the "
                        "_MATI.meta / .MATI.meta files are present." % (
                            os.path.basename(m.path),
                            os.path.basename(cands[0]), os.path.basename(cands[1])))
            return {"CANCELLED"}
        self.report({"INFO"}, "Filled %d slot hash(es) across %d material(s); "
                    "%d .TEXD hash(es) resolved" % (filled, mats_done, n_td))
        return {"FINISHED"}


class GLACIER_OT_decode_model(bpy.types.Operator):
    bl_idname = "glacier.decode_model"
    bl_label = "Decode Textures"
    bl_description = ("Search the Work / Texture / Scan folders (and every sub-folder) "
                      "for .TEXT + matching .TEXD, decode them to images in the Work "
                      "Folder and load them into the .blend's image cache. If a model's "
                      "materials are loaded, their Custom Texture slots are filled too")

    def _texd_for(self, tp, h, texd_by, pairs=None):
        dp = texd_by.get(h)
        if dp:
            return dp
        # the pairing read straight from the .TEXT metas (works even when the meta
        # isn't sitting next to the .TEXT file)
        if pairs and h in pairs and pairs[h] is not None:
            cand = texd_by.get("%016X" % pairs[h])
            if cand:
                return cand
        meta = derive_resource_meta(tp, "TEXT")
        if os.path.exists(meta):
            try:
                mm = parse_meta(bytearray(open(meta, "rb").read()))
                for (rh, _f) in mm["refs"]:
                    cand = texd_by.get("%016X" % rh)
                    if cand:
                        return cand
            except Exception:
                pass
        return None

    def execute(self, context):
        sc = context.scene
        wd = _glacier_work_dir(context)
        try:
            os.makedirs(wd, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.report({"ERROR"}, "Can't use work folder '%s' (%s). Set a writable "
                        "Work Folder." % (wd, type(e).__name__))
            return {"CANCELLED"}

        # search EVERYWHERE we might have textures: the Work Folder (the user often
        # drops .TEXT there), the Texture Folder, the material Scan Folder and any
        # model / material dirs - all walked recursively.
        dirs = [wd, bpy.path.abspath(getattr(sc, "glacier_tex_folder", "") or "")]
        dirs += _glacier_search_dirs(context)
        seen = set()
        dirs = [d for d in dirs if d and not (d in seen or seen.add(d))]
        text_by, texd_by = index_textures_by_hash(dirs)
        pairs = pair_text_to_texd(dirs)

        ext = ".png" if sc.glacier_decode_fmt == "PNG" else ".tga"
        n = fail = 0
        done = set()
        used_texd = set()
        errors = []

        def decode_one(h, tp, dp, slot=None):
            nonlocal n, fail
            try:
                w, hh, rgba = decode_texture_file(tp, dp)
            except Exception as e:
                fail += 1
                errors.append("%s: %s" % (h, e))
                print("[Glacier] decode failed for %s -> %s" % (tp, e))
                return
            if dp:
                used_texd.add(os.path.normcase(os.path.abspath(dp)))
            out = os.path.join(wd, "%s%s" % (h, ext))
            try:
                if ext == ".png":
                    write_png(out, w, hh, rgba)
                else:
                    write_tga(out, w, hh, rgba)
            except (OSError, PermissionError):
                fail += 1
                return
            _load_image_into_blend(out)
            done.add(h)
            n += 1
            if slot is not None:
                slot.tex_source = "IMAGE"
                slot.image_path = out
                slot.use_file = True
                if not slot.file_path:
                    slot.file_path = tp
                if dp and not slot.file_path_texd:
                    slot.file_path_texd = dp
                if not slot.texd_hash:
                    if dp:
                        th = hash_from_path(dp)
                        if th:
                            slot.texd_hash = th
                    elif h in pairs and pairs[h] is not None:
                        slot.texd_hash = "%016X" % pairs[h]

        # 1) loaded material slots first (so they get wired up for swap/preview)
        for ts in sc.glacier_tex_slots:
            if ts.tex_index < 0 or not ts.old_hash:
                continue
            h = ts.old_hash.upper()
            if h in done:
                continue
            tp = bpy.path.abspath(ts.file_path) if ts.file_path else text_by.get(h)
            if not tp or not os.path.isfile(tp):
                continue
            dp = bpy.path.abspath(ts.file_path_texd) if ts.file_path_texd else None
            if not (dp and os.path.isfile(dp)):
                dp = self._texd_for(tp, h, texd_by, pairs)
            decode_one(h, tp, dp, ts)

        # 2) every other .TEXT found in the folders (no materials needed)
        for h, tp in text_by.items():
            if h in done:
                continue
            decode_one(h, tp, self._texd_for(tp, h, texd_by, pairs))

        # 3) any .TEXD with no matching .TEXT (headerless) - auto-detect + decode
        #    standalone, so TEXD-only files still come out. Skip TEXDs already used
        #    as the full-res half of a decoded .TEXT (would just be a duplicate).
        for h, dp in texd_by.items():
            if h in done:
                continue
            if os.path.normcase(os.path.abspath(dp)) in used_texd:
                continue
            try:
                w, hh, rgba, fmt = decode_texd_standalone(dp)
            except Exception as e:
                fail += 1
                errors.append("%s: %s" % (h, e))
                print("[Glacier] TEXD decode failed for %s -> %s" % (dp, e))
                continue
            out = os.path.join(wd, "%s%s" % (h, ext))
            try:
                if ext == ".png":
                    write_png(out, w, hh, rgba)
                else:
                    write_tga(out, w, hh, rgba)
            except (OSError, PermissionError):
                fail += 1
                continue
            _load_image_into_blend(out)
            done.add(h)
            n += 1

        if n == 0 and not text_by and not texd_by:
            self.report({"WARNING"}, "No .TEXT/.TEXD files found. Point the Work Folder "
                        "or Texture Folder at a folder of textures.")
            return {"CANCELLED"}
        msg = "Decoded %d texture(s) into the image cache" % n
        if fail:
            msg += ", %d failed (%s)" % (fail, "; ".join(errors[:2]))
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class GLACIER_OT_decode_folder(bpy.types.Operator):
    bl_idname = "glacier.decode_folder"
    bl_label = "Decode Folder of Textures"
    bl_description = ("Search a folder (and all sub-folders) for every .TEXT, pair "
                      "each with its .TEXD, decode them to images in the Work Folder "
                      "and load them into the .blend's image cache - no model or "
                      "materials needed")

    def execute(self, context):
        sc = context.scene
        wd = _glacier_work_dir(context)
        try:
            os.makedirs(wd, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.report({"ERROR"}, "Can't use work folder (%s). Set a writable Work "
                        "Folder." % type(e).__name__)
            return {"CANCELLED"}
        # which folder to scan: the dedicated texture folder, else the material scan
        # folder, else everywhere we know about (model/material dirs)
        folder = bpy.path.abspath(getattr(sc, "glacier_tex_folder", "") or "")
        if not folder:
            folder = bpy.path.abspath(getattr(sc, "glacier_scan_folder", "") or "")
        dirs = [folder] if (folder and os.path.isdir(folder)) else _glacier_search_dirs(context)
        if not any(d and os.path.isdir(d) for d in dirs):
            self.report({"WARNING"}, "Set a Texture Folder (or Scan Folder) to search "
                        "for .TEXT/.TEXD files")
            return {"CANCELLED"}
        text_by, texd_by = index_textures_by_hash(dirs)
        if not text_by:
            self.report({"WARNING"}, "No .TEXT files found under the folder")
            return {"CANCELLED"}
        ext = ".png" if sc.glacier_decode_fmt == "PNG" else ".tga"
        n = fail = 0
        errors = []
        for h, tp in text_by.items():
            dp = texd_by.get(h)
            if not dp:
                meta = derive_resource_meta(tp, "TEXT")
                if os.path.exists(meta):
                    try:
                        mm = parse_meta(bytearray(open(meta, "rb").read()))
                        for (rh, _f) in mm["refs"]:
                            dp = texd_by.get("%016X" % rh)
                            if dp:
                                break
                    except Exception:
                        dp = None
            try:
                w, hh, rgba = decode_texture_file(tp, dp)
            except Exception as e:
                fail += 1
                errors.append("%s: %s" % (h, e))
                print("[Glacier] decode failed for %s -> %s" % (tp, e))
                continue
            out = os.path.join(wd, "%s%s" % (h, ext))
            try:
                if ext == ".png":
                    write_png(out, w, hh, rgba)
                else:
                    write_tga(out, w, hh, rgba)
            except (OSError, PermissionError):
                fail += 1
                continue
            _load_image_into_blend(out)
            n += 1
        msg = "Decoded %d texture(s) into the image cache" % n
        if fail:
            msg += ", %d failed (%s)" % (fail, "; ".join(errors[:2]))
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class GLACIER_OT_reencode(bpy.types.Operator):
    bl_idname = "glacier.reencode"
    bl_label = "Re-encode to .TEXT/.TEXD"
    bl_description = ("Convert every Custom Texture slot's image back into game "
                      ".TEXT + .TEXD (with metas) in the Work Folder, using the "
                      "chosen BC compression")

    def execute(self, context):
        sc = context.scene
        wd = _glacier_work_dir(context)
        try:
            os.makedirs(wd, exist_ok=True)
        except (OSError, PermissionError) as e:
            self.report({"ERROR"}, "Can't use work folder (%s)" % type(e).__name__)
            return {"CANCELLED"}
        bc = _bc_code_for(sc.glacier_bc_format)
        organize = getattr(sc, "glacier_organize_textures", True)
        # search EVERY folder that might hold the textures/metas (Work, Texture,
        # material dirs, and each slot's recorded .TEXT folder), so the original
        # .TEXT metas (which hold the distinct .TEXD hash) are always found.
        dirs = _all_texture_dirs(context)
        text_by, texd_by = index_textures_by_hash(dirs)
        pairs = _fill_slot_texd_hashes(sc, dirs)   # stamps ts.texd_hash too
        n = fail = 0
        seen = {}
        for ts in sc.glacier_tex_slots:
            if ts.tex_source != "IMAGE" or not ts.image_path:
                continue
            img = bpy.path.abspath(ts.image_path)
            if not os.path.isfile(img):
                continue
            tmpl_path = bpy.path.abspath(ts.file_path) if ts.file_path else ""
            if not (tmpl_path and os.path.exists(tmpl_path)) and ts.old_hash:
                tmpl_path = text_by.get(ts.old_hash.upper(), "")
            tmpl_bytes = None
            if tmpl_path and os.path.exists(tmpl_path):
                try:
                    tmpl_bytes = bytearray(open(tmpl_path, "rb").read())
                except OSError:
                    tmpl_bytes = None
            try:
                target = int(ts.new_hash.strip() or ts.old_hash, 16)
            except ValueError:
                self.report({"WARNING"}, "Slot '%s': needs a 16-hex hash" % ts.slot_name)
                continue
            if target in seen:
                self.report({"WARNING"}, "Slot '%s' and '%s' share hash %016X - skipping "
                            "the duplicate. Use a different hash per texture."
                            % (ts.slot_name, seen[target], target))
                continue
            seen[target] = ts.slot_name
            # resolve the .TEXD hash. `texd_known` = we know this texture's .TEXD
            # situation; `texd_hash` = its distinct hash (None means it has no .TEXD).
            # Priority: the slot's stamped texd_hash, then its recorded .TEXD path,
            # then the TEXT->TEXD pairing (by new hash, else original), then a
            # targeted meta search by hash, then the template meta.
            texd_hash = None
            texd_known = False
            if ts.texd_hash:
                try:
                    texd_hash = int(ts.texd_hash, 16); texd_known = True
                except ValueError:
                    pass
            if not texd_known and ts.file_path_texd:
                th = hash_from_path(bpy.path.abspath(ts.file_path_texd))
                if th:
                    texd_hash = int(th, 16); texd_known = True
            if not texd_known:
                for key in ("%016X" % target,
                            ts.old_hash.upper() if ts.old_hash else None):
                    if key and key in pairs:
                        texd_hash = pairs[key]; texd_known = True
                        break
            if not texd_known:
                # targeted hunt for THIS texture's .TEXT meta anywhere in the dirs
                for hx in ("%016X" % target,
                           ts.old_hash.upper() if ts.old_hash else None):
                    if not hx:
                        continue
                    cm = find_resource_meta("%s.TEXT" % hx, "TEXT", dirs)
                    if cm and os.path.exists(cm):
                        try:
                            mm = parse_meta(bytearray(open(cm, "rb").read()))
                            refs = mm.get("refs", [])
                            texd_hash = next((rh for (rh, fl) in refs if fl == 0x9F),
                                             refs[0][0] if refs else None)
                            texd_known = True
                            break
                        except Exception:
                            pass
            ct = os.path.join(wd, "_tmp.TEXT"); cd = os.path.join(wd, "_tmp.TEXD")
            ok, msg = convert_image_native(img, ct, cd, tmpl_bytes, bc)
            if not ok:
                self.report({"WARNING"}, "Slot '%s': %s" % (ts.slot_name, msg))
                fail += 1
                continue
            texd_bytes = open(cd, "rb").read()
            # has a .TEXD only if we know its hash AND there are big mips to store
            has_texd = (texd_hash is not None) and len(texd_bytes) > 0
            if not texd_known and len(texd_bytes) > 0:
                want = ts.old_hash.upper() or ("%016X" % target)
                self.report({"WARNING"}, "Slot '%s': no .TEXT meta found for hash %s, so "
                            "its .TEXD hash is unknown. Put %s_TEXT.meta (or "
                            "%s.TEXT.meta) in a searched folder. Writing .TEXT only."
                            % (ts.slot_name, want, want, want))
                has_texd = False
            try:
                dest = (lambda fn: organize_texture_dest(wd, fn, organize))
                if has_texd:
                    write_texture_pair(wd, target, open(ct, "rb").read(), texd_bytes,
                                       tmpl_path or None, dest_fn=dest,
                                       texd_hash=texd_hash)
                else:
                    write_texture_only(wd, target, open(ct, "rb").read(),
                                       tmpl_path or None, dest_fn=dest)
                n += 1
            except Exception as e:
                self.report({"WARNING"}, "Slot '%s': write failed (%s)"
                            % (ts.slot_name, e))
                fail += 1
            for tmp in (ct, cd):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        msg = "Re-encoded %d texture(s) to %s" % (n, wd)
        if fail:
            msg += " (%d failed)" % fail
        self.report({"INFO"}, msg)
        return {"FINISHED"}


# =============================================================================
# One-click render materials: turn an imported 007 material into a real Blender
# Principled material with the decoded textures wired up the way the game uses
# them (basecolor -> Base Color, SRM -> roughness/metallic/specular, normal ->
# Normal Map, translucency -> subsurface for skin, params drive strengths).
# =============================================================================
def _slot_role(slot_name):
    """Map a 007 texture-slot name to a render role."""
    s = (slot_name or "").lower()
    if "micronormal" in s or "microbump" in s or "detailnormal" in s:
        return "detail_normal"
    if "basecolor" in s or "albedo" in s or "diffuse" in s or "_color" in s \
            or s.endswith("color"):
        return "base"
    if "translucen" in s or "transmission" in s or "sss" in s or "subsurf" in s:
        return "translucency"
    if "normal" in s or s.endswith("_n") or "bump" in s:
        return "normal"
    if "srm" in s or "rmo" in s or "rough" in s or "specular" in s or "gloss" in s \
            or "_mask" in s or "orm" in s:
        return "srm"
    if "emiss" in s or "glow" in s:
        return "emission"
    if "opacity" in s or "alpha" in s or "transparency" in s:
        return "alpha"
    if "ao" in s or "occlusion" in s:
        return "ao"
    return "other"


def _bsdf_input(bsdf, *names):
    """First matching Principled input socket (names differ across Blender 3/4/5)."""
    for n in names:
        try:
            if n in bsdf.inputs:
                return bsdf.inputs[n]
        except Exception:
            pass
    return None


def _glacier_image_for(eff_hash, image_path=""):
    """Find a loaded Blender image for a texture: by explicit image path first,
    else any image whose name/filepath carries the 16-hex hash."""
    try:
        if image_path:
            base = os.path.basename(bpy.path.abspath(image_path))
            for im in bpy.data.images:
                if os.path.basename(im.filepath or "") == base or im.name == base:
                    return im
        h = (eff_hash or "").upper()
        if h:
            for im in bpy.data.images:
                nm = (im.name or "").upper()
                fp = (im.filepath or "").upper()
                if nm.startswith(h) or h in os.path.basename(fp):
                    return im
    except Exception:
        pass
    return None


def _resolve_render_roles(slots, images_by_slot):
    """Decide which texture drives which shader role. Names are tried first
    (mapTex_Basecolor, SRM, Normal...). Generic names (mapTexture2D_01/03/04,
    common on non-skin shaders) fall through to 'other', so we then use the
    texture FORMAT (BC5 is essentially always a normal map) and finally the
    texture ORDER to fill base -> srm -> normal. Tiny constant maps (<=8px,
    e.g. a 4x4 dummy) never drive base/srm."""
    def fmt_of(img):
        try:
            return (img.get("glacier_fmt") or "").upper()
        except Exception:
            return ""

    def is_tiny(img):
        try:
            return max(int(img.size[0]), int(img.size[1])) <= 8
        except Exception:
            return False

    roles = {}
    leftovers = []
    for ts in slots:
        img = images_by_slot.get(ts)
        if img is None:
            continue
        r = _slot_role(ts.slot_name)
        if r == "other":
            # a BC5 texture is a normal map even when the slot is named generically
            if fmt_of(img) == "BC5":
                r = "normal" if "normal" not in roles else "detail_normal"
                roles.setdefault(r, img)
                continue
            leftovers.append((ts, img))
            continue
        roles.setdefault(r, img)

    # positional fallback for generically-named colour maps, in slot order
    for ts, img in leftovers:
        if is_tiny(img):
            continue
        for r in ("base", "srm", "normal", "detail_normal"):
            if r not in roles:
                roles[r] = img
                break
    return roles


# ---------------------------------------------------------------------------
# Render-material engine (Quartermaster-style, driven by raw .MATI/.TEXT data)
# ---------------------------------------------------------------------------
# This is a port of the "First Light Quartermaster" material approach onto the
# data the 007 Toolkit already extracts from the game itself (texture slots,
# shader parameters and natively-decoded TEXT/TEXD images). Instead of reading
# an external JSON "contract", the material *family* (skin / eye / hair / fabric
# / generic) is inferred from the shader-template name, the slot names and the
# resolved roles, and a family-aware Principled node graph is built from there.
_GLACIER_OWNED = "glacier_owned"

_HAIR_TOKENS = ("hair", "eyebrow", "eyelash", "brow", "lash", "fur", "beard",
                "stubble", "moustache", "mustache")
_EYE_WET_TOKENS = ("wetness", "eye_wet", "wet_eye", "tearline", "tear_line")
_EYE_TOKENS = ("eye", "iris", "sclera", "cornea", "eyeball", "pupil", "ocular")
_SKIN_TOKENS = ("skin", "head", "face", "body", "neck", "torso", "arm", "leg",
                "hand", "flesh")
_FABRIC_TOKENS = ("cloth", "fabric", "shirt", "tshirt", "jacket", "vest", "suit",
                  "henley", "jean", "trouser", "pant", "cuff", "sock", "glove",
                  "boot", "shoe", "trainer", "belt", "holster", "pouch", "leather",
                  "denim", "wool", "cotton", "outfit", "kit", "gear", "tactical",
                  "weave")


def _glacier_material_family(hint, roles, params_by, slots):
    """Best-guess First-Light material family from everything we can see in the
    raw material: the shader-template/friendly name, the texture-slot names, the
    parameter names and which render roles resolved. Hair is tested before eye so
    that 'eyebrow'/'eyelash' classify as hair rather than eye."""
    parts = [(hint or "").lower()]
    for s in (slots or []):
        parts.append((getattr(s, "slot_name", "") or "").lower())
    for k in (params_by or {}):
        parts.append((k or "").lower())
    blob = " ".join(parts)

    def has(tokens):
        return any(t in blob for t in tokens)

    skin_template = "shadertemplate_01" in blob
    if has(_EYE_WET_TOKENS):
        return "eye_wetness"
    if has(_HAIR_TOKENS):
        return "hair"
    if has(_EYE_TOKENS):
        return "eye"
    if skin_template or has(_SKIN_TOKENS) or "translucency" in roles:
        return "skin"
    if has(_FABRIC_TOKENS):
        return "fabric"
    return "generic"


def _glacier_set_blend(mat, mode):
    """Set a material's transparency mode across Blender 4.2 (EEVEE legacy,
    `blend_method`) and 4.3 / 5.x (EEVEE-Next, `surface_render_method`). `mode`
    is one of OPAQUE / HASHED / BLEND. Everything is guarded so a missing
    attribute on any one version is a no-op rather than a crash."""
    for attr, val in (("blend_method", mode),):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass
    # EEVEE-Next: DITHERED ~ alpha-hashed cutout, BLENDED ~ alpha blend
    new_method = {"OPAQUE": "DITHERED", "HASHED": "DITHERED", "BLEND": "BLENDED"}.get(mode)
    if new_method is not None:
        try:
            mat.surface_render_method = new_method
        except Exception:
            pass
    for attr, val in (("show_transparent_back", mode == "BLEND"),
                      ("use_transparency_overlap", mode == "BLEND")):
        try:
            setattr(mat, attr, val)
        except Exception:
            pass


def _glacier_image_alpha_is_real(img):
    """True when an image genuinely carries an alpha channel worth using as
    opacity (4 channels). Avoids forcing transparency on opaque RGB maps."""
    try:
        return int(getattr(img, "channels", 0)) >= 4 or img.depth in (32, 64, 128)
    except Exception:
        return False


def build_glacier_blender_material(name, slots, params, images_by_slot, roles=None):
    """Create / replace a Blender material `name` with a family-aware Principled
    node graph that mirrors how 007 First Light uses its textures.

    The material *family* is detected (skin / eye / hair / fabric / generic) and
    drives the shader defaults and blend policy. Then each render role is wired:
      base        -> Base Color (x AO, x tint)
      srm         -> Separate -> Roughness (skin range) + Specular + Metallic
      normal      -> (BC5 rebuild B=1) -> Normal Map  (+ detail normal blended)
      translucency-> Multiply -> Subsurface
      emission    -> Emission Color/Strength
      alpha       -> Alpha (+ HASHED/BLEND blend policy)
    Material parameters become labelled Value/RGB nodes that drive the graph.
    Branches sit in titled frames; every node we create is tagged glacier_owned.

    Signature is unchanged from earlier versions so all call sites keep working:
    `roles`, when supplied, is a dict of {role_name: image} in the 007 role
    vocabulary (base/srm/normal/detail_normal/translucency/emission/ao/alpha)."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)

    roles = roles if roles is not None else _resolve_render_roles(slots, images_by_slot)
    pby = {(p.name or "").lower(): p for p in params}

    def pval(*keys):
        for k in keys:
            if k in pby:
                return pby[k]
        return None

    family = _glacier_material_family(name, roles, pby, slots)

    # ---- core nodes ---------------------------------------------------------
    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (820, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (440, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])

    def _set(names, value):
        s = _bsdf_input(bsdf, *names)
        if s is not None:
            try:
                s.default_value = value
            except Exception:
                pass

    # Principled defaults (Quartermaster-style preview-friendly base)
    _set(("Metallic",), 0.0)
    _set(("Roughness",), 0.6)
    _set(("Specular IOR Level", "Specular"), 0.5)
    _set(("Coat Weight", "Clearcoat"), 0.0)
    _set(("Alpha",), 1.0)

    # Family-specific shader defaults + blend policy
    blend_mode = "OPAQUE"
    if family == "skin":
        try:
            bsdf.subsurface_method = "RANDOM_WALK"
        except Exception:
            pass
        ssr = _bsdf_input(bsdf, "Subsurface Radius")
        if ssr is not None:
            try:
                ssr.default_value = (1.0, 0.2, 0.1)
            except Exception:
                pass
        _set(("Roughness",), 0.45)
    elif family in ("eye", "eye_wetness"):
        _set(("Roughness",), 0.12 if family == "eye" else 0.05)
        _set(("Specular IOR Level", "Specular"), 0.85)
        _set(("Subsurface Weight", "Subsurface"), 0.0)
        ior = _bsdf_input(bsdf, "IOR")
        if ior is not None:
            try:
                ior.default_value = 1.4
            except Exception:
                pass
    elif family == "hair":
        _set(("Roughness",), 0.5)
        _set(("Subsurface Weight", "Subsurface"), 0.0)

    # shared UV input so every image samples the model's first UV map
    _uv_holder = {}

    def uv_socket():
        n = _uv_holder.get("n")
        if n is None:
            n = nt.nodes.new("ShaderNodeUVMap")
            n.location = (-1640, 360); n.label = "UV"
            n[_GLACIER_OWNED] = True
            _uv_holder["n"] = n
        return n.outputs.get("UV")

    def frame(title, color=None):
        fr = nt.nodes.new("NodeFrame")
        fr.label = title
        fr[_GLACIER_OWNED] = True
        try:
            fr.label_size = 18
            if color:
                fr.use_custom_color = True
                fr.color = color
        except Exception:
            pass
        return fr

    def img_node(image, noncolor, x, y, parent, label):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = image
        n.location = (x, y); n.parent = parent; n.label = label
        n.width = 240
        n[_GLACIER_OWNED] = True
        if noncolor:
            try:
                image.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        else:
            try:
                image.colorspace_settings.name = "sRGB"
            except Exception:
                pass
        us = uv_socket()
        if us is not None:
            try:
                nt.links.new(us, n.inputs["Vector"])
            except Exception:
                pass
        return n

    def value_node(label, v, x, y, parent):
        n = nt.nodes.new("ShaderNodeValue")
        n.label = label; n.location = (x, y); n.parent = parent
        n[_GLACIER_OWNED] = True
        try:
            n.outputs[0].default_value = float(v)
        except Exception:
            pass
        return n

    def set_mode_rgb(n):
        try:
            n.mode = "RGB"
        except Exception:
            pass

    base_color_socket = None   # current node socket feeding Base Color

    # ---- BASE COLOR ---------------------------------------------------------
    base_img = roles.get("base")
    if base_img is not None:
        fr = frame("Base Color", (0.18, 0.12, 0.10))
        t = img_node(base_img, False, -1180, 520, fr, "Basecolor")
        base_color_socket = t.outputs["Color"]
        # Hair cards & other cutouts carry their mask in the diffuse alpha.
        if family == "hair" and "alpha" not in roles and _glacier_image_alpha_is_real(base_img):
            a_in = _bsdf_input(bsdf, "Alpha")
            if a_in is not None:
                nt.links.new(t.outputs["Alpha"], a_in)
                blend_mode = "BLEND"

    # ---- AMBIENT OCCLUSION (multiplied over base) ---------------------------
    if base_color_socket is not None and roles.get("ao") is not None:
        fr = frame("Ambient Occlusion", (0.12, 0.12, 0.14))
        t = img_node(roles["ao"], True, -880, 700, fr, "AO")
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"; mix.location = (-560, 640); mix.parent = fr
        mix.label = "AO x Base"; mix[_GLACIER_OWNED] = True
        try:
            mix.inputs["Fac"].default_value = 1.0
        except Exception:
            pass
        nt.links.new(base_color_socket, mix.inputs["Color1"])
        nt.links.new(t.outputs["Color"], mix.inputs["Color2"])
        base_color_socket = mix.outputs["Color"]

    # ---- TINT / SKIN COLOUR (param) multiplied over base --------------------
    tint = pval("skincolor", "tintcolor", "basecolor", "diffusecolor", "color")
    if tint is not None and getattr(tint, "type", 0) == 0x03:
        fr = frame("Tint", (0.13, 0.13, 0.13))
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.location = (-880, 880); rgb.parent = fr; rgb.label = "Tint Color"
        rgb[_GLACIER_OWNED] = True
        try:
            rgb.outputs[0].default_value = (tint.color[0], tint.color[1], tint.color[2], 1.0)
        except Exception:
            pass
        if base_color_socket is not None:
            mix = nt.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"; mix.location = (-560, 860); mix.parent = fr
            mix.label = "Tint x Base"; mix[_GLACIER_OWNED] = True
            try:
                mix.inputs["Fac"].default_value = 1.0
            except Exception:
                pass
            nt.links.new(base_color_socket, mix.inputs["Color1"])
            nt.links.new(rgb.outputs[0], mix.inputs["Color2"])
            base_color_socket = mix.outputs["Color"]
        else:
            base_color_socket = rgb.outputs[0]

    if base_color_socket is not None:
        bc = _bsdf_input(bsdf, "Base Color")
        if bc is not None:
            nt.links.new(base_color_socket, bc)

    # ---- SRM : Specular / Roughness / Metallic ------------------------------
    if roles.get("srm") is not None:
        fr = frame("SRM  -  Specular / Roughness / Metallic", (0.10, 0.14, 0.18))
        t = img_node(roles["srm"], True, -1180, 150, fr, "SRM")
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-880, 170); sep.parent = fr; sep.label = "Split SRM"
        sep[_GLACIER_OWNED] = True
        set_mode_rgb(sep)
        nt.links.new(t.outputs["Color"], sep.inputs[0])

        # Roughness = green. Skin remaps into the shader's roughness range.
        rmin = pval("roughness_min"); rmax = pval("roughness_max")
        rough_socket = sep.outputs[1]
        if family == "skin" or rmin is not None or rmax is not None:
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.location = (-560, 120); mr.parent = fr; mr.label = "Roughness Range"
            mr[_GLACIER_OWNED] = True
            try:
                mr.clamp = True
            except Exception:
                pass
            nt.links.new(sep.outputs[1], mr.inputs[0])
            if rmin is not None:
                vn = value_node("Roughness Min", rmin.fval, -560, 300, fr)
                nt.links.new(vn.outputs[0], mr.inputs["To Min"])
            else:
                try:
                    mr.inputs["To Min"].default_value = 0.0
                except Exception:
                    pass
            if rmax is not None:
                vn = value_node("Roughness Max", rmax.fval, -560, 230, fr)
                nt.links.new(vn.outputs[0], mr.inputs["To Max"])
            else:
                try:
                    mr.inputs["To Max"].default_value = 1.0
                except Exception:
                    pass
            rough_socket = mr.outputs[0]
        rough = _bsdf_input(bsdf, "Roughness")
        if rough is not None:
            nt.links.new(rough_socket, rough)
        spec = _bsdf_input(bsdf, "Specular IOR Level", "Specular")
        if spec is not None:
            nt.links.new(sep.outputs[0], spec)
        metal = _bsdf_input(bsdf, "Metallic")
        if metal is not None:
            nt.links.new(sep.outputs[2], metal)

    # ---- NORMAL (+ optional detail / micro normal) --------------------------
    def _build_normal(image, x, y, fr, label):
        """Image -> (rebuild B=1 for 2-channel BC5) -> Normal Map. Returns the
        Normal Map node so detail normals can be blended afterwards."""
        t = img_node(image, True, x, y, fr, label)
        src = t.outputs["Color"]
        # Game tangent-space normals are commonly BC5 (only R,G stored); rebuild
        # the blue channel so the Normal Map node gets a valid vector.
        if (image.get("glacier_fmt") or "").upper() == "BC5":
            sep = nt.nodes.new("ShaderNodeSeparateColor")
            sep.location = (x + 300, y + 20); sep.parent = fr; sep.label = "Split " + label
            sep[_GLACIER_OWNED] = True; set_mode_rgb(sep)
            nt.links.new(t.outputs["Color"], sep.inputs[0])
            comb = nt.nodes.new("ShaderNodeCombineColor")
            comb.location = (x + 560, y + 20); comb.parent = fr; comb.label = "Rebuild (B=1)"
            comb[_GLACIER_OWNED] = True; set_mode_rgb(comb)
            nt.links.new(sep.outputs[0], comb.inputs[0])
            nt.links.new(sep.outputs[1], comb.inputs[1])
            try:
                comb.inputs[2].default_value = 1.0
            except Exception:
                pass
            src = comb.outputs[0]
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (x + 820, y); nm.parent = fr; nm.label = label + " Map"
        nm[_GLACIER_OWNED] = True
        nt.links.new(src, nm.inputs["Color"])
        return nm

    if roles.get("normal") is not None:
        fr = frame("Normal", (0.10, 0.16, 0.10))
        nm = _build_normal(roles["normal"], -1480, -260, fr, "Normal")
        bs = pval("norm_bumpscale", "normalstrength", "bumpscale", "normal_intensity")
        if bs is not None:
            vn = value_node("Normal Strength", max(0.0, abs(bs.fval)) or 1.0, -660, -60, fr)
            try:
                nt.links.new(vn.outputs[0], nm.inputs["Strength"])
            except Exception:
                pass
        normal_out = nm.outputs["Normal"]
        # Detail / micro normal: blend (add + normalize) over the primary normal.
        if roles.get("detail_normal") is not None:
            dnm = _build_normal(roles["detail_normal"], -1480, -560, fr, "Detail Normal")
            add = nt.nodes.new("ShaderNodeVectorMath")
            add.operation = "ADD"; add.location = (180, -360); add.parent = fr
            add.label = "Combine Normals"; add[_GLACIER_OWNED] = True
            nt.links.new(nm.outputs["Normal"], add.inputs[0])
            nt.links.new(dnm.outputs["Normal"], add.inputs[1])
            nrm = nt.nodes.new("ShaderNodeVectorMath")
            nrm.operation = "NORMALIZE"; nrm.location = (360, -360); nrm.parent = fr
            nrm.label = "Normalize"; nrm[_GLACIER_OWNED] = True
            nt.links.new(add.outputs[0], nrm.inputs[0])
            normal_out = nrm.outputs[0]
        nin = _bsdf_input(bsdf, "Normal")
        if nin is not None:
            nt.links.new(normal_out, nin)

    # ---- TRANSLUCENCY -> SUBSURFACE -----------------------------------------
    if roles.get("translucency") is not None:
        fr = frame("Translucency  ->  Subsurface", (0.17, 0.10, 0.16))
        t = img_node(roles["translucency"], True, -1180, -940, fr, "Translucency")
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
        mul.location = (-760, -920); mul.parent = fr; mul.label = "Intensity"
        mul[_GLACIER_OWNED] = True
        nt.links.new(t.outputs["Color"], mul.inputs[0])
        ti = pval("translucency_intensity", "translucency", "sss_intensity")
        if ti is not None:
            vn = value_node("Translucency Intensity", max(0.0, ti.fval) or 0.3, -760, -770, fr)
            nt.links.new(vn.outputs[0], mul.inputs[1])
        else:
            try:
                mul.inputs[1].default_value = 0.3
            except Exception:
                pass
        ssw = _bsdf_input(bsdf, "Subsurface Weight", "Subsurface")
        if ssw is not None:
            nt.links.new(mul.outputs[0], ssw)

    # ---- EMISSION -----------------------------------------------------------
    if roles.get("emission") is not None:
        fr = frame("Emission", (0.18, 0.16, 0.08))
        t = img_node(roles["emission"], False, -1180, -1320, fr, "Emission")
        ec = _bsdf_input(bsdf, "Emission Color", "Emission")
        if ec is not None:
            nt.links.new(t.outputs["Color"], ec)
        es = _bsdf_input(bsdf, "Emission Strength")
        if es is not None:
            try:
                es.default_value = 1.0
            except Exception:
                pass

    # ---- ALPHA / OPACITY ----------------------------------------------------
    if roles.get("alpha") is not None:
        t = img_node(roles["alpha"], True, -1180, -1620, None, "Opacity")
        a = _bsdf_input(bsdf, "Alpha")
        if a is not None:
            # a dedicated opacity map: use its colour (greyscale) as alpha
            nt.links.new(t.outputs["Color"], a)
            blend_mode = "BLEND" if family in ("hair", "eye_wetness") else "HASHED"

    _glacier_set_blend(mat, blend_mode)

    # ---- debug stamp --------------------------------------------------------
    try:
        mat["glacier_family"] = family
        mat["glacier_shader_name"] = name
        mat["glacier_roles"] = ",".join(sorted(roles.keys()))
        mat["glacier_blend"] = blend_mode
    except Exception:
        pass
    return mat
def _ensure_slot_images(context, slots):
    """Make sure each slot's texture is decoded into a Blender image; decode any
    that are missing using the indexed .TEXT/.TEXD (or a standalone .TEXD). Returns
    {slot: image|None}."""
    sc = context.scene
    dirs = _all_texture_dirs(context)
    text_by, texd_by = index_textures_by_hash(dirs)
    pairs = pair_text_to_texd(dirs)
    wd = _glacier_work_dir(context)
    ext = ".png" if getattr(sc, "glacier_decode_fmt", "PNG") == "PNG" else ".tga"
    result = {}
    for ts in slots:
        eff = (ts.new_hash.strip() or ts.old_hash or "").upper()
        img = _glacier_image_for(eff, ts.image_path)
        # a Custom Texture slot may point at an image that isn't loaded yet
        if img is None and ts.image_path:
            p = bpy.path.abspath(ts.image_path)
            if os.path.isfile(p):
                img = _load_image_into_blend(p)
        # also accept an already-decoded file sitting in the Work Folder
        if img is None and eff:
            for cand in (os.path.join(wd, eff + ".png"), os.path.join(wd, eff + ".tga")):
                if os.path.isfile(cand):
                    img = _load_image_into_blend(cand)
                    if img is not None:
                        break
        if img is None and eff:
            tp = text_by.get(eff)
            dp = texd_by.get(eff)
            if dp is None and eff in pairs and pairs[eff] is not None:
                dp = texd_by.get("%016X" % pairs[eff])
            fmt_name = ""
            try:
                if tp:                                   # decode TEXT (+TEXD)
                    hdr = parse_text_header(bytearray(open(tp, "rb").read()))
                    if hdr:
                        fmt_name = hdr.get("format_name", "")
                    w, h, rgba = decode_texture_file(tp, dp)
                elif dp:                                 # only a TEXD - decode headerless
                    w, h, rgba, fmt_name = decode_texd_standalone(dp)
                else:
                    rgba = None
                if rgba is not None:
                    os.makedirs(wd, exist_ok=True)
                    outp = os.path.join(wd, "%s%s" % (eff, ext))
                    (write_png if ext == ".png" else write_tga)(outp, w, h, rgba)
                    img = _load_image_into_blend(outp)
            except Exception:
                img = None
            if img is not None and fmt_name:
                try:
                    img["glacier_fmt"] = fmt_name
                except Exception:
                    pass
        # stamp the source texture format on the image (helps role detection,
        # e.g. BC5 == normal map) even when the image was already loaded
        if img is not None and "glacier_fmt" not in img and eff:
            tp2 = text_by.get(eff)
            if tp2:
                try:
                    hdr = parse_text_header(bytearray(open(tp2, "rb").read()))
                    if hdr and hdr.get("format_name"):
                        img["glacier_fmt"] = hdr["format_name"]
                except Exception:
                    pass
        result[ts] = img
    return result


def _decode_texture_image(context, eff_hash, texd_hash="", image_path="",
                          text_by=None, texd_by=None, pairs=None, wd=None, ext=None):
    """Resolve (or decode) a single texture to a Blender image, by hash. Stamps
    the source format on the image (for role detection). Shared by the override
    build path and the reference build path."""
    sc = context.scene
    if text_by is None:
        dirs = _all_texture_dirs(context)
        text_by, texd_by = index_textures_by_hash(dirs)
        pairs = pair_text_to_texd(dirs)
    if wd is None:
        wd = _glacier_work_dir(context)
    if ext is None:
        ext = ".png" if getattr(sc, "glacier_decode_fmt", "PNG") == "PNG" else ".tga"
    eff = (eff_hash or "").upper()
    img = _glacier_image_for(eff, image_path)
    if img is None and image_path:
        p = bpy.path.abspath(image_path)
        if os.path.isfile(p):
            img = _load_image_into_blend(p)
    if img is None and eff:
        for cand in (os.path.join(wd, eff + ".png"), os.path.join(wd, eff + ".tga")):
            if os.path.isfile(cand):
                img = _load_image_into_blend(cand)
                if img is not None:
                    break
    fmt_name = ""
    if img is None and eff:
        tp = text_by.get(eff)
        dp = texd_by.get(eff)
        if dp is None and texd_hash:
            dp = texd_by.get((texd_hash or "").upper())
        if dp is None and eff in pairs and pairs[eff] is not None:
            dp = texd_by.get("%016X" % pairs[eff])
        try:
            if tp:
                hdr = parse_text_header(bytearray(open(tp, "rb").read()))
                if hdr:
                    fmt_name = hdr.get("format_name", "")
                w, h, rgba = decode_texture_file(tp, dp)
            elif dp:
                w, h, rgba, fmt_name = decode_texd_standalone(dp)
            else:
                rgba = None
            if rgba is not None:
                os.makedirs(wd, exist_ok=True)
                outp = os.path.join(wd, "%s%s" % (eff, ext))
                (write_png if ext == ".png" else write_tga)(outp, w, h, rgba)
                img = _load_image_into_blend(outp)
        except Exception:
            img = None
    if img is not None and "glacier_fmt" not in img:
        if not fmt_name and eff and text_by.get(eff):
            try:
                hdr = parse_text_header(bytearray(open(text_by[eff], "rb").read()))
                if hdr:
                    fmt_name = hdr.get("format_name", "")
            except Exception:
                pass
        if fmt_name:
            try:
                img["glacier_fmt"] = fmt_name
            except Exception:
                pass
    return img


def _ordered_model_material_keys(context):
    """Ordered list of MATI hashes the imported model references (PRIM meta ref
    order), for mapping object material ids to materials."""
    src = None
    for o in list(context.selected_objects) + list(context.scene.objects):
        if o.get("glacier_source_prim"):
            src = o["glacier_source_prim"]; break
    if not src:
        return []
    mp = derive_meta_path(src)
    if not mp or not os.path.exists(mp):
        return []
    try:
        m = parse_meta(bytearray(open(mp, "rb").read()))
    except Exception:
        return []
    return ["%016X" % h for (h, _f) in m["refs"]]


def _slot_image_path(ts):
    """On-disk image path for a slot, if any - its Custom Texture image, else a
    cached/decoded image matched by the slot's hash or paired .TEXD hash."""
    if ts.image_path:
        p = bpy.path.abspath(ts.image_path)
        if os.path.isfile(p):
            return p
    eff = (ts.new_hash.strip() or ts.old_hash or "").upper()
    img = _glacier_image_for(eff, ts.image_path)
    if img is None and ts.texd_hash:
        img = _glacier_image_for(ts.texd_hash.upper(), "")
    if img is not None:
        p = bpy.path.abspath(img.filepath or "")
        if os.path.isfile(p):
            return p
    return ""


def _generate_missing_textures(context, out_dir, organize, fmt_code):
    """For every texture slot that has an image but whose original .TEXT is NOT in
    the searched folders, build a brand-new .TEXT + .TEXD (+ metas) from the image
    so the game can load it. Returns (count, warnings)."""
    sc = context.scene
    dirs = _all_texture_dirs(context)
    text_by, _texd_by = index_textures_by_hash(dirs)
    _fill_slot_texd_hashes(sc, dirs)
    os.makedirs(out_dir, exist_ok=True)
    gen = 0
    warns = []
    seen = set()
    for ts in sc.glacier_tex_slots:
        th = (ts.new_hash.strip() or ts.old_hash or "").upper()
        if not th or th in seen:
            continue
        if th in text_by:                 # original .TEXT exists - nothing to make
            continue
        ip = _slot_image_path(ts)
        if not ip:                        # no image to build from
            continue
        seen.add(th)
        text_hash = int(th, 16)
        texd_hash = None
        if ts.texd_hash:
            try:
                texd_hash = int(ts.texd_hash, 16)
            except ValueError:
                texd_hash = None
        if texd_hash is None:
            ih = hash_from_path(ip)       # decoded-from-TEXD images are named <texdhash>
            if ih and ih.upper() != th:
                texd_hash = int(ih, 16)
        ct = os.path.join(out_dir, "_gen.TEXT"); cd = os.path.join(out_dir, "_gen.TEXD")
        dest = (lambda fn: organize_texture_dest(out_dir, fn, organize))
        try:
            if texd_hash is not None:
                ok, msg = convert_image_native(ip, ct, cd, None, fmt_code or 0x4C)
                if not ok:
                    warns.append("%s: %s" % (ts.slot_name, msg)); continue
                write_texture_pair(out_dir, text_hash, open(ct, "rb").read(),
                                   open(cd, "rb").read(), None, dest_fn=dest,
                                   texd_hash=texd_hash)
            else:
                # no .TEXD hash known: bundle every mip in the .TEXT, no .TEXD
                w, h, rgba = read_image_rgba(ip)
                if (w & (w-1)) or (h & (h-1)):
                    warns.append("%s: image must be power-of-two" % ts.slot_name); continue
                tmpl = template_from_scratch(w, h); tmpl["text_scale"] = 0
                text_all, _td = build_texture_v4(rgba, w, h, tmpl, fmt_code or 0x4C)
                write_texture_only(out_dir, text_hash, text_all, None, dest_fn=dest)
                warns.append("%s: no .TEXD hash known - wrote a single-file .TEXT "
                             "(works for smaller textures; verify in-game)" % ts.slot_name)
            gen += 1
        except Exception as e:
            warns.append("%s: generate failed (%s)" % (ts.slot_name, e))
        for tmp in (ct, cd):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return gen, warns


class GLACIER_OT_generate_missing(bpy.types.Operator):
    bl_idname = "glacier.generate_missing"
    bl_label = "Generate Missing .TEXT/.TEXD"
    bl_description = ("For any texture slot that has a decoded image but is MISSING its "
                      "original .TEXT, build a brand-new .TEXT + .TEXD (with metas) from "
                      "the image so the game can load it. Use this when you only have a "
                      "texture's .TEXD (the image) but not its .TEXT half. Set the slot's "
                      "source to Custom Texture and pick the decoded image first")

    def execute(self, context):
        sc = context.scene
        wd = _glacier_work_dir(context)
        bc = _bc_code_for(sc.glacier_bc_format)
        organize = getattr(sc, "glacier_organize_textures", True)
        try:
            gen, warns = _generate_missing_textures(context, wd, organize, bc)
        except (OSError, PermissionError) as e:
            self.report({"ERROR"}, "Can't write to the Work Folder (%s)" % type(e).__name__)
            return {"CANCELLED"}
        for w in warns[:4]:
            self.report({"WARNING"}, w)
        if gen == 0 and not warns:
            self.report({"INFO"}, "Nothing to generate - every slot either already has "
                        "its .TEXT or has no image. Set a slot to Custom Texture and pick "
                        "its decoded image, then try again.")
            return {"CANCELLED"}
        self.report({"INFO"}, "Generated %d missing texture(s) into %s" % (gen, wd))
        return {"FINISHED"}


class GLACIER_UL_render_slots(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        sub = row.row()
        sub.active = item.enabled
        sub.label(text=item.slot_name or item.tex_hash[:10])
        meta = (item.res + "  " + item.fmt).strip()
        if meta:
            r = sub.row(); r.alignment = "RIGHT"
            r.label(text=meta)


class GLACIER_OT_pull_reference(bpy.types.Operator):
    bl_idname = "glacier.pull_reference"
    bl_label = "Pull From Reference"
    bl_description = ("Read the textures the material(s) reference (.MATI/.MATB + their "
                      ".TEXT/.TEXD) and list them here, so you can pick exactly which "
                      "ones load into the render material and what each one drives. This "
                      "is independent of the export texture overrides")

    apply_to: bpy.props.EnumProperty(
        name="From",
        items=[("MODEL", "Whole Model", "Every loaded material"),
               ("ACTIVE", "Active Material", "Just the active material")],
        default="MODEL")

    def execute(self, context):
        sc = context.scene
        mats = [mt for mt in sc.glacier_materials if not mt.is_blueprint]
        if not mats:
            self.report({"WARNING"}, "Load a material first (Materials section)")
            return {"CANCELLED"}
        if self.apply_to == "ACTIVE":
            mats = [mt for mt in mats if mt.key == sc.glacier_active_material] or mats[:1]
        keys = {mt.key for mt in mats}
        for i in range(len(sc.glacier_render_slots) - 1, -1, -1):
            if sc.glacier_render_slots[i].mati_hash in keys:
                sc.glacier_render_slots.remove(i)

        dirs = _all_texture_dirs(context)
        text_by, _texd_by = index_textures_by_hash(dirs)
        n = 0
        for mt in mats:
            for ts in sc.glacier_tex_slots:
                if ts.mati_hash != mt.key:
                    continue
                tex_hash = (ts.old_hash or "").upper()
                fmt, res = "", ""
                tp = text_by.get(tex_hash)
                if tp:
                    try:
                        hdr = parse_text_header(bytearray(open(tp, "rb").read()))
                        if hdr:
                            fmt = hdr.get("format_name", "")
                            res = "%dx%d" % (hdr["width"], hdr["height"])
                    except Exception:
                        pass
                rs = sc.glacier_render_slots.add()
                rs.mati_hash = mt.key
                rs.slot_name = ts.slot_name
                rs.tex_hash = tex_hash
                rs.texd_hash = (ts.texd_hash or "").upper()
                rs.fmt = fmt
                rs.res = res
                rs.enabled = True
                rs.role = "AUTO"
                n += 1
        sc.glacier_render_from_reference = True
        self.report({"INFO"}, "Pulled %d texture(s) from reference. Pick what loads, "
                    "then Build Render Materials." % n)
        return {"FINISHED"}


class GLACIER_OT_build_materials(bpy.types.Operator):
    bl_idname = "glacier.build_materials"
    bl_label = "Build Render Materials"
    bl_description = ("Turn the loaded 007 material(s) into real Blender materials: it "
                      "decodes any missing textures, builds a Principled shader with the "
                      "basecolor, SRM (roughness/metallic/specular), normal and "
                      "translucency maps wired up the way the game uses them, and assigns "
                      "them to your imported meshes. One click to a render-ready model")

    apply_to: bpy.props.EnumProperty(
        name="Build",
        description="Which materials to build and where to put them",
        items=[
            ("MODEL", "Whole Model",
             "Build every loaded material and assign each to the imported meshes that "
             "use it (by material id) - the one-click option for a full character"),
            ("ACTIVE", "Active -> Selected",
             "Build only the active material and assign it to the selected objects"),
        ],
        default="MODEL")

    def execute(self, context):
        sc = context.scene
        mats = [mt for mt in sc.glacier_materials if not mt.is_blueprint]
        if not mats:
            self.report({"WARNING"}, "Load a material first (Materials section)")
            return {"CANCELLED"}
        if self.apply_to == "ACTIVE":
            active = sc.glacier_active_material
            mats = [mt for mt in mats if mt.key == active] or mats[:1]

        built = {}                       # material key -> bpy material
        missing = []                     # slots whose image couldn't be found
        use_ref = getattr(sc, "glacier_render_from_reference", False)
        ref_dirs = ref_text = ref_texd = ref_pairs = ref_wd = ref_ext = None
        # Many model parts share the same shader label (e.g. "gm_mTransform2D_01").
        # Without disambiguation every such part resolves to ONE Blender material
        # datablock and they overwrite each other, so only the last survives.
        # Count labels across ALL loaded materials (not just this build batch) so
        # the suffix is applied even when building one material at a time in
        # "Active" mode - otherwise Active builds would still collide.
        _all_mats = [m for m in sc.glacier_materials if not m.is_blueprint]
        _label_counts = {}
        for _mt in _all_mats:
            _lbl = _mt.label or _mt.key[:8]
            _label_counts[_lbl] = _label_counts.get(_lbl, 0) + 1
        for mt in mats:
            params = [p for p in sc.glacier_params if p.mati_hash == mt.key]
            _lbl = mt.label or mt.key[:8]
            name = "007_%s" % _lbl
            # Disambiguate when the label is shared OR is a raw engine shader
            # template (gm_m... / Transform2D), which is exactly the case that
            # used to collapse many parts onto one material datablock.
            _generic = _lbl.lower().startswith("gm_m") or "transform2d" in _lbl.lower()
            if _label_counts.get(_lbl, 0) > 1 or _generic:
                name = "%s [%s]" % (name, (mt.key or "")[:6])
            ref_slots = [rs for rs in sc.glacier_render_slots
                         if rs.mati_hash == mt.key and rs.enabled and rs.role != "SKIP"]
            try:
                if use_ref and ref_slots:
                    # build strictly from the chosen reference textures
                    if ref_text is None:
                        ref_dirs = _all_texture_dirs(context)
                        ref_text, ref_texd = index_textures_by_hash(ref_dirs)
                        ref_pairs = pair_text_to_texd(ref_dirs)
                        ref_wd = _glacier_work_dir(context)
                        ref_ext = (".png" if getattr(sc, "glacier_decode_fmt", "PNG")
                                   == "PNG" else ".tga")
                    roles, auto_slots, auto_imgs = {}, [], {}
                    for rs in ref_slots:
                        img = _decode_texture_image(
                            context, rs.tex_hash, rs.texd_hash, "",
                            ref_text, ref_texd, ref_pairs, ref_wd, ref_ext)
                        if img is None:
                            missing.append(rs.slot_name or rs.tex_hash[:8])
                            continue
                        if rs.fmt and "glacier_fmt" not in img:
                            try:
                                img["glacier_fmt"] = rs.fmt
                            except Exception:
                                pass
                        if rs.role == "AUTO":
                            auto_slots.append(rs)
                            auto_imgs[rs] = img
                        else:
                            roles.setdefault(rs.role.lower(), img)
                    if auto_slots:
                        for r, img in _resolve_render_roles(auto_slots, auto_imgs).items():
                            roles.setdefault(r, img)
                    bmat = build_glacier_blender_material(name, [], params, {}, roles=roles)
                else:
                    slots = [ts for ts in sc.glacier_tex_slots if ts.mati_hash == mt.key]
                    imgs = _ensure_slot_images(context, slots)
                    for ts in slots:
                        if imgs.get(ts) is None:
                            missing.append(ts.slot_name)
                    bmat = build_glacier_blender_material(name, slots, params, imgs)
            except Exception as e:
                self.report({"WARNING"}, "Couldn't build '%s' (%s)" % (name, e))
                continue
            built[mt.key] = bmat

        if not built:
            self.report({"WARNING"}, "No materials were built")
            return {"CANCELLED"}
        if missing:
            uniq = sorted(set(missing))
            self.report({"WARNING"}, "No decoded image for: %s. Decode those textures "
                        "first (Texture Tools > Decode Textures) or put their .TEXT/.TEXD "
                        "in the Search folder, then Build again." % ", ".join(uniq[:6]))

        assigned = 0
        if self.apply_to == "ACTIVE":
            mt = mats[0]
            bmat = built.get(mt.key)
            for o in context.selected_objects:
                if o.type == "MESH" and bmat is not None:
                    o.data.materials.clear()
                    o.data.materials.append(bmat)
                    assigned += 1
        else:
            order = _ordered_model_material_keys(context)
            mat_keys = [mt.key for mt in mats]
            # objects carry a material id (index); map id -> material by the model's
            # ref order when available, else by load order.
            for o in context.scene.objects:
                if o.type != "MESH" or "glacier_material_id" not in o:
                    continue
                mid = o["glacier_material_id"]
                key = None
                if order:
                    hits = [k for k in order if k in built]
                    if 0 <= mid < len(hits):
                        key = hits[mid]
                if key is None and 0 <= mid < len(mat_keys):
                    key = mat_keys[mid]
                if key is None and len(mat_keys) == 1:
                    key = mat_keys[0]
                bmat = built.get(key)
                if bmat is not None:
                    o.data.materials.clear()
                    o.data.materials.append(bmat)
                    assigned += 1

        self.report({"INFO"}, "Built %d material(s), assigned to %d mesh(es). "
                    "Switch the viewport to Material Preview / Rendered to see them."
                    % (len(built), assigned))
        return {"FINISHED"}


class GLACIER_OT_set_shading(bpy.types.Operator):
    bl_idname = "glacier.set_shading"
    bl_label = "Material Preview"
    bl_description = ("Switch the 3D viewport to Material Preview shading so the built "
                      "render materials are visible")

    def execute(self, context):
        done = False
        try:
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    for sp in area.spaces:
                        if sp.type == "VIEW_3D":
                            sp.shading.type = "MATERIAL"
                            done = True
        except Exception:
            pass
        if not done:
            self.report({"WARNING"}, "Couldn't find a 3D viewport to switch")
            return {"CANCELLED"}
        return {"FINISHED"}


def _model_material_hashes(context):
    """Set of 16-hex hashes referenced by the imported model's .prim meta, or
    None when no imported 007 model is present in the scene."""
    src = None
    for o in list(context.selected_objects) + list(context.scene.objects):
        if o.get("glacier_source_prim"):
            src = o["glacier_source_prim"]
            break
    if not src:
        return None
    mp = derive_meta_path(src)
    if not mp or not os.path.exists(mp):
        return None
    try:
        m = parse_meta(bytearray(open(mp, "rb").read()))
    except Exception:
        return None
    return {"%016X" % h for (h, _f) in m["refs"]}


class GLACIER_OT_override_refresh(bpy.types.Operator):
    bl_idname = "glacier.override_refresh"
    bl_label = "Load Materials From Model"
    bl_description = "Read material references and texture slots from the model's folder"

    def execute(self, context):
        src = None
        pool = list(context.selected_objects) + list(context.scene.objects)
        for o in pool:
            if o.get("glacier_source_prim"):
                src = o["glacier_source_prim"]
                break
        if not src:
            self.report({"WARNING"}, "No imported 007 model found in the scene")
            return {"CANCELLED"}
        mp = derive_meta_path(src)
        if not mp or not os.path.exists(mp):
            self.report({"WARNING"}, "Meta file not found next to the model")
            return {"CANCELLED"}
        m = parse_meta(bytearray(open(mp, "rb").read()))
        sc = context.scene
        sc.glacier_overrides.clear()
        sc.glacier_tex_slots.clear()
        sc.glacier_params.clear()
        sc.glacier_materials.clear()

        # naming: resolve each PRIM ref to a friendly label when it is a MATI
        folder = os.path.dirname(src)
        name_dirs = [folder]
        for attr in ("glacier_scan_folder", "glacier_tex_folder", "glacier_work_dir"):
            p = bpy.path.abspath(getattr(sc, attr, "") or "")
            if p and os.path.isdir(p) and p not in name_dirs:
                name_dirs.append(p)
        name_map = build_resource_name_map(
            name_dirs, bpy.path.abspath(getattr(sc, "glacier_names_file", "") or ""))
        ref_labels = {}
        n_mati = 0
        first_key = ""
        seen_refs = set()
        for (h, flag) in m["refs"]:
            if h in seen_refs:
                continue
            seen_refs.add(h)
            mati_path = os.path.join(folder, "%016X.MATI" % h)
            if os.path.exists(mati_path):
                try:
                    key = _load_mati_into_scene(sc, mati_path, name_map)
                    n_mati += 1
                    if not first_key:
                        first_key = key
                    mt = sc.glacier_materials[-1]
                    ref_labels[h] = mt.label or ("material %s" % key[:8])
                except Exception:
                    pass

        for i, (h, flag) in enumerate(m["refs"]):
            it = sc.glacier_overrides.add()
            nm = ref_labels.get(h)
            if not nm and ("%016X" % h) in name_map:
                nm = _pretty_material_name(name_map["%016X" % h])
            it.label = (nm if nm else "ref %d" % i) + "  [%016X]" % h
            it.old_hash = "%016X" % h
            it.new_hash = ""

        _dedup_materials(sc)
        if first_key:
            sc.glacier_active_material = first_key
        if n_mati == 0:
            self.report({"WARNING"}, "Found %d refs but no .MATI next to the model "
                        "- use 'Load Material File' to pick one" % len(m["refs"]))
        else:
            self.report({"INFO"}, "Loaded %d refs, %d material(s)" % (len(m["refs"]), n_mati))
        return {"FINISHED"}


class GLACIER_OT_load_material_file(bpy.types.Operator):
    bl_idname = "glacier.load_material_file"
    bl_label = "Load Material File"
    bl_description = "Load a .MATI (full) or .MATB (schema only) and add it as a material"

    def execute(self, context):
        sc = context.scene
        path = bpy.path.abspath(sc.glacier_mat_file).replace(os.sep, "/")
        if not path or not os.path.exists(path):
            self.report({"WARNING"}, "Pick an existing .MATI / .MATB file first")
            return {"CANCELLED"}
        if os.path.isdir(path):
            self.report({"WARNING"}, "That's a folder, not a file. Use the 'Scan' "
                        "button to load every material in a folder, or pick a single "
                        ".MATI / .MATB file here.")
            return {"CANCELLED"}
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".mati", ".matb"):
            self.report({"WARNING"}, "Pick a .MATI or .MATB file (got '%s')"
                        % (ext or "no extension"))
            return {"CANCELLED"}
        try:
            if ext == ".matb":
                props = parse_matb(bytearray(open(path, "rb").read()))
                key = hash_from_path(path) or os.path.basename(path).upper()
                _clear_material(sc, key)
                mt = sc.glacier_materials.add()
                mt.key = key
                mt.label = "blueprint"
                mt.path = path
                mt.is_blueprint = True
                for p in props:
                    if p["kind"] == "texture":
                        it = sc.glacier_tex_slots.add()
                        it.mati_hash = key
                        it.slot_name = p["name"]
                        it.tex_index = -1            # schema only: not swappable
                        it.old_hash = ""
                    else:
                        it = sc.glacier_params.add()
                        it.mati_hash = key
                        it.name = p["name"]
                        it.type = 0x03 if p["kind"] == "color" else 0x01
                        it.data_off = -1             # schema only: never exported
                        it.color = (1.0, 1.0, 1.0)   # avoid a misleading black swatch
                        it.changed = False
                # only focus the blueprint if nothing real is loaded yet
                if not any(not m.is_blueprint for m in sc.glacier_materials):
                    sc.glacier_active_material = key
                self.report({"INFO"}, "Loaded blueprint schema (%d properties). It "
                            "has no texture hashes or values - load the matching "
                            ".MATI to actually swap textures/params." % len(props))
            else:
                key = _load_mati_into_scene(sc, path)
                sc.glacier_active_material = key
                self.report({"INFO"}, "Loaded material %s" % key[:8])
            _dedup_materials(sc)
        except Exception as e:
            self.report({"ERROR"}, "Failed to parse: %s" % e)
            return {"CANCELLED"}
        return {"FINISHED"}


class GLACIER_OT_update_names(bpy.types.Operator):
    bl_idname = "glacier.update_names"
    bl_label = "Update Names"
    bl_description = ("Re-read the Names file (and the material folders) and refresh the "
                      "readable names on every loaded material. Use it after you point "
                      "the Names field at a different file/folder, or after the file "
                      "changed on disk - no need to reload the materials")

    def execute(self, context):
        sc = context.scene
        if not len(sc.glacier_materials):
            self.report({"WARNING"}, "Load some materials first")
            return {"CANCELLED"}

        # search the Names file, plus every folder we know about (incl. each
        # material's own folder), so a moved/renamed names file is still found
        dirs = []
        for attr in ("glacier_scan_folder", "glacier_tex_folder", "glacier_work_dir"):
            p = bpy.path.abspath(getattr(sc, attr, "") or "")
            if p and os.path.isdir(p) and p not in dirs:
                dirs.append(p)
        for mt in sc.glacier_materials:
            d = os.path.dirname(mt.path) if mt.path else ""
            if d and os.path.isdir(d) and d not in dirs:
                dirs.append(d)
        explicit = bpy.path.abspath(getattr(sc, "glacier_names_file", "") or "")
        if explicit and os.path.isdir(explicit):
            # allow pointing the Names field at a folder of .meta.json / hashlists
            if explicit not in dirs:
                dirs.append(explicit)
            explicit = ""
        name_map = build_resource_name_map(dirs, explicit)

        changed = 0
        for mt in sc.glacier_materials:
            if mt.is_blueprint:
                continue
            new_label = None
            if name_map and mt.key and mt.key.upper() in name_map:
                new_label = _pretty_material_name(name_map[mt.key.upper()])
            elif mt.path and os.path.isfile(mt.path):
                try:
                    mati = parse_mati(bytearray(open(mt.path, "rb").read()))
                    new_label = _mati_display_name(mati, mt.key)
                except Exception:
                    new_label = None
            if new_label and new_label != mt.label:
                mt.label = new_label
                changed += 1

        if not name_map:
            self.report({"WARNING"}, "No names found. Point the Names field at a hash "
                        "list / dependency .txt, or a folder of .meta.json files.")
            return {"CANCELLED"}
        self.report({"INFO"}, ("Updated %d material name(s)" % changed) if changed
                    else "Names are already up to date")
        return {"FINISHED"}


class GLACIER_OT_scan_folder(bpy.types.Operator):
    bl_idname = "glacier.scan_folder"
    bl_label = "Scan Folder for Materials & Textures"
    bl_description = ("Search a folder and ALL its sub-folders for .MATI / .TEXT / "
                      ".TEXD files, load every material, and auto-fill each texture "
                      "slot's hash and (when the .TEXT is found on disk) its file")

    def execute(self, context):
        sc = context.scene
        root = bpy.path.abspath(sc.glacier_scan_folder or "")
        if not root or not os.path.isdir(root):
            self.report({"WARNING"}, "Pick a folder to scan first")
            return {"CANCELLED"}

        mati_files = []
        text_by_hash, texd_by_hash = {}, {}
        try:
            walker = os.walk(root, onerror=lambda e: None, followlinks=False)
            for dirpath, _dirs, files in walker:
                for fn in files:
                    try:
                        ext = os.path.splitext(fn)[1].lower()
                        if ext not in (".mati", ".text", ".texd"):
                            continue
                        full = os.path.join(dirpath, fn)
                        h = hash_from_path(full)
                        if ext == ".mati":
                            mati_files.append(full)
                        elif ext == ".text" and h:
                            text_by_hash[h] = full
                        elif ext == ".texd" and h:
                            texd_by_hash[h] = full
                    except (OSError, ValueError):
                        continue
        except (OSError, PermissionError) as e:
            self.report({"WARNING"}, "Could not fully read the folder (%s). "
                        "Try copying the files somewhere like your Desktop and "
                        "scanning there." % type(e).__name__)

        if not mati_files:
            self.report({"WARNING"}, "No readable .MATI files found under %s" % root)
            return {"CANCELLED"}

        # optionally keep only the materials the imported model actually uses
        model_hashes = None
        note = ""
        if sc.glacier_scan_model_only:
            model_hashes = _model_material_hashes(context)
            if model_hashes is None:
                note = " (no imported model found - loaded all)"
            else:
                mati_files = [p for p in mati_files
                              if (hash_from_path(p) or "").upper() in model_hashes]
                if not mati_files:
                    self.report({"WARNING"}, "None of the scanned .MATI files are used "
                                "by the imported model. Turn off 'Only This Model's "
                                "Materials' to load them all.")
                    return {"CANCELLED"}

        sc.glacier_overrides.clear()
        sc.glacier_tex_slots.clear()
        sc.glacier_params.clear()
        sc.glacier_materials.clear()

        n_mati = n_auto = n_skip = 0
        first_key = ""
        name_dirs = [root]
        for attr in ("glacier_scan_folder", "glacier_tex_folder", "glacier_work_dir"):
            p = bpy.path.abspath(getattr(sc, attr, "") or "")
            if p and os.path.isdir(p) and p not in name_dirs:
                name_dirs.append(p)
        name_map = build_resource_name_map(
            name_dirs, bpy.path.abspath(getattr(sc, "glacier_names_file", "") or ""))
        for mp in sorted(mati_files):
            try:
                key = _load_mati_into_scene(sc, mp, name_map)
                n_mati += 1
                if not first_key:
                    first_key = key
            except (OSError, PermissionError):
                n_skip += 1
                continue
            except Exception:
                n_skip += 1
                continue
        _dedup_materials(sc)

        for ts in sc.glacier_tex_slots:
            if ts.tex_index < 0 or not ts.old_hash:
                continue
            tp = text_by_hash.get(ts.old_hash.upper())
            if tp:
                ts.tex_source = "CUSTOM"
                ts.use_file = True
                ts.file_path = tp
                try:
                    tmeta = derive_resource_meta(tp, "TEXT")
                    if tmeta and os.path.exists(tmeta):
                        mm = parse_meta(bytearray(open(tmeta, "rb").read()))
                        for (rh, _fl) in mm["refs"]:
                            dp = texd_by_hash.get("%016X" % rh)
                            if dp:
                                ts.file_path_texd = dp
                                break
                except (OSError, PermissionError):
                    pass
                except Exception:
                    pass
                n_auto += 1

        if first_key:
            sc.glacier_active_material = first_key
        sc.glacier_materials_index = 0
        msg = ("Scanned: %d materials, %d .TEXT, %d .TEXD - auto-filled %d slot(s)"
               % (n_mati, len(text_by_hash), len(texd_by_hash), n_auto))
        if n_skip:
            msg += " (%d unreadable skipped)" % n_skip
        msg += note
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class GLACIER_OT_override_add(bpy.types.Operator):
    bl_idname = "glacier.override_add"
    bl_label = "Add Override"

    def execute(self, context):
        context.scene.glacier_overrides.add()
        context.scene.glacier_overrides_index = len(context.scene.glacier_overrides) - 1
        return {"FINISHED"}


class GLACIER_OT_override_remove(bpy.types.Operator):
    bl_idname = "glacier.override_remove"
    bl_label = "Remove Override"

    def execute(self, context):
        sc = context.scene
        i = sc.glacier_overrides_index
        if 0 <= i < len(sc.glacier_overrides):
            sc.glacier_overrides.remove(i)
            sc.glacier_overrides_index = max(0, i - 1)
        return {"FINISHED"}


class GLACIER_OT_lod_show_all(bpy.types.Operator):
    bl_idname = "glacier.lod_show_all"
    bl_label = "Show All LODs"

    def execute(self, context):
        for objs in _glacier_lod_groups(context).values():
            for o in objs:
                o.hide_viewport = False
        return {"FINISHED"}


class GLACIER_OT_lod_show_lod0(bpy.types.Operator):
    bl_idname = "glacier.lod_show_lod0"
    bl_label = "LOD 0 Only"
    bl_description = "Show only the highest-detail mesh in each material group"

    def execute(self, context):
        context.scene.glacier_lod_level = 0
        _apply_lod_level(context, 0)
        return {"FINISHED"}


class GLACIER_OT_inspect_texture(bpy.types.Operator):
    bl_idname = "glacier.inspect_texture"
    bl_label = "Inspect .TEXT"
    bl_description = ("Read a game-format .TEXT header and report its size, format "
                      "and mip count - use it to sanity-check a custom texture")

    def execute(self, context):
        fp = bpy.path.abspath(context.scene.glacier_inspect_tex or "")
        if not fp or not os.path.exists(fp):
            self.report({"WARNING"}, "Pick a .TEXT file first")
            return {"CANCELLED"}
        try:
            info = parse_text_header(bytearray(open(fp, "rb").read()))
        except OSError as e:
            self.report({"ERROR"}, "Cannot read file: %s" % e)
            return {"CANCELLED"}
        if info is None:
            self.report({"WARNING"}, "Not a recognizable 007 .TEXT header")
            return {"CANCELLED"}
        self.report({"INFO"}, "%dx%d  %s  %d mips" % (
            info["width"], info["height"], info["format_name"], info["mips"]))
        return {"FINISHED"}


# =============================================================================
# RPKG Chunk Browser  (mass-import TEXT/TEXD straight from a game chunk)
# =============================================================================
class GlacierChunkEntry(bpy.types.PropertyGroup):
    """One resource row shown in the chunk browser."""
    hash: bpy.props.StringProperty()
    ext: bpy.props.StringProperty()
    size: bpy.props.IntProperty()          # uncompressed size (display only)
    sel: bpy.props.BoolProperty(name="", default=False)


_CHUNK_FILTER_ITEMS = [
    ("TEXTD", "TEXT + TEXD", "Textures only - both halves"),
    ("TEXT", "TEXT only", "Small texture headers"),
    ("TEXD", "TEXD only", "Full-resolution texture data"),
    ("ALL", "Everything", "Every resource type in the chunk"),
]
_CHUNK_FILTER_EXTS = {
    "TEXTD": {"TEXT", "TEXD"},
    "TEXT": {"TEXT"},
    "TEXD": {"TEXD"},
    "ALL": None,
}
# how many rows we put in the on-screen list (extraction is NOT limited by this)
_CHUNK_UI_CAP = 4000


class GLACIER_UL_chunk_entries(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active, prop):
        row = layout.row(align=True)
        row.prop(item, "sel", text="")
        row.label(text=item.hash, icon="FILE")
        r = row.row(); r.alignment = "RIGHT"
        r.label(text="%s  %s" % (item.ext, _human_size(item.size)))


def _human_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return "%.0f%s" % (n, unit) if unit == "B" else "%.1f%s" % (n, unit)
        n /= 1024.0
    return "%dB" % n


def _chunk_populate(context):
    """(Re)build the on-screen list from the cached parse, honouring filter +
    search. Returns (shown, total_matching)."""
    sc = context.scene
    path = bpy.path.abspath(sc.glacier_chunk_path or "")
    sc.glacier_chunk_entries.clear()
    if not path or not os.path.isfile(path):
        return 0, 0
    arc = rpkg_open_cached(path)
    exts = _CHUNK_FILTER_EXTS.get(sc.glacier_chunk_filter, None)
    matches = arc.filter(exts, sc.glacier_chunk_search)
    for e in matches[:_CHUNK_UI_CAP]:
        it = sc.glacier_chunk_entries.add()
        it.hash = e["hash"]; it.ext = e["ext"]
        it.size = min(e["size_uncompressed"], 0x7FFFFFFF)
        it.sel = False
    return min(len(matches), _CHUNK_UI_CAP), len(matches)


class GLACIER_OT_chunk_scan(bpy.types.Operator):
    bl_idname = "glacier.chunk_scan"
    bl_label = "Scan Chunk"
    bl_description = ("Read the chunk's index and list its TEXT/TEXD (or all) "
                     "resources. Point at the full chunkNN.rpkg in the game's "
                     "Runtime folder")
    bl_options = {"REGISTER"}

    def execute(self, context):
        sc = context.scene
        path = bpy.path.abspath(sc.glacier_chunk_path or "")
        if not path or not os.path.isfile(path):
            self.report({"ERROR"}, "Pick a chunk .rpkg file first")
            return {"CANCELLED"}
        try:
            arc = rpkg_open_cached(path)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        shown, total = _chunk_populate(context)
        sc.glacier_chunk_total = len(arc.entries)
        sc.glacier_chunk_shown = shown
        sc.glacier_chunk_matching = total
        note = "" if shown >= total else " (showing first %d - narrow with Search)" % shown
        self.report({"INFO"}, "Chunk has %d resources; %d match filter%s" %
                    (len(arc.entries), total, note))
        return {"FINISHED"}


class GLACIER_OT_chunk_refresh(bpy.types.Operator):
    bl_idname = "glacier.chunk_refresh"
    bl_label = "Apply Filter"
    bl_description = "Re-apply the type filter and hash search to the list"
    bl_options = {"REGISTER"}

    def execute(self, context):
        sc = context.scene
        path = bpy.path.abspath(sc.glacier_chunk_path or "")
        if not path or not os.path.isfile(path):
            self.report({"ERROR"}, "Scan a chunk first")
            return {"CANCELLED"}
        shown, total = _chunk_populate(context)
        sc.glacier_chunk_shown = shown
        sc.glacier_chunk_matching = total
        return {"FINISHED"}


class GLACIER_OT_chunk_select(bpy.types.Operator):
    bl_idname = "glacier.chunk_select"
    bl_label = "Select"
    bl_description = "Tick or untick the shown rows"
    bl_options = {"REGISTER"}
    mode: bpy.props.StringProperty(default="ALL")

    def execute(self, context):
        val = (self.mode == "ALL")
        for it in context.scene.glacier_chunk_entries:
            it.sel = val
        return {"FINISHED"}


_HASH_RE = re.compile(r'[0-9A-Fa-f]{16}')


class GLACIER_OT_chunk_paste_select(bpy.types.Operator):
    bl_idname = "glacier.chunk_paste_select"
    bl_label = "Select From List"
    bl_description = ("Parse the Hash List field for 16-hex hashes and tick every "
                     "matching row. Accepts any separator (newline, comma, space, "
                     "tab) and ignores file extensions")
    bl_options = {"REGISTER"}

    def execute(self, context):
        sc = context.scene
        txt = sc.glacier_chunk_paste_hashes or ""
        want = set(m.group().upper() for m in _HASH_RE.finditer(txt))
        if not want:
            self.report({"WARNING"}, "No 16-hex hashes found in the list. Paste "
                        "hashes like 01673C4916558956, one per line or separated "
                        "by commas/spaces")
            return {"CANCELLED"}
        found = 0
        for it in sc.glacier_chunk_entries:
            if it.hash.upper() in want:
                it.sel = True
                found += 1
        missing = len(want) - found
        msg = "Selected %d of %d pasted hash%s" % (found, len(want),
                                                    "es" if len(want) != 1 else "")
        if missing:
            msg += " (%d not in the current list — check the filter)" % missing
        self.report({"INFO"}, msg)
        return {"FINISHED"}


def _chunk_decode_pair(context, out_dir, text_path, texd_path):
    """Best-effort: decode an extracted TEXT(+paired TEXD) to a PNG via the
    existing codec. Non-fatal."""
    try:
        eff = hash_from_path(text_path) if text_path else ""
        text_by, texd_by = index_textures_by_hash([out_dir])
        pairs = pair_text_to_texd([out_dir])
        img = _decode_texture_image(context, eff or "", "", text_path or "",
                                    text_by, texd_by, pairs, out_dir, "png")
        return img is not None
    except Exception:
        return False


class GLACIER_OT_chunk_extract(bpy.types.Operator):
    bl_idname = "glacier.chunk_extract"
    bl_label = "Import Selected"
    bl_description = ("Extract the ticked resources (descramble + decompress) to "
                     "the output folder as <hash>.<EXT> plus their .meta")
    bl_options = {"REGISTER"}
    # SELECTED = ticked rows; ALLMATCH = every resource matching the filter
    scope: bpy.props.StringProperty(default="SELECTED")

    def execute(self, context):
        sc = context.scene
        path = bpy.path.abspath(sc.glacier_chunk_path or "")
        if not path or not os.path.isfile(path):
            self.report({"ERROR"}, "Scan a chunk first")
            return {"CANCELLED"}
        out_dir = bpy.path.abspath(sc.glacier_chunk_out or sc.glacier_work_dir or "")
        if not out_dir:
            self.report({"ERROR"}, "Set an output folder (Chunk Out or Work)")
            return {"CANCELLED"}
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            self.report({"ERROR"}, "Cannot create output folder: %s" % e)
            return {"CANCELLED"}

        try:
            arc = rpkg_open_cached(path)
        except Exception as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}

        # custom XOR key override (hex), else the default
        key = RPKG_XOR_KEY
        kh = (sc.glacier_chunk_xor_key or "").strip().replace(" ", "")
        if kh:
            try:
                key = bytes.fromhex(kh)
            except ValueError:
                self.report({"ERROR"}, "XOR key must be hex (e.g. DC45A69C...)")
                return {"CANCELLED"}

        if self.scope == "ALLMATCH":
            exts = _CHUNK_FILTER_EXTS.get(sc.glacier_chunk_filter, None)
            targets = arc.filter(exts, sc.glacier_chunk_search)
        else:
            want = {it.hash for it in sc.glacier_chunk_entries if it.sel}
            targets = [arc.by_hash[h] for h in want if h in arc.by_hash]
        if not targets:
            self.report({"WARNING"}, "Nothing selected to extract")
            return {"CANCELLED"}

        organize = sc.glacier_organize_textures
        written = 0
        text_paths = []
        warned_garbled = False
        for e in targets:
            try:
                data = arc.extract(e, key)
            except Exception:
                continue
            sub = os.path.join(out_dir, e["ext"], e["hash"]) if organize else out_dir
            try:
                os.makedirs(sub, exist_ok=True)
            except OSError:
                sub = out_dir
            res_path = os.path.join(sub, "%s.%s" % (e["hash"], e["ext"]))
            try:
                with open(res_path, "wb") as f:
                    f.write(data)
                with open(os.path.join(sub, "%s_%s.meta" % (e["hash"], e["ext"])),
                          "wb") as f:
                    f.write(arc.standalone_meta(e))
                written += 1
                if e["ext"] == "TEXT":
                    text_paths.append(res_path)
                    if not warned_garbled and not _looks_like_text(data):
                        warned_garbled = True
            except OSError:
                continue

        # optional decode of extracted TEXT(+paired TEXD) to PNG
        dec = 0
        if sc.glacier_chunk_decode and text_paths:
            for tp in text_paths:
                if _chunk_decode_pair(context, out_dir, tp, ""):
                    dec += 1

        msg = "Extracted %d resource(s) to %s" % (written, os.path.basename(out_dir) or out_dir)
        if dec:
            msg += ", decoded %d to PNG" % dec
        if warned_garbled:
            msg += ". NOTE: a TEXT header looks wrong - the XOR key may differ; " \
                   "try clearing/changing the XOR Key field"
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, msg)
        return {"FINISHED"}


def _looks_like_text(data):
    """Sanity check an extracted .TEXT so we can warn if the descramble key is
    wrong. Uses the same header parser the rest of the toolkit relies on."""
    try:
        info = parse_text_header(data)
        return bool(info) and 0 < info.get("width", 0) <= 16384 \
            and 0 < info.get("height", 0) <= 16384
    except Exception:
        return False


class GLACIER_OT_chunk_browser(bpy.types.Operator):
    bl_idname = "glacier.chunk_browser"
    bl_label = "Open Chunk Browser"
    bl_description = ("Open a window to browse the chunk's resources and mass-"
                     "import TEXT/TEXD without RPKG-Tool")
    bl_options = {"REGISTER"}

    def invoke(self, context, event):
        sc = context.scene
        path = bpy.path.abspath(sc.glacier_chunk_path or "")
        if path and os.path.isfile(path) and not len(sc.glacier_chunk_entries):
            try:
                rpkg_open_cached(path)
                shown, total = _chunk_populate(context)
                sc.glacier_chunk_shown = shown
                sc.glacier_chunk_matching = total
            except Exception as e:
                self.report({"ERROR"}, str(e))
        return context.window_manager.invoke_props_dialog(self, width=640)

    def draw(self, context):
        sc = context.scene
        layout = self.layout
        box = layout.box()
        box.label(text="RPKG Chunk Browser", icon="FILEBROWSER")
        box.prop(sc, "glacier_chunk_path", text="Chunk")
        row = box.row(align=True)
        row.prop(sc, "glacier_chunk_filter", text="")
        row.prop(sc, "glacier_chunk_search", text="", icon="VIEWZOOM")
        row.operator("glacier.chunk_scan", text="Scan", icon="FILE_REFRESH")
        row.operator("glacier.chunk_refresh", text="", icon="FILTER")

        if len(sc.glacier_chunk_entries):
            info = box.row()
            info.enabled = False
            cap = "" if sc.glacier_chunk_shown >= sc.glacier_chunk_matching \
                else " of %d (narrow with Search)" % sc.glacier_chunk_matching
            info.label(text="Showing %d%s  -  chunk total %d" %
                       (sc.glacier_chunk_shown, cap, sc.glacier_chunk_total))
            box.template_list("GLACIER_UL_chunk_entries", "", sc,
                              "glacier_chunk_entries", sc, "glacier_chunk_index",
                              rows=12)
            selrow = box.row(align=True)
            op = selrow.operator("glacier.chunk_select", text="Select All")
            op.mode = "ALL"
            op = selrow.operator("glacier.chunk_select", text="Select None")
            op.mode = "NONE"

            pb = box.box()
            pb.label(text="Paste Hash List", icon="PASTEDOWN")
            pb.prop(sc, "glacier_chunk_paste_hashes", text="")
            r = pb.row(); r.enabled = False
            r.label(text="Paste hashes (one per line, or comma/space separated)")
            pb.operator("glacier.chunk_paste_select", text="Select Pasted Hashes",
                        icon="CHECKMARK")
        else:
            box.label(text="Pick a chunk and press Scan", icon="INFO")

        outb = layout.box()
        outb.label(text="Extract To", icon="EXPORT")
        outb.prop(sc, "glacier_chunk_out", text="Folder")
        r = outb.row()
        r.enabled = False
        r.label(text="Blank = use the Work folder. Writes <hash>.TEXT/.TEXD + .meta")
        outb.prop(sc, "glacier_chunk_decode")
        outb.prop(sc, "glacier_organize_textures", text="Sort into TYPE/<hash>/")
        adv = outb.row()
        adv.prop(sc, "glacier_chunk_xor_key", text="XOR Key")
        a2 = outb.row(); a2.enabled = False
        a2.label(text="Advanced: leave blank for the default First Light key")

        act = layout.row(align=True)
        op = act.operator("glacier.chunk_extract", text="Import Selected",
                          icon="IMPORT")
        op.scope = "SELECTED"
        op = act.operator("glacier.chunk_extract", text="Import All Matching",
                          icon="IMPORT")
        op.scope = "ALLMATCH"

    def execute(self, context):
        # closing the dialog with OK does nothing destructive; extraction is via
        # the explicit buttons so the window stays open while you work.
        return {"FINISHED"}


class VIEW3D_PT_glacier_mesh_tools(bpy.types.Panel):
    bl_label = "007 Mesh Tools"
    bl_idname = "VIEW3D_PT_glacier_mesh_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "007 Mesh Tools"

    def draw(self, context):
        sc = context.scene
        layout = self.layout

        # Version banner - so you can confirm at a glance the new build loaded.
        ver = ".".join(str(x) for x in bl_info["version"])
        vr = layout.row(align=True)
        vr.alignment = "RIGHT"
        vr.label(text="007 Toolkit  v%s" % ver, icon="CHECKMARK")

        def section(prop, title, icon, badge=None, disabled=False):
            box = layout.box()
            head = box.row(align=True)
            if disabled:
                head.enabled = False        # grey out the whole header + toggle
            head.prop(sc, prop, text="", emboss=False,
                      icon="TRIA_DOWN" if (getattr(sc, prop) and not disabled)
                      else "TRIA_RIGHT")
            head.label(text=title, icon=icon)
            if disabled:
                r = head.row(); r.alignment = "RIGHT"; r.enabled = False
                r.label(text="dev build only")
            elif badge:
                r = head.row(); r.alignment = "RIGHT"; r.label(text=str(badge))
            if disabled:
                return None                 # never expands in the public build
            return box.column() if getattr(sc, prop) else None

        def sub(parent, title, icon="DOT"):
            box = parent.box()
            box.label(text=title, icon=icon)
            return box

        def hint(parent, text, icon="FILE_BLANK"):
            # a greyed-out helper line (e.g. which file types to point at)
            r = parent.row()
            r.enabled = False
            r.label(text=text, icon=icon)

        nmat = len(sc.glacier_materials)
        groups = _glacier_lod_groups(context)
        active = sc.glacier_active_material

        # ============ IMPORT / EXPORT =====================================
        b = section("glacier_show_io", "Import / Export", "IMPORT")
        if b:
            sb = sub(b, "Import", "IMPORT")
            col = sb.column(align=True); col.scale_y = 1.15
            col.operator("import_scene.glacier2_007_prim",
                         text="Import Model  (.prim)", icon="MESH_MONKEY")
            col.operator("import_scene.glacier2_007_borg",
                         text="Import Skeleton  (.borg)", icon="ARMATURE_DATA")
            sb = sub(b, "Export", "EXPORT")
            col = sb.column(align=True); col.scale_y = 1.15
            col.operator("export_scene.glacier2_007_prim",
                         text="Export Model + Edits", icon="EXPORT")
            hint(sb, "Writes .prim / .MATI / .TEXT / .TEXD + metas")

        # ============ MATERIALS ===========================================
        b = section("glacier_show_mats", "Materials", "MATERIAL", nmat or None)
        if b:
            sb = sub(b, "From Imported Model", "FILE_REFRESH")
            sb.operator("glacier.override_refresh",
                        text="Load From Imported Model", icon="FILE_REFRESH")
            hint(sb, "Uses the materials of the .prim you imported")
            sb = sub(b, "Scan a Folder", "VIEWZOOM")
            hint(sb, "Folder of extracted files (.MATI / .TEXT / .TEXD)", "FILE_FOLDER")
            sb.prop(sc, "glacier_scan_folder", text="")
            sb.prop(sc, "glacier_scan_model_only")
            sb.operator("glacier.scan_folder", text="Scan Folder", icon="ZOOM_ALL")
            sb = sub(b, "Single File", "FILE")
            hint(sb, "Pick one .MATI (full material) or .MATB (schema)")
            row = sb.row(align=True)
            row.prop(sc, "glacier_mat_file", text="")
            row.operator("glacier.load_material_file", text="", icon="FILEBROWSER")
            sb = sub(b, "Names File (optional)", "SYNTAX_OFF")
            hint(sb, "Shows real names instead of hashes")
            row = sb.row(align=True)
            row.prop(sc, "glacier_names_file", text="")
            row.operator("glacier.update_names", text="", icon="FILE_REFRESH")
            sb.operator("glacier.update_names", text="Update Names", icon="FILE_REFRESH")
            hint(sb, "A hash list / dependency .txt, or a folder of .meta.json")
            hint(sb, "Get one: RPKG-Tool > Generate Hash List, or your", "QUESTION")
            hint(sb, "extracted .meta.json files (they hold the IOI paths)", "BLANK1")
            if nmat:
                sb = sub(b, "Loaded Materials", "PRESET")
                sb.template_list("GLACIER_UL_materials", "", sc, "glacier_materials",
                                 sc, "glacier_materials_index", rows=5)

        # ============ RENDER MATERIALS ====================================
        b = section("glacier_show_render", "Render Materials", "SHADING_RENDERED",
                    nmat or None)
        if b:
            if not nmat:
                b.label(text="Load a material first", icon="INFO")
            else:
                sb = sub(b, "Build", "NODE_MATERIAL")
                sb.prop(sc, "glacier_build_scope", text="")
                src_on = getattr(sc, "glacier_render_from_reference", False)
                n_ref = len([rs for rs in sc.glacier_render_slots])
                sb.label(text="Source: %s" % ("Reference list" if (src_on and n_ref)
                         else "Texture overrides"),
                         icon="PRESET" if (src_on and n_ref) else "TEXTURE")
                op = sb.operator("glacier.build_materials",
                                 text="Build Render Materials", icon="NODE_MATERIAL")
                op.apply_to = sc.glacier_build_scope
                sb = sub(b, "View", "SHADING_TEXTURE")
                sb.operator("glacier.set_shading",
                            text="Material Preview", icon="SHADING_TEXTURE")

        # ============ RENDER SOURCE (Pull from Reference) =================
        b = section("glacier_show_source", "Render Source", "PRESET",
                    nmat or None)
        if b:
            if not nmat:
                b.label(text="Load a material first", icon="INFO")
            else:
                sb = sub(b, "Pull From Reference", "IMPORT")
                sb.label(text="Auto-fill from .MATI / .MATB + .TEXT / .TEXD",
                         icon="INFO")
                row = sb.row(align=True)
                op = row.operator("glacier.pull_reference",
                                  text="Whole Model", icon="IMPORT")
                op.apply_to = "MODEL"
                op = row.operator("glacier.pull_reference",
                                  text="Active Only", icon="IMPORT")
                op.apply_to = "ACTIVE"
                sb.prop(sc, "glacier_render_from_reference")

                mine = [rs for rs in sc.glacier_render_slots if rs.mati_hash == active]
                sb = sub(b, "Textures For This Material", "TEXTURE")
                sb.prop(sc, "glacier_active_material", text="")
                if not mine:
                    sb.label(text="Press Pull From Reference above", icon="INFO")
                else:
                    sb.label(text="Tick = load it. Set what each one drives:")
                    for rs in sc.glacier_render_slots:
                        if rs.mati_hash != active:
                            continue
                        tb = sb.box()
                        hr = tb.row(align=True)
                        hr.prop(rs, "enabled", text="")
                        nm = hr.row(); nm.active = rs.enabled
                        nm.label(text=rs.slot_name or rs.tex_hash[:10])
                        meta = (rs.res + "  " + rs.fmt).strip()
                        if meta:
                            mr = nm.row(); mr.alignment = "RIGHT"
                            mr.label(text=meta)
                        rr = tb.row(); rr.active = rs.enabled
                        rr.prop(rs, "role", text="")

        # ============ EDIT MATERIAL =======================================
        b = section("glacier_show_edit", "Edit Material", "RESTRICT_SELECT_OFF")
        if b:
            if not nmat:
                b.label(text="Load a material first", icon="INFO")
            else:
                sb = sub(b, "Active Material", "MATERIAL")
                sb.prop(sc, "glacier_active_material", text="")
                mt_active = next((mt for mt in sc.glacier_materials
                                  if mt.key == active), None)
                is_bp = bool(mt_active and mt_active.is_blueprint)

                tbox = sub(b, "Textures", "TEXTURE")
                if not is_bp:
                    tbox.operator("glacier.fill_hashes", text="Fill Hashes",
                                  icon="FILE_REFRESH")
                if is_bp:
                    tbox.label(text="Blueprint - load its .MATI to swap", icon="INFO")
                any_tex = False
                for ts in sc.glacier_tex_slots:
                    if ts.mati_hash != active:
                        continue
                    any_tex = True
                    changed = (bool(ts.new_hash.strip())
                               and ts.new_hash.strip().upper() != ts.old_hash.upper()
                               ) or ts.tex_source != "HASH"
                    sbx = tbox.box()
                    head = sbx.row(align=True)
                    head.label(text=ts.slot_name,
                               icon="CHECKMARK" if changed else "DOT")
                    subr = head.row(); subr.alignment = "RIGHT"
                    subr.label(text=(ts.old_hash or "(none)")[:10])
                    if is_bp:
                        continue
                    sbx.prop(ts, "tex_source", text="")
                    eff = ts.new_hash.strip() or ts.old_hash
                    if ts.tex_source == "HASH":
                        sbx.prop(ts, "new_hash", text="Hash")
                        hint(sbx, "16-hex hash of an existing in-game texture")
                    elif ts.tex_source == "IMAGE":
                        sbx.prop(ts, "image_path", text="Image")
                        hint(sbx, "Your .png / .tga  (encoded on export)")
                        sbx.prop(ts, "new_hash", text="New Hash")
                        sbx.prop(ts, "texd_hash", text=".TEXD hash")
                    else:
                        sbx.prop(ts, "file_path", text=".TEXT")
                        sbx.prop(ts, "file_path_texd", text=".TEXD")
                        hint(sbx, ".TEXD optional - found by hash if blank")
                        sbx.prop(ts, "new_hash", text="New Hash")
                        sbx.prop(ts, "texd_hash", text=".TEXD hash")
                    if ts.tex_source != "HASH":
                        if eff:
                            td = ("  TEXD %s" % ts.texd_hash[:16]) if ts.texd_hash else ""
                            sbx.label(text="exports as %s%s" % (eff[:16], td),
                                      icon="CHECKMARK")
                        else:
                            sbx.label(text="click Fill Hashes", icon="ERROR")
                if not any_tex:
                    tbox.label(text="(no textures)")

                pbox = sub(b, "Parameters", "MODIFIER")
                any_p = False
                for p in sc.glacier_params:
                    if p.mati_hash != active:
                        continue
                    any_p = True
                    pbox.prop(p, "color" if p.type == 0x03 else "fval", text=p.name)
                if not any_p:
                    pbox.label(text="(none)")
                b.label(text="Edits apply on Export", icon="EXPORT")

        # ============ SWAP WHOLE MATERIAL =================================
        b = section("glacier_show_swap", "Swap Whole Material", "UV_SYNC_SELECT")
        if b:
            sb = sub(b, "Material Overrides", "UV_SYNC_SELECT")
            row = sb.row()
            row.template_list("GLACIER_UL_overrides", "", sc, "glacier_overrides",
                              sc, "glacier_overrides_index", rows=3)
            c2 = row.column(align=True)
            c2.operator("glacier.override_add", text="", icon="ADD")
            c2.operator("glacier.override_remove", text="", icon="REMOVE")
            if 0 <= sc.glacier_overrides_index < len(sc.glacier_overrides):
                sb.prop(sc.glacier_overrides[sc.glacier_overrides_index],
                        "new_hash", text="New Hash")

        # ============ TEXTURE TOOLS =======================================
        b = section("glacier_show_conv", "Texture Tools", "IMAGE_DATA")
        if b:
            sb = sub(b, "Folders", "FILE_FOLDER")
            sb.prop(sc, "glacier_work_dir", text="Work")
            hint(sb, "Output folder for decoded / re-encoded files")
            sb.prop(sc, "glacier_tex_folder", text="Search")
            hint(sb, "Folder of .TEXT / .TEXD to read from")

            sb = sub(b, "Decode To Images", "IMPORT")
            hint(sb, "Reads .TEXT / .TEXD  ->  writes .png / .tga")
            sb.prop(sc, "glacier_decode_fmt", text="Save As")
            sb.operator("glacier.decode_model",
                        text="Decode Textures", icon="IMPORT")

            sb = sub(b, "Encode & Generate", "EXPORT")
            hint(sb, "Reads your .png / .tga  ->  writes .TEXT / .TEXD")
            sb.prop(sc, "glacier_bc_format", text="Encode")
            sb.operator("glacier.reencode",
                        text="Re-encode Images", icon="EXPORT")
            sb.operator("glacier.generate_missing",
                        text="Generate Missing .TEXT/.TEXD", icon="FILE_NEW")
            sb.prop(sc, "glacier_organize_textures")

            sb = sub(b, "Single Texture", "FILE_IMAGE")
            hint(sb, "Decode one texture to an image")
            sb.prop(sc, "glacier_decode_text", text=".TEXT")
            sb.prop(sc, "glacier_decode_texd", text=".TEXD")
            sb.operator("glacier.decode_texture", text="Decode to Image",
                        icon="FILE_IMAGE")
            row = sb.row(align=True)
            row.prop(sc, "glacier_inspect_tex", text="Inspect")
            row.operator("glacier.inspect_texture", text="", icon="VIEWZOOM")
            hint(sb, "Inspect = read a .TEXT's size / format / mips")

        # ============ RPKG CHUNK BROWSER ==================================
        b = section("glacier_show_chunk", "RPKG Chunk Browser", "FILEBROWSER",
                    "NEW")
        if b:
            sb = sub(b, "Mass-Import From A Chunk", "PACKAGE")
            hint(sb, "Pull TEXT/TEXD straight out of a game chunk - no RPKG-Tool")
            sb.prop(sc, "glacier_chunk_path", text="Chunk")
            hint(sb, "Point at chunkNN.rpkg in the game's Runtime folder")
            sb.prop(sc, "glacier_chunk_out", text="Extract To")
            hint(sb, "Blank = your Work folder")
            sb.operator("glacier.chunk_browser", text="Open Chunk Browser",
                        icon="FILEBROWSER")
            hint(sb, "Opens a window to filter, pick and bulk-extract")

        # ============ LEVEL OF DETAIL =====================================
        b = section("glacier_show_lod", "Level of Detail", "MOD_DECIM",
                    len(groups) or None)
        if b:
            if groups:
                sb = sub(b, "Show LOD", "MOD_DECIM")
                sb.prop(sc, "glacier_lod_level", slider=True)
                row = sb.row(align=True)
                row.operator("glacier.lod_show_lod0")
                row.operator("glacier.lod_show_all")
            else:
                b.label(text="Import a model to use LOD tools", icon="INFO")

        b = section("glacier_show_rig", "Control Rig", "ARMATURE_DATA", disabled=True)
        if b:
            b.prop(sc, "glacier_rig_target", text="Armature")
            arm = _rig_active_armature(context)
            if arm is None:
                b.label(text="Select / set the imported armature", icon="INFO")
            else:
                rep = arm.get("glacier_rig_report", "")
                if rep:
                    col = b.column(align=True)
                    col.scale_y = 0.8
                    for chunk in [rep[i:i + 42] for i in range(0, len(rep), 42)]:
                        col.label(text=chunk)
            b.operator("glacier.rig_analyze", icon="VIEWZOOM")

            sb = sub(b, "Clean Skeleton", "TRASH")
            sb.prop(sc, "glacier_rig_clean_dryrun")
            sb.operator("glacier.rig_clean", icon="BRUSH_DATA")

            sb = sub(b, "Build Rig", "CON_KINEMATIC")
            row = sb.row(align=True)
            row.prop(sc, "glacier_rig_ik_arms", toggle=True)
            row.prop(sc, "glacier_rig_ik_legs", toggle=True)
            row = sb.row(align=True)
            row.prop(sc, "glacier_rig_foot_roll", toggle=True)
            row.prop(sc, "glacier_rig_twist", toggle=True)
            sb.operator("glacier.rig_build", icon="ARMATURE_DATA")
            sb.operator("glacier.rig_face", icon="MONKEY")
            sb.operator("glacier.rig_remove", icon="X")

        b = section("glacier_show_anim", "Animation (Preview)", "ARMATURE_DATA",
                    disabled=True)
        if b:
            arm = _rig_active_armature(context)
            if arm is None:
                b.label(text="Set the target armature above", icon="INFO")
            else:
                rep = arm.get("glacier_anim_report", "")
                if rep:
                    col = b.column(align=True)
                    col.scale_y = 0.8
                    for chunk in [rep[i:i + 42] for i in range(0, len(rep), 42)]:
                        col.label(text=chunk)
            b.operator("import_scene.glacier2_anim", icon="IMPORT")
            b.operator("glacier.anim_clear", icon="X")
            b.label(text="Preview only - no export yet", icon="INFO")







# =============================================================================
# CONTROL RIG  -  Auto-Rig-Pro-style control rig generator for 007 skeletons
# -----------------------------------------------------------------------------
# Turns an imported .borg game skeleton (raw IOI deform bones) into a poseable
# control rig, all inside the SAME armature:
#   * DEF  - the original deform bones (keep the vertex weights, they FOLLOW the
#            controls via Copy-Transforms - "the deform bones the rest follow")
#   * CTRL - FK controllers for spine / neck / head / clavicles / limbs / fingers
#   * MCH  - hidden machine bones that carry the IK solving for arms & legs
#   * FACE - facial controllers (jaw, eye-aim, lids, brows, lips)
# Plus optional skeleton CLEANUP that strips weightless helper/attacher/end
# bones from the skeleton (and folds any stray weights into the parent).
#
# Detection is name-pattern driven and adaptive: it builds only the chains it
# can find, using the First Light bone vocabulary (L_/R_ sides, spine_01..04,
# neck_01, head, L_thumb_0.., ankle/heel/ball/toe, *_twist, weapon/attacher
# helpers...) with generic fallbacks. The pure logic (rig_* functions) is
# bpy-free and unit-tested; the bpy layer only builds bones/constraints/drivers.
# =============================================================================

RIG_DEF_COLL = "DEF"
RIG_CTRL_COLL = "CTRL"
RIG_MCH_COLL = "MCH"
RIG_FACE_COLL = "FACE"
RIG_ROOT_COLL = "Root"
RIG_WIDGET_COLL = "RIG_Widgets"

_RIG_HELPER_TOKENS = ("ground", "origin", "camera", "attacher", "weapon",
                      "holster", "sheath", "magazine", "ammo", "equip", "militar",
                      "helper", "prop_", "_prop", "ik_", "_ik", "pole", "marker",
                      "cloth", "physics", "dyn_", "accessory")
_RIG_TWIST_TOKENS = ("twist", "_rbf", "rbf_", "xtra", "roll", "bend_")
_RIG_FACE_TOKENS = ("jaw", "eye", "lid", "brow", "lip", "mouth", "cheek", "nose",
                    "tongue", "teeth", "chin", "forehead", "nostril", "sneer",
                    "smile", "frown", "blink", "squint", "dimple", "pucker")
_RIG_FINGERS = ("thumb", "index", "middle", "ring", "little", "pinky")


def rig_side(name):
    """Return 'L', 'R' or '' for a bone name (First Light uses L_/R_ prefixes)."""
    n = name or ""
    if n.startswith(("L_", "l_")):
        return "L"
    if n.startswith(("R_", "r_")):
        return "R"
    low = n.lower()
    if low.endswith((".l", "_l", "-l")) or "_left" in low or low.startswith("left"):
        return "L"
    if low.endswith((".r", "_r", "-r")) or "_right" in low or low.startswith("right"):
        return "R"
    return ""


def _rig_num_suffix(name):
    """Trailing integer in a bone name (spine_04 -> 4), else -1."""
    import re as _re
    m = _re.search(r"(\d+)\s*$", name or "")
    return int(m.group(1)) if m else -1


def rig_classify_bone(name):
    """Coarse role for a bone name. Returns one of: root, helper, twist, face,
    finger, clavicle, upperarm, forearm, hand, thigh, shin, foot, ball, toe,
    heel, pelvis, spine, neck, head, other."""
    low = (name or "").lower()
    if not low:
        return "other"
    if low in ("root", "origin", "ground", "cog", "master"):
        return "root"
    if low.endswith(("_end", "_tip", "_nub", "_marker")):
        return "helper"
    if any(t in low for t in _RIG_TWIST_TOKENS):
        return "twist"
    if any(t in low for t in _RIG_HELPER_TOKENS):
        return "helper"
    if low.endswith("_root"):
        return "root"
    if any(t in low for t in _RIG_FINGERS):
        return "finger"
    if "clavicle" in low or "collar" in low or "shoulder" in low:
        return "clavicle"
    if "upperarm" in low or ("arm" in low and "upper" in low) or "humerus" in low:
        return "upperarm"
    if "forearm" in low or "lowerarm" in low or "forearm" in low or "ulna" in low:
        return "forearm"
    if "hand" in low or "wrist" in low or "palm" in low:
        return "hand"
    if "thigh" in low or "upperleg" in low or "upleg" in low or "femur" in low:
        return "thigh"
    if "calf" in low or "shin" in low or "lowerleg" in low or "tibia" in low or "knee" in low:
        return "shin"
    if "heel" in low:
        return "heel"
    if "ball" in low:
        return "ball"
    if "toe" in low:
        return "toe"
    if "foot" in low or "ankle" in low:
        return "foot"
    if any(t in low for t in _RIG_FACE_TOKENS):
        return "face"
    if "pelvis" in low or low == "hips" or low == "hip" or "cog" in low:
        return "pelvis"
    if "spine" in low or "chest" in low or "abdomen" in low or "waist" in low:
        return "spine"
    if "neck" in low:
        return "neck"
    if low == "head" or low.endswith("_head") or low.endswith("head"):
        return "head"
    return "other"


def _rig_sorted_chain(names):
    """Sort a set of bone names by their trailing number (then alphabetically)."""
    return sorted(names, key=lambda n: (_rig_num_suffix(n), n))


def rig_detect_chains(names):
    """Group a flat list of bone names into rig chains. Returns a dict describing
    spine, neck, head, root, per-side arms/legs/fingers, twist/face/helper sets."""
    cls = {n: rig_classify_bone(n) for n in names}
    out = {
        "root": None,
        "spine": [],
        "neck": [],
        "head": None,
        "arms": {"L": {}, "R": {}},
        "legs": {"L": {}, "R": {}},
        "fingers": {"L": {}, "R": {}},
        "twist": [],
        "face": [],
        "helper": [],
        "classes": cls,
    }
    roots = [n for n in names if cls[n] == "root"]
    out["root"] = roots[0] if roots else None
    out["spine"] = _rig_sorted_chain([n for n in names if cls[n] in ("pelvis", "spine")])
    out["neck"] = _rig_sorted_chain([n for n in names if cls[n] == "neck"])
    heads = [n for n in names if cls[n] == "head"]
    out["head"] = heads[0] if heads else None
    out["twist"] = sorted(n for n in names if cls[n] == "twist")
    out["face"] = sorted(n for n in names if cls[n] == "face")
    out["helper"] = sorted(n for n in names if cls[n] == "helper")

    for n in names:
        c = cls[n]
        side = rig_side(n)
        if c in ("clavicle", "upperarm", "forearm", "hand") and side in ("L", "R"):
            out["arms"][side][c] = n
        elif c in ("thigh", "shin", "foot", "ball", "toe", "heel") and side in ("L", "R"):
            out["legs"][side].setdefault(c, n)
        elif c == "finger" and side in ("L", "R"):
            fam = next((f for f in _RIG_FINGERS if f in n.lower()), "")
            if fam:
                out["fingers"][side].setdefault(fam, []).append(n)
    for side in ("L", "R"):
        for fam, bones in out["fingers"][side].items():
            out["fingers"][side][fam] = _rig_sorted_chain(bones)
    return out


def rig_clean_plan(names, parent_of, weighted, protect=None):
    """Decide which bones to strip when cleaning the skeleton.
    `parent_of`  : {bone: parent_bone or None}
    `weighted`   : set of bone names that actually carry vertex weight.
    `protect`    : extra bone names that must never be removed.
    A bone is removed when it carries no weight, none of its descendants carry
    weight, and it is a helper/attacher or a weightless leaf. Weighted bones, the
    deform chain, ALL facial bones, and anything under the head are always kept
    (facial detail bones are often weightless but must be preserved for the face
    rig). Returns {'remove': [...], 'keep': [...], 'reparent': {child: parent}}."""
    protect = set(protect or ())
    children = {n: [] for n in names}
    for n in names:
        p = parent_of.get(n)
        if p in children:
            children[p].append(n)
    cls = {n: rig_classify_bone(n) for n in names}

    # find the head bone and everything parented under it -> protect the face
    # detail bones (often weightless) but still allow helper/attacher cleanup.
    head = next((n for n in names if cls[n] == "head"), None)
    if head:
        stack = [head]
        while stack:
            cur = stack.pop()
            if cls.get(cur) != "helper":
                protect.add(cur)
            stack.extend(children.get(cur, ()))

    # bones that have a weighted bone anywhere below them (or are weighted)
    carries = {}

    def carries_weight(n):
        if n in carries:
            return carries[n]
        val = n in weighted or any(carries_weight(c) for c in children[n])
        carries[n] = val
        return val
    for n in names:
        carries_weight(n)

    remove = []
    for n in names:
        if n in weighted or n in protect:
            continue
        if carries.get(n):           # keep: something below it deforms
            continue
        # never strip the rig root, a structural bone, or any facial bone
        if cls[n] in ("root", "spine", "neck", "head", "pelvis", "face"):
            continue
        is_leaf = not children[n]
        # helpers/attachers anywhere, plus any weightless leaf (pivots, *_end,
        # weapon points, ik markers), are surplus on a deform skeleton.
        if cls[n] == "helper" or is_leaf:
            remove.append(n)
    remove_set = set(remove)
    reparent = {}
    for n in names:
        if n in remove_set:
            continue
        p = parent_of.get(n)
        while p in remove_set:        # skip removed ancestors
            p = parent_of.get(p)
        if p != parent_of.get(n):
            reparent[n] = p
    keep = [n for n in names if n not in remove_set]
    return {"remove": sorted(remove_set), "keep": keep, "reparent": reparent}


def rig_control_plan(chains, opts):
    """Build a build-plan from detected chains. Produces:
      controls   : [{name, source, parent, coll, color, widget, size}]  edit-bones
      copy       : [{def_bone, ctrl, name}]   DEF follows CTRL (copy transforms)
      ik         : [{mch_chain:[...], target, pole, def_bones:[...], switch, side, kind}]
      twist      : [{bone, follow}]
      props      : [{bone, name, value, min, max}]
    `source` is the deform bone a control is cloned from (head/tail/roll)."""
    controls = []
    copy = []
    ik = []
    twist = []
    props = []

    root_src = chains["root"]
    root_ctrl = "CTRL-root"
    controls.append({"name": root_ctrl, "source": root_src, "parent": None,
                     "coll": RIG_ROOT_COLL, "color": "THEME09", "widget": "circle",
                     "size": 2.5})

    # ---- spine / neck / head FK -------------------------------------------
    prev = root_ctrl
    spine_ctrls = []
    for b in chains["spine"]:
        c = "CTRL-%s" % b
        controls.append({"name": c, "source": b, "parent": prev,
                         "coll": RIG_CTRL_COLL, "color": "THEME03", "widget": "cube",
                         "size": 1.4})
        copy.append({"def_bone": b, "ctrl": c, "name": "RIG-fk"})
        spine_ctrls.append(c)
        prev = c
    chain_top = spine_ctrls[-1] if spine_ctrls else root_ctrl
    prev = chain_top
    for b in chains["neck"]:
        c = "CTRL-%s" % b
        controls.append({"name": c, "source": b, "parent": prev,
                         "coll": RIG_CTRL_COLL, "color": "THEME03", "widget": "cube",
                         "size": 1.2})
        copy.append({"def_bone": b, "ctrl": c, "name": "RIG-fk"})
        prev = c
    if chains["head"]:
        c = "CTRL-%s" % chains["head"]
        controls.append({"name": c, "source": chains["head"], "parent": prev,
                         "coll": RIG_CTRL_COLL, "color": "THEME03", "widget": "circle",
                         "size": 1.6})
        copy.append({"def_bone": chains["head"], "ctrl": c, "name": "RIG-fk"})

    # ---- arms --------------------------------------------------------------
    for side in ("L", "R"):
        arm = chains["arms"][side]
        clav, up, fore, hand = (arm.get("clavicle"), arm.get("upperarm"),
                                arm.get("forearm"), arm.get("hand"))
        parent = chain_top
        if clav:
            c = "CTRL-%s" % clav
            controls.append({"name": c, "source": clav, "parent": chain_top,
                             "coll": RIG_CTRL_COLL, "color": "THEME04", "widget": "circle",
                             "size": 1.0})
            copy.append({"def_bone": clav, "ctrl": c, "name": "RIG-fk"})
            parent = c
        fk_prev = parent
        for b in (up, fore, hand):
            if not b:
                continue
            c = "CTRL-%s" % b
            controls.append({"name": c, "source": b, "parent": fk_prev,
                             "coll": RIG_CTRL_COLL, "color": "THEME04", "widget": "circle",
                             "size": 1.1})
            copy.append({"def_bone": b, "ctrl": c, "name": "RIG-fk"})
            fk_prev = c
        if opts.get("ik_arms", True) and up and fore and hand:
            tgt = "CTRL-ik_hand.%s" % side
            pole = "CTRL-pole_arm.%s" % side
            controls.append({"name": tgt, "source": hand, "parent": root_ctrl,
                             "coll": RIG_CTRL_COLL, "color": "THEME01", "widget": "cube",
                             "size": 1.3})
            controls.append({"name": pole, "source": fore, "parent": root_ctrl,
                             "coll": RIG_CTRL_COLL, "color": "THEME01", "widget": "sphere",
                             "size": 0.5, "pole_for": (up, fore, hand)})
            mch = ["MCH-ik_%s" % up, "MCH-ik_%s" % fore]
            controls.append({"name": mch[0], "source": up, "parent": parent,
                             "coll": RIG_MCH_COLL, "color": "THEME08", "widget": None,
                             "size": 1.0, "deform": False})
            controls.append({"name": mch[1], "source": fore, "parent": mch[0],
                             "coll": RIG_MCH_COLL, "color": "THEME08", "widget": None,
                             "size": 1.0, "deform": False})
            props.append({"bone": tgt, "name": "ik_fk", "value": 0.0, "min": 0.0, "max": 1.0})
            ik.append({"mch_chain": mch, "target": tgt, "pole": pole,
                       "def_bones": [up, fore], "fk_ctrls": ["CTRL-%s" % up, "CTRL-%s" % fore],
                       "hand": hand, "hand_ctrl": "CTRL-%s" % hand,
                       "switch": tgt, "side": side, "kind": "arm"})

    # ---- legs --------------------------------------------------------------
    for side in ("L", "R"):
        leg = chains["legs"][side]
        thigh, shin, foot = leg.get("thigh"), leg.get("shin"), leg.get("foot")
        ball, toe = leg.get("ball"), leg.get("toe")
        fk_prev = root_ctrl
        for b in (thigh, shin, foot, ball, toe):
            if not b:
                continue
            c = "CTRL-%s" % b
            controls.append({"name": c, "source": b, "parent": fk_prev,
                             "coll": RIG_CTRL_COLL, "color": "THEME04", "widget": "circle",
                             "size": 1.1})
            copy.append({"def_bone": b, "ctrl": c, "name": "RIG-fk"})
            fk_prev = c
        if opts.get("ik_legs", True) and thigh and shin and foot:
            tgt = "CTRL-ik_foot.%s" % side
            pole = "CTRL-pole_leg.%s" % side
            controls.append({"name": tgt, "source": foot, "parent": root_ctrl,
                             "coll": RIG_CTRL_COLL, "color": "THEME01", "widget": "cube",
                             "size": 1.4})
            controls.append({"name": pole, "source": shin, "parent": root_ctrl,
                             "coll": RIG_CTRL_COLL, "color": "THEME01", "widget": "sphere",
                             "size": 0.5, "pole_for": (thigh, shin, foot)})
            mch = ["MCH-ik_%s" % thigh, "MCH-ik_%s" % shin]
            controls.append({"name": mch[0], "source": thigh, "parent": root_ctrl,
                             "coll": RIG_MCH_COLL, "color": "THEME08", "widget": None,
                             "size": 1.0, "deform": False})
            controls.append({"name": mch[1], "source": shin, "parent": mch[0],
                             "coll": RIG_MCH_COLL, "color": "THEME08", "widget": None,
                             "size": 1.0, "deform": False})
            props.append({"bone": tgt, "name": "ik_fk", "value": 0.0, "min": 0.0, "max": 1.0})
            ik.append({"mch_chain": mch, "target": tgt, "pole": pole,
                       "def_bones": [thigh, shin], "fk_ctrls": ["CTRL-%s" % thigh, "CTRL-%s" % shin],
                       "hand": foot, "hand_ctrl": "CTRL-%s" % foot,
                       "switch": tgt, "side": side, "kind": "leg"})
            if opts.get("foot_roll", True) and (ball or toe):
                roll = "CTRL-foot_roll.%s" % side
                controls.append({"name": roll, "source": foot, "parent": tgt,
                                 "coll": RIG_CTRL_COLL, "color": "THEME01",
                                 "widget": "arrow", "size": 1.0})
                props.append({"bone": roll, "name": "roll", "value": 0.0,
                              "min": -90.0, "max": 90.0})

    # ---- fingers (FK curl chain) ------------------------------------------
    for side in ("L", "R"):
        hand_ctrl = "CTRL-%s" % chains["arms"][side].get("hand", "") if chains["arms"][side].get("hand") else root_ctrl
        for fam, bones in chains["fingers"][side].items():
            prev = hand_ctrl
            for b in bones:
                c = "CTRL-%s" % b
                controls.append({"name": c, "source": b, "parent": prev,
                                 "coll": RIG_CTRL_COLL, "color": "THEME11",
                                 "widget": "circle", "size": 0.6})
                copy.append({"def_bone": b, "ctrl": c, "name": "RIG-fk"})
                prev = c

    # ---- twist bones follow their parent deform ---------------------------
    if opts.get("twist_follow", True):
        for b in chains["twist"]:
            twist.append({"bone": b})

    return {"controls": controls, "copy": copy, "ik": ik, "twist": twist, "props": props}


def rig_face_regions(face_bones):
    """Group facial bones into a small set of meaningful regions so the face rig
    has ONE controller per region (top lip, bottom lip, jaw, each brow, each
    eyelid, each cheek...) rather than a controller per individual bone."""
    regions = {}
    for b in face_bones:
        low = b.lower()
        side = rig_side(b) or "C"
        if ("eye" in low and "lid" not in low and "brow" not in low
                and "lash" not in low):
            continue                                  # eyes -> aim target, below
        if "jaw" in low or "chin" in low:
            key = "jaw"
        elif "lip" in low or "mouth" in low:
            if "corner" in low or "cnr" in low:
                key = "lip_corner_%s" % side
            elif any(t in low for t in ("upper", "_up", "top")):
                key = "lip_upper"
            elif any(t in low for t in ("lower", "_low", "bottom", "_bot")):
                key = "lip_lower"
            else:
                key = "mouth_%s" % side
        elif "brow" in low:
            key = "brow_%s" % side
        elif "lid" in low or "eyelid" in low or "lash" in low:
            if any(t in low for t in ("lower", "_low", "bottom", "_bot")):
                key = "lid_lower_%s" % side
            else:
                key = "lid_upper_%s" % side
        elif "cheek" in low:
            key = "cheek_%s" % side
        elif "nose" in low or "nostril" in low:
            key = "nose"
        else:
            key = "face_%s" % side
        regions.setdefault(key, []).append(b)
    return regions


def rig_face_plan(chains, head_bone):
    """Plan a clean, ARP-style facial rig: an eye-aim target the eyes track, a
    visible jaw control, and ONE controller per face region driving every bone in
    that region (so e.g. a single control moves the whole top lip)."""
    face = chains["face"]
    controls = []
    copy = []
    aim = []
    jaw = None
    eyes = [b for b in face if "eye" in b.lower() and "lid" not in b.lower()
            and "brow" not in b.lower() and "lash" not in b.lower()]
    if eyes and head_bone:
        master = "CTRL-eyes_target"
        controls.append({"name": master, "source": head_bone, "parent": head_bone,
                         "coll": RIG_FACE_COLL, "color": "THEME01", "widget": "circle",
                         "size": 2.0, "offset_forward": True})
        for e in eyes:
            side = rig_side(e) or "C"
            t = "CTRL-eye_target.%s" % side
            controls.append({"name": t, "source": e, "parent": master,
                             "coll": RIG_FACE_COLL, "color": "THEME01", "widget": "circle",
                             "size": 0.6, "offset_forward": True})
            aim.append({"eye": e, "target": t})

    regions = rig_face_regions(face)
    for key, bones in sorted(regions.items()):
        ctrl = "CTRL-face_%s" % key
        if key == "jaw":
            jaw = bones[0]
            controls.append({"name": ctrl, "from_bones": list(bones), "parent": head_bone,
                             "coll": RIG_FACE_COLL, "color": "THEME06", "widget": "wedge",
                             "size": 1.6})
        else:
            controls.append({"name": ctrl, "from_bones": list(bones), "parent": head_bone,
                             "coll": RIG_FACE_COLL, "color": "THEME06", "widget": "sphere",
                             "size": 0.5})
        for b in bones:
            copy.append({"def_bone": b, "ctrl": ctrl, "name": "RIG-face"})
    return {"controls": controls, "copy": copy, "aim": aim, "jaw": jaw,
            "eyes": eyes, "regions": sorted(regions.keys())}


# ----- bpy construction layer ----------------------------------------------
def _rig_collection(arm, name):
    coll = arm.collections.get(name) if hasattr(arm.collections, "get") else None
    if coll is None:
        try:
            coll = arm.collections.get(name)
        except Exception:
            coll = None
    if coll is None:
        coll = arm.collections.new(name)
    return coll


def _rig_widget_object(name, kind):
    """Create (or reuse) a simple custom-shape mesh for control bones."""
    obj = bpy.data.objects.get(name)
    if obj is not None:
        return obj
    import math as _m
    verts, edges = [], []
    if kind == "circle":
        n = 16
        for i in range(n):
            a = 2 * _m.pi * i / n
            verts.append((_m.cos(a), 0.0, _m.sin(a)))
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif kind == "cube":
        s = 0.5
        verts = [(x * s, y * s, z * s) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        edges = [(0, 1), (0, 2), (0, 4), (3, 1), (3, 2), (3, 7), (5, 1), (5, 4),
                 (5, 7), (6, 2), (6, 4), (6, 7)]
    elif kind == "sphere":
        n = 12
        for plane in range(3):
            for i in range(n):
                a = 2 * _m.pi * i / n
                c, s = _m.cos(a), _m.sin(a)
                verts.append((c, s, 0) if plane == 0 else (c, 0, s) if plane == 1 else (0, c, s))
        edges = []
        for p in range(3):
            base = p * n
            edges += [(base + i, base + (i + 1) % n) for i in range(n)]
    elif kind == "arrow":
        verts = [(0, 0, 0), (0, 1, 0), (-0.2, 0.8, 0), (0.2, 0.8, 0)]
        edges = [(0, 1), (1, 2), (1, 3)]
    elif kind == "wedge":
        # a chunky open V / jaw shape that reads clearly in the viewport
        verts = [(-0.7, 0.0, 0.3), (0.7, 0.0, 0.3), (0.5, -0.6, -0.2),
                 (-0.5, -0.6, -0.2), (0.0, -0.9, -0.1)]
        edges = [(0, 1), (1, 2), (2, 4), (4, 3), (3, 0), (2, 3)]
    else:
        verts = [(0, 0, 0), (0, 1, 0)]
        edges = [(0, 1)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, edges, [])
    me.update()
    obj = bpy.data.objects.new(name, me)
    coll = bpy.data.collections.get(RIG_WIDGET_COLL)
    if coll is None:
        coll = bpy.data.collections.new(RIG_WIDGET_COLL)
        try:
            bpy.context.scene.collection.children.link(coll)
            coll.hide_viewport = True
            coll.hide_render = True
        except Exception:
            pass
    coll.objects.link(obj)
    return obj


def _rig_make_controls(arm_obj, specs):
    """Create the controller edit-bones described by `specs` and parent them.
    Pole targets are placed out along the limb's natural bend direction so the
    IK solve matches the model instead of snapping."""
    from mathutils import Vector
    arm = arm_obj.data
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm.edit_bones
    made = {}
    for sp in specs:
        if sp["name"] in ebs:
            eb = ebs[sp["name"]]
        else:
            eb = ebs.new(sp["name"])
        src = ebs.get(sp["source"]) if sp.get("source") else None
        if sp.get("from_bones"):
            # region face control: sit at the average head of its member bones,
            # with a short tail facing forward, so one controller covers the group
            mem = [ebs.get(n) for n in sp["from_bones"] if ebs.get(n) is not None]
            if mem:
                avg = Vector((0.0, 0.0, 0.0))
                for m in mem:
                    avg = avg + m.head
                avg = avg * (1.0 / len(mem))
                span = max((m.tail - m.head).length for m in mem) or 0.04
                eb.head = avg
                eb.tail = avg + Vector((0.0, -1.0, 0.0)) * (span * 1.2 + 0.02)
                eb.roll = 0.0
            else:
                eb.head = Vector((0, 0, 0)); eb.tail = Vector((0, 0, 1))
        elif sp.get("pole_for"):
            # place the pole in the bend plane, out in front of the joint
            a, b, c = sp["pole_for"]
            ea, eb2, ec = ebs.get(a), ebs.get(b), ebs.get(c)
            if ea and eb2 and ec:
                root_h, mid_h, end_h = ea.head, eb2.head, ec.head
                chord = (root_h + end_h) * 0.5
                bend = (mid_h - chord)
                limb = (end_h - root_h).length or 1.0
                if bend.length < 1e-4:
                    bend = Vector((0, -1, 0))
                bend = bend.normalized()
                eb.head = mid_h + bend * limb * 0.5
                eb.tail = eb.head + bend * (limb * 0.12 + 0.02)
                eb.roll = 0.0
            elif src is not None:
                eb.head = src.head.copy(); eb.tail = src.tail.copy()
        elif src is not None:
            eb.head = src.head.copy()
            eb.tail = src.tail.copy()
            eb.roll = src.roll
            size = sp.get("size", 1.0)
            if size != 1.0:
                d = (eb.tail - eb.head)
                eb.tail = eb.head + d * size
            if sp.get("offset_forward"):
                fwd = Vector((0, -1, 0)) * max(0.1, (src.tail - src.head).length) * 4
                eb.head = src.head + fwd
                eb.tail = eb.head + Vector((0, 0, (src.tail - src.head).length))
        else:
            eb.head = Vector((0, 0, 0))
            eb.tail = Vector((0, 0, 1))
        eb.use_connect = False
        made[sp["name"]] = sp
    for sp in specs:                          # parenting after all exist
        if sp.get("parent") and sp["parent"] in ebs and sp["name"] in ebs:
            ebs[sp["name"]].parent = ebs[sp["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")

    # collections, colors, widgets, deform flag
    for sp in specs:
        bone = arm.bones.get(sp["name"])
        if bone is None:
            continue
        bone.use_deform = bool(sp.get("deform", False))
        coll = _rig_collection(arm, sp["coll"])
        try:
            coll.assign(bone)
        except Exception:
            pass
        try:
            bone.color.palette = sp.get("color", "DEFAULT")
        except Exception:
            pass
        pb = arm_obj.pose.bones.get(sp["name"])
        if pb is not None and sp.get("widget"):
            try:
                pb.custom_shape = _rig_widget_object("WGT-" + sp["widget"], sp["widget"])
                # custom shapes scale with bone length by default, so the widget
                # is always proportional to the model; `size` fine-tunes it.
                pb.custom_shape_scale_xyz = (sp.get("size", 1.0),) * 3
                try:
                    pb.use_custom_shape_bone_size = True
                except Exception:
                    pass
                pb.color.palette = sp.get("color", "DEFAULT")
            except Exception:
                pass
    return made


def _rig_add_prop(arm_obj, bone, name, value, lo, hi):
    pb = arm_obj.pose.bones.get(bone)
    if pb is None:
        return
    pb[name] = float(value)
    try:
        ui = pb.id_properties_ui(name)
        ui.update(min=float(lo), max=float(hi), soft_min=float(lo), soft_max=float(hi))
    except Exception:
        pass


def _rig_copy_constraint(arm_obj, def_bone, ctrl, name, ctype="COPY_TRANSFORMS",
                         space="LOCAL"):
    pb = arm_obj.pose.bones.get(def_bone)
    if pb is None or ctrl not in arm_obj.pose.bones:
        return None
    con = pb.constraints.new(ctype)
    con.name = name
    con.target = arm_obj
    con.subtarget = ctrl
    # LOCAL->LOCAL copy is immune to any rest-pose orientation mismatch between
    # the control and the deform bone: at rest the control's local transform is
    # identity, so the deform bone is untouched (no "noodle"/melt). Only when you
    # actually pose the control does the deform bone follow.
    if space == "LOCAL":
        try:
            con.target_space = "LOCAL"
            con.owner_space = "LOCAL"
        except Exception:
            pass
    return con


def _rig_driver_influence(arm_obj, bone, con_name, prop_bone, prop_name, invert):
    """Drive a constraint's influence from a custom property (IK/FK switch)."""
    path = 'pose.bones["%s"].constraints["%s"].influence' % (bone, con_name)
    try:
        fc = arm_obj.driver_add(path)
    except Exception:
        return
    drv = fc.driver
    drv.type = "SCRIPTED"
    for v in list(drv.variables):
        drv.variables.remove(v)
    var = drv.variables.new()
    var.name = "sw"
    var.type = "SINGLE_PROP"
    tgt = var.targets[0]
    tgt.id = arm_obj
    tgt.data_path = 'pose.bones["%s"]["%s"]' % (prop_bone, prop_name)
    drv.expression = "1.0 - sw" if invert else "sw"


def _rig_apply_relations(arm_obj, plan):
    """Add all constraints + drivers for the body control plan."""
    for c in plan["copy"]:
        _rig_copy_constraint(arm_obj, c["def_bone"], c["ctrl"], c["name"])

    for sw in plan["props"]:
        _rig_add_prop(arm_obj, sw["bone"], sw["name"], sw["value"], sw["min"], sw["max"])

    for ik in plan["ik"]:
        mch = ik["mch_chain"]
        # MCH chain copies the FK controls, last MCH gets the IK constraint
        last = arm_obj.pose.bones.get(mch[-1])
        if last is not None and ik["target"] in arm_obj.pose.bones:
            con = last.constraints.new("IK")
            con.name = "RIG-ik"
            con.target = arm_obj
            con.subtarget = ik["target"]
            con.chain_count = len(mch)
            if ik["pole"] in arm_obj.pose.bones:
                con.pole_target = arm_obj
                con.pole_subtarget = ik["pole"]
                con.pole_angle = -1.5708 if ik["kind"] == "leg" else 1.5708
        # blend each DEF bone between its FK control and the matching MCH-IK bone
        for def_bone, mch_bone in zip(ik["def_bones"], mch):
            con = _rig_copy_constraint(arm_obj, def_bone, mch_bone, "RIG-ik")
            if con is not None:
                _rig_driver_influence(arm_obj, def_bone, "RIG-ik",
                                      ik["switch"], "ik_fk", invert=False)
        # the hand/foot DEF follows the IK target directly when in IK. This one
        # is WORLD space so the hand matches the target's orientation.
        if ik.get("hand"):
            con = _rig_copy_constraint(arm_obj, ik["hand"], ik["target"], "RIG-ik",
                                       space="WORLD")
            if con is not None:
                _rig_driver_influence(arm_obj, ik["hand"], "RIG-ik",
                                      ik["switch"], "ik_fk", invert=False)

    for tw in plan["twist"]:
        pb = arm_obj.pose.bones.get(tw["bone"])
        if pb is None or pb.parent is None:
            continue
        con = pb.constraints.new("COPY_ROTATION")
        con.name = "RIG-twist"
        con.target = arm_obj
        con.subtarget = pb.parent.name
        con.use_x = False
        con.use_z = False
        con.influence = 0.5
        con.mix_mode = "ADD"
        # LOCAL spaces so the twist bone only follows its parent's *local* roll;
        # copying world rotation here is what melted the arms.
        try:
            con.target_space = "LOCAL"
            con.owner_space = "LOCAL"
        except Exception:
            pass


def _rig_apply_face(arm_obj, plan):
    for c in plan["copy"]:
        _rig_copy_constraint(arm_obj, c["def_bone"], c["ctrl"], c["name"])
    for a in plan["aim"]:
        pb = arm_obj.pose.bones.get(a["eye"])
        if pb is None or a["target"] not in arm_obj.pose.bones:
            continue
        con = pb.constraints.new("DAMPED_TRACK")
        con.name = "RIG-eye-aim"
        con.target = arm_obj
        con.subtarget = a["target"]
        con.track_axis = "TRACK_Y"


def _rig_weighted_bones(arm_obj):
    """Bones that actually carry weight on any mesh skinned to this armature."""
    weighted = set()
    for ob in bpy.data.objects:
        if ob.type != "MESH":
            continue
        if not any(m.type == "ARMATURE" and m.object == arm_obj for m in ob.modifiers):
            continue
        names = {vg.index: vg.name for vg in ob.vertex_groups}
        present = set()
        for v in ob.data.vertices:
            for g in v.groups:
                if g.weight > 0.0001:
                    present.add(g.group)
        for gi in present:
            if gi in names:
                weighted.add(names[gi])
    return weighted


def _rig_skinned_meshes(arm_obj):
    out = []
    for ob in bpy.data.objects:
        if ob.type == "MESH" and any(
                m.type == "ARMATURE" and m.object == arm_obj for m in ob.modifiers):
            out.append(ob)
    return out


def _rig_transfer_group(ob, src_name, dst_name):
    """Fold vertex-group `src_name`'s weights into `dst_name`, then delete src."""
    src = ob.vertex_groups.get(src_name)
    if src is None:
        return
    dst = ob.vertex_groups.get(dst_name)
    if dst is None and dst_name:
        dst = ob.vertex_groups.new(name=dst_name)
    si = src.index
    if dst is not None:
        for v in ob.data.vertices:
            for g in v.groups:
                if g.group == si and g.weight > 0.0:
                    dst.add([v.index], g.weight, "ADD")
    ob.vertex_groups.remove(src)


def _rig_survivor_parent(arm, name, remove_set):
    """Nearest ancestor of `name` that is NOT being removed (or None)."""
    b = arm.bones.get(name)
    p = b.parent if b else None
    while p is not None and p.name in remove_set:
        p = p.parent
    return p.name if p is not None else None


def _rig_clean_apply(arm_obj, plan):
    """Delete the planned bones. Any weight a removed bone carries is first folded
    into its nearest surviving parent so the mesh keeps deforming - "redo weights
    on skeleton change" - instead of silently losing influence."""
    arm = arm_obj.data
    remove_set = set(plan["remove"])
    for ob in _rig_skinned_meshes(arm_obj):
        for rem in plan["remove"]:
            survivor = _rig_survivor_parent(arm, rem, remove_set)
            _rig_transfer_group(ob, rem, survivor)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = arm.edit_bones
    for child, new_parent in plan["reparent"].items():
        if child in ebs:
            ebs[child].parent = ebs.get(new_parent) if new_parent else None
    for rem in plan["remove"]:
        if rem in ebs:
            ebs.remove(ebs[rem])
    bpy.ops.object.mode_set(mode="OBJECT")


def _rig_organize_collections(arm_obj):
    """Put every controlled (deform) bone in the DEF collection and hide it, hide
    the MCH machine bones, and keep CTRL / FACE / Root visible. This is what lets
    you hide the skeleton and pose with just the controllers."""
    arm = arm_obj.data
    defc = _rig_collection(arm, RIG_DEF_COLL)
    for b in arm.bones:
        if b.name.startswith(("CTRL-", "MCH-")):
            continue
        try:
            defc.assign(b)
        except Exception:
            pass
    vis = {RIG_DEF_COLL: False, RIG_MCH_COLL: False, RIG_WIDGET_COLL: False,
           RIG_CTRL_COLL: True, RIG_FACE_COLL: True, RIG_ROOT_COLL: True}
    for name, visible in vis.items():
        coll = arm.collections.get(name) if hasattr(arm.collections, "get") else None
        if coll is not None:
            try:
                coll.is_visible = visible
            except Exception:
                pass


def _rig_active_armature(context):
    sc = context.scene
    arm = getattr(sc, "glacier_rig_target", None)
    if arm is not None and arm.type == "ARMATURE":
        return arm
    ob = context.active_object
    if ob is not None and ob.type == "ARMATURE":
        return ob
    if ob is not None and ob.parent is not None and ob.parent.type == "ARMATURE":
        return ob.parent
    return None


class GLACIER_OT_rig_analyze(bpy.types.Operator):
    bl_idname = "glacier.rig_analyze"
    bl_label = "Analyze Skeleton"
    bl_description = ("Scan the selected armature and report the chains it found "
                      "(spine, arms, legs, fingers, face) plus weightless bones "
                      "that cleanup could remove. Changes nothing")

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None:
            self.report({"WARNING"}, "Select the imported armature first")
            return {"CANCELLED"}
        names = [b.name for b in arm.data.bones]
        chains = rig_detect_chains(names)
        weighted = _rig_weighted_bones(arm)
        parent_of = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
        clean = rig_clean_plan(names, parent_of, weighted)
        n_arms = sum(1 for s in ("L", "R") if chains["arms"][s])
        n_legs = sum(1 for s in ("L", "R") if chains["legs"][s])
        n_fing = sum(len(chains["fingers"][s]) for s in ("L", "R"))
        msg = ("%d bones | spine %d, neck %d, head %s | arms %d, legs %d, fingers %d "
               "| face %d | weightless removable %d" % (
                   len(names), len(chains["spine"]), len(chains["neck"]),
                   "yes" if chains["head"] else "no", n_arms, n_legs, n_fing,
                   len(chains["face"]), len(clean["remove"])))
        arm["glacier_rig_report"] = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class GLACIER_OT_rig_clean(bpy.types.Operator):
    bl_idname = "glacier.rig_clean"
    bl_label = "Clean Skeleton"
    bl_description = ("Remove weightless helper / attacher / end bones from the "
                      "skeleton and delete their empty vertex groups. Surviving "
                      "bones are re-parented across the gaps")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None:
            self.report({"WARNING"}, "Select the imported armature first")
            return {"CANCELLED"}
        names = [b.name for b in arm.data.bones]
        weighted = _rig_weighted_bones(arm)
        parent_of = {b.name: (b.parent.name if b.parent else None) for b in arm.data.bones}
        plan = rig_clean_plan(names, parent_of, weighted)
        if not plan["remove"]:
            self.report({"INFO"}, "Nothing to clean - no weightless removable bones")
            return {"FINISHED"}
        if context.scene.glacier_rig_clean_dryrun:
            self.report({"INFO"}, "Dry run: would remove %d bones (%s%s)" % (
                len(plan["remove"]), ", ".join(plan["remove"][:8]),
                "..." if len(plan["remove"]) > 8 else ""))
            return {"FINISHED"}
        context.view_layer.objects.active = arm
        _rig_clean_apply(arm, plan)
        self.report({"INFO"}, "Removed %d weightless bones" % len(plan["remove"]))
        return {"FINISHED"}


class GLACIER_OT_rig_build(bpy.types.Operator):
    bl_idname = "glacier.rig_build"
    bl_label = "Build Control Rig"
    bl_description = ("Generate FK + IK controllers on the armature. The deform "
                      "bones follow the controls; arms and legs get IK with pole "
                      "targets and an IK/FK switch; twist bones follow their parent")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None:
            self.report({"WARNING"}, "Select the imported armature first")
            return {"CANCELLED"}
        sc = context.scene
        names = [b.name for b in arm.data.bones]
        chains = rig_detect_chains(names)
        opts = {
            "ik_arms": sc.glacier_rig_ik_arms,
            "ik_legs": sc.glacier_rig_ik_legs,
            "foot_roll": sc.glacier_rig_foot_roll,
            "twist_follow": sc.glacier_rig_twist,
        }
        plan = rig_control_plan(chains, opts)
        context.view_layer.objects.active = arm
        _rig_make_controls(arm, plan["controls"])
        _rig_apply_relations(arm, plan)
        _rig_organize_collections(arm)
        arm["glacier_has_control_rig"] = True
        try:
            arm.show_in_front = True
        except Exception:
            pass
        self.report({"INFO"}, "Built %d controllers, %d IK chain(s) - deform bones "
                    "hidden in the DEF collection" % (len(plan["controls"]), len(plan["ik"])))
        return {"FINISHED"}


class GLACIER_OT_rig_face(bpy.types.Operator):
    bl_idname = "glacier.rig_face"
    bl_label = "Build Facial Rig"
    bl_description = ("Add facial controllers from the head's face bones: an "
                      "eye-aim target the eyes track, a jaw control, and FK "
                      "controllers for lids, brows, lips and cheeks")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None:
            self.report({"WARNING"}, "Select the imported armature first")
            return {"CANCELLED"}
        names = [b.name for b in arm.data.bones]
        chains = rig_detect_chains(names)
        if not chains["face"]:
            self.report({"WARNING"}, "No facial bones detected on this skeleton")
            return {"CANCELLED"}
        plan = rig_face_plan(chains, chains["head"] or (chains["neck"][-1] if chains["neck"] else None))
        context.view_layer.objects.active = arm
        _rig_make_controls(arm, plan["controls"])
        _rig_apply_face(arm, plan)
        _rig_organize_collections(arm)
        arm["glacier_has_face_rig"] = True
        self.report({"INFO"}, "Built %d face controllers (%d eye aims) - face "
                    "deform bones hidden" % (len(plan["controls"]), len(plan["aim"])))
        return {"FINISHED"}


class GLACIER_OT_rig_remove(bpy.types.Operator):
    bl_idname = "glacier.rig_remove"
    bl_label = "Remove Control Rig"
    bl_description = ("Delete every generated CTRL-/MCH- controller bone and the "
                      "constraints that point at them, leaving the clean deform "
                      "skeleton")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None:
            return {"CANCELLED"}
        # strip generated constraints from deform bones
        for pb in arm.pose.bones:
            for con in list(pb.constraints):
                if con.name in ("RIG-fk", "RIG-ik", "RIG-twist", "RIG-face",
                                "RIG-eye-aim"):
                    pb.constraints.remove(con)
        context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = arm.data.edit_bones
        for eb in list(ebs):
            if eb.name.startswith(("CTRL-", "MCH-")):
                ebs.remove(eb)
        bpy.ops.object.mode_set(mode="OBJECT")
        # un-hide the deform skeleton again
        for nm in (RIG_DEF_COLL, RIG_MCH_COLL):
            coll = arm.data.collections.get(nm) if hasattr(arm.data.collections, "get") else None
            if coll is not None:
                try:
                    coll.is_visible = True
                except Exception:
                    pass
        arm["glacier_has_control_rig"] = False
        arm["glacier_has_face_rig"] = False
        self.report({"INFO"}, "Control rig removed")
        return {"FINISHED"}




# =============================================================================
# ANIMATION IMPORT  -  preview Glacier 2 animations on the imported skeleton
# -----------------------------------------------------------------------------
# Goal: load an animation and PLAY it on the armature in Blender (preview only,
# no export yet). The pipeline has three layers:
#
#   1. A bpy-free animation model (GlacierAnim) + a JSON loader. This is the
#      working preview path today: per-bone local pose keyframes -> a Blender
#      Action you can scrub. Fully unit-tested.
#   2. A binary "probe" that reads the IOI container header (u64 header_offset
#      then the header table) and reports the fields it finds, so the native
#      track layout can be mapped from a real sample file.
#   3. The apply step that writes the keyframes onto the pose bones.
#
# The native Glacier 2 / KNT animation resource is not documented in any of the
# project references and is most likely Havok-compressed, so the binary decoder
# is intentionally a calibrated-on-sample scaffold rather than a guess that would
# produce wrong motion. The JSON bridge lets you preview now (dump keyframes from
# any source into the documented schema) while the binary path is finished from a
# sample.
# =============================================================================

ANIM_JSON_SCHEMA = (
    '{ "fps": 30, "frame_start": 1, "bones": { '
    '"spine_01": [ {"frame":1, "rotation":[1,0,0,0], "location":[0,0,0], '
    '"scale":[1,1,1]}, ... ] } }  - rotation is quaternion w,x,y,z in the bone\'s '
    'local pose space; location/scale optional.')


class GlacierAnimError(Exception):
    pass


class GlacierAnim:
    """fps, frame range and a dict of {bone_name: [keyframes]} where each key is
    (frame:int, location:(x,y,z)|None, quat_wxyz:(w,x,y,z)|None, scale:(x,y,z)|None)."""
    def __init__(self, fps=30.0, frame_start=1):
        self.fps = float(fps)
        self.frame_start = int(frame_start)
        self.tracks = {}

    @property
    def frame_end(self):
        end = self.frame_start
        for keys in self.tracks.values():
            for k in keys:
                if k[0] > end:
                    end = k[0]
        return end

    @property
    def bone_count(self):
        return len(self.tracks)

    @property
    def key_count(self):
        return sum(len(v) for v in self.tracks.values())


def _anim_vec(seq, n, default):
    if seq is None:
        return None
    try:
        vals = [float(x) for x in seq]
    except (TypeError, ValueError):
        return None
    if len(vals) < n:
        vals = vals + list(default[len(vals):])
    return tuple(vals[:n])


def anim_from_json(data):
    """Build a GlacierAnim from the documented JSON schema (dict already parsed)."""
    if not isinstance(data, dict):
        raise GlacierAnimError("animation JSON must be an object")
    bones = data.get("bones")
    if not isinstance(bones, dict) or not bones:
        raise GlacierAnimError("animation JSON has no 'bones' map")
    anim = GlacierAnim(fps=data.get("fps", 30), frame_start=data.get("frame_start", 1))
    for bone_name, keys in bones.items():
        if not isinstance(keys, list):
            continue
        track = []
        for k in keys:
            if not isinstance(k, dict):
                continue
            frame = int(k.get("frame", anim.frame_start))
            loc = _anim_vec(k.get("location"), 3, (0.0, 0.0, 0.0))
            rot = _anim_vec(k.get("rotation"), 4, (1.0, 0.0, 0.0, 0.0))
            scl = _anim_vec(k.get("scale"), 3, (1.0, 1.0, 1.0))
            track.append((frame, loc, rot, scl))
        if track:
            track.sort(key=lambda t: t[0])
            anim.tracks[str(bone_name)] = track
    if not anim.tracks:
        raise GlacierAnimError("no usable keyframes found in animation JSON")
    return anim


def anim_match_report(anim, bone_names):
    """How well the animation's tracks line up with an armature's bones."""
    have = set(bone_names)
    matched = [b for b in anim.tracks if b in have]
    missing = [b for b in anim.tracks if b not in have]
    return {"matched": sorted(matched), "missing": sorted(missing),
            "matched_count": len(matched), "missing_count": len(missing)}


def anim_probe_binary(raw):
    """Read the IOI container header without decoding tracks, returning the raw
    fields so the native layout can be identified. Mirrors the PRIM/BORG pattern
    (leading u64 header_offset, then a small header table)."""
    import struct
    if len(raw) < 16:
        raise GlacierAnimError("file too small to be a Glacier container")
    header_offset = struct.unpack_from("<Q", raw, 0)[0]
    info = {"size": len(raw), "header_offset": header_offset}
    if not (0 < header_offset < len(raw) - 8):
        info["note"] = "header_offset out of range - not a recognised container"
        return info
    fields = []
    for i in range(8):
        off = header_offset + i * 4
        if off + 4 <= len(raw):
            fields.append(struct.unpack_from("<I", raw, off)[0])
    info["header_u32"] = fields
    info["header_f32"] = [struct.unpack_from("<f", raw, header_offset + i * 4)[0]
                          for i in range(8) if header_offset + i * 4 + 4 <= len(raw)]
    return info


def anim_parse_binary(raw, bone_names, bind_poses=None):
    """Best-effort native parse. Until the track layout is calibrated from a
    sample this raises with the probe diagnostics rather than emitting wrong
    motion. The probe info is attached so the importer can surface it."""
    info = anim_probe_binary(raw)
    raise GlacierAnimError(
        "Native Glacier 2 animation decoding needs a sample file to calibrate. "
        "Header probe: %r. Use the JSON preview path for now." % info)


# ----- bpy apply ------------------------------------------------------------
def _anim_apply(arm_obj, anim, action_name="GlacierAnim", clear=True):
    """Write the animation's per-bone local pose keyframes onto `arm_obj` as a
    new Action and return (matched_bone_count, inserted_key_count)."""
    if arm_obj is None or getattr(arm_obj, "type", "") != "ARMATURE":
        raise GlacierAnimError("no armature to apply the animation to")
    if arm_obj.animation_data is None:
        arm_obj.animation_data_create()
    action = bpy.data.actions.new(action_name)
    arm_obj.animation_data.action = action

    matched = 0
    inserted = 0
    for bone_name, keys in anim.tracks.items():
        pb = arm_obj.pose.bones.get(bone_name)
        if pb is None:
            continue
        matched += 1
        try:
            pb.rotation_mode = "QUATERNION"
        except Exception:
            pass
        for frame, loc, rot, scl in keys:
            if loc is not None:
                pb.location = loc
                pb.keyframe_insert(data_path="location", frame=frame)
                inserted += 1
            if rot is not None:
                pb.rotation_quaternion = rot
                pb.keyframe_insert(data_path="rotation_quaternion", frame=frame)
                inserted += 1
            if scl is not None:
                pb.scale = scl
                pb.keyframe_insert(data_path="scale", frame=frame)
                inserted += 1
    sc = bpy.context.scene
    try:
        sc.render.fps = max(1, int(round(anim.fps)))
        sc.frame_start = min(sc.frame_start, anim.frame_start)
        sc.frame_end = max(sc.frame_end, anim.frame_end)
        sc.frame_set(anim.frame_start)
    except Exception:
        pass
    return matched, inserted


def _anim_read_file(filepath):
    """Load a file as either the JSON preview schema or a binary container.
    Returns a GlacierAnim (JSON) or raises GlacierAnimError (binary, with probe)."""
    import json
    with open(filepath, "rb") as f:
        raw = f.read()
    stripped = raw.lstrip()[:1]
    if stripped in (b"{", b"["):
        try:
            return anim_from_json(json.loads(raw.decode("utf-8")))
        except GlacierAnimError:
            raise
        except Exception as e:
            raise GlacierAnimError("could not parse JSON animation: %s" % e)
    return anim_parse_binary(raw, [])


class IMPORT_SCENE_OT_glacier2_anim(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.glacier2_anim"
    bl_label = "Import Glacier 2 Animation"
    bl_description = ("Load an animation onto the active/target armature for "
                      "preview. Accepts the documented JSON keyframe schema now; "
                      "native .anim files are probed and reported")
    bl_options = {"REGISTER", "UNDO"}
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json;*.anim;*. animset;*.*", options={"HIDDEN"})

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None:
            self.report({"WARNING"}, "Select / set the target armature first")
            return {"CANCELLED"}
        try:
            anim = _anim_read_file(self.filepath)
        except GlacierAnimError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        names = [b.name for b in arm.data.bones]
        rep = anim_match_report(anim, names)
        if rep["matched_count"] == 0:
            self.report({"WARNING"}, "None of the animation's %d bone tracks match "
                        "this armature's bone names" % anim.bone_count)
            return {"CANCELLED"}
        context.view_layer.objects.active = arm
        import os as _os
        try:
            matched, inserted = _anim_apply(
                arm, anim, action_name=_os.path.basename(self.filepath))
        except GlacierAnimError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        arm["glacier_anim_report"] = ("%s: %d/%d bone tracks matched, %d keys, "
                                      "frames %d-%d @ %dfps" % (
                                          _os.path.basename(self.filepath), matched,
                                          anim.bone_count, inserted, anim.frame_start,
                                          anim.frame_end, int(round(anim.fps))))
        self.report({"INFO"}, arm["glacier_anim_report"])
        return {"FINISHED"}


class GLACIER_OT_anim_clear(bpy.types.Operator):
    bl_idname = "glacier.anim_clear"
    bl_label = "Clear Animation"
    bl_description = "Remove the active animation Action from the target armature"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        arm = _rig_active_armature(context)
        if arm is None or arm.animation_data is None:
            return {"CANCELLED"}
        arm.animation_data.action = None
        self.report({"INFO"}, "Animation cleared")
        return {"FINISHED"}


_panel_classes = (
    GlacierRefOverride,
    GlacierTexSlot,
    GlacierRenderSlot,
    GlacierMatParam,
    GlacierMaterial,
    GLACIER_UL_overrides,
    GLACIER_UL_materials,
    GLACIER_UL_render_slots,
    GLACIER_OT_pull_reference,
    GLACIER_OT_decode_texture,
    GLACIER_OT_fill_hashes,
    GLACIER_OT_decode_model,
    GLACIER_OT_decode_folder,
    GLACIER_OT_reencode,
    GLACIER_OT_generate_missing,
    GLACIER_OT_build_materials,
    GLACIER_OT_set_shading,
    GLACIER_OT_override_refresh,
    GLACIER_OT_scan_folder,
    GLACIER_OT_load_material_file,
    GLACIER_OT_update_names,
    GLACIER_OT_override_add,
    GLACIER_OT_override_remove,
    GLACIER_OT_lod_show_all,
    GLACIER_OT_lod_show_lod0,
    GLACIER_OT_inspect_texture,
    GlacierChunkEntry,
    GLACIER_UL_chunk_entries,
    GLACIER_OT_chunk_scan,
    GLACIER_OT_chunk_refresh,
    GLACIER_OT_chunk_select,
    GLACIER_OT_chunk_paste_select,
    GLACIER_OT_chunk_extract,
    GLACIER_OT_chunk_browser,
    GLACIER_OT_rig_analyze,
    GLACIER_OT_rig_clean,
    GLACIER_OT_rig_build,
    GLACIER_OT_rig_face,
    GLACIER_OT_rig_remove,
    IMPORT_SCENE_OT_glacier2_anim,
    GLACIER_OT_anim_clear,
    VIEW3D_PT_glacier_mesh_tools,
)


# =============================================================================
# Registration
# =============================================================================
def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_glacier2_prim.bl_idname,
                         text="Glacier 2 007 Model (.prim)")
    self.layout.operator(IMPORT_SCENE_OT_glacier2_borg.bl_idname,
                         text="Glacier 2 007 Skeleton (.borg)")
    # Animation import is held back from the public build (kept registered for the
    # dev panel only); re-add the File > Import entry when it ships.


def menu_func_export(self, context):
    self.layout.operator(EXPORT_SCENE_OT_glacier2_prim.bl_idname,
                         text="Glacier 2 007 Model (.prim)")


classes = (
    IMPORT_SCENE_OT_glacier2_prim,
    IMPORT_SCENE_OT_glacier2_borg,
    EXPORT_SCENE_OT_glacier2_prim,
)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    for c in _panel_classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.glacier_overrides = bpy.props.CollectionProperty(type=GlacierRefOverride)
    bpy.types.Scene.glacier_overrides_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.glacier_tex_slots = bpy.props.CollectionProperty(type=GlacierTexSlot)
    bpy.types.Scene.glacier_tex_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.glacier_params = bpy.props.CollectionProperty(type=GlacierMatParam)
    bpy.types.Scene.glacier_materials = bpy.props.CollectionProperty(type=GlacierMaterial)
    bpy.types.Scene.glacier_active_material = bpy.props.EnumProperty(
        name="Material",
        description="Which loaded material's textures and parameters the Edit Material "
                    "section below shows. Switch between the model's materials here",
        items=_material_enum_items)
    bpy.types.Scene.glacier_mat_file = bpy.props.StringProperty(
        name="Material File",
        description="Pick one .MATI (a full material - shader + textures + params) or "
                    ".MATB (just the schema) to load it into the list below",
        subtype="FILE_PATH", default="")
    bpy.types.Scene.glacier_lod_level = bpy.props.IntProperty(
        name="LOD Level", description="Show this LOD in each material group "
        "(clamped to each group's lowest LOD)", min=0, max=7, default=0,
        update=_update_lod)
    bpy.types.Scene.glacier_inspect_tex = bpy.props.StringProperty(
        name="Inspect .TEXT", description="A game-format .TEXT to read header info from",
        subtype="FILE_PATH", default="")
    bpy.types.Scene.glacier_scan_folder = bpy.props.StringProperty(
        name="Scan Folder", subtype="DIR_PATH", default="",
        description="Folder to search (including every sub-folder) for the model's "
                    ".MATI materials and .TEXT/.TEXD textures. Point this at your "
                    "extracted RPKG output, then press Scan Folder")
    bpy.types.Scene.glacier_materials_index = bpy.props.IntProperty(
        default=0, update=_sync_active_material)
    bpy.types.Scene.glacier_scan_model_only = bpy.props.BoolProperty(
        name="Only This Model's Materials", default=True,
        description="When scanning, keep only the materials the imported/selected "
                    "model actually uses. Turn off to load every .MATI in the folder")
    bpy.types.Scene.glacier_bc_format = bpy.props.EnumProperty(
        name="Compression", default="AUTO",
        description="Block compression used when converting your image to a texture",
        items=[
            ("AUTO", "Auto (match original)",
             "Use the same format as the texture you are replacing - safest"),
            ("BC1", "BC1 (color, opaque)", "RGB, no alpha - basecolor, light, specular"),
            ("BC3", "BC3 (color + alpha)", "RGBA - color with smooth alpha"),
            ("BC4", "BC4 (1 channel)", "Single grayscale channel - roughness, AO, masks"),
            ("BC5", "BC5 (2 channel)", "Two channels - normal maps (RG)"),
            ("BC7", "BC7 (high quality RGBA)",
             "Best quality color+alpha (mode 6). Use for 2K/4K basecolor/normal. "
             "Slower to encode in pure Python"),
        ])
    bpy.types.Scene.glacier_decode_text = bpy.props.StringProperty(
        name=".TEXT", subtype="FILE_PATH", default="",
        description="A single game .TEXT to decode to an image. The .TEXT holds the "
                    "header (size + format) and the small mips; pair it with its .TEXD "
                    "below for full resolution")
    bpy.types.Scene.glacier_decode_texd = bpy.props.StringProperty(
        name=".TEXD", subtype="FILE_PATH", default="",
        description="Optional matching .TEXD (the full-resolution half). Leave blank to "
                    "decode just the .TEXT's low-res mips. The .TEXD has a DIFFERENT "
                    "hash to its .TEXT - the addon finds it from the .TEXT meta")
    bpy.types.Scene.glacier_decode_fmt = bpy.props.EnumProperty(
        name="Save As", default="PNG",
        description="Image format decoded textures are written as in the Work Folder",
        items=[("PNG", "PNG", "Lossless PNG with alpha - the safe default"),
               ("TGA", "TGA", "Targa - handy for some external tools")])
    bpy.types.Scene.glacier_work_dir = bpy.props.StringProperty(
        name="Work Folder", subtype="DIR_PATH", default="",
        description="Where decoded/re-encoded textures are written. Blank = a "
                    "'007_textures' folder next to your .blend (or temp if unsaved)")
    bpy.types.Scene.glacier_tex_folder = bpy.props.StringProperty(
        name="Texture Folder", subtype="DIR_PATH", default="",
        description="Folder to search (with sub-folders) for .TEXT/.TEXD to decode. "
                    "Blank = use the Materials 'Scan Folder'")
    bpy.types.Scene.glacier_organize_textures = bpy.props.BoolProperty(
        name="Sort Into Hash Folders", default=True,
        description="Write each re-encoded file into TYPE/<hash>/ so the .TEXT and "
                    ".TEXD (which have DIFFERENT hashes) land in their own separate "
                    "folders. Off = write everything flat into the Work Folder")
    bpy.types.Scene.glacier_names_file = bpy.props.StringProperty(
        name="Names File", subtype="FILE_PATH", default="",
        description="Optional. Maps hashes to readable IOI paths so materials show real "
                    "names instead of hashes. HOW TO GET ONE: in RPKG-Tool use 'Generate "
                    "Hash List' (or any hashlist / dependency .txt with lines like "
                    "'...[assembly:/.../head_bond_v1.mi]'); or just point this at the "
                    "folder of .meta.json files RPKG-Tool wrote when you extracted - they "
                    "contain the paths. Leave blank to auto-scan the material folders")
    bpy.types.Scene.glacier_show_io = bpy.props.BoolProperty(default=True)
    bpy.types.Scene.glacier_show_mats = bpy.props.BoolProperty(default=True)
    bpy.types.Scene.glacier_show_render = bpy.props.BoolProperty(default=True)
    bpy.types.Scene.glacier_build_scope = bpy.props.EnumProperty(
        name="Build", default="MODEL",
        description="Which materials to build into Blender render materials",
        items=[
            ("MODEL", "Whole Model",
             "Build every loaded material and assign each to the imported meshes that "
             "use it - one click to a render-ready character"),
            ("ACTIVE", "Active -> Selected",
             "Build only the active material and apply it to the selected objects"),
        ])
    bpy.types.Scene.glacier_show_edit = bpy.props.BoolProperty(default=True)
    bpy.types.Scene.glacier_show_source = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_render_slots = bpy.props.CollectionProperty(
        type=GlacierRenderSlot)
    bpy.types.Scene.glacier_render_slots_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.glacier_render_from_reference = bpy.props.BoolProperty(
        name="Use This List When Building", default=True,
        description="Build the render material strictly from the textures pulled from "
                    "reference (above), instead of from the export texture overrides. "
                    "Turn off to go back to using the Edit Material overrides")
    bpy.types.Scene.glacier_show_swap = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_show_conv = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_show_lod = bpy.props.BoolProperty(default=False)
    # ---- RPKG chunk browser ----
    bpy.types.Scene.glacier_show_chunk = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_chunk_path = bpy.props.StringProperty(
        name="Chunk", subtype="FILE_PATH",
        description="The game's packed chunkNN.rpkg (in the Runtime folder)")
    bpy.types.Scene.glacier_chunk_out = bpy.props.StringProperty(
        name="Extract To", subtype="DIR_PATH",
        description="Where extracted .TEXT/.TEXD go. Blank uses the Work folder")
    bpy.types.Scene.glacier_chunk_filter = bpy.props.EnumProperty(
        name="Show", items=_CHUNK_FILTER_ITEMS, default="TEXTD")
    bpy.types.Scene.glacier_chunk_search = bpy.props.StringProperty(
        name="Search", description="Show only hashes containing this text")
    bpy.types.Scene.glacier_chunk_decode = bpy.props.BoolProperty(
        name="Also decode to PNG",
        description="After extracting, decode each TEXT (+paired TEXD) to a PNG",
        default=False)
    bpy.types.Scene.glacier_chunk_xor_key = bpy.props.StringProperty(
        name="XOR Key",
        description="Advanced: hex XOR descramble key. Blank = default key")
    bpy.types.Scene.glacier_chunk_paste_hashes = bpy.props.StringProperty(
        name="Hash List",
        description="Paste a list of 16-hex hashes to auto-select in the browser. "
                    "Any separator works (newlines, commas, spaces, tabs)")
    bpy.types.Scene.glacier_chunk_entries = bpy.props.CollectionProperty(
        type=GlacierChunkEntry)
    bpy.types.Scene.glacier_chunk_index = bpy.props.IntProperty(default=0)
    bpy.types.Scene.glacier_chunk_total = bpy.props.IntProperty(default=0)
    bpy.types.Scene.glacier_chunk_shown = bpy.props.IntProperty(default=0)
    bpy.types.Scene.glacier_chunk_matching = bpy.props.IntProperty(default=0)

    # ----- Control Rig properties -----
    def _rig_is_armature(self, obj):
        return getattr(obj, "type", None) == "ARMATURE"
    bpy.types.Scene.glacier_show_rig = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_rig_target = bpy.props.PointerProperty(
        name="Armature", type=bpy.types.Object, poll=_rig_is_armature,
        description="The imported skeleton to rig (defaults to the active armature)")
    bpy.types.Scene.glacier_rig_ik_arms = bpy.props.BoolProperty(
        name="IK Arms", default=True,
        description="Build IK chains + pole targets and an IK/FK switch for arms")
    bpy.types.Scene.glacier_rig_ik_legs = bpy.props.BoolProperty(
        name="IK Legs", default=True,
        description="Build IK chains + pole targets and an IK/FK switch for legs")
    bpy.types.Scene.glacier_rig_foot_roll = bpy.props.BoolProperty(
        name="Foot Roll", default=True,
        description="Add a foot-roll control when heel/ball/toe bones are present")
    bpy.types.Scene.glacier_rig_twist = bpy.props.BoolProperty(
        name="Twist Follow", default=True,
        description="Make twist/roll bones follow their parent's rotation")
    bpy.types.Scene.glacier_rig_clean_dryrun = bpy.props.BoolProperty(
        name="Dry Run", default=True,
        description="Only report which bones cleanup would remove, don't delete")
    bpy.types.Scene.glacier_show_anim = bpy.props.BoolProperty(default=False)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for _p in ("glacier_show_anim", "glacier_rig_clean_dryrun", "glacier_rig_twist",
               "glacier_rig_foot_roll", "glacier_rig_ik_legs", "glacier_rig_ik_arms",
               "glacier_rig_target", "glacier_show_rig"):
        try:
            delattr(bpy.types.Scene, _p)
        except Exception:
            pass
    for _p in ("glacier_chunk_matching", "glacier_chunk_shown",
               "glacier_chunk_total", "glacier_chunk_index", "glacier_chunk_entries",
               "glacier_chunk_xor_key", "glacier_chunk_paste_hashes",
               "glacier_chunk_decode", "glacier_chunk_search",
               "glacier_chunk_filter", "glacier_chunk_out", "glacier_chunk_path",
               "glacier_show_chunk"):
        try:
            delattr(bpy.types.Scene, _p)
        except Exception:
            pass
    del bpy.types.Scene.glacier_lod_level
    del bpy.types.Scene.glacier_show_lod
    del bpy.types.Scene.glacier_show_conv
    del bpy.types.Scene.glacier_show_swap
    del bpy.types.Scene.glacier_show_edit
    del bpy.types.Scene.glacier_show_source
    del bpy.types.Scene.glacier_show_mats
    del bpy.types.Scene.glacier_show_io
    del bpy.types.Scene.glacier_show_render
    del bpy.types.Scene.glacier_build_scope
    del bpy.types.Scene.glacier_decode_fmt
    del bpy.types.Scene.glacier_tex_folder
    del bpy.types.Scene.glacier_names_file
    del bpy.types.Scene.glacier_organize_textures
    del bpy.types.Scene.glacier_work_dir
    del bpy.types.Scene.glacier_decode_texd
    del bpy.types.Scene.glacier_decode_text
    del bpy.types.Scene.glacier_bc_format
    del bpy.types.Scene.glacier_scan_model_only
    del bpy.types.Scene.glacier_materials_index
    del bpy.types.Scene.glacier_scan_folder
    del bpy.types.Scene.glacier_inspect_tex
    del bpy.types.Scene.glacier_mat_file
    del bpy.types.Scene.glacier_active_material
    del bpy.types.Scene.glacier_materials
    del bpy.types.Scene.glacier_params
    del bpy.types.Scene.glacier_tex_index
    del bpy.types.Scene.glacier_tex_slots
    del bpy.types.Scene.glacier_overrides_index
    del bpy.types.Scene.glacier_overrides
    for c in reversed(_panel_classes):
        bpy.utils.unregister_class(c)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
