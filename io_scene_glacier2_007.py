bl_info = {
    "name": "Glacier 2 — 007 First Light Toolkit",
    "description": (
        "Full modding toolkit for 007 First Light (Glacier / KNT engine). "
        "Import .prim models and .borg skeletons with skin weights and shape keys; "
        "reshape meshes and export back to game-valid .prim (+meta). Load and edit "
        "materials (.MATI/.MATB), swap or repoint textures, and one-click build real "
        "Blender render materials from the game's skin shader (basecolor, SRM, normal, "
        "translucency, with parameters wired in). Native pure-Python texture codec: "
        "decode and encode .TEXT/.TEXD (BC1/BC3/BC4/BC5/BC7) to and from PNG/TGA, "
        "auto-detect formats, generate a missing .TEXT/.TEXD from an image, and package "
        "everything with correct DISTINCT TEXT/TEXD hashes and metas. Plus LOD tools and "
        "material-name resolution from IOI paths."),
    "author": "Glacier modding community",
    "version": (1, 14, 0),
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

    def read(self, br, count, mesh):
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

        if sub == 2:  # WEIGHTED: 8B sub-A | 16B NTB+UV | 4B colour (stream-major)
            for v in self.vertices:
                br.u8vec(8)  # sub-A (unidentified) - skipped for import
            for v in self.vertices:
                v.normal = decode_unit_vec4ub(br.u8vec(4))
                br.u8vec(4)  # tangent
                br.u8vec(4)  # bitangent
                u = br.i16(); vv = br.i16()
                v.uv[0][0] = (u / 32767.0) * tsb[0] + tsb[2]
                v.uv[0][1] = (vv / 32767.0) * tsb[1] + tsb[3]
            for v in self.vertices:
                v.color = br.u8vec(4)
        elif sub in (0, 1):  # LINKED/STANDARD: 16 B/vert interleaved NTB+UV
            for v in self.vertices:
                v.normal = decode_unit_vec4ub(br.u8vec(4))
                br.u8vec(4)
                br.u8vec(4)
                u = br.i16(); vv = br.i16()
                v.uv[0][0] = (u / 32767.0) * tsb[0] + tsb[2]
                v.uv[0][1] = (vv / 32767.0) * tsb[1] + tsb[3]
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
        br.u32()                                   # aux/collision (unused on import)
        br.u32()                                   # unknown_18

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
            self.vertexBuffer.read(br, self.num_vertices, self)

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

    def draw(self, context):
        layout = self.layout
        layout.label(text="Import options:")
        layout.prop(self, "import_shapekeys")
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
        self.report({"INFO"}, "Imported %d object(s)%s" % (
            prim.num_objects(), " + rig" if arma is not None else ""))
        return {"FINISHED"}


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
        sub = meta["sub_type"]
        if sub == 2:            # weighted: positions(8N) + subA(8N) then NTB+UV(16)
            nrm_base, stride = vbo + 16 * n, 16
        elif sub in (0, 1):     # linked/standard: positions(8N) then NTB+UV(16)
            nrm_base, stride = vbo + 8 * n, 16
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
    if m["refs_table_size"] > 0:
        cnt = struct.unpack_from("<H", data, 40)[0]
        m["dummy"] = struct.unpack_from("<H", data, 42)[0]
        flags = data[44:44 + cnt]
        hbase = 44 + cnt
        for i in range(cnt):
            h = struct.unpack_from("<Q", data, hbase + i * 8)[0]
            m["refs"].append((h, flags[i]))
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

    # Single-subset modes (4/5/6) decode exactly. Partitioned modes need the
    # partition/anchor tables; until those are verified against real files we
    # approximate a partitioned block by the average of its endpoints.
    if ns > 1:
        avg = [sum(e[c] for e in ep) // len(ep) for c in range(4)]
        return [list(avg) for _ in range(16)]

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

        verbatim = (not changed) and weighted and sub == 2 and n == o_nv

        # PRIM_OBJECT (44 B) copied verbatim; only patch bbox when topology changed
        po = bytearray(template[orig_off:orig_off + 44])
        if changed:
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
        b += struct.pack("<4f", *o_ps)                  # ORIGINAL pos scale
        b += struct.pack("<4f", *o_pb)                  # ORIGINAL pos bias
        b += struct.pack("<4f", *o_tsb)                 # ORIGINAL tex scale/bias
        b += struct.pack("<I", o_cloth)                 # preserve cloth id
        if weighted:
            b += b"\x00" * 20                           # +124 weighted trailer

        # --- index buffer (always re-emitted; identical bytes for unchanged) ---
        _align16(b)
        ibo = len(b)
        for idx in od["indices"]:
            b += struct.pack("<H", idx & 0xFFFF)

        # --- vertex buffer ---
        _align16(b)
        vbo = len(b)
        ts = (o_tsb[0], o_tsb[1]); tb = (o_tsb[2], o_tsb[3])
        if verbatim:
            # copy the entire original vertex buffer (pos + Sub-A + NTB+UV + colour)
            b += template[o_vbo:o_vbo + o_nv * 36]
        else:
            for i, p in enumerate(od["positions"]):
                wlane = int(od["joints"][i][3]) if (weighted and od["joints"]) else 0
                b += struct.pack("<hhh", _q_i16(p[0], o_pb[0], o_ps[0]),
                                         _q_i16(p[1], o_pb[1], o_ps[1]),
                                         _q_i16(p[2], o_pb[2], o_ps[2]))
                b += struct.pack("<h", wlane)
            tans, bitans = _compute_tangents(od["positions"], od["normals"],
                                             od["uvs"], od["indices"])
            if weighted and sub == 2:
                if n == o_nv:                            # keep original Sub-A bytes
                    b += template[o_vbo + o_nv * 8:o_vbo + o_nv * 8 + n * 8]
                else:                                    # synth Sub-A from UVs
                    for i in range(n):
                        u = _q_i16(od["uvs"][i][0], tb[0], ts[0])
                        v = _q_i16(od["uvs"][i][1], tb[1], ts[1])
                        b += struct.pack("<hhhh", u, v, u, v)
                for i in range(n):
                    b += _enc_normal(od["normals"][i])
                    b += _enc_normal(tans[i])
                    b += _enc_normal(bitans[i])
                    b += struct.pack("<hh", _q_i16(od["uvs"][i][0], tb[0], ts[0]),
                                            _q_i16(od["uvs"][i][1], tb[1], ts[1]))
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
        _align16(b)
        if weighted:
            if verbatim:
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
    b += template[th + 24:th + 48]     # preserve ORIGINAL global bounds (no culling)

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

        if self.custom_topology:
            out_data, patched, note = self._rebuild_custom(data, objs, obj_metas, weighted)
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

        self.report({"INFO"}, "Exported %d object(s) to %s%s%s%s" %
                    (patched, os.path.basename(out_prim), note, wrote_meta, mat_note))
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

    def _rebuild_custom(self, template, objs, obj_metas, weighted):
        by_index = {int(o["glacier_prim_index"]): o for o in objs}
        orig_prim = read_prim_bytes(template)
        rebuilt = []
        changed = 0
        for idx, meta in enumerate(obj_metas):
            if idx in by_index:
                o = by_index[idx]
                d = self._extract_mesh(o, meta["off"], meta["sub_type"], weighted)
                d["changed"] = (len(o.data.vertices) != meta["num_vertices"])
                if d["changed"]:
                    changed += 1
                rebuilt.append(d)
            else:
                d = self._extract_original(orig_prim, idx, meta, weighted)
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
            ".meta.json", ".log", ".md")
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
        try:
            txt = open(f, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for m in _MI_NAME_RE.finditer(txt):
            names.setdefault(m.group(1).upper(), m.group(2).strip())
        # JSON form (e.g. an RPKG .meta.json that carries IOI paths per reference)
        if f.lower().endswith(".json"):
            try:
                import json as _json
                obj = _json.loads(txt)
            except Exception:
                obj = None
            if obj is not None:
                _harvest_json_names(obj, names)
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


def build_glacier_blender_material(name, slots, params, images_by_slot):
    """Create/replace a Blender material `name` with a node graph that mirrors the
    game's skin shader: basecolor -> Base Color, SRM -> Separate -> Map Range (skin
    roughness range) + specular/metallic, normal -> Separate -> Combine (B=1) ->
    Normal Map, translucency -> Multiply -> Subsurface. Material parameters become
    labelled Value nodes that drive the graph. Everything sits in titled frames."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (760, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled"); bsdf.location = (380, 0)
    bsdf.location = (380, 0)
    nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    # skin defaults
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

    pby = {(p.name or "").lower(): p for p in params}

    def pval(*keys):
        for k in keys:
            if k in pby:
                return pby[k]
        return None

    roles = {}
    for ts in slots:
        img = images_by_slot.get(ts)
        if img is not None:
            roles.setdefault(_slot_role(ts.slot_name), img)

    def frame(title, color=None):
        fr = nt.nodes.new("NodeFrame")
        fr.label = title
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
        if noncolor:
            try:
                image.colorspace_settings.name = "Non-Color"
            except Exception:
                pass
        return n

    def value_node(label, v, x, y, parent):
        n = nt.nodes.new("ShaderNodeValue")
        n.label = label; n.location = (x, y); n.parent = parent
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

    # ---- BASE COLOR ----------------------------------------------------------
    if "base" in roles:
        fr = frame("Base Color", (0.18, 0.12, 0.10))
        t = img_node(roles["base"], False, -1180, 520, fr, "Basecolor")
        bc = _bsdf_input(bsdf, "Base Color")
        if bc is not None:
            nt.links.new(t.outputs["Color"], bc)
        a_in = _bsdf_input(bsdf, "Alpha")
        # leave Alpha at default unless a dedicated opacity map exists

    # ---- SRM : Specular / Roughness / Metallic -------------------------------
    if "srm" in roles:
        fr = frame("SRM  -  Specular / Roughness / Metallic", (0.10, 0.14, 0.18))
        t = img_node(roles["srm"], True, -1180, 150, fr, "SRM")
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-880, 170); sep.parent = fr; sep.label = "Split SRM"
        set_mode_rgb(sep)
        nt.links.new(t.outputs["Color"], sep.inputs[0])

        # Roughness = green, remapped into the skin roughness range
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.location = (-560, 120); mr.parent = fr; mr.label = "Roughness Range"
        try:
            mr.clamp = True
        except Exception:
            pass
        nt.links.new(sep.outputs[1], mr.inputs[0])         # Value <- Green
        rmin = pval("roughness_min"); rmax = pval("roughness_max")
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
        rough = _bsdf_input(bsdf, "Roughness")
        if rough is not None:
            nt.links.new(mr.outputs[0], rough)
        # Specular from red, Metallic from blue
        spec = _bsdf_input(bsdf, "Specular IOR Level", "Specular")
        if spec is not None:
            nt.links.new(sep.outputs[0], spec)
        metal = _bsdf_input(bsdf, "Metallic")
        if metal is not None:
            nt.links.new(sep.outputs[2], metal)

    # ---- NORMAL : Separate -> Combine (B=1) -> Normal Map --------------------
    if "normal" in roles:
        fr = frame("Normal", (0.10, 0.16, 0.10))
        t = img_node(roles["normal"], True, -1180, -260, fr, "Normal")
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-880, -240); sep.parent = fr; sep.label = "Split Normal"
        set_mode_rgb(sep)
        nt.links.new(t.outputs["Color"], sep.inputs[0])
        comb = nt.nodes.new("ShaderNodeCombineColor")
        comb.location = (-620, -240); comb.parent = fr; comb.label = "Rebuild (B=1)"
        set_mode_rgb(comb)
        nt.links.new(sep.outputs[0], comb.inputs[0])       # R
        nt.links.new(sep.outputs[1], comb.inputs[1])       # G
        try:
            comb.inputs[2].default_value = 1.0             # B = 1
        except Exception:
            pass
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-360, -240); nm.parent = fr; nm.label = "Normal Map"
        nt.links.new(comb.outputs[0], nm.inputs["Color"])
        bs = pval("norm_bumpscale", "normalstrength", "bumpscale")
        if bs is not None:
            vn = value_node("Normal Strength", max(0.0, abs(bs.fval)) or 1.0,
                            -360, -60, fr)
            try:
                nt.links.new(vn.outputs[0], nm.inputs["Strength"])
            except Exception:
                pass
        nin = _bsdf_input(bsdf, "Normal")
        if nin is not None:
            nt.links.new(nm.outputs["Normal"], nin)

    # ---- TRANSLUCENCY -> SUBSURFACE -----------------------------------------
    if "translucency" in roles:
        fr = frame("Translucency  ->  Subsurface", (0.17, 0.10, 0.16))
        t = img_node(roles["translucency"], True, -1180, -640, fr, "Translucency")
        mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"
        mul.location = (-760, -620); mul.parent = fr; mul.label = "Intensity"
        nt.links.new(t.outputs["Color"], mul.inputs[0])
        ti = pval("translucency_intensity", "translucency")
        if ti is not None:
            vn = value_node("Translucency Intensity", max(0.0, ti.fval) or 0.3,
                            -760, -470, fr)
            nt.links.new(vn.outputs[0], mul.inputs[1])
        else:
            try:
                mul.inputs[1].default_value = 0.3
            except Exception:
                pass
        ssw = _bsdf_input(bsdf, "Subsurface Weight", "Subsurface")
        if ssw is not None:
            nt.links.new(mul.outputs[0], ssw)

    # ---- EMISSION / ALPHA (only if the material actually has them) -----------
    if "emission" in roles:
        fr = frame("Emission", (0.18, 0.16, 0.08))
        t = img_node(roles["emission"], False, -1180, -1020, fr, "Emission")
        ec = _bsdf_input(bsdf, "Emission Color", "Emission")
        if ec is not None:
            nt.links.new(t.outputs["Color"], ec)
        es = _bsdf_input(bsdf, "Emission Strength")
        if es is not None:
            try:
                es.default_value = 1.0
            except Exception:
                pass
    if "alpha" in roles:
        t = img_node(roles["alpha"], True, -1180, -1320, None, "Opacity")
        a = _bsdf_input(bsdf, "Alpha")
        if a is not None:
            nt.links.new(t.outputs["Color"], a)
            try:
                mat.blend_method = "HASHED"
            except Exception:
                pass

    # ---- SKIN COLOUR (game tint param - exposed for the user) ----------------
    sk = pval("skincolor", "tintcolor")
    if sk is not None and getattr(sk, "type", 0) == 0x03:
        fr = frame("Parameters", (0.13, 0.13, 0.13))
        rgb = nt.nodes.new("ShaderNodeRGB")
        rgb.location = (-1180, 760); rgb.parent = fr; rgb.label = "Skin Color"
        try:
            rgb.outputs[0].default_value = (sk.color[0], sk.color[1], sk.color[2], 1.0)
        except Exception:
            pass
        # connect to Specular Tint when that socket exists (otherwise just exposed)
        st = _bsdf_input(bsdf, "Specular Tint")
        if st is not None:
            try:
                nt.links.new(rgb.outputs[0], st)
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
            try:
                if tp:                                   # decode TEXT (+TEXD)
                    w, h, rgba = decode_texture_file(tp, dp)
                elif dp:                                 # only a TEXD - decode headerless
                    w, h, rgba, _f = decode_texd_standalone(dp)
                else:
                    rgba = None
                if rgba is not None:
                    os.makedirs(wd, exist_ok=True)
                    outp = os.path.join(wd, "%s%s" % (eff, ext))
                    (write_png if ext == ".png" else write_tga)(outp, w, h, rgba)
                    img = _load_image_into_blend(outp)
            except Exception:
                img = None
        result[ts] = img
    return result


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
        for mt in mats:
            slots = [ts for ts in sc.glacier_tex_slots if ts.mati_hash == mt.key]
            params = [p for p in sc.glacier_params if p.mati_hash == mt.key]
            imgs = _ensure_slot_images(context, slots)
            for ts in slots:
                if imgs.get(ts) is None:
                    missing.append(ts.slot_name)
            name = "007_%s" % (mt.label or mt.key[:8])
            try:
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

        def section(prop, title, icon, badge=None):
            box = layout.box()
            head = box.row(align=True)
            head.prop(sc, prop, text="", emboss=False,
                      icon="TRIA_DOWN" if getattr(sc, prop) else "TRIA_RIGHT")
            head.label(text=title, icon=icon)
            if badge:
                r = head.row(); r.alignment = "RIGHT"; r.label(text=str(badge))
            return box.column() if getattr(sc, prop) else None

        nmat = len(sc.glacier_materials)
        groups = _glacier_lod_groups(context)

        # ============ IMPORT / EXPORT =====================================
        b = section("glacier_show_io", "Import / Export", "IMPORT")
        if b:
            col = b.column(align=True)
            col.scale_y = 1.2
            col.operator("import_scene.glacier2_007_prim",
                         text="Import Model", icon="MESH_MONKEY")
            col.operator("import_scene.glacier2_007_borg",
                         text="Import Skeleton", icon="ARMATURE_DATA")
            b.separator()
            col = b.column(align=True)
            col.scale_y = 1.2
            col.operator("export_scene.glacier2_007_prim",
                         text="Export Model + Edits", icon="EXPORT")

        # ============ MATERIALS ===========================================
        b = section("glacier_show_mats", "Materials", "MATERIAL", nmat or None)
        if b:
            b.operator("glacier.override_refresh",
                       text="Load From Imported Model", icon="FILE_REFRESH")
            box = b.box()
            box.label(text="Scan a folder", icon="VIEWZOOM")
            box.prop(sc, "glacier_scan_folder", text="")
            box.prop(sc, "glacier_scan_model_only")
            box.operator("glacier.scan_folder", text="Scan Folder", icon="ZOOM_ALL")
            box = b.box()
            box.label(text="Single file / names", icon="FILE")
            row = box.row(align=True)
            row.prop(sc, "glacier_mat_file", text="")
            row.operator("glacier.load_material_file", text="", icon="FILEBROWSER")
            box.prop(sc, "glacier_names_file", text="Names")
            if nmat:
                b.separator()
                b.template_list("GLACIER_UL_materials", "", sc, "glacier_materials",
                                sc, "glacier_materials_index", rows=5)

        # ============ RENDER ==============================================
        b = section("glacier_show_render", "Render Materials", "SHADING_RENDERED",
                    nmat or None)
        if b:
            if not nmat:
                b.label(text="Load a material first", icon="INFO")
            else:
                b.prop(sc, "glacier_build_scope", text="")
                op = b.operator("glacier.build_materials",
                                text="Build Render Materials", icon="NODE_MATERIAL")
                op.apply_to = sc.glacier_build_scope
                b.operator("glacier.set_shading",
                           text="Material Preview", icon="SHADING_TEXTURE")

        # ============ EDIT MATERIAL =======================================
        b = section("glacier_show_edit", "Edit Material", "RESTRICT_SELECT_OFF")
        if b:
            if not nmat:
                b.label(text="Load a material first", icon="INFO")
            else:
                b.prop(sc, "glacier_active_material", text="")
                active = sc.glacier_active_material
                mt_active = next((mt for mt in sc.glacier_materials
                                  if mt.key == active), None)
                is_bp = bool(mt_active and mt_active.is_blueprint)

                tbox = b.box()
                trow = tbox.row(align=True)
                trow.label(text="Textures", icon="TEXTURE")
                if not is_bp:
                    trow.operator("glacier.fill_hashes", text="Fill Hashes",
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
                    sb = tbox.box()
                    head = sb.row(align=True)
                    head.label(text=ts.slot_name,
                               icon="CHECKMARK" if changed else "DOT")
                    sub = head.row(); sub.alignment = "RIGHT"
                    sub.label(text=(ts.old_hash or "(none)")[:10])
                    if is_bp:
                        continue
                    sb.prop(ts, "tex_source", text="")
                    eff = ts.new_hash.strip() or ts.old_hash
                    if ts.tex_source == "HASH":
                        sb.prop(ts, "new_hash", text="Hash")
                    elif ts.tex_source == "IMAGE":
                        sb.prop(ts, "image_path", text="Image")
                        sb.prop(ts, "new_hash", text="New Hash")
                        sb.prop(ts, "texd_hash", text=".TEXD hash")
                    else:
                        sb.prop(ts, "file_path", text=".TEXT")
                        sb.prop(ts, "file_path_texd", text=".TEXD")
                        sb.prop(ts, "new_hash", text="New Hash")
                        sb.prop(ts, "texd_hash", text=".TEXD hash")
                    if ts.tex_source != "HASH":
                        if eff:
                            td = ("  TEXD %s" % ts.texd_hash[:16]) if ts.texd_hash else ""
                            sb.label(text="exports as %s%s" % (eff[:16], td),
                                     icon="CHECKMARK")
                        else:
                            sb.label(text="click Fill Hashes", icon="ERROR")
                if not any_tex:
                    tbox.label(text="(no textures)")

                pbox = b.box()
                pbox.label(text="Parameters", icon="MODIFIER")
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
            row = b.row()
            row.template_list("GLACIER_UL_overrides", "", sc, "glacier_overrides",
                              sc, "glacier_overrides_index", rows=3)
            c2 = row.column(align=True)
            c2.operator("glacier.override_add", text="", icon="ADD")
            c2.operator("glacier.override_remove", text="", icon="REMOVE")
            if 0 <= sc.glacier_overrides_index < len(sc.glacier_overrides):
                b.prop(sc.glacier_overrides[sc.glacier_overrides_index],
                       "new_hash", text="New Hash")

        # ============ TEXTURE TOOLS =======================================
        b = section("glacier_show_conv", "Texture Tools", "IMAGE_DATA")
        if b:
            box = b.box()
            box.label(text="Folders", icon="FILE_FOLDER")
            box.prop(sc, "glacier_work_dir", text="Work")
            box.prop(sc, "glacier_tex_folder", text="Search")

            box = b.box()
            box.label(text="Decode / Re-encode", icon="IMAGE_RGB")
            row = box.row(align=True)
            row.prop(sc, "glacier_decode_fmt", text="As")
            box.operator("glacier.decode_model",
                         text="Decode Textures", icon="IMPORT")
            box.prop(sc, "glacier_bc_format", text="Encode")
            box.operator("glacier.reencode",
                         text="Re-encode Images", icon="EXPORT")
            box.operator("glacier.generate_missing",
                         text="Generate Missing .TEXT/.TEXD", icon="FILE_NEW")
            box.prop(sc, "glacier_organize_textures")

            box = b.box()
            box.label(text="Single texture", icon="FILE_IMAGE")
            box.prop(sc, "glacier_decode_text", text=".TEXT")
            box.prop(sc, "glacier_decode_texd", text=".TEXD")
            box.operator("glacier.decode_texture", text="Decode to Image",
                         icon="FILE_IMAGE")
            row = box.row(align=True)
            row.prop(sc, "glacier_inspect_tex", text="Inspect")
            row.operator("glacier.inspect_texture", text="", icon="VIEWZOOM")

        # ============ LEVEL OF DETAIL =====================================
        b = section("glacier_show_lod", "Level of Detail", "MOD_DECIM",
                    len(groups) or None)
        if b:
            if groups:
                b.prop(sc, "glacier_lod_level", slider=True)
                row = b.row(align=True)
                row.operator("glacier.lod_show_lod0")
                row.operator("glacier.lod_show_all")
            else:
                b.label(text="Import a model to use LOD tools", icon="INFO")



_panel_classes = (
    GlacierRefOverride,
    GlacierTexSlot,
    GlacierMatParam,
    GlacierMaterial,
    GLACIER_UL_overrides,
    GLACIER_UL_materials,
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
    GLACIER_OT_override_add,
    GLACIER_OT_override_remove,
    GLACIER_OT_lod_show_all,
    GLACIER_OT_lod_show_lod0,
    GLACIER_OT_inspect_texture,
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
        description="Optional text / hash-list / dependency file that maps hashes to "
                    "IOI paths (e.g. lines like '... [assembly:/.../head_bond_v1.mi]'). "
                    "The addon reads the real material name from it. Leave blank to "
                    "auto-scan the material folders for such a file")
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
    bpy.types.Scene.glacier_show_swap = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_show_conv = bpy.props.BoolProperty(default=False)
    bpy.types.Scene.glacier_show_lod = bpy.props.BoolProperty(default=False)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    del bpy.types.Scene.glacier_lod_level
    del bpy.types.Scene.glacier_show_lod
    del bpy.types.Scene.glacier_show_conv
    del bpy.types.Scene.glacier_show_swap
    del bpy.types.Scene.glacier_show_edit
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
