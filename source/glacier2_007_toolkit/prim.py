"""
PRIM — 007 First Light RenderPrimitive parser and low-level mesh helpers.
Read-only parsing plus in-place patch utilities (positions, normals, colours).
"""
import struct
from .prim_layout import scale_shift, skip_scale_sentinel
import os

from .reader import Reader


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def decode_unit_vec4ub(b):
    # 007FL packed unit vector: xyz = (byte-128)/127.5, 4th byte = handedness.
    return [(b[0] - 128) / 127.5, (b[1] - 128) / 127.5, (b[2] - 128) / 127.5, b[3]]


def _enc_unit_byte(c):
    return max(0, min(255, int(round(c * 127.5 + 128.0))))


def _uv_oob_penalty(uvs):
    """Penalise a UV channel for how far it strays outside the [0,1] texture
    square. A real texture UV mostly lives in [0,1]; a fake/flow channel often
    does not. Sampled, cheap, and only used to break near-ties between channels."""
    if not uvs:
        return 0.0
    step = max(1, len(uvs) // 1500)
    tot = 0.0; n = 0
    for i in range(0, len(uvs), step):
        u, v = uvs[i]
        if u < 0.0:
            tot += -u
        elif u > 1.0:
            tot += u - 1.0
        if v < 0.0:
            tot += -v
        elif v > 1.0:
            tot += v - 1.0
        n += 1
    return (tot / n) if n else 0.0


# ---------------------------------------------------------------------------
# Header flags
# ---------------------------------------------------------------------------

class HeaderFlags:
    def __init__(self, value):
        self.bitfield = value

    def hasBones(self):           return self.bitfield & 0b1 == 1
    def isWeightedObject(self):   return self.bitfield & 0b1000 == 8
    def isLinkedObject(self):     return self.bitfield & 0b100 == 4


# ---------------------------------------------------------------------------
# Vertex / VertexBuffer
# ---------------------------------------------------------------------------

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
        self.uv_sets = []          # all decoded UV channels (for the Fix-UVs tool)

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

        # Cloth parts (shirts, capes, skirts) carry an EXTRA UV channel that the
        # subtype alone does NOT advertise: their per-vertex NTB+UV record is 20
        # bytes (two UVs), not 16. Reading them as uvc=1 used a 16-byte stride and
        # progressively scrambled the UVs (the shirt body came in with UVs ranging
        # far outside [0,1]). Derive the real channel count from the on-disk vertex
        # region size so the stride is correct; the channel scorer below then keeps
        # the right one. Region = [vbo .. cloth_data_offset | aux_offset).
        if weighted and count:
            region_end = (getattr(mesh, "cloth_data_offset", 0)
                          or getattr(mesh, "aux_offset", 0))
            vbo0 = getattr(mesh, "vbo", 0)
            if region_end and region_end > vbo0:
                bpv = (region_end - vbo0) / count
                # weighted vertex = 8 pos + 8 skin + (12 NTB + 4*uvc) + 4 colour
                uvc_region = int(round((bpv - 32) / 4.0))
                if 1 <= uvc_region <= 4:
                    uvc = uvc_region

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

            def _spread(uvs):
                if not uvs:
                    return 0.0
                us = [p[0] for p in uvs]; vs = [p[1] for p in uvs]
                return max(max(us) - min(us), max(vs) - min(vs))

            # SINGLE-UV: the channel sits at +12 (face/body: N T B UV) or +8
            # (eyelash / eyebrow / hair: N T UV B). Reading the wrong slot pulls the
            # UV out of the bitangent bytes -> scrambled, out-of-square garbage.
            # MULTI-UV uses the standard layout (+12, +16, ...), no +8 variant.
            if uvc == 1:
                def _rank1(uvs):
                    if _spread(uvs) < 1e-3:
                        return 1e9
                    return _score(uvs) + 2.0 * _uv_oob_penalty(uvs)

                def _axes(uvs):
                    us = [p[0] for p in uvs]; vs = [p[1] for p in uvs]
                    return (max(us) - min(us), max(vs) - min(vs))

                def _degenerate(uvs):
                    # a real texture UV spreads in BOTH axes; a wrong-slot read
                    # (UV taken from bitangent bytes) collapses one axis to a line.
                    ax = _axes(uvs)
                    return ax[0] < 0.02 or ax[1] < 0.02

                std_bad = _degenerate(uv_std)
                alt_bad = _degenerate(uv_alt)
                if std_bad and not alt_bad:
                    uv_at_8 = True
                elif alt_bad and not std_bad:
                    uv_at_8 = False          # +8 collapsed -> it's garbage, use +12
                elif std_bad and alt_bad:
                    uv_at_8 = False          # both poor: default to the common slot
                else:
                    # both carry a genuine 2D UV: prefer +12, take +8 only when it
                    # is clearly cleaner (shorter edges + less out-of-square).
                    uv_at_8 = _rank1(uv_alt) < _rank1(uv_std) * 0.9
                chosen = uv_alt if uv_at_8 else uv_std
                uv_sets = [chosen]
                best_i = 0
                # Colour-stream presence is a SEPARATE question from the UV slot, and
                # getting it wrong misaligns the NEXT object. Keep the historically
                # correct, conservative gate so byte consumption is byte-identical to
                # before - only the UV *values* move to the right slot.
                use_alt = _score(uv_alt) < _score(uv_std) * 0.6
            else:
                use_alt = False
                # Multi-UV objects (hair, cloth) carry several UV channels packed
                # back-to-back (channel j sits at +12 + 4*j). For HAIR CARDS the two
                # channels are (a) the real texture-atlas UV - each card mapped onto a
                # vertical strip of the hair sheet - and (b) a per-strand NORMALISED
                # flow/clump UV that remaps every card into ~[0,1], producing hundreds
                # of tiny overlapping "pill" islands. The flow channel's triangles are
                # roughly an order of magnitude SHORTER than the atlas's, so the old
                # "shortest-edge wins" rule always grabbed the flow channel - and since
                # only one channel was kept, the real hair UV never reached Blender.
                # Decode every channel, flag the normalised-flow channel(s) by their
                # much smaller triangle scale, and pick the texture atlas instead.
                uv_sets = [_decode_uv(12 + j * 4) for j in range(uvc)]

                edges = [_score(u) for u in uv_sets]          # mean tri edge length
                spreads = [_spread(u) for u in uv_sets]
                live = [j for j in range(uvc) if spreads[j] >= 1e-3] or list(range(uvc))
                max_edge = max((edges[j] for j in live), default=0.0)

                # A channel whose triangles are an order of magnitude smaller than the
                # coarsest live channel is a per-primitive normalised map (hair flow /
                # clump), never the texture atlas.
                def _is_flow(j):
                    return max_edge > 0.0 and edges[j] < 0.35 * max_edge

                texture_ch = [j for j in live if not _is_flow(j)] or live

                # Among the genuine texture channel(s), keep the cleanest: least
                # out-of-[0,1] excursion first, then the coarsest (largest islands ->
                # the atlas rather than a tighter secondary set).
                best_i = min(texture_ch,
                             key=lambda j: (_uv_oob_penalty(uv_sets[j]), -edges[j]))
                chosen = uv_sets[best_i]

            self.uv_sets = uv_sets          # every decoded channel (imported as maps)
            self.chosen_uv_index = best_i   # which channel became the active UVMap
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


# ---------------------------------------------------------------------------
# PRIM mesh classes
# ---------------------------------------------------------------------------

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
        self.lod_mask = 0xFF              # PRIM_OBJECT lodMask: bit i = present at LOD i

    def read(self, br, weighted):
        # PRIM_OBJECT (44 bytes)
        br.u8(); br.u8(); br.u16()                 # PRIM_HEADER
        self.sub_type = br.u8()
        self.lod_mask = br.u8()                    # lodMask: bit i set => in LOD i
        br.u8()                                    # properties / flags
        br.u8(); br.u8(); br.u8()                  # variant, zbias, zoffset
        self.material_id = br.u16()
        br.u32()                                   # wire colour
        br.u8vec(4)                                # color1
        self.min = br.fvec(3)
        self.max = br.fvec(3)

        # Flattened submesh fields (7 uint32)
        self.num_vertices = br.u32()
        vbo = br.u32()
        self.vbo = vbo
        self.num_indices = br.u32()
        br.u32()                                   # unknown_0C
        ibo = br.u32()
        self.aux_offset = br.u32()                 # aux/collision stream offset
        self.cloth_data_offset = br.u32()          # cloth sim blob offset (0 = none)

        self.path_tracing_layout = skip_scale_sentinel(br)
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


# ---------------------------------------------------------------------------
# File-level read functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# On-disk walk / patch helpers (operate on raw bytearray, no full parse)
# ---------------------------------------------------------------------------

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
        sub_type = data[off + 4]
        nv = u32(off + 44)
        vbo = u32(off + 48)
        aux = u32(off + 64)            # aux_offset
        cloth = u32(off + 68)          # cloth_data_offset (nonzero = cloth/hair)
        # UV channel count. Subtype 3/4/5 advertise 2/3/4 channels directly; for
        # weighted (sub 2) meshes, derive it from the NTB+UV region size:
        #   region = pos(8N) + skin(8N) + (12 NTB + 4*uvc UV)*N  -> per-vert = 32 + 4*uvc
        uvc = {3: 2, 4: 3, 5: 4}.get(sub_type, 1)
        region_end = cloth or aux
        if weighted and nv and region_end > vbo:
            u = int(round(((region_end - vbo) / nv - 32) / 4.0))
            if 1 <= u <= 4:
                uvc = u
        objs.append({
            "off": off,
            "sub_type": sub_type,
            "num_vertices": nv,
            "vbo": vbo,
            "num_indices": u32(off + 52),
            "aux_offset": aux,
            "cloth_data_offset": cloth,
            "uvc": uvc,
            "pos_scale_off": off + 72 + scale_shift(data, off),
            "pos_bias_off": off + 88 + scale_shift(data, off),
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
        # The vertex buffer is TIGHTLY packed (no 16-byte inter-stream padding).
        # Each NTB+UV record is 12 NTB bytes + 4*uvc UV bytes; the normal is the
        # first 4 bytes of that record. uvc=1 for most meshes, but hair carries 2
        # UV channels (20-byte records) — a hardcoded stride of 16 corrupts UVs.
        sub = meta["sub_type"]
        stride = 12 + 4 * int(meta.get("uvc", 1))
        if sub == 2:            # weighted: pos(8N) + skin(8N) + NTB+UV(stride*N)
            nrm_base = vbo + n * 16
        elif sub in (0, 1):     # unweighted: pos(8N) + NTB+UV(stride*N)
            nrm_base = vbo + n * 8
        else:
            return
        for i in range(n):
            o = nrm_base + i * stride
            w = data[o + 3]     # preserve handedness byte
            data[o + 0] = _enc_unit_byte(normals[i][0])
            data[o + 1] = _enc_unit_byte(normals[i][1])
            data[o + 2] = _enc_unit_byte(normals[i][2])
            data[o + 3] = w


def _color_stream_offset(meta):
    """Byte offset of the per-vertex colour stream (4 bytes/vert) for one object,
    or None if the layout has no colour stream room."""
    vbo = meta["vbo"]; n = meta["num_vertices"]; sub = meta.get("sub_type", 2)
    rec = 12 + 4 * int(meta.get("uvc", 1))   # NTB+UV record stride (tight)
    if sub == 2:
        nrm = vbo + n * 16                   # pos(8N) + skin(8N)
    elif sub in (0, 1):
        nrm = vbo + n * 8                     # pos(8N)
    else:
        return None
    return nrm + n * rec                      # colour stream follows NTB+UV region


def patch_object_colors(data, meta, colors):
    """Overwrite the per-vertex colour stream (cloth pin / sim weights) for one
    object, in place. Returns the number of vertices written, or 0."""
    n = meta["num_vertices"]
    base = _color_stream_offset(meta)
    if base is None or base + n * 4 > len(data):
        return 0
    for i in range(min(n, len(colors))):
        c = colors[i]
        data[base + i * 4 + 0] = int(c[0]) & 0xFF
        data[base + i * 4 + 1] = int(c[1]) & 0xFF
        data[base + i * 4 + 2] = int(c[2]) & 0xFF
        data[base + i * 4 + 3] = int(c[3]) & 0xFF
    return min(n, len(colors))
