"""
PRIM builder — geometry helpers and custom-topology PRIM export for
007 First Light. Requires bpy at runtime (Blender), but can be imported
outside Blender for unit testing of the pure-math helpers.
"""
import struct
from .prim_layout import scale_shift

try:
    import bpy
    _HAS_BPY = True
except ImportError:
    _HAS_BPY = False

from .prim import read_prim_bytes, _enc_unit_byte, quantize_positions


# ---------------------------------------------------------------------------
# Face-bone token lists (used by _conform_face_bones)
# ---------------------------------------------------------------------------

_RIG_FACE_TOKENS = ("jaw", "eye", "lid", "brow", "lip", "mouth", "cheek", "nose",
                    "tongue", "teeth", "chin", "forehead", "nostril", "sneer",
                    "smile", "frown", "blink", "squint", "dimple", "pucker")

_CONFORM_RIGID_TOKENS = ("eyeball", "eye_ball", "iris", "sclera", "cornea", "pupil",
                         "eyewetness", "eye_wetness")
_CONFORM_SURFACE_TOKENS = ("lash", "eyelash", "eyeshadow", "eye_shadow", "shadow",
                           "eyelid", "lid", "tearline", "waterline", "occlusion")


# ---------------------------------------------------------------------------
# Low-level write helpers
# ---------------------------------------------------------------------------

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


def _normalise_influences(joints, weights):
    """Return four non-negative influences, merged by bone and sorted strongest
    first. Blender vertex groups are free-form; the PRIM writer needs a compact
    four-slot game skin record with a sane weight sum."""
    pairs = {}
    for j, w in zip(joints or (), weights or ()):
        try:
            ji = int(j)
            wf = float(w)
        except Exception:
            continue
        if ji < 0 or wf <= 0.0:
            continue
        pairs[ji] = pairs.get(ji, 0.0) + wf
    if not pairs:
        pairs = {0: 1.0}
    items = sorted(pairs.items(), key=lambda it: -it[1])[:4]
    total = sum(w for _j, w in items) or 1.0
    js = [j for j, _w in items]
    ws = [w / total for _j, w in items]
    while len(js) < 4:
        js.append(0)
        ws.append(0.0)
    return tuple(js), tuple(ws)


# ---------------------------------------------------------------------------
# Tangent computation
# ---------------------------------------------------------------------------

def _compute_tangents(positions, normals, uvs, indices):
    """Per-vertex tangent/bitangent from UV gradients (Lengyel), Gram-Schmidt
    orthonormalised against the normal. Without these the skin shader's normal
    mapping degenerates and the mesh renders blank/black."""
    n = len(positions)
    acc = [[0.0, 0.0, 0.0] for _ in range(n)]     # tangent direction  (dP/dU)
    acc2 = [[0.0, 0.0, 0.0] for _ in range(n)]    # bitangent direction (dP/dV)
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
        sb = ((e2[0] * du1 - e1[0] * du2) * f,
              (e2[1] * du1 - e1[1] * du2) * f,
              (e2[2] * du1 - e1[2] * du2) * f)
        for i in (i0, i1, i2):
            acc[i][0] += s[0]; acc[i][1] += s[1]; acc[i][2] += s[2]
            acc2[i][0] += sb[0]; acc2[i][1] += sb[1]; acc2[i][2] += sb[2]
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
        # Bitangent = handedness * (N x T). 007 First Light stores the bitangent
        # EXPLICITLY with the sign of the UV winding - verified on real game
        # meshes, the stored bitangent matches Lengyel UV handedness on 99.6%% of
        # vertices. The old code emitted +(N x T) unconditionally, which flips
        # handedness on every UV-mirrored face (the majority here). That inverts
        # tangent-space normal mapping: subtle head-on, but it is exactly the
        # corrupted/"burnt" shading seen in mirror reflections and on the lower
        # LODs the intro renders. Derive the sign from the UV gradient instead.
        bx = ny * tz - nz * ty
        by = nz * tx - nx * tz
        bz = nx * ty - ny * tx
        hx, hy, hz = acc2[i]
        if bx * hx + by * hy + bz * hz < 0.0:
            bx, by, bz = -bx, -by, -bz
        tans.append((tx, ty, tz)); bitans.append((bx, by, bz))
    return tans, bitans


# ---------------------------------------------------------------------------
# LOD propagation: transfer LOD 0 edits to lower LODs
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Head-conform: shift eyeballs / eyelashes / eye-shadow (and face bones) to
# follow edits made to the head mesh
# ---------------------------------------------------------------------------

def _similarity_transform(src, dst):
    """Best-fit scale s, rotation R (3x3, row-major), translation t mapping
    src -> dst in a least-squares sense (Horn's quaternion method, pure Python).
    Keeps a rigid body rigid (no shear), so an eyeball stays a sphere. Returns
    (s, R, t); falls back to identity rotation when degenerate."""
    n = min(len(src), len(dst))
    if n == 0:
        return 1.0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0]
    cs = [sum(src[i][k] for i in range(n)) / n for k in range(3)]
    cd = [sum(dst[i][k] for i in range(n)) / n for k in range(3)]
    xs = [[src[i][k] - cs[k] for k in range(3)] for i in range(n)]
    xd = [[dst[i][k] - cd[k] for k in range(3)] for i in range(n)]
    M = [[0.0] * 3 for _ in range(3)]
    for i in range(n):
        for a in range(3):
            for b in range(3):
                M[a][b] += xs[i][a] * xd[i][b]
    Sxx, Sxy, Sxz = M[0]
    Syx, Syy, Syz = M[1]
    Szx, Szy, Szz = M[2]
    N = [
        [Sxx + Syy + Szz, Syz - Szy,        Szx - Sxz,        Sxy - Syx],
        [Syz - Szy,        Sxx - Syy - Szz, Sxy + Syx,        Szx + Sxz],
        [Szx - Sxz,        Sxy + Syx,       -Sxx + Syy - Szz, Syz + Szy],
        [Sxy - Syx,        Szx + Sxz,        Syz + Szy,       -Sxx - Syy + Szz],
    ]
    tr = N[0][0] + N[1][1] + N[2][2] + N[3][3]
    for i in range(4):
        N[i][i] += abs(tr) + 1.0
    q = [1.0, 0.0, 0.0, 0.0]
    for _ in range(64):
        nq = [sum(N[r][c] * q[c] for c in range(4)) for r in range(4)]
        mag = sum(v * v for v in nq) ** 0.5
        if mag < 1e-12:
            break
        nq = [v / mag for v in nq]
        if sum(abs(nq[i] - q[i]) for i in range(4)) < 1e-12:
            q = nq
            break
        q = nq
    w, x, y, z = q
    R = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ]
    num = den = 0.0
    for i in range(n):
        rx = [sum(R[a][b] * xs[i][b] for b in range(3)) for a in range(3)]
        num += sum(rx[a] * xd[i][a] for a in range(3))
        den += sum(xs[i][a] * xs[i][a] for a in range(3))
    s = (num / den) if den > 1e-12 else 1.0
    if not (s == s) or s <= 1e-6:        # NaN / collapse guard
        s = 1.0
    t = [cd[a] - s * sum(R[a][b] * cs[b] for b in range(3)) for a in range(3)]
    return s, R, t


def _apply_similarity(s, R, t, pts):
    out = []
    for p in pts:
        out.append((
            s * (R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2]) + t[0],
            s * (R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2]) + t[1],
            s * (R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2]) + t[2],
        ))
    return out


def _build_point_grid(pts):
    """Spatial hash for repeated nearest / radius queries against `pts`."""
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; zs = [p[2] for p in pts]
    ext = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1e-6)
    cell = max(ext / max(1, int(len(pts) ** 0.33)), 1e-8)
    inv = 1.0 / cell
    grid = {}
    for i, p in enumerate(pts):
        grid.setdefault((int(p[0] * inv), int(p[1] * inv), int(p[2] * inv)), []).append(i)
    return (grid, inv, pts)


def _grid_knn(g, p, k):
    """Indices of the (up to) k nearest grid points to p, growing the ring until
    at least k candidates are gathered."""
    if g is None:
        return []
    grid, inv, pts = g
    cx = int(p[0] * inv); cy = int(p[1] * inv); cz = int(p[2] * inv)
    cand = []
    r = 1
    while r <= 16:
        cand = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    cand.extend(grid.get((cx + dx, cy + dy, cz + dz), []))
        if len(cand) >= k:
            break
        r += 1
    cand.sort(key=lambda j: (pts[j][0]-p[0])**2 + (pts[j][1]-p[1])**2 + (pts[j][2]-p[2])**2)
    return cand[:k]


def _grid_within(g, c, radius):
    """Indices of all grid points within `radius` of centre c."""
    if g is None:
        return []
    grid, inv, pts = g
    rr = radius * radius
    cr = int(radius * inv) + 1
    cx = int(c[0] * inv); cy = int(c[1] * inv); cz = int(c[2] * inv)
    out = []
    for dx in range(-cr, cr + 1):
        for dy in range(-cr, cr + 1):
            for dz in range(-cr, cr + 1):
                for j in grid.get((cx + dx, cy + dy, cz + dz), []):
                    q = pts[j]
                    if (q[0]-c[0])**2 + (q[1]-c[1])**2 + (q[2]-c[2])**2 <= rr:
                        out.append(j)
    return out


def _knn_surface_deform(dep_pts, head_orig, head_delta, grid, k=6):
    """Move each dependent vertex by the inverse-distance-weighted displacement of
    its k nearest ORIGINAL head vertices. Smooth, robust, and topology-free, so a
    surface-hugging mesh (eyelashes, eye-shadow) rides the reshaped head."""
    out = []
    for p in dep_pts:
        nbr = _grid_knn(grid, p, k)
        if not nbr:
            out.append(tuple(p)); continue
        sw = 0.0; dd = [0.0, 0.0, 0.0]
        for j in nbr:
            q = head_orig[j]
            d2 = (q[0]-p[0])**2 + (q[1]-p[1])**2 + (q[2]-p[2])**2
            w = 1.0 / (d2 + 1e-9)
            sw += w
            for a in range(3):
                dd[a] += w * head_delta[j][a]
        out.append((p[0] + dd[0]/sw, p[1] + dd[1]/sw, p[2] + dd[2]/sw))
    return out


def _bbox(P):
    xs = [p[0] for p in P]; ys = [p[1] for p in P]; zs = [p[2] for p in P]
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _centroid(P):
    n = len(P) or 1
    return (sum(p[0] for p in P)/n, sum(p[1] for p in P)/n, sum(p[2] for p in P)/n)


def _split_clusters(pts):
    """Split a point set into spatially separated clusters (e.g. the left/right
    eyeballs) by recursively cutting at the largest gap along the widest axis.
    Returns a list of index lists; one list when the part is a single blob."""
    clusters = []
    stack = [list(range(len(pts)))]
    while stack:
        grp = stack.pop()
        if len(grp) < 8:
            clusters.append(grp); continue
        best_axis = 0; best_ext = -1.0
        for a in range(3):
            vals = [pts[i][a] for i in grp]
            ext = max(vals) - min(vals)
            if ext > best_ext:
                best_ext = ext; best_axis = a
        grp.sort(key=lambda i: pts[i][best_axis])
        gap = 0.0; cut = -1
        for kk in range(1, len(grp)):
            d = pts[grp[kk]][best_axis] - pts[grp[kk-1]][best_axis]
            if d > gap:
                gap = d; cut = kk
        if cut > 0 and gap > 0.25 * best_ext and (len(clusters) + len(stack)) < 6:
            stack.append(grp[:cut]); stack.append(grp[cut:])
        else:
            clusters.append(grp)
    return clusters


def _part_is_rigid(pts):
    """A part is rigid (an eyeball) when its largest cluster is chunky in all
    three axes; a thin shell (lash / shadow) is not."""
    cl = _split_clusters(pts)
    sub = [pts[i] for i in max(cl, key=len)]
    b = _bbox(sub)
    ext = sorted([b[3]-b[0], b[4]-b[1], b[5]-b[2]])
    return ext[2] > 1e-6 and ext[0] / ext[2] > 0.45


def _conform_role_from_name(name):
    """'rigid' / 'surface' / '' from a material or object name's tokens."""
    s = (name or "").lower()
    if any(t in s for t in _CONFORM_RIGID_TOKENS):
        return "rigid"
    if any(t in s for t in _CONFORM_SURFACE_TOKENS):
        return "surface"
    # a bare 'eye' usually means the eyeball
    if "eye" in s and "brow" not in s:
        return "rigid"
    return ""


def _conform_classify(parts):
    """Pick the head driver and assign each candidate dependent a conform mode.

    `parts` is a list of dicts: {idx, mat, name, lod, pts}. Returns
    (driver_part, [(part, mode), ...]) where mode is 'rigid' (eyeballs - kept a
    rigid body) or 'surface' (lashes / shadow - hug the surface). Detection uses
    the material/object name first, then geometry: the biggest LOD-0 mesh is the
    head; a small part is rigid when it is chunky in all 3 axes (an eyeball),
    else surface."""
    lod0 = [p for p in parts if p["lod"] == 0 and p["pts"]]
    if not lod0:
        return None, []

    def vol(p):
        b = _bbox(p["pts"])
        return (b[3]-b[0]) * (b[4]-b[1]) * (b[5]-b[2])
    driver = max(lod0, key=vol)
    dvol = vol(driver) or 1e-9
    db = _bbox(driver["pts"])
    ext = [db[3]-db[0], db[4]-db[1], db[5]-db[2]]
    up = max(range(3), key=lambda a: ext[a])      # tallest head axis = vertical
    up_c = _centroid(driver["pts"])[up]
    _EXCL = ("mouth", "teeth", "tooth", "tongue", "gum", "neck", "throat", "jawbag")
    deps = []
    for p in parts:
        if p is driver or p["mat"] == driver["mat"] or not p["pts"]:
            continue
        low = (p["name"] or "").lower()
        role = _conform_role_from_name(p["name"])
        if role:
            deps.append((p, role))               # named eye part: always include
            continue
        if any(t in low for t in _EXCL):
            continue                             # named mouth/neck part: skip
        # unnamed: keep only small parts sitting in the UPPER (eye) region, so the
        # eyeballs / lashes / shadow are caught but the mouth, teeth and neck aren't
        if vol(p) > 0.25 * dvol:
            continue
        if _centroid(p["pts"])[up] < up_c - 0.05 * ext[up]:
            continue
        deps.append((p, "rigid" if _part_is_rigid(p["pts"]) else "surface"))
    return driver, deps


def _conform_head_parts(template_data, blender_objs, do_bones=False, armature=None):
    """Bake the head reshape onto the dependent parts. Compares the head's LOD-0
    Blender positions to the original .prim, then for every dependent object (all
    LODs) returns new vertex positions: a best-fit similarity for rigid eyeballs,
    a k-NN surface deform for lashes / shadow. Returns
    (overrides {prim_index: [(x,y,z),...]}, info-dict)."""
    orig = read_prim_bytes(template_data)
    table = orig.header.object_table
    by_idx = {int(o["glacier_prim_index"]): o for o in blender_objs
              if "glacier_prim_index" in o}

    parts = []
    for idx, o in by_idx.items():
        if idx >= len(table):
            continue
        pts = [(v.position[0], v.position[1], v.position[2])
               for v in table[idx].vertexBuffer.vertices]
        name = ""
        try:
            if o.data.materials and o.data.materials[0]:
                name = o.data.materials[0].name
        except Exception:
            pass
        parts.append({"idx": idx, "mat": o.get("glacier_material_id", -1),
                      "lod": o.get("glacier_lod_index", 0), "name": name,
                      "pts": pts, "obj": o})
    driver, deps = _conform_classify(parts)
    info = {"driver": None, "rigid": 0, "surface": 0, "bones": 0}
    if driver is None or not deps:
        return {}, info
    info["driver"] = driver["mat"]

    head_orig = driver["pts"]
    head_cur = _mesh_vertex_coords(driver["obj"].data)
    n0 = min(len(head_orig), len(head_cur))
    if n0 == 0:
        return {}, info
    head_orig = head_orig[:n0]
    head_cur = head_cur[:n0]
    head_delta = [(head_cur[i][0]-head_orig[i][0],
                   head_cur[i][1]-head_orig[i][1],
                   head_cur[i][2]-head_orig[i][2]) for i in range(n0)]
    if max(abs(d[0])+abs(d[1])+abs(d[2]) for d in head_delta) < 1e-6:
        return {}, info                      # head unchanged - nothing to do
    grid = _build_point_grid(head_orig)

    overrides = {}
    eye_centroids = []                       # (orig_centroid, new_centroid) for bones
    for part, mode in deps:
        if mode == "rigid":
            # eyeballs come in pairs - transform each one with its OWN socket so
            # widening/narrowing the face moves them apart correctly, and each
            # stays a rigid sphere.
            new = list(part["pts"])
            for cl in _split_clusters(part["pts"]):
                sub = [part["pts"][i] for i in cl]
                c = _centroid(sub)
                b = _bbox(sub)
                rad = 0.5 * max(b[3]-b[0], b[4]-b[1], b[5]-b[2]) * 1.6 + 1e-4
                sock = _grid_within(grid, c, rad)
                if len(sock) < 4:
                    sock = _grid_knn(grid, c, 12)
                s, R, t = _similarity_transform([head_orig[j] for j in sock],
                                                [head_cur[j] for j in sock])
                moved = _apply_similarity(s, R, t, sub)
                for kk, i in enumerate(cl):
                    new[i] = moved[kk]
                eye_centroids.append((c, _centroid(moved)))
            overrides[part["idx"]] = new
            info["rigid"] += 1
        else:
            overrides[part["idx"]] = _knn_surface_deform(
                part["pts"], head_orig, head_delta, grid)
            info["surface"] += 1

    if do_bones and armature is not None:
        try:
            info["bones"] = _conform_face_bones(armature, head_orig, head_cur,
                                                grid, eye_centroids)
        except Exception:
            info["bones"] = 0
    return overrides, info


def _conform_face_bones(armature, head_orig, head_cur, grid, eye_centroids):
    """Shift the armature's face bones to follow the reshaped geometry, in EDIT
    mode. A face bone is moved by the local head displacement around its head
    position; an eye bone is additionally re-seated so its pivot sits at the new
    eyeball centre (stops eyes 'popping out' when animated). Returns the count
    moved. Must be called with a real Blender armature object."""
    moved = 0
    prev = armature.mode if hasattr(armature, "mode") else "OBJECT"
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        ebones = armature.data.edit_bones

        def local_shift(co):
            nbr = _grid_knn(grid, co, 6)
            if not nbr:
                return (0.0, 0.0, 0.0)
            sw = 0.0; dd = [0.0, 0.0, 0.0]
            for j in nbr:
                q = head_orig[j]
                w = 1.0 / ((q[0]-co[0])**2 + (q[1]-co[1])**2 + (q[2]-co[2])**2 + 1e-9)
                sw += w
                for a in range(3):
                    dd[a] += w * (head_cur[j][a] - q[a])
            return (dd[0]/sw, dd[1]/sw, dd[2]/sw)

        for eb in ebones:
            low = eb.name.lower()
            if not any(tok in low for tok in _RIG_FACE_TOKENS):
                continue
            is_eye = ("eye" in low and "lid" not in low and "brow" not in low
                      and "lash" not in low)
            if is_eye and eye_centroids:
                # re-seat the eye bone's pivot at the nearest new eyeball centre
                h = (eb.head.x, eb.head.y, eb.head.z)
                oc, nc = min(eye_centroids,
                             key=lambda pair: (pair[0][0]-h[0])**2 +
                             (pair[0][1]-h[1])**2 + (pair[0][2]-h[2])**2)
                off = (nc[0]-oc[0], nc[1]-oc[1], nc[2]-oc[2])
            else:
                off = local_shift((eb.head.x, eb.head.y, eb.head.z))
            eb.head = (eb.head.x + off[0], eb.head.y + off[1], eb.head.z + off[2])
            eb.tail = (eb.tail.x + off[0], eb.tail.y + off[1], eb.tail.z + off[2])
            moved += 1
    finally:
        try:
            bpy.ops.object.mode_set(mode=prev if prev in ("OBJECT", "POSE", "EDIT") else "OBJECT")
        except Exception:
            pass
    return moved


# ---------------------------------------------------------------------------
# Main PRIM builder
# ---------------------------------------------------------------------------

def build_custom_prim(template, objects, weighted, preserve_cloth=True):
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
        layout_shift = scale_shift(template, orig_off)
        o_ps = struct.unpack_from("<4f", template, orig_off + 72 + layout_shift)
        o_pb = struct.unpack_from("<4f", template, orig_off + 88 + layout_shift)
        o_tsb = struct.unpack_from("<4f", template, orig_off + 104 + layout_shift)
        o_cloth = struct.unpack_from("<I", template, orig_off + 120 + layout_shift)[0]
        # cloth-sim preservation: cages store a self-sized cloth blob (magic
        # 0x0002009C) at cloth_data_offset (PRIM_OBJECT +68); aux_offset (+64)
        # marks its end. We copy that blob verbatim and repoint both fields so the
        # simulation survives a re-export. Links reference the cage's own vertices
        # by index, so this is valid as long as the vertex COUNT is unchanged.
        o_aux_field = struct.unpack_from("<I", template, orig_off + 64)[0]
        o_clothdat = struct.unpack_from("<I", template, orig_off + 68)[0]
        _is_cloth_cage = bool(
            o_clothdat and o_aux_field > o_clothdat and o_clothdat + 8 <= len(template)
            and struct.unpack_from("<I", template, o_clothdat + 4)[0] == 0x0002009C)

        use_ps = o_ps; use_pb = o_pb; use_tsb = o_tsb

        # --- copy the 72-byte PRIM_OBJECT header verbatim ----------------------
        # The header fields (sub_type, material_id, lod_mask, etc.) must not
        # change; only the data pointers (vbo/ibo/aux/skin) are patched after.
        b += template[orig_off:orig_off + 44]        # PRIM_OBJECT fixed header

        # --- flattened submesh fields (+44 to +72) ----------------------------
        fields_off = len(b)
        b += struct.pack("<I", n)                    # num_vertices
        b += struct.pack("<I", 0)                    # vbo (patched)
        b += struct.pack("<I", len(od["indices"]))   # num_indices
        b += struct.pack("<I", 0)                    # unknown_0C
        b += struct.pack("<I", 0)                    # ibo (patched)
        b += struct.pack("<I", 0)                    # aux_offset (patched)
        b += struct.pack("<I", 0)                    # cloth_data_offset (patched)

        # pos_scale, pos_bias, tex_scale_bias, cloth_id (+72 to +120)
        if layout_shift:
            b += b"\xff" * 4
        b += struct.pack("<4f", *use_ps)
        b += struct.pack("<4f", *use_pb)
        b += struct.pack("<4f", *use_tsb)
        b += struct.pack("<I", o_cloth)              # cloth_id

        # --- weighted-mesh trailer (+120 to +140) ----------------------------
        if weighted and sub == 2:
            b += struct.pack("<IIIII", 0, 0, 0, 0, 0)  # patched later

        # update global AABB
        for p in od["positions"]:
            for axis in range(3):
                if p[axis] < g_min[axis]: g_min[axis] = p[axis]
                if p[axis] > g_max[axis]: g_max[axis] = p[axis]

        _align16(b)

        # --- index buffer ---------------------------------------------------
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
                o_bi, o_binfo, _cc, _co, o_skin = struct.unpack_from("<5I", template, orig_off + 124 + layout_shift)
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
                # Regenerate the skin partition for a different vertex count.
                # Format verified against original PRIMs:
                #   - Batching is by TRIANGLE (IBO entry triplet), not by vertex.
                #   - BoneIndices contains IBO rendering indices grouped by batch
                #     (NOT vertex indices 0..n-1).
                #   - BoneInfo batch entries (cursor, count) sit at the LAST
                #     nacc*8 bytes of the BoneInfo block, at a fixed base offset
                #     of 2052 from the block start. The 1792 bytes between the
                #     bone_remap and the batch entries are 0xFF (runtime fill).
                #   - total_size = 2052 + nacc * 8  (constant 2052 from format).
                for i in range(n):
                    j, ws = _normalise_influences(od["joints"][i], od["weights"][i])
                    od["joints"][i] = j
                    od["weights"][i] = ws
                w255 = [_weights_to_255(list(od["weights"][i])) for i in range(n)]

                # Build per-vertex bone sets (for triangle-based palette check)
                vbones = []
                for i in range(n):
                    j = od["joints"][i]
                    s = set(int(j[k]) for k in range(4) if w255[i][k] > 0)
                    vbones.append(s if s else {int(j[0])})

                # Group TRIANGLES into batches so each batch uses <= _SKIN_PALETTE bones
                indices = od["indices"]
                num_tris = len(indices) // 3
                tri_batches, pal, cur_tris = [], set(), []
                for t in range(num_tris):
                    t_bones = (vbones[indices[t*3]]
                               | vbones[indices[t*3+1]]
                               | vbones[indices[t*3+2]])
                    if cur_tris and len(pal | t_bones) > _SKIN_PALETTE:
                        tri_batches.append(cur_tris)
                        pal, cur_tris = set(), []
                    pal |= t_bones
                    cur_tris.append(t)
                if cur_tris:
                    tri_batches.append(cur_tris)

                # BoneInfo block: 2052 fixed base + nacc*8 batch entries at end
                # Layout: 4B header | 255B bone_remap | 1B pad | 1792B fill | nacc*8B batches
                # Batch entries are always at offset 2052 from block start.
                nacc = len(tri_batches)
                total_size = 2052 + nacc * 8
                bone_info_off = len(b)
                b += struct.pack("<H", total_size)
                b += struct.pack("<H", nacc)
                # bone_remap: identity mapping for every bone actually used.
                # All-0xFF causes the engine to reference bone #255 (invalid)
                # for every GPU palette slot → vertices fly to garbage positions.
                _used_bones = set()
                for _ri in range(n):
                    _wv = w255[_ri]
                    _jv = od["joints"][_ri]
                    for _ki in range(4):
                        if _wv[_ki] > 0:
                            _bid = int(_jv[_ki]) & 0xFF
                            if _bid < 255:
                                _used_bones.add(_bid)
                _remap = bytearray([0xFF] * 255)
                for _bid in _used_bones:
                    _remap[_bid] = _bid
                b += bytes(_remap)
                b += bytes([0])            # padding
                b += bytes([0xFF] * 1792)  # fill: always 1792 bytes (offsets 260..2052)
                # Batch entries at offset 2052: (cursor, count) per batch
                # cursor = flat index into BoneIndices (after 2 sentinels)
                cursor_flat = 2
                for bt in tri_batches:
                    ibo_count = len(bt) * 3
                    b += struct.pack("<II", cursor_flat, ibo_count)
                    cursor_flat += ibo_count
                _align16(b)

                # BoneIndices: flat IBO entries grouped by batch (triangle order)
                bone_indices_off = len(b)
                total_ibo = sum(len(bt) * 3 for bt in tri_batches)
                b += struct.pack("<I", 2 + total_ibo)   # count including sentinels
                b += struct.pack("<HH", 0, 0)           # 2 leading sentinels
                for bt in tri_batches:
                    for t in bt:
                        b += struct.pack("<HHH",
                                         indices[t*3] & 0xFFFF,
                                         indices[t*3+1] & 0xFFFF,
                                         indices[t*3+2] & 0xFFFF)
                _align16(b)

                # Per-vertex skin data: 4 weight bytes + packed 3×10-bit bone indices
                skin_off = len(b)
                for i in range(n):
                    b += bytes(w255[i])
                    j = od["joints"][i]
                    packed = (int(j[0]) & 0x3FF) | ((int(j[1]) & 0x3FF) << 10) | ((int(j[2]) & 0x3FF) << 20)
                    b += struct.pack("<I", packed)
                _align16(b)
            struct.pack_into("<IIIII", b, off + 124 + layout_shift,
                             bone_indices_off, bone_info_off, 0, 0, skin_off)

        struct.pack_into("<I", b, fields_off + 4, vbo)
        struct.pack_into("<I", b, fields_off + 16, ibo)

        # --- aux block (REQUIRED on every object) -----------------------------
        # Each object has an aux_offset (+0x40) pointing to a per-object cull
        # cluster table: u16 clusterCount, u16 type(0x20), then count*6 bytes,
        # padded to 16. The engine dereferences this on EVERY object, so leaving
        # aux_offset 0 (as older builds did) crashes custom meshes. Preserve the
        # original table byte-exact when the vertex count is unchanged; for a
        # changed count emit an empty (0-cluster) table so the pointer is valid.
        def _emit_aux_block():
            _align16(b)
            _new = len(b)
            if n == o_nv and o_aux_field and o_aux_field + 4 <= len(template):
                _ac = struct.unpack_from("<H", template, o_aux_field)[0]
                _asz = (4 + _ac * 6 + 15) & ~15
                b.extend(template[o_aux_field:o_aux_field + _asz])
            else:
                b.extend(struct.pack("<HH", 0, 0x20))
                _align16(b)
            return _new

        # preserve the cloth-sim blob for cages whose vertex count is unchanged
        if _is_cloth_cage and n == o_nv and preserve_cloth:
            _align16(b)
            new_cloth_off = len(b)
            b += template[o_clothdat:o_aux_field]      # the self-sized cloth blob
            struct.pack_into("<I", b, fields_off + 24, new_cloth_off)  # cloth_data_offset
            aux_new = len(b)                           # aux block sits right after cloth
            if o_aux_field and o_aux_field + 4 <= len(template):
                _ac = struct.unpack_from("<H", template, o_aux_field)[0]
                _asz = (4 + _ac * 6 + 15) & ~15
                b += template[o_aux_field:o_aux_field + _asz]
            struct.pack_into("<I", b, fields_off + 20, aux_new)        # aux_offset
        else:
            struct.pack_into("<I", b, fields_off + 20, _emit_aux_block())

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
