"""
META — 007 First Light resource metadata parser/builder.
Handles .PRIM.meta, .BORG.meta and any other .meta sidecar files.
"""
import struct
import os


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


# ---------------------------------------------------------------------------
# .meta filename helpers
# ---------------------------------------------------------------------------

def meta_path_candidates(res_path, ext):
    """Return (dot_style_path, underscore_style_path) for a resource's meta."""
    base, _ = os.path.splitext(res_path)
    return [res_path + ".meta", base + "_" + ext.upper() + ".meta"]


def meta_uses_dot_style(res_path, ext):
    """True when <hash>.<EXT>.meta exists; False when _<EXT>.meta exists.
    Defaults to dot style (RPKG-Tool first-light convention)."""
    dot, und = meta_path_candidates(res_path, ext)
    if res_path and os.path.exists(dot):
        return True
    if res_path and os.path.exists(und):
        return False
    return True


def meta_out_name(hash_int, ext, dot_style):
    if dot_style:
        return "%016X.%s.meta" % (hash_int, ext.upper())
    return "%016X_%s.meta" % (hash_int, ext.upper())


def derive_resource_meta(res_path, ext):
    cands = meta_path_candidates(res_path, ext)
    for c in cands:
        if os.path.exists(c):
            return c
    return cands[0]


def find_resource_meta(res_path, ext, search_dirs):
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


def derive_sibling_resource(prim_path, hash_int, ext):
    """Path to <hash>.<EXT> next to the given resource file."""
    folder = os.path.dirname(prim_path)
    return os.path.join(folder, "%016X.%s" % (hash_int, ext.upper()))
