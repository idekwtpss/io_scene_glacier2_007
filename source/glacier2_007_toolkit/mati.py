"""
MATI / MATB / TEMP — 007 First Light material instance and blueprint parsers,
plus quickentity TEMP JSON colour reader.
"""
import struct
import json as _json
import os


# MATI record type → data-payload length in bytes.
# type 0x01 = float(4B), 0x02 = texture(8B), 0x03 = color/vec3(12B), 0x08 = matrix/blob(32B)
_MATI_TYPELEN = {0x01: 4, 0x02: 8, 0x03: 12, 0x08: 32}

# MATB property kind labels
_MATB_KIND = {1: "texture", 2: "color", 4: "float", 8: "int"}


def parse_mati(data):
    """Parse a MATI binary blob.

    Returns::
        {
            "strings": {offset: str, ...},
            "records": [(off, name_off, type, sub, data_off, data_len), ...],
            "textures": [{"name": str, "index": int, "data_off": int}, ...],
            "params":   [{"name": str, "type": int, "data_off": int, "values": [float, ...]}, ...],
        }
    """
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

    # Fallback for NODE-GRAPH materials (hair / haircap): recover texture slots
    # from the 16-byte node-graph record format when the flat walk yields nothing.
    if not textures and strings:
        texlike = {off: nm for off, nm in strings.items()
                   if ("texture" in nm.lower() or nm.lower().startswith("map"))}
        if texlike:
            seen = set()
            for pos in range(i, n - 1):
                noff = struct.unpack_from("<H", data, pos)[0]
                if noff in texlike and pos - 14 >= 0 and \
                        struct.unpack_from("<H", data, pos - 14)[0] == 0x0002:
                    idx = struct.unpack_from("<H", data, pos - 4)[0]
                    key = (noff, idx)
                    if key in seen:
                        continue
                    seen.add(key)
                    textures.append({"name": texlike[noff], "index": idx,
                                     "data_off": pos - 4, "node_graph": True})
            textures.sort(key=lambda t: t["index"])

    return {"strings": strings, "records": records,
            "textures": textures, "params": params}


def parse_matb(data):
    """Parse a MATB binary blob.

    Returns a list of property definitions::
        [{"name": str, "type": int, "kind": str}, ...]
    """
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


def parse_temp_entity_colors(json_path):
    """Read SColorRGB tint colours from a quickentity-format TEMP JSON file.

    Returns a list of dicts::
        {"sub_idx": int, "entity_id": str, "prop_id": str,
         "r": float, "g": float, "b": float, "location": str}

    Mirrors the OutfitEditor's _scan_sub_colors / _scan_outfit_colors.
    Returns [] on any parse error.
    """
    try:
        with open(json_path, "r", encoding="utf-8", errors="replace") as fh:
            data = _json.load(fh)
    except Exception:
        return []
    colors = []
    for idx, sub in enumerate(data.get("subEntities", [])):
        eid = str(sub.get("entityId", sub.get("entity_id", "")))
        for kind in ("propertyValues", "postInitPropertyValues"):
            for pv in sub.get(kind, []):
                val = pv.get("value", {})
                if val.get("$type") != "SColorRGB":
                    continue
                v = val.get("$val", {})
                try:
                    r = float(v.get("r", 0.0))
                    g = float(v.get("g", 0.0))
                    b = float(v.get("b", 0.0))
                except (TypeError, ValueError):
                    continue
                colors.append({
                    "sub_idx": idx,
                    "entity_id": eid,
                    "prop_id": str(pv.get("nPropertyID", pv.get("propertyID", ""))),
                    "r": r, "g": g, "b": b,
                    "location": kind,
                })
    return colors
