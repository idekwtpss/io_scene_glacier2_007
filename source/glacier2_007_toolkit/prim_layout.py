"""PRIM object pre-scale sentinel, shared by import and template-based export.

Legacy float4 fields: +72/+88/+104. Path-tracing: +76/+92/+108.
The inserted u32 also shifts the following cloth/declaration and skin trailer.
No vertex stream or pre-sentinel field layout changes are inferred here.
"""
import struct


def scale_shift(data, off):
    if off < 0 or off + 76 > len(data):
        raise ValueError("Truncated PRIM object")
    shift = 4 if struct.unpack_from("<I", data, off + 72)[0] == 0xFFFFFFFF else 0
    if off + 124 + shift > len(data):
        raise ValueError("Truncated PRIM scale/bias fields")
    return shift


def skip_scale_sentinel(reader):
    start = reader.tell()
    sentinel = reader.u32() == 0xFFFFFFFF
    if not sentinel:
        reader.seek(start)
    return sentinel
