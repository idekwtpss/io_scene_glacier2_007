"""
Binary stream reader — little-endian, read-only.
Used by all Glacier 2 / 007 First Light format parsers.
"""
import struct


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
