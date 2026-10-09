"""
BORG — 007 First Light BoneRig / skeleton parser.
Read-only; parses bones and bind poses from a .BORG file.
"""
import os

from .reader import Reader


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
