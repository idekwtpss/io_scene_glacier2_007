"""
resourcelib_native.py - optional ctypes bridge to ResourceLib_HM3.dll.

Converts Glacier 2 resource bodies (here: TEMP / TBLU entity files) between their
packed BIN1 binary form and ResourceLib's RT-JSON. This is the SAFE engine for the
template / part editor: the game's own serializer round-trips the file, so a part
edit never has to be written "blind" (the failure mode that crashes the game).

Recovered exports (extern "C", ResourceLib public FFI ABI):
    ResourceConverter* HM3_GetConverterForResource(const char* type)
    ResourceGenerator* HM3_GetGeneratorForResource(const char* type)
    bool               HM3_IsResourceTypeSupported(const char* type)
    void               HM3_FreeJsonString(JsonString*)

    struct JsonString { const char* JsonData;   size_t StrSize; }
    struct ResourceMem{ const void* ResourceData; size_t DataSize; }
    struct ResourceConverter {
        JsonString* (*FromMemoryToJsonString)(const void* data, size_t size);
        bool        (*FromMemoryToJsonFile)(const void* data, size_t size, const char* path);
    }
    struct ResourceGenerator {
        ResourceMem* (*FromJsonStringToResourceMem)(const char* json, size_t size, bool compatible);
        bool         (*FromJsonFileToResourceFile)(const char* jsonPath, const char* outPath, bool compatible);
    }

SAFETY: a wrong ctypes signature can crash the host and Python can't catch that,
so this module is OPT-IN. Before any WRITE, callers must pass the unmodified
resource through `roundtrip_ok()` (BIN1 -> JSON -> BIN1 -> JSON, require the two
JSONs to match). If the DLL can't load, an export is missing, or the round-trip
disagrees, the bridge marks itself unavailable and the editor refuses to write.

No Blender dependency -> the logic is import-testable standalone (the actual DLL
calls only run on a 64-bit Windows Blender).
"""

import os
import ctypes
import json as _json

MAIN_DLL = "ResourceLib_HM3.dll"
DEP_DLLS = ("ResourceLib_HM2016.dll", "ResourceLib_HM2.dll")


class JsonString(ctypes.Structure):
    _fields_ = [("JsonData", ctypes.c_char_p), ("StrSize", ctypes.c_size_t)]


class ResourceMem(ctypes.Structure):
    _fields_ = [("ResourceData", ctypes.c_void_p), ("DataSize", ctypes.c_size_t)]


_FROM_MEM_TO_JSON = ctypes.CFUNCTYPE(
    ctypes.POINTER(JsonString), ctypes.c_void_p, ctypes.c_size_t)
_FROM_MEM_TO_JSON_FILE = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
_FROM_JSON_TO_MEM = ctypes.CFUNCTYPE(
    ctypes.POINTER(ResourceMem), ctypes.c_char_p, ctypes.c_size_t, ctypes.c_bool)
_FROM_JSON_FILE_TO_FILE = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool)


class ResourceConverter(ctypes.Structure):
    _fields_ = [("FromMemoryToJsonString", _FROM_MEM_TO_JSON),
                ("FromMemoryToJsonFile", _FROM_MEM_TO_JSON_FILE)]


class ResourceGenerator(ctypes.Structure):
    _fields_ = [("FromJsonStringToResourceMem", _FROM_JSON_TO_MEM),
                ("FromJsonFileToResourceFile", _FROM_JSON_FILE_TO_FILE)]


class ResourceLibError(Exception):
    pass


class ResourceLib:
    """Thin, defensive wrapper around one loaded ResourceLib_HM3.dll."""

    def __init__(self, dll_dir):
        self.dll_dir = dll_dir
        self.lib = None
        self.ok = False           # DLL loaded + bound
        self.error = ""
        self._cookie = None

    def available_files(self):
        found, missing = [], []
        for name in (MAIN_DLL,) + DEP_DLLS:
            (found if os.path.isfile(os.path.join(self.dll_dir, name))
             else missing).append(name)
        return found, missing

    # -- loading -----------------------------------------------------------
    def load(self):
        """Load + bind. Returns True on success else sets self.error. Never raises."""
        main = os.path.join(self.dll_dir, MAIN_DLL)
        if not os.path.isfile(main):
            self.error = "%s not found in %s" % (MAIN_DLL, self.dll_dir)
            return False
        if hasattr(os, "add_dll_directory"):
            try:
                self._cookie = os.add_dll_directory(self.dll_dir)
            except OSError:
                pass
        try:
            self.lib = ctypes.CDLL(main)
        except OSError as e:
            self.error = ("Could not load %s (needs a 64-bit Windows Blender): %s"
                          % (MAIN_DLL, e))
            return False
        try:
            self._bind()
        except AttributeError as e:
            self.error = "%s missing an expected export: %s" % (MAIN_DLL, e)
            self.lib = None
            return False
        self.ok = True
        return True

    def _bind(self):
        c, L = ctypes, self.lib
        L.HM3_GetConverterForResource.argtypes = [c.c_char_p]
        L.HM3_GetConverterForResource.restype = c.POINTER(ResourceConverter)
        L.HM3_GetGeneratorForResource.argtypes = [c.c_char_p]
        L.HM3_GetGeneratorForResource.restype = c.POINTER(ResourceGenerator)
        L.HM3_IsResourceTypeSupported.argtypes = [c.c_char_p]
        L.HM3_IsResourceTypeSupported.restype = c.c_bool
        L.HM3_FreeJsonString.argtypes = [c.POINTER(JsonString)]
        L.HM3_FreeJsonString.restype = None

    @staticmethod
    def _b(s):
        return s if isinstance(s, bytes) else (s or "").encode("utf-8", "replace")

    def supports(self, rtype):
        if not self.ok:
            return False
        try:
            return bool(self.lib.HM3_IsResourceTypeSupported(self._b(rtype)))
        except Exception:
            return False

    # -- BIN1 -> JSON ------------------------------------------------------
    def to_json(self, rtype, data):
        """Bytes of a packed resource body -> RT-JSON str. Raises on failure."""
        if not self.ok:
            raise ResourceLibError("ResourceLib not loaded")
        conv = self.lib.HM3_GetConverterForResource(self._b(rtype))
        if not conv:
            raise ResourceLibError("no converter for %r" % rtype)
        buf = (ctypes.c_char * len(data)).from_buffer_copy(data)
        js = conv.contents.FromMemoryToJsonString(
            ctypes.cast(buf, ctypes.c_void_p), len(data))
        if not js:
            raise ResourceLibError("FromMemoryToJsonString returned null (bad %s body?)" % rtype)
        try:
            raw = ctypes.string_at(js.contents.JsonData, js.contents.StrSize)
        finally:
            try:
                self.lib.HM3_FreeJsonString(js)
            except Exception:
                pass
        return raw.decode("utf-8", "replace")

    # -- JSON -> BIN1 ------------------------------------------------------
    def to_resource(self, rtype, json_str, compatible=True):
        """RT-JSON str -> packed resource bytes. Raises on failure.
        (ResourceMem is intentionally not freed here - copying out and skipping a
        possibly-mismatched free avoids any chance of a bad-free crash; the small
        per-save leak is bounded and harmless.)"""
        if not self.ok:
            raise ResourceLibError("ResourceLib not loaded")
        gen = self.lib.HM3_GetGeneratorForResource(self._b(rtype))
        if not gen:
            raise ResourceLibError("no generator for %r" % rtype)
        js = self._b(json_str)
        rm = gen.contents.FromJsonStringToResourceMem(js, len(js), bool(compatible))
        if not rm:
            raise ResourceLibError("FromJsonStringToResourceMem returned null (bad JSON?)")
        n = int(rm.contents.DataSize)
        if n <= 0 or n > (1 << 31):
            raise ResourceLibError("implausible generated size %d" % n)
        return ctypes.string_at(rm.contents.ResourceData, n)

    # -- safety gate -------------------------------------------------------
    def roundtrip_ok(self, rtype, data):
        """BIN1 -> JSON -> BIN1 -> JSON; require the two JSON docs to be equal.
        This proves the converter/generator agree on THIS exact file before we
        ever trust a real edit-and-write. Returns (ok, detail)."""
        try:
            j1 = self.to_json(rtype, data)
            back = self.to_resource(rtype, j1)
            j2 = self.to_json(rtype, back)
        except Exception as e:
            return False, "round-trip raised: %s" % e
        try:
            if _json.loads(j1) == _json.loads(j2):
                return True, "json stable across round-trip"
        except Exception:
            if j1 == j2:
                return True, "json byte-identical across round-trip"
        return False, "round-trip JSON differs - refusing to write"
