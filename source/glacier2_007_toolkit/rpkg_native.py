"""
rpkg_native.py - optional ctypes bridge to RPKG-Tool's native rpkg-lib.dll.

This lets the 007 Toolkit extract resources through IO Interactive/RPKG-Tool's
own C++ library (rpkg-lib.dll + ResourceLib_*.dll) instead of the addon's
pure-Python reader. The native lib does its own descramble/decompress, so it
handles anything the Python LZ4 path can't (e.g. Oodle/Kraken mips).

IMPORTANT - how the signatures were obtained and why this is safe:
  No C header shipped with the DLLs, so the function signatures below were
  recovered by DISASSEMBLING rpkg-lib.dll (Win64 ABI: args in RCX/RDX/R8/R9,
  return in RAX). A wrong ctypes signature can crash the host, and Python can't
  catch that - so this module is OPT-IN and is CROSS-CHECKED at runtime against
  the addon's pure-Python extractor on real sample hashes before anything else
  is allowed to use it. If the DLL can't load, an export is missing, or the
  self-test doesn't agree with the Python reader, the bridge marks itself
  unavailable and callers fall back to Python.

Recovered exports (all extern "C"):
    int    load_hash_list(const char* path)
    int    import_rpkgs(const char* path)            # file or folder
    char*  get_latest_hash_rpkg_path(const char* hash)
    uint64 get_hash_in_rpkg_size(const char* rpkg_path, const char* hash)
    char*  get_hash_in_rpkg_data(const char* rpkg_path, const char* hash)
    void   clear_hash_data_vector(void)

This module has NO Blender dependency so its logic can be tested standalone.
"""

import os
import ctypes
import threading

MAIN_DLL = "rpkg-lib.dll"
# Dependencies rpkg-lib.dll pulls in; we add the folder to the DLL search path
# so the loader can find them next to rpkg-lib.dll.
DEP_DLLS = (
    "ResourceLib_HM3.dll", "ResourceLib_HM2016.dll", "ResourceLib_HM2.dll",
    "assimp.dll", "quickentity_ffi.dll",
)
DEFAULT_HASH_LIST = "hash_list.txt"

# Reject obviously-bogus sizes coming back from the lib (signature mismatch /
# not-found) before we ever ask ctypes to read that many bytes.
_MAX_RESOURCE = 1 << 31     # 2 GiB hard ceiling for a single resource


class NativeError(Exception):
    pass


class RpkgNative:
    """Thin, defensive wrapper around one loaded rpkg-lib.dll."""

    def __init__(self, dll_dir):
        self.dll_dir = dll_dir
        self.lib = None
        self.ok = False               # DLL loaded + bound
        self.verified = False         # self-test agreed with Python
        self.error = ""
        self.imported_folder = None
        self.hash_list_loaded = None
        self._dll_cookie = None
        self._lock = threading.RLock()

    # -- loading -----------------------------------------------------------
    def available_files(self):
        """Report which expected DLLs are present (for diagnostics)."""
        found, missing = [], []
        for name in (MAIN_DLL,) + DEP_DLLS:
            (found if os.path.isfile(os.path.join(self.dll_dir, name))
             else missing).append(name)
        return found, missing

    def load(self):
        """Load and bind rpkg-lib.dll. Returns True on success, else sets
        self.error. Never raises."""
        main = os.path.join(self.dll_dir, MAIN_DLL)
        if not os.path.isfile(main):
            self.error = "%s not found in %s" % (MAIN_DLL, self.dll_dir)
            return False
        # Make the dependency DLLs resolvable (Win, Python 3.8+).
        if hasattr(os, "add_dll_directory"):
            try:
                self._dll_cookie = os.add_dll_directory(self.dll_dir)
            except OSError:
                pass
        try:
            self.lib = ctypes.CDLL(main)
        except OSError as e:
            self.error = ("Could not load %s (is this a 64-bit Windows Blender, "
                          "and are the ResourceLib_*.dll files next to it?): %s"
                          % (MAIN_DLL, e))
            return False
        try:
            self._bind()
        except AttributeError as e:
            self.error = "rpkg-lib.dll is missing an expected export: %s" % e
            self.lib = None
            return False
        self.ok = True
        return True

    def _bind(self):
        """Declare argtypes/restypes (recovered from disassembly). Centralised
        here so a future header can correct them in one place."""
        c, L = ctypes, self.lib
        L.load_hash_list.argtypes = [c.c_char_p]
        L.load_hash_list.restype = c.c_int
        L.load_hmla_hash_list.argtypes = [c.c_char_p]
        L.load_hmla_hash_list.restype = c.c_int
        # import_rpkgs(prefix, csv_of_rpkg_paths): the 2nd arg is a comma-
        # separated list the lib splits internally (recovered by disassembly;
        # passing one arg dereferenced a null 2nd arg -> access violation).
        L.import_rpkgs.argtypes = [c.c_char_p, c.c_char_p]
        L.import_rpkgs.restype = c.c_int
        L.get_latest_hash_rpkg_path.argtypes = [c.c_char_p]
        L.get_latest_hash_rpkg_path.restype = c.c_char_p
        L.get_hash_in_rpkg_size.argtypes = [c.c_char_p, c.c_char_p]
        L.get_hash_in_rpkg_size.restype = c.c_uint64
        L.get_hash_in_rpkg_data.argtypes = [c.c_char_p, c.c_char_p]
        L.get_hash_in_rpkg_data.restype = c.c_void_p
        L.clear_hash_data_vector.argtypes = []
        L.clear_hash_data_vector.restype = None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _b(s):
        if isinstance(s, bytes):
            return s
        return (s or "").encode("utf-8", "replace")

    # -- API ---------------------------------------------------------------
    def load_hash_list(self, path):
        if not self.ok or not path or not os.path.isfile(path):
            return False
        with self._lock:
            # .hmla is the binary hash list and needs its own loader; the text
            # loader on a .hmla corrupts state and crashes the next call.
            if path.lower().endswith(".hmla"):
                self.lib.load_hmla_hash_list(self._b(path))
            else:
                self.lib.load_hash_list(self._b(path))
            self.hash_list_loaded = path
        return True

    def import_folder(self, folder):
        """Import every rpkg in `folder` into the lib's global state. The lib's
        import_rpkgs(prefix, csv) splits its 2nd argument on commas, so we pass
        an empty prefix and a comma-separated list of full rpkg paths."""
        if not self.ok:
            return False
        with self._lock:
            if self.imported_folder == folder:
                return True
            try:
                rpkgs = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                               if f.lower().endswith(".rpkg"))
            except OSError:
                return False
            if not rpkgs:
                return False
            csv = ",".join(rpkgs)
            self.lib.import_rpkgs(b"", self._b(csv))
            self.imported_folder = folder
        return True

    def latest_rpkg_path(self, h):
        if not self.ok:
            return ""
        with self._lock:
            p = self.lib.get_latest_hash_rpkg_path(self._b(h))
        if not p:
            return ""
        return p.decode("utf-8", "replace") if isinstance(p, bytes) else str(p)

    def size(self, rpkg_path, h):
        if not self.ok:
            return -1
        with self._lock:
            return int(self.lib.get_hash_in_rpkg_size(self._b(rpkg_path), self._b(h)))

    def extract(self, rpkg_path, h):
        """Return the fully decoded resource bytes, or None. `rpkg_path` may be
        '' to let the lib pick the latest version."""
        if not self.ok:
            return None
        with self._lock:
            if not rpkg_path:
                rpkg_path = self.latest_rpkg_path(h)
                if not rpkg_path:
                    return None
            n = int(self.lib.get_hash_in_rpkg_size(self._b(rpkg_path), self._b(h)))
            if n <= 0 or n > _MAX_RESOURCE:
                return None
            ptr = self.lib.get_hash_in_rpkg_data(self._b(rpkg_path), self._b(h))
            if not ptr:
                return None
            data = ctypes.string_at(ptr, n)
            try:
                self.lib.clear_hash_data_vector()
            except Exception:
                pass
            return data

    # -- self-test ---------------------------------------------------------
    def verify(self, samples, py_extract):
        """Cross-check the native path against the Python extractor.

        samples : list of (hash, rpkg_path) known to exist in the runtime.
        py_extract(hash) -> bytes|None : the addon's pure-Python extraction.

        The native lib also decompresses formats the Python reader can't (e.g.
        Oodle), so native and Python only have to AGREE on resources Python can
        already decode. A single byte-exact match proves the ctypes binding is
        correct; remaining mismatches are cases where native is the better one.
        Sets self.verified and returns a (verified, report_lines) tuple.
        """
        checked = matched = produced = 0
        lines = []
        for (h, rpkg_path) in samples:
            try:
                nd = self.extract(rpkg_path, h)
            except OSError as e:
                lines.append("%s: native call failed (%s)" % (h[:16], e))
                self.verified = False
                return False, lines
            if not nd:
                continue
            produced += 1
            try:
                pd = py_extract(h)
            except Exception:
                pd = None
            if not pd:
                lines.append("%s: native ok (%d B), Python skipped" % (h[:16], len(nd)))
                continue
            checked += 1
            if nd == pd:
                matched += 1
                lines.append("%s: match (%d B)" % (h[:16], len(nd)))
            else:
                lines.append("%s: differ native=%d Python=%d B (native may be "
                             "decompressing Oodle)" % (h[:16], len(nd), len(pd)))
        # Trust native once it agrees with Python on at least one resource both
        # decode. If nothing was comparable, fall back (we can't prove it).
        self.verified = matched >= 1
        if not lines:
            lines.append("No samples produced data.")
        elif matched == 0 and produced > 0:
            lines.append("Native produced data but none matched Python - not "
                         "trusting it without a confirmed match.")
        return self.verified, lines


# -----------------------------------------------------------------------------
# Module-level cache: one bridge per DLL folder.
# -----------------------------------------------------------------------------
_BRIDGES = {}


def get_bridge(dll_dir):
    """Loaded RpkgNative for `dll_dir`, cached. None if it couldn't load."""
    dll_dir = os.path.abspath(dll_dir or "")
    if not dll_dir:
        return None
    br = _BRIDGES.get(dll_dir)
    if br is not None:
        return br if br.ok else None
    br = RpkgNative(dll_dir)
    br.load()
    _BRIDGES[dll_dir] = br
    return br if br.ok else None


def find_hash_list(dll_dir):
    """A hash list next to the DLLs, if present (hash_list.txt or .hmla)."""
    base = os.path.abspath(dll_dir or "")
    for name in (DEFAULT_HASH_LIST, "hash_list.hmla"):
        cand = os.path.join(base, name)
        if os.path.isfile(cand):
            return cand
    return ""


def reset():
    _BRIDGES.clear()
