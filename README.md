# 007 First Light Toolkit — Blender Addon

A complete modding toolkit for **007 First Light** (Glacier 2 / KNT engine) built into Blender. Import game models, rebuild their materials, swap textures, reshape meshes, and export everything back as game-ready files.

**Latest:** v1.14 | **Blender:** 4.2+ | **Status:** Stable

---

## Features

- **Import & Export** — `.prim` meshes and `.borg` skeletons with skin weights and shape keys
- **One-Click Materials** — Reconstruct the game's skin shader as a native Blender material
- **Native Texture Pipeline** — Decode/encode `.TEXT`/`.TEXD` (BC1–BC7) without external tools
- **Safe Mesh Editing** — Reshape geometry while keeping vertex weights intact
- **Material Tools** — Load, edit, swap, and repoint materials with full parameter control
- **Level of Detail** — Switch between LODs or jump to full detail
- **Search Folder on Export** — Robust texture pairing with nested file structures

---

## Install

1. Download `io_scene_glacier2_007.py` from [Releases](https://github.com/glacier-modding/io_scene_glacier2_007/releases)
2. **Edit → Preferences → Add-ons → Install from Disk**
3. Enable and **restart Blender**
4. Press **N**, click **007 Mesh Tools**

---

## How It Works

Import `.prim` models and `.borg` skeletons. Load their materials. Build real Blender shaders in one click. Decode and edit textures natively (BC1–BC7, no external tools). Reshape meshes safely. Export everything back to the game with correct hashes and metadata.

**Full documentation, step-by-step walkthroughs, and troubleshooting:** https://007toolsguide.netlify.app/

---

## Requirements

- **Blender 4.2+** (tested on 5.1)
- Your extracted `.prim`, `.borg`, `.MATI`, `.MATB`, `.TEXT`, `.TEXD` files
- RPKG-Tool to repack results

---

## Known Issues

See `KNOWN_ISSUES.md` for the full list. Key limitations:

- BC7 encoding produces slightly different output (visual difference imperceptible)
- Custom Mesh mode only works on unskinned geometry
- Shape keys don't animate in Material Preview
- Large textures (>8192px) may have issues

---

## Contributing

Contributions welcome:

1. Fork the repo
2. Create a feature branch
3. Test thoroughly in Blender 4.2+ and 5.1
4. Submit a pull request

---

## License

This addon is provided as-is for modding purposes. Reverse-engineered from the Glacier 2 engine.

---

## Credits

Built by the Glacier modding community.

---

## Links

- **RPKG-Tool:** https://github.com/glacier-modding/RPKG-Tool
- **Issues:** https://github.com/glacier-modding/io_scene_glacier2_007/issues


