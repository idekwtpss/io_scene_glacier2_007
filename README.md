# 007 First Light Toolkit — Blender Addon

A complete modding toolkit for **007 First Light** (Glacier 2 / KNT engine) built into Blender. Import game models, rebuild their materials, swap textures, reshape meshes, and export everything back as game-ready files — all without leaving Blender.

**Latest:** v1.14 | **Blender:** 4.2+ | **Status:** Stable

---

## Features

- **Import & Export** — `.prim` meshes and `.borg` skeletons with skin weights and shape keys
- **One-Click Materials** — Reconstruct the game's skin shader as a native Blender material
- **Native Texture Pipeline** — Decode/encode `.TEXT`/`.TEXD` (BC1–BC7) without external tools
- **Safe Mesh Editing** — Reshape geometry while keeping vertex weights intact
- **Material Tools** — Load, edit, swap, and repoint materials with full parameter control
- **Level of Detail** — Switch between LODs or jump to full detail
- **Search Folder on Export** — Robust texture pairing even with nested file structures
- **Full Documentation** — Step-by-step guides, visual references, and troubleshooting

---

## Quick Start

### Install

1. Download `io_scene_glacier2_007.py`
2. **Edit → Preferences → Add-ons → Install from Disk**
3. Enable the addon and **restart Blender**
4. Press **N** in the viewport, click **007 Mesh Tools**

### Example: Replace a Head Texture

```
1. Import the head .prim
2. Materials → Load From Imported Model
3. Texture Tools → set Work/Search folders → Decode Textures
4. Render Materials → Build Render Materials → Material Preview
5. Edit the basecolor image
6. Edit Material → mapTex_Basecolor → Custom Texture → your image
7. Export → Texture Replacement → Search Folder → your WorkingFile → Export
```

Done. Two files in `TEXT/<hash>/` and `TEXD/<hash>/` folders, ready to repack.

---

## What It Does

### Import
- Load `.prim` models (vertices, UVs, vertex colors, tangents)
- Load `.borg` skeletons with full bone hierarchy and skin weights
- Preserve shape keys and morph targets
- Auto-tag meshes with material ID and LOD info

### Materials
- Load materials from imported models, folders, or single files
- Resolve friendly names from IOI paths (hash lists, RPKG `.meta.json`)
- Edit textures (Hash / Custom Texture / TEXT Override)
- Edit parameters (roughness range, bump scale, translucency intensity, etc.)
- Swap material references wholesale
- Write changes back to game format

### Render Materials
- One-click reconstruction of the game's skin shader
- Correct node wiring: basecolor → Base Color, SRM split (green→roughness, red→specular, blue→metallic), normal map, translucency → subsurface
- Parameters become labelled nodes (Roughness Min/Max, Normal Strength, etc.)
- Assign to whole model or selected objects

### Textures
- Decode `.TEXT`/`.TEXD` to PNG/TGA (auto-detect format)
- Re-encode PNG/TGA back to `.TEXT`/`.TEXD` (BC1–BC7)
- Generate missing `.TEXT` from images when only mips are available
- Standalone `.TEXD` decode (no header needed)
- Texture pairing by metadata — always writes TEXT/TEXD under correct, distinct hashes

### Mesh
- Safe reshaping (move vertices, keep count the same)
- Recalculate normals after deformation
- Level of Detail switching (slider or buttons)
- Experimental custom topology (unskinned props only)

### Export
- Four modes: Full Model + Edits, Texture Replacement, Textures Only, Mesh Only
- Recompute normals, only export changed materials, generate missing textures
- Organize output into TYPE/<hash>/ folders (recommended)
- Write `.meta` and `.meta.json` with correct hashes
- Search Folder on export finds texture pairs in nested directories

---

## Panel Layout

All tools live in one sidebar tab (**N → 007 Mesh Tools**):

| Section | What |
|---|---|
| **Import / Export** | Bring models/skeletons in, export edits out |
| **Materials** | Load and list materials |
| **Render Materials** | Build shader in one click |
| **Edit Material** | Swap textures, tweak parameters |
| **Swap Whole Material** | Repoint material references |
| **Texture Tools** | Decode, encode, generate |
| **Level of Detail** | Switch LODs |

---

## File Format Support

| Type | Support |
|---|---|
| `.prim` | Import/Export (mesh + metadata) |
| `.borg` | Import (skeleton + weights) |
| `.MATI` | Read/Write (full material) |
| `.MATB` | Read (material schema) |
| `.TEXT` | Decode/Encode (header + small mips) |
| `.TEXD` | Decode/Encode (full-res mips) |
| `.meta` | Rebuild with correct hashes |

**Texture Formats:** BC1, BC3, BC4, BC5, BC7 (all native, no Oodle)

---

## Documentation

- **`007_Toolkit_StepByStep.md`** — Complete step-by-step guide (14 sections)
- **`007_Guide.html`** — Interactive HTML guide with scroll-spy nav
- **`007_Toolkit_Guide.html`** — Professional 007-styled visual reference
- **`KNOWN_ISSUES.md`** — Known issues, workarounds, roadmap

All included in the release.

---

## Requirements

- **Blender 4.2+** (tested on 5.1)
- Your extracted `.prim`, `.borg`, `.MATI`, `.MATB`, `.TEXT`, `.TEXD` files
- RPKG-Tool (or equivalent) to repack results back into `.rpkg`

---

## Installation

### From Source

```bash
git clone https://github.com/glacier-modding/io_scene_glacier2_007.git
cd io_scene_glacier2_007
```

Then in Blender:
1. **Edit → Preferences → Add-ons → Install from Disk**
2. Pick `io_scene_glacier2_007.py`
3. Restart Blender

### From Release

Download the latest `.py` file from [Releases](https://github.com/glacier-modding/io_scene_glacier2_007/releases) and follow the same install steps.

---

## Usage Examples

### Import a Head and Build Its Material

```python
# Manual scripting example (mostly for reference)
# The UI is the intended way to use the toolkit

import bpy
from io_scene_glacier2_007 import (
    IMPORT_OT_glacier2_prim,
    GLACIER_OT_load_materials,
    GLACIER_OT_build_materials
)

# Import
op = IMPORT_OT_glacier2_prim()
op.filepath = "/path/to/head.prim"
op.execute(bpy.context)

# Load materials
op = GLACIER_OT_load_materials()
bpy.context.scene.glacier_source_prim = "/path/to/head.prim"
op.execute(bpy.context)

# Build render materials
op = GLACIER_OT_build_materials()
op.apply_to = "MODEL"
op.execute(bpy.context)
```

**Recommended:** Use the UI panels. All features are click-based and don't require scripting.

### Decode a Texture

1. Set **Texture Tools → Work** to your output folder
2. Set **Texture Tools → Search** to your extracted files
3. Click **Decode Textures**
4. PNG/TGA files appear in the Work folder and load into Blender

### Export a Texture Replacement

1. Edit Material → set **mapTex_Basecolor** → **Custom Texture** → your image
2. Export → **Texture Replacement** → set **Search Folder** → Export
3. Output: `TEXT/<hash>/...TEXT` and `TEXD/<hash>/...TEXD` with metas
4. Repack with RPKG-Tool

---

## Known Issues

See `KNOWN_ISSUES.md` for a full list. Key limitations:

- **BC7 encoding** produces slightly different output (visual difference imperceptible)
- **Custom Mesh mode** only works on unskinned geometry (skinned meshes must keep vertex count)
- **Shape keys** don't animate in Material Preview (use Rendered mode)
- **Large textures** (>8192px) may have issues (rare, planned for v1.16)

---

## Architecture

```
io_scene_glacier2_007.py
├── Operators (Import/Export/Decode/Encode/etc)
├── UI Panels (sidebar sections)
├── Properties (scene & texture-slot data)
└── Codecs
    ├── Texture I/O (BC1–BC7, LZ4, mip handling)
    ├── Mesh builders (PRIM/BORG parsing)
    ├── Material tools (MATI/MATB)
    └── Metadata (parse/build .meta binary)
```

Everything is pure Python — no external dependencies beyond Blender's bpy API.

---

## Roadmap

### v1.15 (planned)
- [ ] Material parameters update render material in real-time
- [ ] Metadata reconstruction for new vertex colors
- [ ] Better error messages for UNC paths (Windows)
- [ ] Validate Fill Hashes input

### v1.16
- [ ] Support for textures >8192px
- [ ] Improved LOD detection
- [ ] Real-time normal map preview in viewport

### v2.0
- [ ] Weight painting tools integrated into the panel
- [ ] Multi-material mesh splitting
- [ ] Batch texture conversion
- [ ] Support for other Glacier 2 games (if applicable)

---

## Contributing

Contributions welcome. To contribute:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Test thoroughly in Blender 4.2+ and 5.1
4. Update `KNOWN_ISSUES.md` if relevant
5. Submit a pull request with a clear description

**Before major changes**, open an issue to discuss the direction.

---

## Troubleshooting

### Old Build Still Loaded
If the panel header doesn't show **v1.14 ✓**, the old addon is cached:
1. Remove it in **Add-ons**
2. Restart Blender
3. Install the new file
4. Restart again

### TEXD Not Found on Export
Set the export **Search Folder** to your texture directory. The exporter searches recursively for `.TEXT` metas to read the paired `.TEXD` hash.

### Material Parameters Look Wrong
After editing a material in the slot, delete the render material and rebuild it (**Build Render Materials** again).

### Mesh Exploded In-Game
If you changed the vertex count, that's the issue. Use safe reshaping only (move vertices, keep the count). `Custom Mesh` is experimental and unskinned only.

See `KNOWN_ISSUES.md` and the included guides for more solutions.

---

## License

This addon is provided as-is for modding purposes. Reverse-engineered from the Glacier 2 engine. Use at your own risk.

---

## Credits

Built by the Glacier modding community. Thanks to:
- RPKG-Tool team (for the extraction format)
- The Hitman/007 modding community for testing and feedback
- Blender team for the amazing API

---

## Links

- **Nexus Mods:** [007 First Light Toolkit](https://nexusmods.com)
- **RPKG-Tool:** https://github.com/glacier-modding/RPKG-Tool
- **Glacier Modding Discord:** [Join](https://discord.gg/glacier)
- **Issues:** https://github.com/glacier-modding/io_scene_glacier2_007/issues

---

**007 First Light Toolkit v1.14** — The complete Blender modding toolkit for 007 First Light.
