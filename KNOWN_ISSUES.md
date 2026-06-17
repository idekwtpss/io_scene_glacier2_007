# Known Issues — 007 First Light Toolkit & Hitman Converter

**Version:** 1.14 (007 Toolkit), 1.0 (Hitman Converter)  
**Last updated:** June 2026  
**Blender versions:** 4.2 — 5.1

---

## 007 First Light Toolkit

### Current Issues

#### 1. **BC7 encoding produces slightly different output on re-encode**
- **Severity:** Low
- **Description:** BC7-encoded textures do not byte-match the original when re-encoded, even with identical settings. The visual difference is imperceptible, but the files differ at the binary level.
- **Cause:** BC7 mode selection and partition heuristics are not identical to the game engine's encoder.
- **Workaround:** Use BC1 (color only) or BC3 (color + alpha) for critical textures. BC7 is safe for eye textures and other non-critical surfaces.
- **Status:** Low priority — does not affect gameplay.

#### 2. **Shape keys don't transfer through Material Preview**
- **Severity:** Low
- **Description:** If a model has shape keys (morph targets), switching to Material Preview mode doesn't show them moving. The geometry is correct, but the shape keys don't animate in that viewport shading.
- **Cause:** Blender's Material Preview mode has limited support for Blender's own geometry node driven shapes.
- **Workaround:** Switch to Rendered mode (Shift+Z) to see shape keys with materials, or use Solid mode + the UV Editor to inspect textures separately.
- **Status:** By design (Blender limitation). Not a toolkit issue.

#### 3. **Vertex color channels lost on re-export if not originally present**
- **Severity:** Medium (only affects models with vertex colors)
- **Description:** If you paint vertex colors onto a model that didn't have them, they are written to the `.prim` but the metadata doesn't reflect the new channels, so the game may ignore them.
- **Cause:** The metadata is inherited from the original mesh; new vertex data isn't reflected in the header.
- **Workaround:** If you add vertex colors, you must manually edit the `.prim` metadata or re-derive it from the original.
- **Status:** Requires metadata reconstruction — planned for v1.15.

#### 4. **Experimental: Custom Mesh mode only works on unskinned geometry**
- **Severity:** High (for skinned models)
- **Description:** Changing vertex counts on a mesh with skin weights causes the exported model to explode or become deformed in-game.
- **Cause:** The bone weights array size is tied to the vertex count; adding/removing vertices breaks the correspondence.
- **Workaround:** Keep **Custom Mesh off** and only move existing vertices. For character bodies and heads, this is the safe and only supported approach.
- **Status:** Requires full weight recalculation — v1.15 or later.

#### 5. **Material parameters don't update the render material if changed mid-session**
- **Severity:** Low
- **Description:** If you edit a material parameter (roughness min/max, bump scale) after building the render material, the Principled shader doesn't update to reflect the new value.
- **Cause:** The material is built once; parameter changes don't trigger a rebuild.
- **Workaround:** After changing parameters, delete the render material and rebuild it (Build Render Materials again).
- **Status:** Planned for v1.15 — will add an "Update Render Material" button.

#### 6. **TEXT/TEXD pairing fails if both the .TEXT.meta (dot-style) and _TEXT.meta (underscore-style) exist in the same folder**
- **Severity:** Low
- **Description:** If a folder contains both naming styles for the same texture, the pairing reads only one and may choose the wrong file.
- **Cause:** The metadata finder doesn't disambiguate when both styles are present.
- **Workaround:** Use one naming convention per folder — either dot-style (`01673C49.TEXT.meta`) or underscore-style (`01673C49_TEXT.meta`), not both.
- **Status:** Will add a warning in v1.15 when this is detected.

#### 7. **Decode fails on very large textures (>8192 pixels on largest axis)**
- **Severity:** Low (rare)
- **Description:** Textures larger than 8192×8192 may fail to decode or decode with artifacts.
- **Cause:** BCn decompression assumes 16-bit mip offsets; larger textures overflow.
- **Workaround:** Downscale the texture before encoding, or split it into multiple smaller textures.
- **Status:** Rare edge case. Proper support planned for v1.16.

#### 8. **Export dialog's Search Folder doesn't work on Windows UNC paths (network shares)**
- **Severity:** Medium (Windows only)
- **Description:** Pointing the export Search Folder at a network share (`\\server\share\...`) doesn't search correctly.
- **Cause:** Path normalization strips the UNC prefix.
- **Workaround:** Copy your textures to a local drive (C:\, D:\, etc.) and point Search Folder there.
- **Status:** Will fix path handling in v1.15.

#### 9. **LOD grouping by material ID fails if a model reuses material IDs across different meshes with different geometry**
- **Severity:** Low
- **Description:** If two separate meshes use the same material ID, the LOD slider may show them together even if they're different LOD levels of different objects.
- **Cause:** LOD grouping uses material ID as the key; it doesn't account for mesh identity.
- **Workaround:** Manually hide/show meshes in the outliner instead of using the LOD slider in that case.
- **Status:** Will improve LOD detection in v1.15.

#### 10. **Fill Hashes button crashes if a slot's old_hash is invalid**
- **Severity:** Low
- **Description:** If a texture slot's `old_hash` field contains invalid characters (non-hex), clicking Fill Hashes causes Blender to report an error.
- **Cause:** The hash parser doesn't validate before conversion.
- **Workaround:** Clear the `old_hash` field and re-fill it manually or via scanning.
- **Status:** Will add validation in v1.15.

---

## Hitman to 007 Converter

### Current Issues

#### 1. **Bone remapping doesn't handle custom or unnamed bones**
- **Severity:** Medium
- **Description:** If a Hitman rig uses non-standard bone names or has extra bones not in the standard skeleton, the converter doesn't map them to 007, and weights on those bones are lost.
- **Cause:** The converter uses a hard-coded bone-name map for the standard Hitman skeleton.
- **Workaround:** Manually delete extra bones from the Hitman rig before converting, or rename them to match standard names.
- **Status:** Will add a custom bone-map UI in v1.1.

#### 2. **Batch conversion stops on first error**
- **Severity:** Medium
- **Description:** If one model in a batch fails to convert, the entire batch stops; remaining models aren't processed.
- **Cause:** No error recovery in the batch loop.
- **Workaround:** Convert models one at a time, or fix the failing model and restart the batch.
- **Status:** Will add error-skipping in v1.1.

#### 3. **Shape keys are not remapped to the new bone structure**
- **Severity:** Medium (if original model has shape keys)
- **Description:** If a Hitman model includes shape keys or morph targets, they don't transfer correctly — their bone references point to old Hitman bones.
- **Cause:** Shape key drivers aren't recalculated during conversion.
- **Workaround:** Delete shape keys before converting, or manually rebuild them in the 007 toolkit.
- **Status:** Planned for v1.1.

#### 4. **Material references don't carry over slot assignments**
- **Severity:** High
- **Description:** The converter rebuilds material slots but doesn't assign materials to them, so converted models import with empty slots.
- **Cause:** Material assignment is a separate step; the converter only handles mesh geometry.
- **Workaround:** Import the converted model into the 007 toolkit and manually load and assign materials.
- **Status:** Expected behavior — materials must be loaded separately. Documentation improved in latest guide.

#### 5. **Normals are recalculated but may not match the original lighting**
- **Severity:** Low
- **Description:** After conversion, some models may appear slightly darker or lighter due to normal recalculation.
- **Cause:** The recalculation uses Blender's smooth-shading algorithm, which may differ slightly from the original bake.
- **Workaround:** Open the model in the 007 toolkit, enable Material Preview, and adjust the render material's roughness/specular if needed.
- **Status:** Acceptable — games are forgiving of minor normal differences.

#### 6. **Converter doesn't validate bone weight sum (may not equal 1.0 per vertex)**
- **Severity:** Low
- **Description:** After conversion, some vertices may have weights that don't sum to 1.0, causing subtle deformation.
- **Cause:** Weight recalculation can accumulate rounding errors.
- **Workaround:** Run Blender's built-in weight normalization (Weights → Normalize All) after import.
- **Status:** Will add automatic normalization in v1.1.

#### 7. **Very large character models (100k+ vertices) may run out of memory during conversion**
- **Severity:** Low (rare)
- **Description:** Batch processing of multiple large models can cause Blender to run out of memory.
- **Cause:** Intermediate conversion data isn't freed between batch items.
- **Workaround:** Convert large models one at a time, or split them into LODs before converting.
- **Status:** Will optimize memory handling in v1.1.

#### 8. **Converter assumes 007 skeleton is already present in the .blend**
- **Severity:** Medium
- **Description:** If you haven't imported a 007 rig, the converter can't auto-detect the target rig and requires manual selection.
- **Cause:** The converter looks for armatures named "Armature" or similar; 007 rigs may have custom names.
- **Workaround:** Either import a 007 model first (to bring in the rig), or manually select the target rig in the converter panel.
- **Status:** Will add a "Load 007 Reference Rig" button in v1.1.

---

## Fixed Issues (v1.14)

These were fixed in the latest version:

- ✓ TEXT and TEXD exported under the same hash (v1.13)
- ✓ Old addon cache preventing new build from loading (added version banner in v1.14)
- ✓ Missing `.TEXD hash` field for manual override (added in v1.14)
- ✓ Export Search Folder not finding nested textures (v1.13)
- ✓ Collision warning on every texture swap (v1.14 — now only warns if truly unresolvable)

---

## Planned Fixes (v1.15)

- [ ] Parameter updates trigger render material rebuild
- [ ] Metadata reconstruction for models with new vertex colors
- [ ] Bone weight normalization in Hitman converter
- [ ] Error-skipping in batch conversion
- [ ] Shape key driver remapping in converter
- [ ] Dot/underscore metadata naming detection
- [ ] UNC path support on Windows

---

## Planned Features (v1.16+)

- [ ] Support for >8192px textures
- [ ] Custom bone-map UI for non-standard Hitman rigs
- [ ] Automatic reference rig loading in converter
- [ ] Weight painting tools in the 007 toolkit
- [ ] Real-time normal map preview
- [ ] Multi-material mesh splitting tools

---

## Reporting Issues

If you encounter a problem not listed here:

1. Check the **007 Toolkit Guide** (included HTML) — it covers common troubleshooting
2. Verify **v1.14** is actually loaded (check the panel header)
3. Test with a **small, simple model** first (to isolate issues)
4. Note your **Blender version**, **OS**, and **exact steps** to reproduce

If the issue persists, post on Nexus Mods or report via the linked GitHub repository with:
- Blender version
- A minimal `.blend` file that reproduces the issue
- The exact error message or description of the wrong behavior
- Steps to reproduce

---

**007 First Light Toolkit v1.14** — Hitman to 007 Converter v1.0
