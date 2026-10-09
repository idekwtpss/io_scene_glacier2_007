# Toolkit PRIM format update

Based on repository commit 34f5912557cc7a8d7893a98b5b9b631eea39d839,
glacier2_007_toolkit_clean_v2.zip. Packaged addon version: 2.9.2.

Both the active monolithic Blender addon and the modular prim/prim_builder
copies now detect u32 0xFFFFFFFF at object+72. The three float4 fields then
start at +76/+92/+108 instead of +72/+88/+104. Following declaration/skin
trailer reads and writes follow the same four-byte insertion; no pre-sentinel
or vertex-stream offsets change. Existing builder alignment remains unchanged.

Safe-edit export uses detected scale/bias offsets. Custom-mesh export preserves
the template's sentinel, reads its shifted trailer and writes the corresponding
layout. Old templates remain old format: this is not a file-generation converter.
Import a fresh current-game PRIM as the template for current-game exports.

Validation runs both code copies inside Blender 5.0 on the matched old/current
stock head pair (20 objects). Original imports match positions, UVs, normals,
indices, joints and weights. Safe-edit and unchanged custom export/reimport pass;
edited custom exports also produce identical geometry across the two generations.
See validation.json and test_sentinel_blender.py. Rebuilding may relocate blocks;
whole-file byte equality is not expected from the existing custom-mesh writer.

No material, cloth simulation, skinning algorithm or RPKG code was changed.
This fixes format handling, not unrelated custom-topology behavior. No in-game
render test or interactive UI-operator test is claimed. Existing installed addons
and game files were not changed. Install the supplied ZIP through Blender's
Preferences > Add-ons > Install from Disk, then restart Blender.
