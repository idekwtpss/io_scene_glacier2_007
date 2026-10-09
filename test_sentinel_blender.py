"""Run with Blender --background --factory-startup --python this_file."""
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import glacier2_007_toolkit as addon
from glacier2_007_toolkit import prim,prim_builder

WORK=ROOT.parent
paths=[WORK/'007fl_mod_patcher/evidence/01C75259EEAD9C5B.old.PRIM',
       WORK/'007-EntityViewer/evidence/01C75259EEAD9C5B.updated.PRIM']

def snapshot(parsed):
    return [(m.indices,[(v.position,v.normal,v.uv,v.joint,v.weight) for v in m.vertexBuffer.vertices])
            for m in parsed.header.object_table]

results=[]
for module,builder in ((addon,addon),(prim,prim_builder)):
    edited_snapshots=[]
    parsed=[module.read_prim_bytes(p.read_bytes()) for p in paths]
    assert snapshot(parsed[0])==snapshot(parsed[1]),module.__name__
    for path,model in zip(paths,parsed):
        data=path.read_bytes()
        _,weighted,metas=module.walk_prim_objects(data)
        new=path==paths[1]
        parts=[]
        for meta,m in zip(metas,model.header.object_table):
            vs=m.vertexBuffer.vertices
            assert meta['pos_scale_off']==meta['off']+72+4*new
            parts.append(dict(orig_off=meta['off'],sub_type=meta['sub_type'],changed=False,from_original=True,
                positions=[v.position[:3] for v in vs],normals=[v.normal[:3] for v in vs],
                uvs=[v.uv[0] for v in vs],colors=[v.color for v in vs],
                joints=[v.joint for v in vs],weights=[v.weight for v in vs],indices=m.indices))
        # Existing safe-edit path must retain the sentinel and skin weights.
        edited=bytearray(data)
        m=metas[0]
        module.patch_object(edited,m,parts[0]['positions'],None)
        assert edited[m['off']+72:m['off']+76]==data[m['off']+72:m['off']+76] if new else True
        reimport=module.read_prim_bytes(edited)
        assert [v.joint for v in reimport.header.object_table[0].vertexBuffer.vertices]==parts[0]['joints']
        # Existing custom topology writer, with unchanged objects.
        output=builder.build_custom_prim(data,parts,weighted)
        rebuilt=module.read_prim_bytes(output)
        assert len(rebuilt.header.object_table)==len(parts)
        _,_,out_metas=module.walk_prim_objects(output)
        for i,(meta,mesh) in enumerate(zip(out_metas,rebuilt.header.object_table)):
            assert (struct.unpack_from('<I',output,meta['off']+72)[0]==0xffffffff)==new
            assert mesh.indices==model.header.object_table[i].indices
            error=max(abs(a.position[k]-b.position[k]) for a,b in zip(mesh.vertexBuffer.vertices,model.header.object_table[i].vertexBuffer.vertices) for k in range(3))
            assert error<1e-5,(module.__name__,str(path),i,error)
            assert [tuple(v.joint) for v in mesh.vertexBuffer.vertices]==[tuple(v) for v in parts[i]['joints']],(module.__name__,new,i,next((k,v.joint,parts[i]['joints'][k]) for k,v in enumerate(mesh.vertexBuffer.vertices) if tuple(v.joint)!=tuple(parts[i]['joints'][k])))
            assert [tuple(v.weight) for v in mesh.vertexBuffer.vertices]==[tuple(v) for v in parts[i]['weights']]
        results.append(dict(module=module.__name__,input=str(path),sha256=hashlib.sha256(data).hexdigest(),
                            objects=len(parts),safe_export_reimport=True,custom_export_reimport=True,
                            vertices_indices_weights_preserved=True))
        # Exercise the actual edited-mesh branch, not only untouched passthrough.
        for part in parts:
            part['from_original']=False
            part['changed']=True
            part['positions']=[(p[0]+0.001,p[1],p[2]) for p in part['positions']]
        edited_output=builder.build_custom_prim(data,parts,weighted)
        edited_model=module.read_prim_bytes(edited_output)
        edited_snapshots.append(snapshot(edited_model))
        results[-1]['edited_custom_export_reimport']=True
    assert edited_snapshots[0]==edited_snapshots[1],module.__name__+' edited layouts differ'
(ROOT/'validation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('SENTINEL_VALIDATION_PASS',json.dumps(results))
