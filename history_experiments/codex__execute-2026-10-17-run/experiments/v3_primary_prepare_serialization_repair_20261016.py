import pathlib,json,hashlib,time,ast
r=pathlib.Path('/home/xzf/Code/phase0-3-20261016-v3-git');o=r/'research_plans/2026-10-16_v3/execution_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
source=r/'experiments/v3_test_summary_20261016.py';freeze=o/'test_readout_frozen.json'
assert sha(source)=='149430001d72e57f5de6e745afb7614462b7c9c5f6a8c5b19482fef3ab58c2aa'
assert sha(freeze)=='91739132c2dd825b77dff5e0b88281621c6ce51e68eaf56219be0d4f2091c0e5'
state=json.loads((o/'test_all14_summary_CPU_manifest.status.json').read_text());assert len(state['tasks'])==14 and all(x['exit_code']==1 for x in state['tasks'].values())
spec=dict(status='DESIGNED',scope='serialization-only engineering correction after all14 frozen summary outputs failed before completion',hypothesis_changes=False,design='Run the unchanged frozen summary module through a separate wrapper. Recursively convert numpy scalar leaves with .item() only at base.dump. No changed values, metrics, selection, statistical function, manifests or frozen sources. Move partial output directories intact to an explicit failed-attempt subdirectory, rerun same14 manifests and original output paths. Compare all preexisting CSV and JSON file bytes after rerun; then independently reconstruct all results using frozen v2 verifier.',acceptance='all14 real child exit0; old partial CSV/JSON files byte-identical; original source/freeze SHA unchanged; frozen numeric reconstruction PASS',independent_review=False,time=time.time())
dump(o/'test_summary_serialization_repair_design.json',spec)
p=r/'experiments/v3_run_frozen_summary_serializable_20261016.py'
p.write_text('''"""Execute unchanged frozen summary with numpy-scalar JSON output compatibility."""\nimport hashlib\nimport json\nfrom pathlib import Path\nimport time\nimport numpy as np\nimport v3_test_summary_20261016 as frozen\n\ndef normalize(value):\n    if isinstance(value, np.generic):\n        return value.item()\n    if isinstance(value, dict):\n        return {normalize(k): normalize(v) for k,v in value.items()}\n    if isinstance(value, (list, tuple)):\n        return [normalize(v) for v in value]\n    return value\n\ndef main():\n    original_dump = frozen.base.dump\n    source = Path(frozen.__file__)\n    assert frozen.base.sha(source) == '149430001d72e57f5de6e745afb7614462b7c9c5f6a8c5b19482fef3ab58c2aa'\n    frozen.base.dump = lambda path, value: original_dump(path, normalize(value))\n    frozen.main()\n    import sys\n    output = Path(sys.argv[sys.argv.index('--out')+1])\n    original_dump(output/'serialization_wrapper_provenance.json', dict(status='COMPLETE', wrapper_source_sha=frozen.base.sha(__file__), unchanged_frozen_summary_source_sha=frozen.base.sha(source), scope='numpy scalar to native scalar conversion only at JSON output boundary', time=time.time()))\n\nif __name__ == '__main__':\n    main()\n''')
ast.parse(p.read_text())
# Minimal type/value check independent of research metrics.
import importlib.util,sys
sys.path.insert(0,str(r/'experiments'))
s=importlib.util.spec_from_file_location('serialization_fix',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
import numpy as np
x={'int':np.int64(2),'float':np.float64(.25),'nested':[np.bool_(True),{'k':np.int32(3)}]};y=m.normalize(x)
assert json.loads(json.dumps(y,allow_nan=False))=={'int':2,'float':.25,'nested':[True,{'k':3}]}
old=json.loads((o/'test_all14_summary_CPU_manifest.json').read_text());new=json.loads(json.dumps(old));preserved=[]
failroot=o/'analysis/failed_test_summary_numpy_int64_attempt';failroot.mkdir(exist_ok=False)
for t in new['tasks']:
 out=pathlib.Path(t['command'][t['command'].index('--out')+1]);assert out.is_dir() and not (out/'receipt.json').exists()
 files={q.name:sha(q) for q in out.iterdir() if q.is_file()};dest=failroot/out.name;out.rename(dest);preserved.append(dict(output=str(out),failed_copy=str(dest),files=files))
 t['name']+='_serialization_retry';t['command'][1]=str(p.relative_to(r));t['log']=str(o/'logs'/f"{t['name']}.log")
dump(o/'test_summary_serialization_failed_partial_index.json',dict(records=preserved))
mp=o/'test_all14_summary_serialization_retry_CPU_manifest.json';dump(mp,new)
dump(o/'test_summary_serialization_repair_implementation_report.json',dict(status='IMPLEMENTED',source=str(p),source_sha=sha(p),original_source_sha=sha(source),freeze_sha=sha(freeze),basic_check='syntax + recursive int64/float64/bool/int32 type/value exact JSON roundtrip PASS',preserved_partial_outputs=14))
dump(o/'test_summary_serialization_repair_primary_self_review.json',dict(status='PASS',independent_review=False,review='Separate wrapper only normalizes scalar leaves at serialization. Unchanged frozen source, all14 original frozen manifest paths and outputs retained. Failed outputs preserved. No result-driven choices or scientific/statistical code changes.',source_sha=sha(p),manifest_sha=sha(mp),freeze_sha=sha(freeze),time=time.time()))
print('PASS serialization repair design/implementation/self-review; original frozen source preserved; ready',mp)
