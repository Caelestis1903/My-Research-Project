"""Exact code identity stability checks; never execute archived function bodies."""
from pathlib import Path
import ast,hashlib,importlib._bootstrap_external,json,marshal,opcode,types
from experiments.mdmt_rcf_20261026.raw_reconstruction_observer import semantic_code_fingerprint,callable_fingerprint,raw_features,raw_source,FINGERPRINT_SCHEME,_constant_semantics
root=Path('experiments/mdmt_rcf_20261026').resolve()
oldpath=Path('research_reports/2026-10-26_v2/evidence/pre_raw_reconstruction_observer/baselines.py')
oldbytes=oldpath.read_bytes();compiled=compile(oldbytes,str(root/'baselines.py'),'exec')
roundtrip=marshal.loads(marshal.dumps(compiled))
pyc=importlib._bootstrap_external._code_to_timestamp_pyc(compiled,mtime=123456,source_size=len(oldbytes))
pyc_code=marshal.loads(pyc[16:])
get=lambda module:{c.co_name:c for c in module.co_consts if isinstance(c,types.CodeType) and c.co_name in ('raw_features','_raw_source')}
original=get(compiled);module_rt=get(roundtrip);pyc_functions=get(pyc_code)
records={};old_semantic={};old_marshal={}
for name,code in original.items():
    function=types.FunctionType(code,{'__builtins__':__builtins__},name)
    variants={'compiled':code,'code_marshal_roundtrip':marshal.loads(marshal.dumps(code)),
              'module_marshal_roundtrip':module_rt[name],'pyc_shape':pyc_functions[name],
              'function_type':function.__code__,
              'relocated_metadata':code.replace(co_filename='/metadata/only/different.py',co_firstlineno=code.co_firstlineno+701,co_linetable=b'')}
    fingerprints={key:semantic_code_fingerprint(value) for key,value in variants.items()}
    assert len(set(fingerprints.values()))==1
    expected=fingerprints['compiled'];negative={}
    changed=bytearray(code.co_code);changed[0]=opcode.opmap['NOP'] if changed[0]!=opcode.opmap['NOP'] else opcode.opmap['RESUME']
    negative['different_opcode']=code.replace(co_code=bytes(changed))
    constants=list(code.co_consts);index=next(i for i,x in enumerate(constants) if isinstance(x,str));constants[index]+=' changed-semantic-constant'
    negative['different_constant']=code.replace(co_consts=tuple(constants))
    names=list(code.co_names);names[0]+='_changed_global'
    negative['different_global_name']=code.replace(co_names=tuple(names))
    assert all(semantic_code_fingerprint(value)!=expected for value in negative.values())
    assert callable_fingerprint(function)==expected
    old_semantic[name]=expected;old_marshal[name]=hashlib.sha256(marshal.dumps(code)).hexdigest()
    records[name]=dict(semantic_fingerprints=fingerprints,all_equivalent=True,
        marshal_fingerprints={key:hashlib.sha256(marshal.dumps(value)).hexdigest() for key,value in variants.items()},
        all_instruction_constant_name_changes_rejected=True,negative_cases=list(negative))
assert _constant_semantics(frozenset(('beta','alpha')))==_constant_semantics(frozenset(('alpha','beta')))
assert _constant_semantics(0.)!=_constant_semantics(-0.)
assert _constant_semantics(True)!=_constant_semantics(1)
asts={}
for name in ('baselines.py','readers.py'):
    old=ast.parse((oldpath.parent/name).read_text());new=ast.parse((root/name).read_text())
    defs=lambda tree:{a.name:ast.dump(a,include_attributes=False) for a in tree.body if isinstance(a,(ast.FunctionDef,ast.ClassDef))}
    aa,bb=defs(old),defs(new);changed=[k for k in aa if aa[k]!=bb.get(k)]
    assert changed==(['_raw_source','raw_features'] if name=='baselines.py' else [])
    asts[name]=dict(changed_original_definitions=changed,all_other_original_definitions_unchanged=True)
new_functions={'raw_features':raw_features,'_raw_source':raw_source}
result=dict(status='BASIC_CHECKS_COMPLETE',fingerprint_scheme=FINGERPRINT_SCHEME,archived_function_bodies_executed=False,
    formal_training=False,resident_service_contacted=False,records=records,metadata_not_identity=True,
    recursive_typed_constants=True,float_sign_bits_preserved=True,frozenset_canonical_order=True,
    original_baseline_source_sha256=hashlib.sha256(oldbytes).hexdigest(),
    old_callable_fingerprints=old_semantic,old_marshal_fingerprints=old_marshal,
    new_callable_fingerprints={k:callable_fingerprint(v) for k,v in new_functions.items()},
    new_marshal_fingerprints={k:hashlib.sha256(marshal.dumps(v.__code__)).hexdigest() for k,v in new_functions.items()},
    source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('raw_reconstruction_observer.py','baselines.py','readers.py')},
    original_definition_AST=asts,original_tolerance=1e-6,model_files_written=0)
Path('research_reports/2026-10-26_v2/OBSERVER_SEMANTIC_FINGERPRINT_BASIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
