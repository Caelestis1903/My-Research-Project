"""Verified archival source activation and common population injection."""
from pathlib import Path
import hashlib
import json
import sys
HERE=Path(__file__).resolve().parent
ARCHIVE=HERE/'source_archive'
CORE=HERE.parent/'natural_ml_reassessment'
ART=Path('/home_new/xzf_natural_movielens_reassessment')
SEEDS=(2022,2023,2024,2025)
# Freeze only dependencies of this batch; other studies can continue implementation.
V2_WRAPPERS=('__init__.py','bootstrap.py','v2.py','manifest.py')
CORE_WRAPPERS=('__init__.py','bootstrap.py','data.py')
CORE_PRODUCERS=('stage_F0_data','stage_S1_data','stage_S1_protocol')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');temp.replace(path)

def dependency_inventory(label='v2'):
    """Return verified records for executed archives and exact wrapper dependencies."""
    result={}
    inventory=json.loads((ARCHIVE/'inventory.json').read_text())
    for name,record in inventory.items():
        if not name.startswith(label+'/'):continue
        actual=sha(ARCHIVE/name)
        if actual!=record['sha256']:raise RuntimeError('archived producer drift: '+name)
        result['extensions/source_archive/'+name]=dict(record,actual_path=str(ARCHIVE/name))
    wrappers=V2_WRAPPERS if label=='v2' else ('__init__.py','bootstrap.py')
    for name in wrappers:
        path=HERE/name
        result['extensions/'+name]=dict(source_path=str(path),actual_path=str(path),sha256=sha(path),kind='study_wrapper')
    core_inventory=json.loads((CORE/'source_archive/inventory.json').read_text())
    for module in CORE_PRODUCERS:
        name='core/experiments/'+module+'.py';record=core_inventory[name]
        path=CORE/'source_archive'/name;actual=sha(path)
        if actual!=record['sha256'] or actual!=record['git_blob_sha256']:
            raise RuntimeError('core population archive drift: '+name)
        if not record['source_commit'] or not record['source_path']:
            raise RuntimeError('core population provenance missing: '+name)
        result['core/source_archive/'+name]=dict(record,actual_path=str(path))
    for name in CORE_WRAPPERS:
        path=CORE/name
        result['core/'+name]=dict(source_path=str(path),actual_path=str(path),sha256=sha(path),kind='population_wrapper')
    return result

def sources(label=None):
    # Pin the selected inventory records rather than unrelated studies' inventory entries.
    records=dependency_inventory(label or 'v2')
    result={name:record['sha256'] for name,record in records.items()}
    result['dependency_inventory_sha256']=hashlib.sha256(
        json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result

def actual_module_paths(label='v2',require_population=True):
    """Assert loaded modules resolve to the archives whose hashes are frozen."""
    expected={module:CORE/'source_archive/core/experiments'/(module+'.py') for module in CORE_PRODUCERS}
    expected.update({'experiments.natural_ml_reassessment.'+name[:-3]:CORE/name for name in CORE_WRAPPERS if name!='__init__.py'})
    expected['experiments.natural_ml_reassessment']=CORE/'__init__.py'
    if require_population:
        for name,path in expected.items():
            module=sys.modules.get(name)
            if module is None or Path(module.__file__).resolve()!=path.resolve():
                raise RuntimeError('unexpected executed population module: '+name)
    roots=(ARCHIVE/label,CORE)
    observed={}
    for name,module in sorted(sys.modules.copy().items()):
        file=getattr(module,'__file__',None)
        if file is None:continue
        path=Path(file).resolve()
        if any(path.is_relative_to(root.resolve()) for root in roots):
            observed[name]=dict(path=str(path),sha256=sha(path))
    return observed

def activate(label):
    sources(label)
    root=ARCHIVE/label
    for path in (root/'m3oe/examples',root/'m3oe',root/'experiments',root/'research_R2/src'):
        sys.path.insert(0,str(path))
    return root

def population(name='movielens_natural'):
    # Verify archives before importing or executing any population producer.
    dependency_inventory('v2')
    from experiments.natural_ml_reassessment.data import load_population
    actual_module_paths('v2')
    result=load_population(name)
    actual_module_paths('v2')
    return result
