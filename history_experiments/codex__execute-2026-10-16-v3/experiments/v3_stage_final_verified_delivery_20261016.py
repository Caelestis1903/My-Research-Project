"""Produce an explicit compact artifact allowlist after mandatory final checks."""
from pathlib import Path
import json,subprocess
R=Path.cwd();prefix='research_plans/2026-10-16_v3/execution_20261016/';O=R/prefix
assert json.loads((O/'final_delivery_primary_verification.json').read_text())['status']=='PASS_ALL_REQUIRED_SCIENCE_REPORTS_AND_LOCAL_LINKS'
paths=set()
for args in [['git','diff','--name-only','-z'],['git','ls-files','--others','--exclude-standard','-z']]:paths.update(x.decode() for x in subprocess.check_output(args).split(b'\0') if x)
chosen=[];excluded=[]
for name in sorted(paths):
 p=R/name;ok=False
 if name.startswith('experiments/v3_') and p.suffix=='.py' and '__pycache__' not in name:ok=True
 if name.startswith(prefix):
  rel=name[len(prefix):];parts=Path(rel).parts
  if len(parts)==1 and p.suffix in ['.md','.json'] and not name.endswith('.status.json') and 'progress' not in name.lower():ok=True
  elif rel.startswith('archive/formal/') and p.suffix in ['.csv','.json']:ok=True
  elif rel.startswith('figures/') and p.suffix in ['.png','.pdf']:ok=True
  elif len(parts)>=3 and parts[0]=='analysis' and p.suffix in ['.json','.csv','.png','.pdf']:
   if not parts[1].startswith('failed_') and p.name!='progress.json' and not p.name.endswith('.status.json'):
    receipt=O/'analysis'/parts[1]/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status','').startswith(('PASS','COMPLETE','VERIFIED')):ok=True
  if rel.startswith('analysis/Q34_final_all_training_moment_refresh/') and p.name not in ['receipt.json','compact_moment_summary.csv']:ok=False
 if not ok:continue
 assert p.is_file() and not p.is_symlink(),name
 assert p.stat().st_size<=20*1024**2,(name,p.stat().st_size)
 if p.suffix=='.json':json.loads(p.read_text())
 chosen.append(name)
assert not any('__pycache__' in x or x.endswith(('.pt','.npz','.lock','.log')) for x in chosen)
Path('/tmp/v3_final_verified_delivery_paths.nul').write_bytes(b'\0'.join(x.encode() for x in chosen)+b'\0')
print('Final verified compact allowlist files',len(chosen),'MiB',sum((R/x).stat().st_size for x in chosen)/1024**2)
