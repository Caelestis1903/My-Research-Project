"""Separate cumulative continuation counters from newly executed N17 budget."""
from pathlib import Path
import hashlib,json
import pandas as pd
ART=Path('/home_new/xzf_natural_movielens_reassessment/N17')
OUT=Path(__file__).resolve().parent/'n17_training_accounting'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 configs={}
 for name in ['base_manifest.json','post_pilot_manifest.json','H200_manifest.json']:
  for c in json.loads((ART/name).read_text())['configurations']:
   if c.get('run_dir'):configs[c['id']]=c
 assert len(configs)==88;rows=[]
 for c in configs.values():
  run=Path(c['run_dir']);receipt=run/'receipt.json';r=json.loads(receipt.read_text());assert r['status']=='COMPLETE_TRAINING_TEST_DEFERRED'
  parent=Path(c['resume_from'])/'receipt.json' if c.get('resume_from') else None
  old=json.loads(parent.read_text()) if parent else dict(epochs=0,steps=0,examples=0)
  rows.append(dict(run_id=c['id'],group=c['group'],seed=c['seed'],arm=c['arm'],lr=c['lr'],population=c['population'],
   cumulative_epochs=r['epochs'],cumulative_steps=r['steps'],cumulative_examples=r['examples'],
   new_epochs=r['epochs']-old['epochs'],new_steps=r['steps']-old['steps'],new_examples=r['examples']-old['examples'],
   from_scratch=parent is None,parent_receipt=str(parent) if parent else '',parent_receipt_sha256=sha(parent) if parent else '',receipt_path=str(receipt),receipt_sha256=sha(receipt)))
 f=pd.DataFrame(rows);OUT.mkdir(exist_ok=True);f.to_csv(OUT/'configuration_budget_accounting.csv',index=False)
 cols=['cumulative_epochs','cumulative_steps','cumulative_examples','new_epochs','new_steps','new_examples']
 f.groupby('group')[cols].sum().to_csv(OUT/'group_budget_totals.csv')
 (OUT/'accounting_status.json').write_text(json.dumps(dict(complete=True,nominal_configurations=88,pilots=16,formal_budget_entries=72,
  from_scratch_configurations=int(f.from_scratch.sum()),continuations=int((~f.from_scratch).sum()),
  executed_new_steps=int(f.new_steps.sum()),executed_new_examples=int(f.new_examples.sum()),
  failed_attempts_excluded=True,interpretation='new counters exclude shared H100 prefix; configuration counts are not independent statistical replicates'),indent=2)+'\n')
 print((OUT/'accounting_status.json').read_text())
if __name__=='__main__':main()
