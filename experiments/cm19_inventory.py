"""Write the explicit planned/complete/alias ledger from raw receipts."""
import json,time
from collections import Counter
import pandas as pd
from cm19_train import OUT,dump

def main():
    manifest=json.loads((OUT/'run_manifest.json').read_text());queue=OUT/'queue_formal.status.json';running={r['run_id']:r for r in json.loads(queue.read_text())['running']} if queue.exists() else {};rows=[];actual={}
    for c in manifest['runs']:
        out=OUT/'runs'/c['run_id'];receipt=out/'receipt.json';test=out/'test_receipt.json';r=json.loads(receipt.read_text()) if receipt.exists() else {};failure=out/'failure.json'
        status='COMPLETE_FROZEN_TEST' if test.exists() else 'COMPLETE_TRAINING_TEST_DEFERRED' if receipt.exists() else 'FAILED' if failure.exists() else 'RUNNING' if c['run_id'] in running else 'PENDING'
        actual[c['run_id']]=status;row={**c,'execution_status':status,'equivalent_to':'','independent_training':True,'completed_epochs':r.get('epochs'),'completed_steps':r.get('steps'),'wall_seconds':r.get('wall_seconds'),'peak_reserved_mib':r.get('peak_reserved_mib'),'test_complete':test.exists(),'gpu':running.get(c['run_id'],{}).get('gpu'),'state_directory':''}
        cfg=out/'config.json'
        if cfg.exists():row['state_directory']=json.loads(cfg.read_text())['state_directory']
        rows.append(row)
    for a in manifest['aliases']:
        source=a['source'];done=actual[source] in ['COMPLETE_FROZEN_TEST','COMPLETE_TRAINING_TEST_DEFERRED']
        rows.append(dict(run_id=a['run_id'],stage=a['stage'],execution_status='EQUIVALENT_COMPLETED' if done else 'EQUIVALENT_PENDING',equivalent_to=source,independent_training=False,reason=a['reason'],test_complete=actual[source]=='COMPLETE_FROZEN_TEST'))
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'RUN_LEDGER.csv',index=False)
    counts=Counter(actual.values());complete=sum(v for k,v in counts.items() if k.startswith('COMPLETE_'));tests=counts['COMPLETE_FROZEN_TEST'];summary=dict(updated_unix=time.time(),independent_training_planned=len(manifest['runs']),nominal_configurations=len(rows),aliases=len(manifest['aliases']),formal_training_completed=complete,formal_training_running=counts['RUNNING'],formal_training_pending=counts['PENDING'],failed=counts['FAILED'],formal_test_completed=tests,by_status=dict(counts),all_training_complete=complete==len(manifest['runs']))
    dump(OUT/'RUN_SUMMARY.json',summary);status=json.loads((OUT/'EXECUTION_STATUS.json').read_text());status.update(summary);status['test_stage']='COMPLETE_FROZEN_TEST' if tests==len(manifest['runs']) else 'PARTIAL_TEST' if tests else 'DEFERRED';dump(OUT/'EXECUTION_STATUS.json',status)
    print(json.dumps(summary))
if __name__=='__main__':main()
