import copy
import json
from pathlib import Path
import shutil
from experiments.gradient_aug_time_select_20261102.protocol import ROOT,HISTORY,protocol
from experiments.gradient_aug_time_select_20261102.recovery import (
    attempt_endpoint_rows,historical_endpoint_core,task_clock)


def test_actual_history_interface_and_reject_missing_identity_or_changed_hash(tmp_path):
    p=protocol();p['history_dir']=str(HISTORY)
    old=ROOT/'artifacts/gradient_aug_time_select_20261102/attempt_20261102_0005'
    rows=attempt_endpoint_rows(old)
    real=historical_endpoint_core(rows,p,tmp_path/'real')
    assert len(rows)==48 and real['matched_endpoints']==48 and all(real['verified'].values())
    assert any(r['historical_field']=='data_manifest.movielens.features.encoder_hash' and r['matches']
        for r in real['identity_checks'])
    missing=tmp_path/'missing_history';missing.mkdir()
    for name in real['source_files']:shutil.copyfile(HISTORY/name,missing/name)
    data=json.loads((missing/'data_manifest.json').read_text());del data['movielens']['features']['encoder_hash']
    (missing/'data_manifest.json').write_text(json.dumps(data))
    rejected=historical_endpoint_core(rows,dict(p,history_dir=str(missing)),tmp_path/'missing_receipt')
    assert rejected['verified']==dict(movielens=False,kuairand=True)
    assert any('encoder_hash: missing required identity' in r for r in rejected['reasons']['movielens'])
    changed=copy.deepcopy(rows);changed[0]['model_hash']='0'*64
    rejected=historical_endpoint_core(changed,p,tmp_path/'changed_receipt')
    assert rejected['verified'][changed[0]['dataset']] is False
    assert rejected['matched_endpoints']==47 and rejected['historical_test_accessed_for_decision'] is False


def test_original_clock_deadlines_are_not_extended_by_recovery():
    origin=1793549242.1426222;limits=protocol()['limits']
    first=task_clock(origin,limits,origin+100)
    later=task_clock(origin,limits,origin+20000)
    assert first['stop_dispatch_deadline']==later['stop_dispatch_deadline']==origin+36000
    assert first['dispatch_deadline']==later['dispatch_deadline']==origin+43200
    assert later['elapsed_before_recovery_seconds']==20000
    assert origin+35999<later['stop_dispatch_deadline']<=origin+36000
    assert origin+43199<later['dispatch_deadline']<=origin+43200
