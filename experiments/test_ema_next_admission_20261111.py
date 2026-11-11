from experiments.ema_next_admission_20261111 import reservations,Admission,parent_cohort_forecast


def test_pending_and_growing_workers_reserve_full_deficit():
    prepared={'calibration':{'movielens':{'gpu_peak_bytes':1000}}}
    workers=[dict(identity={'pid':1},key='j1',device=0,dataset='movielens',rss_bytes=1,allocated_peak_bytes=900),dict(identity={'pid':2},key='j2',device=0,dataset='movielens',rss_bytes=1,allocated_peak_bytes=0)]
    adjusted,reserve,highwater,entries=reservations(prepared,workers,[dict(index=0,used=5000)],{(1,0):1300},{})
    assert reserve['movielens']==1400
    assert adjusted[0]['used']==6500
    assert entries[1]['observed_gpu_bytes'] is None and entries[1]['unallocated_reservation_bytes']==1400
    again=reservations(prepared,[],[dict(index=0,used=1)],{},highwater)
    assert again[1]['movielens']==1400


def test_pending_same_job_cannot_duplicate_after_controller_restart(monkeypatch,tmp_path):
    import experiments.ema_next_admission_20261111 as module
    worker=dict(identity={'pid':3},key='pending',device=1,dataset='movielens',rss_bytes=1,allocated_peak_bytes=0)
    monkeypatch.setattr(module,'worker_inventory',lambda _: [worker])
    monkeypatch.setattr(module,'gpu_inventory',lambda _: ([dict(index=1,used=1)],{}))
    a=Admission(tmp_path,persist=False)
    device,receipt=a({'calibration':{'movielens':{'gpu_peak_bytes':1000}}},{},{'job_id':'pending','dataset':'movielens'},[1],0)
    assert device is None and receipt['reason']=='same_job_pending_or_running'


def kr_case():
    from experiments.ema_next_20261111.protocol import ROOT,resolve_plan,resource_forecast
    prepared=resolve_plan(ROOT/'research_plans/26-11-11_ema_next_phase')
    prepared['resource_estimate']=resource_forecast(prepared)
    prepared['groups']={ds:{'runtime_buffer_bytes':0} for ds in prepared['manifest']['dataset_order']}
    prepared['calibration']={ds:{'gpu_peak_bytes':1000} for ds in prepared['groups']}
    job=next(j for j in prepared['jobs'] if j['dataset']=='kuairand' and j['family']=='PARENT')
    status={'payload_bytes':prepared['resource_estimate']['datasets']['kuairand']['prior_dataset_endpoint_bytes']}
    return prepared,status,job


def test_parent_cohort_four_admitted_five_rejected_including_pending(monkeypatch,tmp_path):
    import experiments.ema_next_admission_20261111 as module
    prepared,status,job=kr_case();peak=dict(prepared['resource_estimate']['datasets']['kuairand']['peak'])
    workers=[dict(identity={'pid':n+1},key=f'other{n}',device=0,dataset='kuairand',family='PARENT',rss_bytes=1,allocated_peak_bytes=0) for n in range(3)]
    monkeypatch.setattr(module,'worker_inventory',lambda _:workers)
    monkeypatch.setattr(module,'gpu_inventory',lambda _:([dict(index=0,used=1,total=24<<30,power=1,power_limit=100)],{}))
    a=Admission(tmp_path,persist=False);a.highwater={'nonallocated_overhead_bytes':100}
    device,receipt=a(prepared,status,job,[0],0)
    assert device==0 and receipt['parent_cohort']['cohort_count']==4
    assert receipt['guardian_forecast']<4<<30 and receipt['physical_forecast']<7<<30
    assert all(w['unallocated_reservation_bytes']==1100 for w in receipt['worker_reservations'])
    workers.append(dict(workers[0],identity={'pid':9},key='fourth'))
    device,receipt=a(prepared,status,job,[0],0)
    assert device is None and receipt['reason']=='guardian_commit_4GiB'
    assert receipt['parent_cohort']['cohort_count']==5 and receipt['physical_forecast']>7<<30
    assert prepared['resource_estimate']['datasets']['kuairand']['peak']==peak


def test_retained_payload_accumulates_and_other_families_keep_original_bound():
    prepared,status,job=kr_case()
    first=parent_cohort_forecast(prepared,status,job,[],0)
    retained=dict(status,payload_bytes=status['payload_bytes']+2*first['immutable_closure_bytes'])
    later=parent_cohort_forecast(prepared,retained,job,[],0)
    for key in ('guardian_predicate','physical_tensor_bytes'):
        assert later[key]-first[key]==2*first['immutable_closure_bytes']
    assert parent_cohort_forecast(prepared,status,dict(job,family='A'),[],0) is None
    assert parent_cohort_forecast(prepared,status,job,[{'family':'A','dataset':'kuairand'}],1) is None
    assert parent_cohort_forecast(prepared,status,job,[],1) is None


def test_frozen_gate_reentry_normalizes_seed_keys_and_rejects_changed_value(tmp_path):
    import pytest
    from experiments.ema_next_admission_20261111 import gate_write
    value={'points':{'F':{2022:{'macro_auc':.75}}},'winner':'F'}
    gate_write(tmp_path,'movielens','PARENT_BC',value)
    path=tmp_path/'decisions/movielens/PARENT_BC.json';before=path.read_bytes();modified=path.stat().st_mtime_ns
    gate_write(tmp_path,'movielens','PARENT_BC',value)
    assert path.read_bytes()==before and path.stat().st_mtime_ns==modified
    value['points']['F'][2022]['macro_auc']=.75000001
    with pytest.raises(ValueError,match='Frozen decision changed'):
        gate_write(tmp_path,'movielens','PARENT_BC',value)
    assert path.read_bytes()==before and path.stat().st_mtime_ns==modified


def test_actual_baseline_completed_and_pending_cohorts(monkeypatch):
    import experiments.ema_next_admission_20261111 as module
    prepared,_,job=kr_case();base={'payload_bytes':288103704}
    monkeypatch.setattr(module.scheduler,'gpu_state',lambda _: [dict(index=1,total=24<<30,used=1,power=1,power_limit=100)])
    for done,n,admitted in [(0,4,True),(0,5,False),(3,2,True),(4,1,True)]:
        workers=[dict(key=f'pending{i}',family='PARENT',dataset='kuairand') for i in range(n-1)]
        status={'payload_bytes':1464696528,'jobs':{f'done{i}':{'job':job,'status':'COMPLETE'} for i in range(done)}}
        f=parent_cohort_forecast(prepared,status,job,workers,0,base)
        assert f['cohort_count']==n and f['completed_parent_count']==done
        resolved=dict(prepared,resource_estimate={'datasets':dict(prepared['resource_estimate']['datasets'])})
        resolved['resource_estimate']['datasets']['kuairand']=dict(resolved['resource_estimate']['datasets']['kuairand'],peak=f)
        device,receipt=module.ORIGINAL_ADMIT(resolved,status,job,[1],0)
        assert (device is not None)==admitted
        if admitted:assert receipt['guardian_forecast']<4<<30 and receipt['physical_forecast']<7<<30
    old=parent_cohort_forecast(prepared,status,job,[],0)
    assert old['retained_payload_bytes']==status['payload_bytes'] and old['completed_parent_count']==0


def test_baseline_receipt_binding_and_nonapplicable_fallback(monkeypatch,tmp_path):
    import experiments.ema_next_admission_20261111 as module
    from experiments.overnight_ab_20261103.state import write_json,sha_file,digest
    prepared,_,job=kr_case();prepared['identity']={'implementation_hash':'science'}
    owner={'pid':123,'start':'1','boot':'boot'};status={'pid':123,'jobs':{}}
    record={'job_id':job['job_id'],'time':1.,'reason':'ADMITTED','worker_reservations':[],'discovered_worker_count':0,'parent_cohort':{'retained_payload_bytes':288103704}}
    (tmp_path/'RESOURCE_ADMISSION.jsonl').write_text(module.json.dumps(record)+'\n')
    complete=tmp_path/'decisions/movielens/DATASET_COMPLETE.json';write_json(complete,{'status':'COMPLETE'})
    proof=dict(attempt=str(tmp_path),scientific_hash='science',guardian_owner=owner,previous_dataset='movielens',previous_complete_sha256=sha_file(complete),first_job_id=job['job_id'],first_admission_time=1.,first_admission_sha256=digest(record),payload_bytes=288103704)
    path=tmp_path/'baseline.json';write_json(path,proof);monkeypatch.setattr(module,'BASELINE_PATH',path);monkeypatch.setattr(module,'process_identity',lambda _:owner)
    assert module.verified_parent_baseline(tmp_path,prepared,status,job,[])['payload_bytes']==288103704
    assert module.verified_parent_baseline(tmp_path,prepared,status,dict(job,objective='gaussian_d0'),[]) is None
    orphan=dict(status,jobs={'unseen':{'job':job,'status':'RUNNING'}})
    assert module.verified_parent_baseline(tmp_path,prepared,orphan,job,[]) is None
    write_json(complete,{'status':'CHANGED'})
    assert module.verified_parent_baseline(tmp_path,prepared,status,job,[]) is None
    write_json(complete,{'status':'COMPLETE'});record['time']=2.
    (tmp_path/'RESOURCE_ADMISSION.jsonl').write_text(module.json.dumps(record)+'\n')
    assert module.verified_parent_baseline(tmp_path,prepared,status,job,[]) is None


def a_case():
    prepared,status,parent=kr_case()
    b=next(r for r in prepared['readouts']['kuairand'] if r['id']=='B_H277_all_logit')
    job=dict(parent,job_id='proposed_A',family='A_MATCH',start_epoch=20,end_epoch=26,anchor=20,frozen_B=b)
    return prepared,dict(status,payload_bytes=1367218872),job


def test_a_four_workers_admitted_five_rejected_with_pending(monkeypatch,tmp_path):
    import experiments.ema_next_admission_20261111 as module
    prepared,status,job=a_case();workers=[dict(identity={'pid':i+1},key=f'A{i}',device=1,dataset='kuairand',family='A_MATCH',objective='clean',job=dict(job,job_id=f'A{i}'),rss_bytes=1,allocated_peak_bytes=0) for i in range(3)]
    monkeypatch.setattr(module,'worker_inventory',lambda _:workers)
    monkeypatch.setattr(module,'gpu_inventory',lambda _:([dict(index=1,total=24<<30,used=1,power=1,power_limit=100)],{}))
    a=Admission(tmp_path,persist=False);a.highwater={'nonallocated_overhead_bytes':100}
    device,receipt=a(prepared,status,job,[1],0)
    assert device==1 and receipt['a_cohort']['cohort_count']==4
    assert receipt['guardian_forecast']==2831471640 and receipt['physical_forecast']==7076674744
    workers.append(dict(workers[0],key='fourth',identity={'pid':9}))
    device,receipt=a(prepared,status,job,[1],0)
    assert device is None and receipt['reason']=='physical_tensor_7GiB'


def test_a_completed_payload_and_scope_fallback():
    from experiments.ema_next_admission_20261111 import a_cohort_forecast
    prepared,status,job=a_case();before=a_cohort_forecast(prepared,status,job,[],0)
    kept=dict(status,payload_bytes=status['payload_bytes']+100000000,jobs={'e20':{'status':'COMPLETE'}})
    after=a_cohort_forecast(prepared,kept,job,[],0)
    released=dict(kept,payload_bytes=kept['payload_bytes']-50000000,jobs={'e20':{'status':'COMPLETE'},'e35':{'status':'RELEASED'}})
    final=a_cohort_forecast(prepared,released,job,[],0)
    for key in ('guardian_predicate','physical_tensor_bytes'):
        assert after[key]-before[key]==100000000 and after[key]-final[key]==50000000
    continuation=dict(job,family='A_RAW',continuation=True,start_epoch=26,end_epoch=50)
    assert a_cohort_forecast(prepared,status,continuation,[],0) is not None
    for other in (dict(job,family='KD_LABEL'),dict(job,readout_menu=[]),dict(job,objective='gaussian_d0')):
        assert a_cohort_forecast(prepared,status,other,[],0) is None
    assert a_cohort_forecast(prepared,status,job,[{'job':dict(job,family='PARENT')}],1) is None
    assert a_cohort_forecast(prepared,status,job,[],1) is None
