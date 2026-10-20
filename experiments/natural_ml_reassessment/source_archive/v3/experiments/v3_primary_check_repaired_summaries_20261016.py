from pathlib import Path
import json,hashlib,time
R=Path.cwd();O=R/'research_plans/2026-10-16_v3/execution_20261016';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();load=lambda p:json.loads(p.read_text())
s=load(O/'test_all14_summary_serialization_retry_CPU_manifest.status.json');assert s['status']=='COMPLETE' and len(s['tasks'])==14 and all(x['exit_code']==0 and x['status']=='COMPLETE' for x in s['tasks'].values())
n=0
for x in load(O/'test_summary_serialization_failed_partial_index.json')['records']:
 for name,h in x['files'].items():
  if name.endswith('.tmp'):continue
  assert sha(Path(x['output'])/name)==h,(name,x['output']);n+=1
assert sha(O/'test_readout_frozen.json')=='91739132c2dd825b77dff5e0b88281621c6ce51e68eaf56219be0d4f2091c0e5'
assert sha(R/'experiments/v3_test_summary_20261016.py')=='149430001d72e57f5de6e745afb7614462b7c9c5f6a8c5b19482fef3ab58c2aa'
proof=dict(status='PASS_SERIALIZATION_ONLY_ACTUAL14_EXIT0',time=time.time(),independent_review=False,preserved_preexisting_output_files_byte_identical=n,all14_child_exit0=True,original_source_and_freeze_unchanged=True,full_numeric_verification='NEXT_REQUIRED_NOT_YET_PASSED')
(O/'test_summary_serialization_repair_actual_primary_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
source=R/'experiments/v3_verify_frozen_test_summaries_v2_20261016.py';assert sha(source)=='506297ff341eefe2347276e5d5e6d3274df11f05868689af32dc3288ba85d6b6'
m=dict(cwd=str(R),max_cpu_processes=1,tasks=[dict(name='frozen_test_v2_full_numeric_reconstruction',command=['/home/xzf/miniconda3/envs/phase0-3-env/bin/python',str(source.relative_to(R)),'--manifest',str(O/'test_summary_verifier_manifest.json'),'--out',str(O/'analysis/test_all_cohorts_primary_verification')],log=str(O/'logs/frozen_test_v2_full_numeric_reconstruction.log'))])
p=O/'test_all14_numeric_verifier_CPU_manifest.json';assert not p.exists();p.write_text(json.dumps(m,indent=2)+'\n')
review=dict(status='PASS',independent_review=False,manifest_sha=sha(p),source_sha=sha(source),pretest_review_sha=sha(O/'frozen_test_summary_verifier_partial_support_v2_self_review.json'),summary_repair_proof=proof,scope='Activate exact pretest frozen verifier only. No condition, estimator, endpoint, selection or contrast changes.',time=time.time())
(O/'test_all14_numeric_verifier_activation_review.json').write_text(json.dumps(review,indent=2)+'\n');print(proof)
