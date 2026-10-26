"""Check mandatory completed artifacts and local report links before publication."""
from pathlib import Path
import hashlib,json,re,time
R=Path(__file__).resolve().parents[1];O=R/'research_plans/2026-10-16_v3/execution_20261016'
load=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(O/'test_readout_frozen.json')=='91739132c2dd825b77dff5e0b88281621c6ce51e68eaf56219be0d4f2091c0e5'
assert sha(R/'experiments/v3_verify_frozen_test_summaries_v2_20261016.py')=='506297ff341eefe2347276e5d5e6d3274df11f05868689af32dc3288ba85d6b6'
v=load(O/'analysis/test_all_cohorts_primary_verification/receipt.json');assert v['status']=='PASS' and v['cohorts']==14
q=load(O/'question_registry.json');assert len(q['questions'])==46 and {x['id'] for x in q['questions']}=={f'Q{i}' for i in range(1,47)}
assert all(x['status'] in ['RESOLVED','FALSIFIED','BOUNDARY','NOT_IDENTIFIABLE_WITH_CURRENT_DESIGN'] for x in q['questions'])
l=load(O/'execution_ledger.json');assert len(l)==15 and all(x['status']=='COMPLETE' for x in l.values())
a=load(O/'final_test_archive_primary_verification.json');assert a['status']=='PASS_ALL156_TEST_ARCHIVES_AND_RAW_INDEX' and a['required_source_paths']==156 and a['native_test_prediction_files']==13500 and a['reuse_provenance_files']==36
for n in ['test_native_oracle_manifest.status.json','test_native_bulk_manifest.status.json','test_remaining_prefix_projection_CPU_manifest.status.json','test_all14_summary_serialization_retry_CPU_manifest.status.json','test_all14_numeric_verifier_CPU_manifest.status.json']:
 s=load(O/n);assert s['status']=='COMPLETE' and all(x['status']=='COMPLETE' and x['exit_code']==0 for x in s['tasks'].values()),n
f=O/'FINAL_REPORT.md';t=f.read_text();assert [int(x) for x in re.findall(r'^## (\d+)\.',t,re.M)]==list(range(1,15));assert '# Newly Exposed Questions' in t
new=re.split(r'^### (Q\d+)：',t,flags=re.M);ids=[]
for i in range(1,len(new),2):
 ids.append(new[i]);part=new[i+1]
 for field in ['原始 observation','为何仍有边界','是否阻塞核心','已完成','当前 boundary','下一最小实验']:assert field in part,(new[i],field)
assert ids==[f'Q{i}' for i in range(28,47)]
reports=['FINAL_REPORT.md','execution_report.md','experiment_results.md','mechanism_analysis.md','V12_final_causal_synthesis.md','V13_final_method_decision.md','final_question_closure.md']
links=[]
for name in reports:
 p=O/name;s=p.read_text()
 for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',s):
  if target.startswith(('https://','http://','#','mailto:')):continue
  target=target.split('#')[0];dest=(p.parent/target).resolve();assert dest.exists(),(name,target);links.append([name,target])
figures=['final_test_selected_macro_gaps.png','final_test_selected_macro_gaps.pdf','final_test_all_six_cells.png','final_test_all_six_cells.pdf','final_frozen_covariance_spectra.png','final_frozen_covariance_spectra.pdf']
assert all((O/'figures'/x).is_file() for x in figures)
d=load(O/'final_scientific_decision.json');assert d['status']=='FINAL_APPROVED_AFTER_ALL_REQUIRED_RESULTS' and q['scientific_classification']==d['classification'] and q['method_decision']==d['method_decision']
proof=dict(status='PASS_ALL_REQUIRED_SCIENCE_REPORTS_AND_LOCAL_LINKS',time=time.time(),independent_review=False,source_sha=sha(Path(__file__)),report_hashes={x:sha(O/x) for x in reports},numbered_final_chapters=14,terminal_questions=46,new_questions=19,new_question_required_fields=114,completed_phases=15,verified_local_report_links=len(links),numeric_verification_receipt_sha=sha(O/'analysis/test_all_cohorts_primary_verification/receipt.json'),archive_verification_sha=sha(O/'final_test_archive_primary_verification.json'),classification=d['classification'],method_decision=d['method_decision'],publication_status='PENDING_ACTUAL_GIT_COMMIT_PUSH_AND_REMOTE_MATCH',scope='This check verifies completed scientific artifacts; it does not claim publication before actual push and remote readback.')
(O/'final_delivery_primary_verification.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n');print(json.dumps(proof,ensure_ascii=False))
