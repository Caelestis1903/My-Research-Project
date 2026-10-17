"""Record actual primary reporting subprocess provenance and exit status."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'research_plans/2026-10-16_v3/execution_20261016'
p=argparse.ArgumentParser();p.add_argument('script');a=p.parse_args();source=R/'experiments'/a.script
allowed={'v3_summarize_frozen_tests_20261016.py','v3_final_test_figures_20261016.py','v3_finalize_question_closure_20261016.py','v3_archive_final_test_evidence_20261016.py','v3_render_final_report_20261016.py','v3_render_final_main_reports_20261016.py','v3_final_delivery_check_20261016.py'};assert a.script in allowed
assert json.loads((O/'analysis/test_all_cohorts_primary_verification/receipt.json').read_text())['status']=='PASS'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();report=O/(source.stem+'_actual_primary_execution.json');assert not report.exists()
cmd=[sys.executable,str(source.relative_to(R))];start=time.time();proc=subprocess.Popen(cmd,cwd=R,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'})
record=dict(status='RUNNING',actual_pid=proc.pid,command=cmd,cwd=str(R),source_sha=sha(source),launcher_source_sha=sha(Path(__file__)),started=start,finished=None,exit_code=None);report.write_text(json.dumps(record,indent=2)+'\n')
output,_=proc.communicate();log=O/'logs'/(source.stem+'_actual_primary_execution.log');log.write_text(output);assert record['source_sha']==sha(source)
record.update(status='COMPLETE' if proc.returncode==0 else 'FAILED',finished=time.time(),exit_code=proc.returncode,log=str(log));report.write_text(json.dumps(record,indent=2)+'\n');print(output,end='');print(json.dumps(record));sys.exit(proc.returncode)
