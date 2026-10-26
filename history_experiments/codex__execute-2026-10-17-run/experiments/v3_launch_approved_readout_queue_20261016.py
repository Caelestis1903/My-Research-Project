"""Launch only primary-approved immutable paired readouts; compute every hash."""
import hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
R=Path('/home/xzf/Code/phase0-3-20261016-v3-git');O=R/'research_plans/2026-10-16_v3/execution_20261016'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_text())

def main():
 stem=sys.argv[1];assert Path(stem).name==stem
 manifest=O/(stem+'_manifest.json');review=O/(stem+'_activation_review.json');priorpath=O/(stem+'_predecessors.json');status=manifest.with_suffix('.status.json');launch=O/(stem+'_queue_launch.json')
 assert not status.exists() and not launch.exists()
 a=load(review);assert a['status']=='PASS' and a['formal_readouts_authorized'] is True and a['test_access'] is False
 assert a['manifest_sha']==sha(manifest)
 m=load(manifest);engine=R/'experiments/v3_gpu_readout_20261016.py';serializer=R/'experiments/v3_readout_20261016.py';queue=R/'experiments/v3_readout_gpu_queue_20261016.py'
 assert sha(engine)==a['engine_sha'] and sha(serializer)==a['readout_sha']
 for t in m['tasks']:
  c=t['command'];assert R/c[1]==engine and c[c.index('--mode')+1]=='pair' and c[c.index('--device')+1]=='{device}'
  assert not Path(c[c.index('--out')+1]).exists()
 prior=load(priorpath);registered=load(O/'orchestrator_checkpoint.json')['active_gpu_status_files'];assert registered[:len(prior)]==prior and registered[len(prior)]==status.name
 assert all((O/x).exists() for x in prior)
 command=['/home/xzf/miniconda3/envs/phase0-3-env/bin/python',str(queue),'--manifest',str(manifest)]
 for name in prior:command.extend(['--preceding-status',str(O/name)])
 log=O/'logs'/(stem+'_queue.log')
 with log.open('ab') as stream:p=subprocess.Popen(command,cwd=R,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
 result=dict(pid=p.pid,command=command,command_shlex=shlex.join(command),cwd=str(R),code_commit_at_queue_launch=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip(),manifest=str(manifest),manifest_sha256=sha(manifest),activation_review=str(review),activation_review_sha256=sha(review),predecessors_file=str(priorpath),predecessors_sha256=sha(priorpath),predecessor_count=len(prior),source_sha256={'queue':sha(queue),'readout':sha(engine),'serializer':sha(serializer),'launcher':sha(Path(__file__))},started_unix=time.time(),stdout_log=str(log),status=str(status),start_new_session=True,returncode=None)
 launch.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['pid','started_unix','code_commit_at_queue_launch','manifest_sha256','predecessor_count','source_sha256','status']}))
if __name__=='__main__':main()
