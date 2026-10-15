"""Bounded test launcher using the existing reserved-slot protocol.
The original run_f0_python_task.sh is absent from this repository snapshot.
No production queue, claim, hold, model or GPU is touched.
"""
import sys,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments'))
from stage_F0_queue import acquire_process_slot
parent=acquire_process_slot(role='20261015-data-test-supervisor',blocking=False)
try:
 child=acquire_process_slot(role='20261015-data-pytest',blocking=False)
 try:
  fd=child.lock.handle.fileno();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',F0_RESERVED_PROCESS_SLOT_FD=str(fd),F0_RESERVED_PROCESS_SLOT_INDEX=str(child.index),F0_RESERVED_PROCESS_SLOT_OWNER_PID=str(os.getpid()),F0_RESERVED_PROCESS_SLOT_ROLE='20261015-data-pytest')
  r=subprocess.run([sys.executable,'-m','pytest','-q','experiments/tests/test_stage_S1_data.py'],cwd=ROOT,env=env,pass_fds=(fd,))
 finally:child.close()
finally:parent.close()
sys.exit(r.returncode)
