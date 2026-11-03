"""Use the pinned RAM guardian with this study's process entrypoint."""
import json
import subprocess
import sys
import time
from pathlib import Path
from .vendor.v3.guardian import request, serve, alive
from .protocol import ROOT


def ensure_guardian(attempt):
    attempt=Path(attempt).resolve();attempt.mkdir(parents=True,exist_ok=True)
    if (attempt/'guardian.json').exists():return request(attempt,'status')
    with (attempt/'guardian.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-m',__package__+'.run','_guardian','--attempt',str(attempt)],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    for _ in range(600):
        if (attempt/'guardian.json').exists():return request(attempt,'status')
        if process.poll() is not None:break
        time.sleep(.1)
    raise RuntimeError('RAM guardian startup failed; inspect guardian.log')
