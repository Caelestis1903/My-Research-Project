# Durable resident service launch

Launch only after the independent review authorizations match the actual source/config. Preserve the old process-loss archive; it is not evidence of a model result or an OOM cause. No model/optimizer state exists on disk.

Run the following project-Python launcher in an ordinary short-lived tool call. The child has a new session, ordinary file-backed stdout/stderr, closed inherited input, and no tool PTY. Do not directly run `serve` as a long-running tool PTY.

```python
from pathlib import Path
import os, subprocess, time
repo=Path('/home/xzf/Code/phase0-3')
root=repo/'artifacts/mdmt_rcf_20261026/v1'
log=root/f'service_replay_{time.time_ns()}.log'
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with log.open('ab',buffering=0) as output:
    process=subprocess.Popen(
        ['/home/xzf/miniconda3/envs/phase0-3-env/bin/python','-m',
         'experiments.mdmt_rcf_20261026.cli','serve','--devices','cuda:0','cuda:1'],
        cwd=repo,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
        start_new_session=True,close_fds=True,env=env)
print({'pid':process.pid,'log':str(log)})
```

Verify the fresh `SERVICE.json` PID exists and the authenticated `cli status` works before sending training commands. A stale endpoint must never be overwritten while its recorded process is alive. If PID is absent, remove only the owned stale socket after verifying owner/type, as already documented in the process-loss receipt.

The first replay command can be `cli capture-fixture --dataset movielens --seed 5201`. This runs one native/source/local prefix plus B OOF/refit, exports the real group fixture, and keeps all objects in the service worker. After the numerical designer and reviewer approve the actual fixture remedy, refresh the normal review token; `cli reload-readers` additionally requires `READER_RELOAD_AUTHORIZATION.json`. Then `cli prefix --datasets movielens kuairand --seeds 5201 5202 --resume` reuses the already-resident ML5201 prefix and completes only the remaining prefixes. `read-probe` reuses its completed B and any completed S1 method objects.

If the service disappears again, saved prediction/data files do not constitute model resume. Record the loss and missing state honestly before any same-identity replay; never claim disk restoration.
