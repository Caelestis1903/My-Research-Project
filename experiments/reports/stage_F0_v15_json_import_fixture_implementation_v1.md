# Stage F0 v1.5 child-fixture JSON import fix

Only `experiments/tests/test_stage_F0_successor_v15.py` changed. The generated `v15_race_child.py` referenced `json.dumps` from the worker-0 coordinated deferral path, while the previous `json` import was scoped inside the peer-only branch. The generated script now imports `json` at module scope before `fake_live`, so both peer and non-peer workers can write their marker records. The existing fixture-integrity test also asserts that this module import appears before `fake_live`.

No runtime, runner, algorithm, or production-state file changed. No long E2E, release, migration, or training was started.

Validation used `/home/xzf/miniconda3/envs/phase0-3-env/bin/python`. `py_compile experiments/tests/test_stage_F0_successor_v15.py` passed. The focused fixture test ran through `run_f0_python_task.sh`, with scheduler root `/tmp/f0-v15-json-import-fix-20261014-b/scheduler` and basetemp `/tmp/f0-v15-json-import-fix-20261014-b/basetemp`: `1 passed in 0.09s`.

One earlier focused attempt exposed an incorrect assertion spelling (`import json` did not match the combined `import faulthandler, json, os, sys` statement). That stdout is retained along with the final passing stdout in `/tmp/f0-v15-json-import-evidence-20261014/`; manifest SHA-256 is `8625fb9c6649f6d78cbb8d90bab710333dc48c802245a82c0139130ddf054c1a`. Final passing stdout SHA-256: `59907462a8d5fd93406cdf4c7d1eb2f7b7a96664ae85c4ec202693d419cd019e`.

Final source SHA-256: `4bd5431b7e16952e0cd9c234ecbb4d88be58ac5516999163b296dc7dca4a40f9`.
