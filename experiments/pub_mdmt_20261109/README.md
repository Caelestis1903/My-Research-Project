# PUB-MDMT

Independent fixed1 M3oE package. Source model, natural population, raw probability
metrics and twelve coordinate groups are frozen to `1391affa`; source hashes and
import-only changes are in `SOURCE_MANIFEST.json`. The paper's Eq.(4) CCP replaces
the audited broken author solver. Public update balancing uses raw Gram scale,
shared pre-step Adam moments and one common coupled-WD correction. G0/G1/G2 replay
the matching C0/C1/C2 coefficients and fallback decisions.

Use the project environment and repository root:

```sh
PY=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
$PY -m experiments.pub_mdmt_20261109.run audit
$PY -m experiments.pub_mdmt_20261109.run check
$PY -m experiments.pub_mdmt_20261109.run smoke --device cuda:0
# Independent REVIEW_REPORT.md must be PASS before the following commands.
$PY -m experiments.pub_mdmt_20261109.run calibrate --device cuda:0
$PY -m experiments.pub_mdmt_20261109.run pilot --devices 0,1
$PY -m experiments.pub_mdmt_20261109.run train --devices 0,1
$PY -m experiments.pub_mdmt_20261109.run report
```

`--out` and `--cache` set artifact and natural array-cache directories. Defaults
are recorded in `run.py`/`data.py`; the raw CSV locations are the migrated paths.
Audit verifies the raw SHA, ordered array contents, row IDs, train-only encoders,
scaler, and split/domain counts. Training workers open Test labels after their
Validation selections have been frozen. Solver/frequency and combination
selection functions receive Validation or numerical inputs only.

The finite matrix contains 40 two-epoch pilot jobs and 70 fifty-epoch formal
jobs. The original A trajectory maintains F simultaneously. PUB+EMA is added
only for a topology meeting the specified five-seed Validation gate. Dispatch
fills GPUs independently while memory/power are below 85% and host RAM below
90%, with measured full-batch memory headroom. Each GPU process sets its assigned
device before model construction.

Selected/E10/E50 states and optimizer/RNG/loader snapshots stay in RAM. An
interrupted job restarts from its original seed; failed scalar logs are retained
as attempt files. No model/checkpoint files are produced. `RESULT.json` contains
metrics and state hashes, never weights. The report command produces raw seed,
cell/task, Macro/Traffic/Pooled, paired SD/CI, cost and offline frequency100
coefficient-staleness tables; it does not select configurations from Test.
