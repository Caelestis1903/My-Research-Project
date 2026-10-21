# Complete M3oE package

`MDMTRec/` contains the original STAR + shared/domain/task expert backbone,
original learned fusion (`original.py`), controlled fixed fusion (`fixed.py`),
and supplied specificity/noise companion (`model.py`). The executable entry is
`python experiments/data_generate/run.py --dataset movielens --seed 2022`.
Use the Python environment and reviewed workflow in
`../reproduction_20261021/README.md`.

The root uploaded files remain byte-identical archival references. Their
relative imports assume placement inside MDMTRec and they are not entrypoints.
Complete independent package copies live under this directory. Shared protocol
and dataset helpers live under `../reproduction_20261021` to prevent diverging
preprocessing. Grad's model is explicitly paired from the supplied data model;
historical identity as an original gradient companion is unproven.
