# Cycle02 numeric data and source

The complete numeric archive has **5305 files**, including all30 new trajectories, six engineering smoke cases, all tables, admission/cost/resource/ACK records and exact result plus50-epoch records for ten reused CLEAN controls. Uncompressed members total391354662 bytes. Code, design, analysis, report, figures and main research-context updates are supplied separately in the repository.

Archive: `RAW_RUNS.tar.gz`,146878639 bytes, split into18 parts `RAW_RUNS.tar.gz.part-00` through `part-17`. Whole SHA256:

`86b7cf1faa0cccd096fbc4e0a72b532322f4f60f4a0a36afa969abeb0606cef0`

Parts and [RAW_RUNS_MANIFEST.json](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1/RAW_RUNS_MANIFEST.json) are under `artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1/`. The manifest records every member and part's byte count and SHA256. [CONTROL_ARCHIVE_MAP.json](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1/CONTROL_ARCHIVE_MAP.json) maps the ten original CLEAN paths/hashes to archived `reused_clean/<key>/` records.

From a directory containing the18 parts:

```bash
cat RAW_RUNS.tar.gz.part-* > RAW_RUNS.tar.gz
sha256sum RAW_RUNS.tar.gz
mkdir cycle02_numeric
tar -xzf RAW_RUNS.tar.gz -C cycle02_numeric
```

Archive member paths include `runs/`, `smoke/`, `jobs/`, `tables/` and `reused_clean/`. Large tables, including216692314-byte `tables/WEIGHTS.csv` and26075558-byte `tables/STATISTICS.csv`, are distributed inside this archive. Other compact numerical summaries are also supplied directly. The archive contains numeric/log files; RAM model/optimizer/checkpoint tensors and the private `guardian.json` credential file are excluded. Source/evaluation identities and the Test=false record accompany all formal outputs.

Export counts:40 trajectories (30 new,10 reused),320 endpoints,4000 estimator-epoch curves,24000 cell-epoch rows,310500 weight rows,99000 statistic rows,1950 diagnostic/group rows and51750 useful training-batch records. [EXPORT_RECEIPT.json](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_02/attempt1/tables/EXPORT_RECEIPT.json) binds numerical inputs and table hashes. Analysis tables and the [numerical appendix](NUMERICAL_APPENDIX.md) preserve every seed's endpoint, selected epoch and selected-state hash.
