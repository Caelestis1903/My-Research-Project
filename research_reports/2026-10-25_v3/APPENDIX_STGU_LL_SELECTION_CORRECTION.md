# STGU LL独立选点修正完整表

仅旧桥接seed3101的零训练更新重评分。旧AUC-primary输出保留；此表依据validation eligible≥12最小macroLL，精确平局最早，未用Q选点。n=1不提供seed CI。六格明确domain/task，Q_explore已暴露。

## kuairand / selected

epoch=14; macroAUC=0.6978586643; macroLL=0.4663680605。

checkpoint `/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_kuairand_STGU_DOMAIN_ADAM_3101/ll_best.pt`；SHA256 `3115a02e9a33c105459a407fbb94c352fb5b89d4ccd87282ae734ad818cbefb3`。

|domain|task|n|positive|AUC|LL|
|---|---|---:|---:|---:|---:|
|0|0|8639|817|0.6657710246|0.2990591015|
|1|0|87195|32928|0.7359767418|0.5778019449|
|2|0|4473|2758|0.6865433607|0.6144211085|
|0|1|8639|385|0.6668914560|0.1753216164|
|1|1|87195|22257|0.7451742593|0.4932482697|
|2|1|4473|2149|0.6867951431|0.6383563223|

## kuairand / endpoint24

epoch=24; macroAUC=0.7031668218; macroLL=0.4678337836。

checkpoint `/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_kuairand_STGU_DOMAIN_ADAM_3101/epoch24.pt`；SHA256 `9fb3ae1298ed1dcc7a91a43f54ecb6b6db2821f310b7ca51ef87001d2c38ca6f`。

|domain|task|n|positive|AUC|LL|
|---|---|---:|---:|---:|---:|
|0|0|8639|817|0.6676879104|0.3015728042|
|1|0|87195|32928|0.7449946698|0.5727701725|
|2|0|4473|2758|0.6951774536|0.6172860982|
|0|1|8639|385|0.6690586854|0.1780302528|
|1|1|87195|22257|0.7536363792|0.4899220373|
|2|1|4473|2149|0.6884458328|0.6474213367|

```json
{
  "status": "COMPLETE",
  "method_id": "STGU_DOMAIN_ADAM_LL",
  "parent_run_id": "V3_RESUME_BASELINE_PILOT_kuairand_STGU_DOMAIN_ADAM_3101",
  "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
  "parent_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
  "validation_journal_sha256": "61ab729918865913fd593ccaa8dcf6bab3cdb851ffe58787c59f3a10f8b23d2e",
  "selection": "eligible>=12 minimum validation macro LL; exact ties earliest",
  "selected_epoch": 14,
  "selected_validation_macro_logloss": 0.4365090643517105,
  "selected_checkpoint_sha256": "3115a02e9a33c105459a407fbb94c352fb5b89d4ccd87282ae734ad818cbefb3",
  "historical_AUC_primary_preserved": true,
  "new_training_updates": 0,
  "selection_uses_Q_explore": false
}
```

## movielens / selected

epoch=12; macroAUC=0.8076594377; macroLL=0.4760514807。

checkpoint `/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_movielens_STGU_DOMAIN_ADAM_3101/ll_best.pt`；SHA256 `8fdb7de9d2c0a9dc565df0fa37c536562e64411d0cc05b3624d31dad6a6f1515`。

|domain|task|n|positive|AUC|LL|
|---|---|---:|---:|---:|---:|
|0|0|7797|4434|0.8029188396|0.5303601736|
|1|0|15930|8981|0.8053868187|0.5282316335|
|2|0|16281|9868|0.7969042286|0.5271636440|
|0|1|7797|1864|0.8117625990|0.4293265756|
|1|1|15930|3565|0.8201994500|0.4094642370|
|2|1|16281|3897|0.8087846905|0.4317626203|

## movielens / endpoint24

epoch=24; macroAUC=0.8147099616; macroLL=0.4772772186。

checkpoint `/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_movielens_STGU_DOMAIN_ADAM_3101/epoch24.pt`；SHA256 `84c5adaa46bb57224c4e40a0b387218c1395da6c9b97d82006e55a19605d78a2`。

|domain|task|n|positive|AUC|LL|
|---|---|---:|---:|---:|---:|
|0|0|7797|4434|0.8106586831|0.5325151410|
|1|0|15930|8981|0.8112780713|0.5301809868|
|2|0|16281|9868|0.8030432790|0.5292758228|
|0|1|7797|1864|0.8212631358|0.4263607484|
|1|1|15930|3565|0.8275961251|0.4137580860|
|2|1|16281|3897|0.8144204753|0.4315725265|

```json
{
  "status": "COMPLETE",
  "method_id": "STGU_DOMAIN_ADAM_LL",
  "parent_run_id": "V3_RESUME_BASELINE_PILOT_movielens_STGU_DOMAIN_ADAM_3101",
  "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
  "parent_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
  "validation_journal_sha256": "46c9e0ef2c176111e1961d4eef4eacd992a22c41cfb746f315d603486f9c4ac5",
  "selection": "eligible>=12 minimum validation macro LL; exact ties earliest",
  "selected_epoch": 12,
  "selected_validation_macro_logloss": 0.535544709014436,
  "selected_checkpoint_sha256": "8fdb7de9d2c0a9dc565df0fa37c536562e64411d0cc05b3624d31dad6a6f1515",
  "historical_AUC_primary_preserved": true,
  "new_training_updates": 0,
  "selection_uses_Q_explore": false
}
```

服务器总收据 `artifacts/mdmt_next_v3/v3_0/STGU_LL_SELECTION_CORRECTION.json`；SHA256 `0198e614dd69876c732d2fe9c617687c16d5aadd02b4d7627e44cd17c26e93b3`。
