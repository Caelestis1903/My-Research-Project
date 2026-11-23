# EXPERIMENT EXECUTION REPORT — A_HISTORY

## 1. Experiment ID

A_HISTORY — A21/A22 conditional-history controls.

## 2. Git/worktree state

Workspace `/home/xzf/Code/phase0-3` has no `.git` directory. Reviewed source hash: `05f22d38ebb9a5a7f494757d721c1b59a033b0f33e46713d074d33757405c62e`; review verdict: PASS.

## 3. Config

- A21: A21/PCOLL_STATIC_099, static condition sampling, λ=0.05, shared dense history scope.
- A22: A22/ADAM_S99, native Adam β1=0.99 for STAR_SKIP and EXPERT; β1=0.9 for other parameters.
- Both arms: original MovieLens and KuaiRand data, fixed M3oE configuration and 50 epochs; fixed EMA and online Validation checkpoints retained.
- Seeds per arm and dataset: 2022, 2023, 2024, 2025, 2026.
- Global steps per run: MovieLens 1,000; KuaiRand 2,450. Test evaluations: 0.

## 4. Seeds

All five registered seeds completed for both arms and datasets.

## 5. Run status

- Supervisor status: `COMPLETE`; failed keys: `[]`; pending keys: `[]`.
- Verified 50-epoch completions with matching RAM completion hashes: 20/20. Missing runs: none.
- Worker exits: 20; nonzero exits: 0.
- Selected model states were verified against the current RAM guardian before completion.

## 6. Raw metrics

Selected fixed-EMA Validation endpoints follow. All six cell AUC/LogLoss values, counts, domain/task metrics, pooled metrics, identities, and runtime fields are in [A_HISTORY_SELECTED_FIXED_EMA_RAW.csv](../../artifacts/conditional_history_shared_update_20261122/A_HISTORY_SELECTED_FIXED_EMA_RAW.csv).

| Dataset | Arm | Seed | Selected epoch | Macro AUC | Macro LogLoss | Domain 0 AUC | Domain 1 AUC | Domain 2 AUC | Task 0 AUC | Task 1 AUC | Log |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| kuairand | A21 | 2022 | 31 | 0.6836303029239902 | 0.4293324527457223 | 0.662727898817283 | 0.7159729950814635 | 0.6721900148732238 | 0.6768508654421012 | 0.690409740405879 | `artifacts/conditional_history_shared_update_20261122/9414fc92d8c2a0bd17551ef737a9b442c740ee611935fa8aa6efc7bd1bf62313.log` |
| kuairand | A21 | 2023 | 41 | 0.6912162069780635 | 0.42894770407293087 | 0.6838246416915358 | 0.7186289479664569 | 0.6711950312761978 | 0.682036864838262 | 0.7003955491178652 | `artifacts/conditional_history_shared_update_20261122/426a317b5dce58b3de80261645d324eda42754b636bbd655bf208e0497a3bf75.log` |
| kuairand | A21 | 2024 | 30 | 0.6917670242765777 | 0.43143066314241924 | 0.6928881811276698 | 0.7129825332818533 | 0.6694303584202097 | 0.6852385735845966 | 0.6982954749685586 | `artifacts/conditional_history_shared_update_20261122/ea59e585ad56e11ed25c03057427610d0a8a77d47306368d25b8f973eae90606.log` |
| kuairand | A21 | 2025 | 27 | 0.6897629667041404 | 0.4268465270380637 | 0.6782785172858544 | 0.7119662640727372 | 0.6790441187538295 | 0.6831675748916789 | 0.6963583585166019 | `artifacts/conditional_history_shared_update_20261122/e20b1f93690b085a0a0efc93b7a11bdced5a02f7f61b12f07826bdc40b158dc2.log` |
| kuairand | A21 | 2026 | 32 | 0.6912855868336503 | 0.4299843883908263 | 0.6838485778444781 | 0.7142254758098696 | 0.6757827068466032 | 0.683478008002408 | 0.6990931656648928 | `artifacts/conditional_history_shared_update_20261122/c7137b73758b0b9282532c309fa0f31126bc738a2cd971f62e56a0b4b06103ed.log` |
| kuairand | A22 | 2022 | 34 | 0.6898691513155822 | 0.42991104495356885 | 0.6782255599280356 | 0.7160394675791409 | 0.6753424264395702 | 0.6798973789221051 | 0.6998409237090594 | `artifacts/conditional_history_shared_update_20261122/55da6828f11f655a7f985a6b5f9d2bb10c1e18f0307f77ce382e6907e0c6e2c5.log` |
| kuairand | A22 | 2023 | 40 | 0.691491755845033 | 0.4291907262054518 | 0.6896117531797117 | 0.7179343218894826 | 0.6669291924659049 | 0.6814756507285713 | 0.7015078609614949 | `artifacts/conditional_history_shared_update_20261122/cdc829b439c4ab757ca0aa54e6e8d4dd0c3e8af4f36c87242ec7311e120c0a6c.log` |
| kuairand | A22 | 2024 | 50 | 0.6926371411503603 | 0.43034677502621177 | 0.6821890806919673 | 0.7184561314450789 | 0.6772662113140351 | 0.6819171338867888 | 0.7033571484139319 | `artifacts/conditional_history_shared_update_20261122/cdda3c10114251270fc22f9cc11616494d5e59269a4d65ee551e27538bd52889.log` |
| kuairand | A22 | 2025 | 32 | 0.6911976822858373 | 0.4463195956971347 | 0.6882557391658448 | 0.7137112195205273 | 0.6716260881711398 | 0.6837843437401373 | 0.6986110208315371 | `artifacts/conditional_history_shared_update_20261122/36196b0b30a18486d38b7b400ce05abc9d4d3cbf4133e582006c1e12075b4d01.log` |
| kuairand | A22 | 2026 | 33 | 0.6877645656501955 | 0.424912579420664 | 0.6733207300496655 | 0.7133479166686589 | 0.6766250502322622 | 0.6801990620813002 | 0.6953300692190908 | `artifacts/conditional_history_shared_update_20261122/e87c4e8eb7973c9a0da01d96fd0ef7914863e1e9d1cf2782c5904545e51a32e2.log` |
| movielens | A21 | 2022 | 21 | 0.760488466873542 | 0.5223597966593391 | 0.7702133568967837 | 0.7569665665607638 | 0.7542854771630784 | 0.76027660417114 | 0.7607003295759439 | `artifacts/conditional_history_shared_update_20261122/3a7fd40e13456d11e70156b52c99ceb9ff10b894f49291f034adec893e4397ef.log` |
| movielens | A21 | 2023 | 20 | 0.7599321286698472 | 0.5241025494920136 | 0.768904034464504 | 0.7568201281279359 | 0.7540722234171016 | 0.7597802888554964 | 0.7600839684841979 | `artifacts/conditional_history_shared_update_20261122/4d7ec2bae1fb7aa953fa9e9c9de03292efe5f779a109dd7d0e5d7fd70014b9ee.log` |
| movielens | A21 | 2024 | 22 | 0.7592404765050874 | 0.5235873806203406 | 0.7687890448682394 | 0.7549490369047489 | 0.7539833477422742 | 0.7590872885754344 | 0.7593936644347407 | `artifacts/conditional_history_shared_update_20261122/56b541f85cb3738b8622f70d00c47843e7cfe1a89dfdaff1457fa1f2a6a3f3d1.log` |
| movielens | A21 | 2025 | 21 | 0.7594891881654262 | 0.5238320661070797 | 0.7678944176387958 | 0.756660937461483 | 0.7539122093959995 | 0.7588145740990649 | 0.7601638022317875 | `artifacts/conditional_history_shared_update_20261122/d1e54ac267d307aaa86347a8d35124b028dd9a2910cfde3bec90ab5d9498f217.log` |
| movielens | A21 | 2026 | 22 | 0.7597995630844067 | 0.5269348823109966 | 0.7684912239654496 | 0.7566067492994764 | 0.7543007159882936 | 0.7590066540144312 | 0.7605924721543817 | `artifacts/conditional_history_shared_update_20261122/9cb89e3cf8d1e73486a5b096d7f0c0f9797070131155d65797e0807acdd40b6c.log` |
| movielens | A22 | 2022 | 35 | 0.7597641402957258 | 0.5222203212032892 | 0.7690766293727676 | 0.7564722047194103 | 0.7537435867949995 | 0.7596430519391976 | 0.759885228652254 | `artifacts/conditional_history_shared_update_20261122/348a5fa24c060ca264ce999179c79e813ae577fe39734753b0538b65b089c6b9.log` |
| movielens | A22 | 2023 | 38 | 0.7591494930958468 | 0.5225005316672208 | 0.7683433259428243 | 0.756043116686463 | 0.7530620366582531 | 0.7591763975502465 | 0.7591225886414471 | `artifacts/conditional_history_shared_update_20261122/f665f166b4fcc71d3c0a36477de90cbc5a93b06759f960bf44e09b52f78f921b.log` |
| movielens | A22 | 2024 | 38 | 0.7592624568076461 | 0.5231007516512042 | 0.7691944330367275 | 0.7552418512938271 | 0.7533510860923842 | 0.7598901967828392 | 0.7586347168324533 | `artifacts/conditional_history_shared_update_20261122/b8a5f8cc1bdbe15938843c861c15599cddb18c7705c09b2c681a681e9706f8d7.log` |
| movielens | A22 | 2025 | 32 | 0.7601472240455659 | 0.5217169687900495 | 0.7702511469176387 | 0.7566567675947133 | 0.7535337576243455 | 0.7603146933059138 | 0.7599797547852178 | `artifacts/conditional_history_shared_update_20261122/ff477b203dd1b69369b24b84c476e1bacff1f6731b001cefc606c772afb2b351.log` |
| movielens | A22 | 2026 | 31 | 0.7597642449384687 | 0.520145782687857 | 0.7682844982596162 | 0.7565746628684843 | 0.7544335736873056 | 0.7596996709563019 | 0.7598288189206355 | `artifacts/conditional_history_shared_update_20261122/82fd89f5fbd767a0c7d4ed4e7a1c77b430002915fbeb50207a0410c49c568439.log` |
## 7. Log locations

- Full per-worker stdout/stderr logs: `artifacts/conditional_history_shared_update_20261122/{job_key}.log` (20 files).
- Full per-job raw result records and RAM verification markers: `artifacts/conditional_history_shared_update_20261122/attempt1/runs/{job_key}/result.json` and `COMPLETE_VERIFIED.json`.
- Supervisor receipt: `artifacts/conditional_history_shared_update_20261122/A_HISTORY_RUN.json`.
- Admission and exit telemetry: `artifacts/conditional_history_shared_update_20261122/A_HISTORY_admissions.jsonl`.

## 8. Runtime errors

None recorded. No `A_HISTORY_PAUSE` file was present at completion.

## 9. Missing runs

None. All 20 registered runs completed and verified.
