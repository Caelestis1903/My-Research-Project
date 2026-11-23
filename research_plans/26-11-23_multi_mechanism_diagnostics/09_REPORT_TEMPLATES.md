# 09 — 报告与机器工件模板

## 1. 机器可读清单

### RUN_MANIFEST.json
包含 study_id、plan_commit、source_branch、source_commit、code_sha、data_hash、protocol_hash、datasets、seeds、arms、anchors、panel identities、stage status。

### SEED_MANIFEST.csv
每行一个 dataset × seed，含 baseline trajectory ID、state receipt、source/config/data hashes。

### BASELINE_MANIFEST.csv
每行一个 baseline trajectory，含 model/Adam/EMA/RNG/sampler identity、epoch、global step、selected checkpoint。

### SNAPSHOT_MANIFEST.csv
每行一个 anchor，含 parent run、epoch、model hash、Adam hash、EMA hash、RNG hash、loader cursor、next batch IDs、live readback status。

### FORK_ENDPOINTS.csv
每行一个 fork，含 arm、dataset、seed、anchor、parent snapshot、A/B batch IDs、step count、optimizer transitions、endpoint hash、validation metrics、costs、status。

### CELL_METRICS.csv
每行一个 cell metric，包含 AUC、LogLoss、support、positive count、negative count、paired delta。

### M1_GRADIENT_CONSISTENCY.csv
每行一个 seed × dataset × anchor × arm × cell，包含 cosine、norm、direction residual、L_cons、P/U displacement。

### M2_SAMPLE_TASK_SCORES.csv
每行一个 sampled row × task × cell，包含 score、score rank、label support、predicted margin delta、observed fork delta reference。

### M3_GROUP_INTERACTIONS.csv
每行一个 group pair × anchor × seed，包含 A/B membership hash、batch size、five arm metrics、interaction residual。

### COSTS.csv
记录 wall seconds、GPU seconds、CPU seconds、peak VRAM、host RAM、forward/backward/optimizer steps、retry count。

### DECISION_RECORD.json
每个机制一个独立 decision，记录 stage、status、primary contrast、thresholds、seed values、limitations、next gate。

## 2. 人类可读报告

- S0_AUDIT_REPORT.md：来源、协议、模型、数据、参数组、资源、语义审计。
- S1_SCREENING_REPORT.md：M1/M2/M3 所有 arm 与候选，不只报告胜者。
- S2_CONFIRMATION_REPORT.md：V_B 独立确认及所有冻结候选。
- S3_SHORT_TRAJECTORY_REPORT.md：5 epoch 轨迹、阶段变化与成本。
- FINAL_DECISION_REPORT.md：机制排序、证据强度、NO_GO/blocked/triggered、是否进入 S4。

## 3. 必须报告的失败

所有失败 run、重试、OOM、缺类 cell、无效 pair panel、NOT_APPLICABLE 标签组合、技术暂停和未触发阶段均需列出。不得静默删除。
