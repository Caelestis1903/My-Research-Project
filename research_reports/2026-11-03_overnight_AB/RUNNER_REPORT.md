# RUNNER_REPORT — Overnight A/B 2026-11-03

## 运行与完成状态

- Attempt: `artifacts/mdmt_overnight_ab_20261103/attempt1`。
- 运行区间（UTC）：2026-11-03 16:30:08 UTC — 2026-11-03 19:47:53 UTC；墙钟时长 11864.2 秒（3.296 小时）。
- 执行环境：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`；设备参数 `0,1`；Smoke 检查 6/6 PASS。
- 数据集：MovieLens、KuaiRand；种子：2022–2026；新训练 family：HG、AutoIFS、CP。
- 冻结预算与已提交epochs：HG 500/500（10轨迹×50）；AutoIFS 600/600（10轨迹×搜索10+最终50）；CP 500/500（10轨迹×50）。
- Queue 最终状态：`FINAL_COMPLETE`；核心新训练身份 DONE 30/30；全部 queue 状态 CONDITIONAL_NOT_ENABLED=20, DONE=30, REUSE_DONE=30.
- 端点覆盖：A 30/30；B 40/40；`missing_A=[]`, `missing_B=[]`。
- 家族 DONE 轨迹：HG 10/10；AutoIFS 10/10；CP 10/10。MovieLens/KuaiRand 均各为 5/5 seeds。
- Coordinator：PID 1189780 已退出；`progress.json`终态更新时间 2026-11-03 19:47:53 UTC.
- 本 attempt guardian：PID 1189863 收到 `shutdown` 并返回 `SHUTDOWN`；进程已退出，Unix socket 已删除；关机前状态为30/30 DONE、snapshot_bytes=0、无活跃worker。
- 权重/优化器/checkpoint：attempt下扫描 `.pt/.pth/.ckpt/.safetensors/.model` 文件数为0；guardian释放后 snapshot_bytes=0。

## 重放、错误与调度记录

- Epoch commit ledger：1600行；唯一 `(job_id, commit_index)` 为1600/1600；身份commit_index连续到冻结预算。state-loss replay：0。
- Guardian `failure_count`：0；`failures.jsonl`及launch `failure.json`：均无记录。
- Worker dispatch：1612次；记录物理launch/epoch commit均1600；另有12次在资源门禁处以 `WAIT_RESOURCE: GPU has an unapproved external owner` 退出，未提交epoch，协调器随后继续派发。
- worker.log 中对应上述WAIT_RESOURCE的Traceback共12个；它们计为资源门禁拒发，不计入训练失败。结束时所有核心job均DONE。

## 累计实际成本（新训练）

- 新训练轨迹成本记录：30条；轨迹 `worker_seconds` 合计 115395.220 秒；逐epoch物理 `physical_worker_seconds` 合计 119283.414 秒（1600条）。
- A_HG/AutoIFS（20条轨迹）只提供汇总 `worker_seconds`，不拆分 training/validation；A 的 `steps` 合计 891300，`optimizer_calls` 合计 925600（这是20条发表模型轨迹的 Adam 调用数）。
- B_CP（10条轨迹）单独记录 `training_seconds` 合计 7326.894 秒、`validation_seconds` 合计 517.664 秒、训练 GPU event 时间合计 7209.624 秒；`optimizer_steps` 合计 17250。CP 的 optimizer_steps 计共享自定义更新与 ordinary Adam 更新。
- 按配置字段相加的全局训练步/更新数为 A `steps` 891300 + CP `optimizer_steps` 17250 = 908550；此口径与 A 的 925600 `optimizer_calls` 不同。`label_exposures` 合计 4490078400。单轨迹和物理launch成本见 `COSTS.csv`。

## 每轨迹原始结果

以下endpoint标量均为 `checkpoint=selected, split=test`；`estimator` 列明确标注表中估计器。CP 每条轨迹同时有 online 与 fixed_ema 的 selected/test 端点，本表列 online；CP fixed_ema 及其余逐端点记录见 [`ENDPOINTS.csv`](ENDPOINTS.csv)。表格含宏观AUC/logloss、逐domain/task宏平均AUC/logloss及endpoint model hash，成本见 [`COSTS.csv`](COSTS.csv)。

| job_id | 状态 | epochs | steps | endpoints | estimator | 轨迹worker秒 | 物理worker秒 | macro AUC | macro logloss | D0 AUC | D1 AUC | D2 AUC | T0 AUC | T1 AUC | D0 LL | D1 LL | D2 LL | T0 LL | T1 LL | model_hash |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| kuairand.AUTOIFS.s2022 | DONE | 60 | 29400 | 6 | online | 7558.133 | 7703.309 | 0.681693 | 0.428069 | 0.654098 | 0.714614 | 0.676366 | 0.671062 | 0.692324 | 0.232467 | 0.415974 | 0.635767 | 0.472329 | 0.383810 | 24e0b5f2f54bf433fac5b2258a8a34774ba3c1b1803354dd8593050db838375a |
| kuairand.C_MIX_P01_SERIAL_V1.s2022 | DONE | 50 | 2450 | 12 | online | 1093.438 | 1256.619 | 0.668603 | 0.433611 | 0.631727 | 0.710297 | 0.663786 | 0.656154 | 0.681053 | 0.236831 | 0.418115 | 0.645888 | 0.477340 | 0.389883 | 4eff0cfc9f019c6d1c193fbf3a1bf6e14d53883b944fdaa0aeed15f2ff618afa |
| kuairand.HYPERGATE.s2022 | DONE | 50 | 98000 | 6 | online | 10799.220 | 10990.982 | 0.684181 | 0.437831 | 0.669689 | 0.713407 | 0.669448 | 0.674231 | 0.694131 | 0.233653 | 0.435131 | 0.644710 | 0.482111 | 0.393552 | 884e0bb8a62f25ce13a3951f8451551af501a0f605908691c63a9a7562eda51e |
| kuairand.AUTOIFS.s2023 | DONE | 60 | 29400 | 6 | online | 7445.039 | 7592.381 | 0.673517 | 0.436171 | 0.639658 | 0.712828 | 0.668064 | 0.664114 | 0.682919 | 0.233957 | 0.431877 | 0.642680 | 0.480916 | 0.391426 | 9cfa9b222b5ec9cad017e8ca7ae99172789e9bc1606eb88021a9a35293bcf0eb |
| kuairand.C_MIX_P01_SERIAL_V1.s2023 | DONE | 50 | 2450 | 12 | online | 1115.993 | 1267.767 | 0.670975 | 0.436829 | 0.637794 | 0.710833 | 0.664298 | 0.655236 | 0.686714 | 0.236409 | 0.426263 | 0.647815 | 0.484709 | 0.388949 | 30c5760ba8a00b683845abb2028396fc8f9f998c5ac3a0dde9b79ce92c877af8 |
| kuairand.HYPERGATE.s2023 | DONE | 50 | 98000 | 6 | online | 11075.566 | 11221.102 | 0.682633 | 0.435689 | 0.666130 | 0.708539 | 0.673231 | 0.671329 | 0.693938 | 0.233644 | 0.430460 | 0.642964 | 0.482633 | 0.388745 | 301d683802cbb1aa979a7ec36c2f816e387520fe473668b2d50a1b6426896f81 |
| kuairand.AUTOIFS.s2024 | DONE | 60 | 29400 | 6 | online | 7445.370 | 7578.573 | 0.676996 | 0.433047 | 0.644277 | 0.713885 | 0.672827 | 0.666883 | 0.687109 | 0.233761 | 0.427328 | 0.638050 | 0.478185 | 0.387908 | e4c3f016b0775daf01430be427ef7ed34c434301367cbd4441171a4aac508a07 |
| kuairand.C_MIX_P01_SERIAL_V1.s2024 | DONE | 50 | 2450 | 12 | online | 1174.787 | 1376.282 | 0.676894 | 0.441009 | 0.648580 | 0.712218 | 0.669883 | 0.663934 | 0.689853 | 0.236413 | 0.440904 | 0.645710 | 0.486335 | 0.395683 | 090a2fe342ee02412fc40184e8a7cac39f8fba7dde06d5378d4f8292bce9ebea |
| kuairand.HYPERGATE.s2024 | DONE | 50 | 98000 | 6 | online | 11002.447 | 11114.901 | 0.680227 | 0.441762 | 0.658524 | 0.716362 | 0.665797 | 0.671300 | 0.689154 | 0.236831 | 0.435940 | 0.652516 | 0.487968 | 0.395556 | 0b37c7cd01e73d234bd7dbcea279ea221a7330ac15f638b89758000aa57cc60e |
| kuairand.AUTOIFS.s2025 | DONE | 60 | 29400 | 6 | online | 7068.067 | 7185.321 | 0.679449 | 0.432044 | 0.645025 | 0.715293 | 0.678028 | 0.670045 | 0.688852 | 0.233829 | 0.427748 | 0.634554 | 0.476838 | 0.387249 | cea728b06b2268cb93d6f283a65156402c6bb480333a4c785c703fa8c6b139d5 |
| kuairand.C_MIX_P01_SERIAL_V1.s2025 | DONE | 50 | 2450 | 12 | online | 1159.806 | 1340.967 | 0.673146 | 0.440431 | 0.644078 | 0.713160 | 0.662200 | 0.664588 | 0.681704 | 0.237224 | 0.429635 | 0.654435 | 0.483670 | 0.397193 | dea56814f912b38a10b49a0e197832c85081a51b287d23b8282d7696d6dfa19c |
| kuairand.HYPERGATE.s2025 | DONE | 50 | 98000 | 6 | online | 10544.685 | 10699.981 | 0.681615 | 0.445444 | 0.662241 | 0.714276 | 0.668329 | 0.672757 | 0.690473 | 0.238199 | 0.438819 | 0.659315 | 0.491591 | 0.399297 | af1678bde4626700fcdd123e2eedfd0ed908ea9aeabf1c70bb784bfce87148f5 |
| kuairand.AUTOIFS.s2026 | DONE | 60 | 29400 | 6 | online | 6262.751 | 6373.535 | 0.681331 | 0.435630 | 0.664965 | 0.712917 | 0.666110 | 0.666429 | 0.696233 | 0.231250 | 0.431105 | 0.644534 | 0.482463 | 0.388796 | 409079849a79cb91002411ddb1e99f69b319d0fb5d010177e5f7df015e2f4844 |
| kuairand.C_MIX_P01_SERIAL_V1.s2026 | DONE | 50 | 2450 | 12 | online | 1163.894 | 1368.970 | 0.670049 | 0.438700 | 0.639814 | 0.712545 | 0.657789 | 0.658228 | 0.681871 | 0.236623 | 0.425738 | 0.653740 | 0.484145 | 0.393255 | 50fac14f48501e30e74561a0f778429c7d9448c01f6b8b4347aa5030e74d864e |
| kuairand.HYPERGATE.s2026 | DONE | 50 | 98000 | 6 | online | 9904.590 | 9974.322 | 0.681906 | 0.442221 | 0.669074 | 0.715880 | 0.660763 | 0.669580 | 0.694231 | 0.238516 | 0.430996 | 0.657152 | 0.485473 | 0.398970 | 6843ab9408dada6fb5eb2124d252d68f756f7a10b3650627e37ad3343a066160 |
| movielens.AUTOIFS.s2022 | DONE | 60 | 11760 | 6 | online | 1264.847 | 1348.253 | 0.772704 | 0.494248 | 0.776813 | 0.782938 | 0.758363 | 0.765012 | 0.780397 | 0.493409 | 0.479645 | 0.509691 | 0.579945 | 0.408552 | c9f8caa234c98aff468c8f8e53d25c41b8f3311e2d4e71d305497cad5ad88453 |
| movielens.C_MIX_P01_SERIAL_V1.s2022 | DONE | 50 | 1000 | 12 | online | 396.244 | 503.713 | 0.772328 | 0.497079 | 0.776983 | 0.785584 | 0.754416 | 0.764295 | 0.780360 | 0.496640 | 0.481499 | 0.513098 | 0.582248 | 0.411910 | bcd6c82a660b081e53ffbb9028eb49f8f47621053306461d691939326a179f70 |
| movielens.HYPERGATE.s2022 | DONE | 50 | 39100 | 6 | online | 2503.696 | 2592.039 | 0.771282 | 0.496069 | 0.773343 | 0.782936 | 0.757567 | 0.763303 | 0.779261 | 0.500759 | 0.479638 | 0.507809 | 0.583065 | 0.409073 | aa3473efabd7fa4e39955dd40c49e49c918bc723717a0badc5c1757361d63702 |
| movielens.AUTOIFS.s2023 | DONE | 60 | 11760 | 6 | online | 1329.220 | 1430.628 | 0.771410 | 0.496185 | 0.776055 | 0.781777 | 0.756399 | 0.763387 | 0.779433 | 0.494796 | 0.482496 | 0.511265 | 0.582378 | 0.409993 | d9361e425e80b6d00798f994e5b6dfbd8be4dad29dc6d267776cb35b5365a023 |
| movielens.C_MIX_P01_SERIAL_V1.s2023 | DONE | 50 | 1000 | 12 | online | 443.628 | 584.483 | 0.773192 | 0.500689 | 0.773031 | 0.787109 | 0.759436 | 0.764311 | 0.782072 | 0.507384 | 0.483952 | 0.510730 | 0.586571 | 0.414807 | 377db3113396417211ad6f4aaf48e423cebb99fbb91a06395366aad166938ea5 |
| movielens.HYPERGATE.s2023 | DONE | 50 | 39100 | 6 | online | 2750.492 | 2864.497 | 0.773587 | 0.499396 | 0.779347 | 0.784159 | 0.757255 | 0.765310 | 0.781864 | 0.501173 | 0.481838 | 0.515178 | 0.590000 | 0.408793 | 133f9cf8ecc0b82c15833c2333dd3f852abf717bb05860a6868db7ff7f9d4c80 |
| movielens.AUTOIFS.s2024 | DONE | 60 | 11760 | 6 | online | 1294.566 | 1385.136 | 0.773643 | 0.494412 | 0.777323 | 0.783755 | 0.759850 | 0.765362 | 0.781924 | 0.493956 | 0.480458 | 0.508821 | 0.580612 | 0.408211 | 0b81f737804f4d451b911b43bfd30dbcf61a10a57a88297bfb6f3e3675082951 |
| movielens.C_MIX_P01_SERIAL_V1.s2024 | DONE | 50 | 1000 | 12 | online | 451.271 | 603.774 | 0.770963 | 0.492305 | 0.772844 | 0.783577 | 0.756469 | 0.762988 | 0.778938 | 0.492129 | 0.477108 | 0.507680 | 0.579209 | 0.405402 | 06930c567e47caae7fdbac1a74ebf2f8e79182e81457b1d3e5d5e5a81deedc6e |
| movielens.HYPERGATE.s2024 | DONE | 50 | 39100 | 6 | online | 2601.086 | 2711.736 | 0.770614 | 0.496393 | 0.773393 | 0.783189 | 0.755259 | 0.762756 | 0.778472 | 0.499551 | 0.480002 | 0.509626 | 0.582094 | 0.410692 | 0f2971af9433b0adda9347c0d18dee66cba9ed61029da7192e7a0358c3d3c1dd |
| movielens.AUTOIFS.s2025 | DONE | 60 | 11760 | 6 | online | 1171.974 | 1264.253 | 0.772572 | 0.496959 | 0.776922 | 0.783016 | 0.757777 | 0.764979 | 0.780165 | 0.497118 | 0.482057 | 0.511702 | 0.582384 | 0.411534 | 586e0f5a9d9489404bf59e20446cf1a648d1eb739ffb4a6dfca6f5d199fa1ee9 |
| movielens.C_MIX_P01_SERIAL_V1.s2025 | DONE | 50 | 1000 | 12 | online | 423.554 | 567.724 | 0.772832 | 0.493584 | 0.774536 | 0.785967 | 0.757993 | 0.764506 | 0.781158 | 0.494635 | 0.479159 | 0.506958 | 0.576630 | 0.410538 | 4e7d2091cdc5e36c175478243d6eafbcd4bf6056d6dd50bf25b114497970ee8e |
| movielens.HYPERGATE.s2025 | DONE | 50 | 39100 | 6 | online | 2497.729 | 2609.440 | 0.770431 | 0.499422 | 0.771393 | 0.785517 | 0.754385 | 0.761542 | 0.779321 | 0.508475 | 0.480880 | 0.508910 | 0.590441 | 0.408402 | fe835abe9f747245ea41f8bd9fd65d079e314e166cf8bd2c385ea07d125e522a |
| movielens.AUTOIFS.s2026 | DONE | 60 | 11760 | 6 | online | 923.881 | 1005.123 | 0.771793 | 0.498667 | 0.776227 | 0.781997 | 0.757156 | 0.763514 | 0.780073 | 0.499543 | 0.484510 | 0.511948 | 0.586132 | 0.411202 | 493a5402ddb7fb709793a964291e426200829769db89a76d218033074a93ea76 |
| movielens.C_MIX_P01_SERIAL_V1.s2026 | DONE | 50 | 1000 | 12 | online | 421.947 | 557.954 | 0.771849 | 0.494686 | 0.775778 | 0.781522 | 0.758247 | 0.762670 | 0.781029 | 0.494891 | 0.484268 | 0.504898 | 0.584068 | 0.405303 | 85a3fab3c03f8def42c8f47f3335a1753363f72885562421a5d1f9afbdce3900 |
| movielens.HYPERGATE.s2026 | DONE | 50 | 39100 | 6 | online | 2107.297 | 2209.650 | 0.771144 | 0.501490 | 0.774673 | 0.784744 | 0.754017 | 0.763711 | 0.778578 | 0.503789 | 0.480037 | 0.520643 | 0.588403 | 0.414577 | be0d2d3eed222a9fbcf84e4fae26bea4ed75cd62685a13a8fd5e92581b077010 |

## 日志与原始数据位置

- Coordinator日志：`artifacts/mdmt_overnight_ab_20261103/attempt1/coordinator.log`。
- Guardian日志：`artifacts/mdmt_overnight_ab_20261103/attempt1/guardian.log`；各轨迹的endpoint seal/hash在 `runs/<job_id>/DONE.json`、`endpoints.json`、`cost.json`。
- Scheduler记录：`queue_state.json`, `events.jsonl`, `epoch_commit_ledger.jsonl`, `progress.json`。
- 原始端点和汇总CSV：`ENDPOINTS.csv`, `ABSOLUTE_SUMMARY.csv`, `COSTS.csv`；逐seed记录均保留。
- Smoke记录：`artifacts/mdmt_overnight_ab_20261103/attempt1/SMOKE.json`。
- FAILED job/epoch：0；未完成或缺失核心轨迹：0。
