# 正式运行记录

- 采集源码提交：`05f75a73f3036694ee414b4483c878eac832f5b1`。
- 采集实现 hash：`e371a4c00352a9eba8f76b11c4f50f790c8c80cd02552feb4eb5a86502810aaf`。
- 自然人口、batch 40960、50 epoch，seeds 2022–2026。
- MovieLens/KuaiRand × clean/Gaussian D0，共20条逻辑轨迹。
- 主代理读取 guardian 核验：20条 COMPLETE、20次 worker 启动、1000次 epoch 提交。
- 正式逻辑 Adam：34,500次。训练进程退出码0。
- 正式训练墙钟：9,980.239秒；原始起止 Unix 时间为1793972102.111702和1793982082.3508604。
- 两数据集 smoke 均通过，共276次工程 Adam，单列为 `ENGINEERING_ONLY`。
- 无正式 worker 重启或逻辑轨迹重放。
- 采集完成时 guardian 参数 payload：2,550,757,800 bytes，保留至控制和冻结 Test 完成。
- 全程使用 GPU 0，最多两个 worker；调度按显存/功耗85%、主机RAM90%条件准入。
- 模型、Adam、影子和端点仅经 RAM IPC 保存。

原始依据为 `TRAINING_COMPLETE.json`、各job的50份epoch事务、`epoch_commit_ledger.jsonl`、两份 `SMOKE_*.json` 及 worker 日志。原始观测目录：`artifacts/ema_mechanism_20261106/attempt1/`。

预检首轮遇到 KuaiRand 验证域人数不等的表示错误；修订规则和测试见设计规格第11节及独立审查R3。该次预检发生在工程和正式 Adam 启动前。
