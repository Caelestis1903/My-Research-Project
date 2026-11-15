# STAGE 2 CODER IMPLEMENTATION REPORT

实现完成：2026-11-15。最终 Stage2 训练源码 hash：`831cba9947738652cc6d4e5abf864e1ab4e32545b304936cb39aba28b227893b`。
本报告记录实现和工程检查；正式10条训练未启动。

## 范围与实现

代码只新增于 `experiments/auxiliary_objective_attribution_20261115/stage2/`：

| 文件 | 设计映射 |
|---|---|
| `core.py` | 全自然 batch 的两个 task-BCE 敏感度视图；近零方向 delta=0 并保留 entry；`.75*clean+.25*.5*(view0t0+view1t1)`；12cell-label 自然权重与全2B支持；CUDA同步计时与紧凑扰动/drift摘要 |
| `protocol.py` | 独立 NDP-A 配置/job/source identity；Stage1 final decision/review、设计、源和support/weight表 provenance；只读 cache；索引10条既有 CLEAN 证据 |
| `runner.py` | 独立10条完整预算调度与worker；继承原loader/Adam/EMA/val选点/guardian；真实 full-batch NDP 与 zero/source-clean smoke；无release/HPO/Test入口 |
| `tables.py` | 20条轨迹（10新+10CLEAN）的selected/E10/E30/E50、两readout、六格/域/任务/Macro/Traffic/Pooled/worst；NDP-A−CLEAN五seed配对原值、mean/SD/95%CI及总EMA/clean-online比较 |
| `test_core.py` | 5项紧凑CPU检查，覆盖新目标、自然支持、zero counterpart和恢复身份 |
| `README.md` | 执行命令、数据日志与RAM状态生命周期 |

未改 Stage1 五个冻结训练文件、vendor、Stage1已冻结输出/报告/decision或全局指标台账。Stage1 hash仍为 `a78bce2f713cf62f9af32548b50bc360b134420ff9ed6ac6990dca22d76a2ad0`。未重扫 Stage1 原始support流。

Stage1 最终decision的原文件 SHA256为 `5845910e92cd37a97725dca86878c077546e8dca5a465b818f06549a68f652c1`；独立审查PASS且无阻塞项，明确推进固定alpha=.03、mixture lambda=.25的NDP-A。Stage2清单绑定该decision/review、冻结设计、Stage1源收据和支持/权重表的路径及hash。

## 目标、状态和日志

每个原始 row-task 的clean/view系数分别为 `.75/(2B)` 和 `.25/(2B)`，总系数保持 `1/(2B)`；每task质量.5、全部质量1。没有pair采样、acceptance剔除、cell二次归一化、projection或额外BCE质量。最后短batch按实际B归一；近零方向保留完整自然计数和权重。

只复用Stage1已审查的forward/sensitivity，probe临时重接gate取得完整数值Jacobian，参数训练维持原gate detach。两view的delta均detach；仅各view自身target-task BCE进入loss。每batch一次zero_grad、combined backward、Adam、after-Adam EMA。两套delta都零时直接返回source clean BCE图，避免FP32不同归约/反传顺序影响CLEAN复用身份。

batch日志只保存原始行序hash、B/2B、12cell-label计数/自然系数质量/BCE贡献，以及每task/cell的zero数量和delta/target/non-target drift的count/mean/median/p90/p99/max摘要。没有81920-entry数组。probe、构造与训练阶段使用CUDA同步wall time；validation单独计时。每epoch保留两readout的完整val指标、独立最早严格最大选点及state/order/log hash、loss/权重汇总、RAM/吞吐。

model、Adam、EMA、RNG、loader、best和selected状态只提交给独立RAM guardian；Stage2没有release命令。没有写模型、优化器或checkpoint文件，也没有复制CSV/cache。

## 正确性检查与实测资源

项目Python：`/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

- `python -m pytest -q experiments/auxiliary_objective_attribution_20261115/stage2/test_core.py`：**5 passed，5.75秒**。覆盖显式2B凸混合loss/参数梯度、task-specific目标、.75/.25/总质量1、短batch/缺域/单标签及近零fallback全支持、zero显式loss/gradient、zero/source clean exact单步状态、NDP和zero的小步RAM恢复/selected身份；zero连续轨迹的model/Adam/EMA与source clean完全相同。
- `runner prepare`：**PREPARED**。Stage1触发/provenance核验通过；10条CLEAN的已复制result按SHA256索引，不复制旧trace或数据。
- 工程直接调用 `runner.smoke(stage2_attempt1, 'cuda:0')`：**PASS**。等待cuda0功耗/显存准入窗口后，两个数据集均完成真实40960行NDP单次backward，另验证canonical zero与source clean的model/Adam/EMA三个state hash完全相同。每项更新数为1，初始hash匹配相应复用CLEAN seed2022。实际full-batch 2B=81920，clean/view/total质量=.75/.25/1。

| 数据集 | NDP峰值allocated bytes | NDP峰值reserved bytes | NDP step wall秒 | probe秒 | 构造秒 | 训练/ledger/Adam/EMA秒 |
|---|---:|---:|---:|---:|---:|---:|
| MovieLens | 2500704256 | 2520776704 | 1.360319 | .274149 | .054810 | .817471 |
| KuaiRand | 2905653248 | 3017801728 | 1.300548 | .232658 | .039776 | .758781 |

冻结Stage1检查helper用于实际四行probe：两数据FP64有限差分最大绝对误差为2.776e-11/2.870e-11，FP32/FP64梯度最大差1.607e-7/2.480e-7，无关行梯度最大差1.490e-7/2.384e-7，均通过继承容差。没有重新扩展旧数值矩阵。

检查收据位于 `artifacts/auxiliary_objective_attribution_20261115/stage2_attempt1/` 的 `CPU_CHECKS.json`、`SMOKE_REPORT.json`、`zero_counterpart_receipt.json`、`reused_clean_index.json`、`source_receipt.json`、`run_manifest.json` 和两份data manifest。`zero_counterpart_receipt.json` 为PASS，记录CLEAN同时作为canonical zero-delta counterpart的完整检查与state hash证据。

## 交接状态

独立reviewer检查最终源码hash、设计、此报告和smoke后，主agent可派发 **10条新轨迹、500epochs、17250次Adam/EMA更新**。batch40960、50epochs、两数据集及五seed预算保持不变。调度按每卡显存/功耗<85%、系统RAM<90%准入，并保留完成后的selected RAM状态。

HPO、Test、NDP-B和新保护机制未实现/未运行；原始数值导出不写scientific decision，后续科学分析由主agent完成。
