# AC-V9：完整阶段评价补充规范

## DESIGN

历史 V9 的 Q41 比较统一 E1–16、完整 E1–50 的 AULC；自然迁移目前的 selected/final 和五个公共点不足以回答早期效应是否持续。八次训练已经全部完成，原生产器保留每轮模型。本补充只评价这些冻结模型，不增加训练、选模或超参数。

- 固定全部 SN/SB × 2022–2025，各自 E1–50，共 400 个模型。
- 一次性冻结各 run 的 config、训练 receipt、selected rule 及 50 个模型文件 SHA256。评价写入独立 `V9_trajectory_readout/`，不得更新原训练目录或其 receipt。
- 使用原 fixed_model、原 predict、原 safe_auc/safe_logloss、原自然 test 数组；每个检查点不构造 optimizer，不调用 backward/step。保存完整预测及逐域任务指标。
- 对应原窗口，汇总 E1–16、E1–50 的算术平均 AUC/LogLoss；同时保留原 E1–10、E1–20 和 E50/selected。窗口固定，不根据自然 test 结果改边界。
- AULC 是各轮指标的均值，不是汇合预测 AUC，也不是梯形积分。逐任务/域、四种子及不一致方向全部报告。
- 历史人工 MovieLens 每轮 5 步，自然每轮 20 步；16 轮对应 80/320 步，不将共同轮次解释为共同优化进度。SGD .3 和 Adam、SGD .1 协议不得混合。
- 历史 test 已被反复使用；本次也已经读取自然 selected/final，完整曲线补读属于探索性复现，不能称新的盲测或独立确认。仅按冻结历史窗口比较，不能用于调整配置。

## IMPLEMENT / REVIEW

实现：`experiments/natural_ml_extensions/v9_readout.py`。主代理审查必须确认全八 run 完成、每轮文件存在、SHA 绑定、只 load model/eval/predict、与已完成训练工件分目录、输出逐域任务和预测。根据 `start.md`，由主代理自审，`independent_review=false`。

## RUN / ANALYZE

主代理完成 freeze 后，luna_worker 仅机械执行八个 readout。遵守每卡最多五个研究进程及显存限制。主代理复算预测指标、比对原公共点，再进行阶段分析。无新增训练；不得把 400 个模型当 400 个独立随机重复。
