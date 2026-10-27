# CODER IMPLEMENTATION REPORT — pipeline

实现依据为本目录 `DESIGN_SPECIFICATION.md`；所有新增实现位于 `experiments/mdmt_basp_v1/`，历史代码与历史结果未修改。

负责文件：`__init__.py`、`__main__.py`、`config.py`、`identity.py`、`roles.py`、`prefix.py`、`metrics.py`、`trainer.py`、`campaign.py`、`export.py`、原始收集器 `report.py`、`tests/test_pipeline.py`。模型、收益、关系、损失、权限、冲突检测和树事务由另一coder实现，已按共同API联调。

## 实现

- 原08/09加载并严格核对schema；开发种子2022/2023，锁定2024/2025/2026。P5实际最多24轨迹、10个共同前缀。条件候选B/C、连续方法、语义置乱、目标监督库与机会诊断未触发时不实现；CLI会明确拒绝这些未实现方法，避免将BASP-A冒充其他方法。
- 直接复用两个现有prepared目录，逐chunk对比真实CSV的标签、域和编码特征；重算T0词表集合/scaler并逐值比较，核对全部历史role数组。未运行新的摘要计算。
- 原始ML user_id用于probe两折/S三窗，KR使用row身份；S按域分层、组不跨窗。标签API检查purpose和行成员。S消费按每条方法轨迹独立记录，进入完整RAM快照；四候选只共同读取一次该窗。
- 固定融合native保留原初始化坐标及逐tensor相等检查，24个T0 epoch、40960 batch；V在12–24每轮选macroLL最小。owner pilot为12 pass，转换成64维标准化源表示并检查原预测等价。全部模型、优化器、selected/final和teacher只留RAM，磁盘仅测量、行数据、日志、配置和代码。
- 自然F batch2048保留尾批，连续排列流消费主批及6/10/14各8额外步；20轮总更新数强制核对。probe/owner/shadow流与teacher版本进入快照。FULL_MODEL_JOINT实时更新native，并保留冻结h0/z0用于固定坐标路由。
- 权限模块执行一次Adam；source全T0与全foreign F检查后投影，再评价V。结构分支继承完整Adam分组，新增center学习率保持.0003，旧m/v/step不重置。
- NEED的S覆盖使用冻结母状态的真实收益距离门（关闭收益聚类时使用固定几何距离门）及父责任；SOURCE使用F编译仿射门、foreign目标覆盖与父读取质量。四臂共同覆盖权重计算局部风险与自然贡献，检查S支持；被拒父冷却下一结构窗。
- 同状态selected/final导出F probe的实测收益、当前mu/q、动作身份与F群体画像，再导出对应Q状态。当前student与上一teacher的折内预测器诊断分别命名。所有对照可按row_id使用同一候选Q责任比较。
- 导出逐epoch/V、Q/E selected/final的原始logit/标签/row ID、六格/宏/自然/pooled风险、原型责任和软AUC、G及ESS/计数/未知先验、动作身份、source约束、CL及梯度权限、结构事务和成本。
- 固定状态干预和reader续训以已批准 `ADDENDUM_READ_INTERVENTIONS.md` 为准，由另一coder的 `interventions.py` 实现。已撤出先前临时两臂诊断；主管线在六臂核心训练全部完成后等待独立 `INTERVENTION_REVIEW_RECEIPT.json`，再单处调用 `run_read_interventions(trainer,which)`，不改变主训练轨迹。附录有自己的最小验收。
- 真实raw-X与缓存适配各测batch1/256的P50/P95；记录共享硬件条件。source-free部署跳过source前缀，实际source调用和线性MAC分项另列。

## 实际检查

`python -m pytest experiments/mdmt_basp_v1/tests -q`：核心28项通过。pipeline包含9项小测试，覆盖schema/种子、角色与每trajectory单次S、FP64 logit指标/单类AUC、完整RAM重放、自然尾批、分裂后的Adam分组恢复、source/need共同覆盖，以及实际Trainer.step→snapshot→KEEP→下一步loss/参数/Adam/RNG/全部游标一致；同一小用例还检查F群体注册、Q机制导出及原始收集器。

两个数据集直接验收均通过：

|数据|自然行数|T0|F|S|Q|V|E|
|---|---:|---:|---:|---:|---:|---:|---:|
|MovieLens|1000209|640133|48010|32007|40008|100021|100021|
|KuaiRand|2507666|1604904|120367|80246|100307|250767|250767|

legacy Q_confirm另保留ML40009/KR100308行。S三窗ML为11307/10313/10387，KR为26749/26749/26748；两数据组身份均不跨S窗。完整收据位于 `artifacts/mdmt_basp_v1/data_validation/`，worker读取 `artifacts/mdmt_basp_v1/data/<dataset>/DIRECT_VALIDATION.json`。

## 运行接口

`python -m experiments.mdmt_basp_v1 --artifact-root artifacts/mdmt_basp_v1 worker --dataset movielens --seed 2022 --device cuda:0`。

训练必须先有 `REVIEW_RECEIPT.json`，verdict为PASS/PASS WITH FIXES且blocking_open=false。worker顺序完成六臂P1、保留全部RAM状态并写READY，然后等待主agent的带证据阶段COMMAND；不会自行判断科研推进条件。最终freeze需要全部已启动worker的完整登记，evaluate单独释放历史E。

本实现阶段未启动正式GPU研究训练。B/C及其他未触发特殊候选需各自实现和审查后方可注册；RAM进程终止后相应共同前缀和方法状态需要重训。
