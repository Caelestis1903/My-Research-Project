# CODER IMPLEMENTATION REPORT — 模型、收益、权限与事务

实现依据：`DESIGN_SPECIFICATION.md`执行版及原02—05、08合同。所有新增实现位于`experiments/mdmt_basp_v1/`，历史代码未改写。

## 实际实现

- `model.py`：标签无关Packet前向、目标残差、收益需求路由、双角色source/need树、任务有向读出、NULL、六开关、保函数NEED/SOURCE分裂、READER/WIDE容量对照与实际新增参数/MAC。
- `benefit.py`：固定状态精确logit BCE差、纯测量接口、独立冻结teacher、固定收益RMS、teacher责任的关系测量、合法对比资格、RAM TeacherCache与事务继承坐标对齐。
- `relation.py`：带任务/方向/版本的原量纲G、二阶矩、unique row计数、ESS、observed/prior/unknown、守恒读质量、全局有向关系矩阵的确定性加权ALS工具。
- `losses.py`：自然有效标签BCE、gain/cluster/合格contrast、真实owner与pilot KL、只训练轮转reader的shadow，以及完整T0/foreign F支持的风险/RMS投影。
- `permissions.py`：逐loss参数白名单、共享trunk辅助梯度上限、阶段冻结、稳定命名Adam组与动态新增参数（旧状态保留，新状态首次更新前不存在）。
- `conflict.py`：BCE-only逐样本source U梯度的16维固定草图、两折标准化ridge、可部署分区、支持/ESS/时间稳定与F半面板风险方向、父责任加权、跨target动作身份独立、拒绝冷却排除接口。
- `transaction.py`：从同一RAM状态fork KEEP/READER/WIDE/SPLIT，同预算八步、投影后一次S面板读取，按预设风险与容量门选择并返回赢家完整快照；WIDE胜明确记录普通容量增量。

`model.forward(packet, domain)`不接收标签/真实gain。普通NONE部署前向不执行source instrument；训练probe通过独立`measure_output`测量相同source动作库。SOURCE_WRITE=false只detach foreign内容，owner仍训练。FULL_MODEL_JOINT的current_native由pipeline挂载；辅助分支detach当前h/z，固定phi和h0/z0保留参考坐标。

## 与pipeline确认的接口

`make_model(cfg,arm,teacher_w,teacher_b,seed)`；Packet为h/z/phi及可选domain/h0/z0。

`refresh_teacher(...,panels=每行0/1,version=明确版本,freeze_scale=...)`返回`TeacherCache`，支持`.subset()`、`.to()`；`measure_benefits`纯测量，不能拟合Q/E关系。`align_cache`只复制冻结事务中继承的动作坐标，不新增独立观测数。

`training_losses`返回未加系数的七项loss；`apply_gradients`按阶段加系数、合成梯度并执行一次Adam和reader半径投影。`project_sources`接受可重复调用的完整owner/foreign数据chunk factory，返回逐source投影记录。

`propose(...,excluded_nodes=...)`返回proposal/diagnostics；`run_transaction`使用pipeline注入的fork/train/project/evaluate/capture，完整sampler、RNG、teacher与Adam状态由pipeline快照负责。

## 已运行验收

指定项目Python下模型合同测试：**19 passed**。涵盖初始可训练参数63680与TARGET width41的63966；标签无关/非法标签入口拒绝；空batch；手算BCE符号与OFF；收益观测支持/计数与unknown；冻结teacher不共享可变参数；逐loss实际autograd权限及native/phi辅助隔离；全NULL仍shadow训练reader；U=0/V非零冷启动；FP32≤1e-6与FP64≤1e-12的非0.5 NEED/SOURCE保函数；完整NULL/本域mask/动作质量；旧Adam状态及参数对象保留、新参数状态0语义；辅助norm cap；完整支持投影与gamma0；六单关和非法组合；相同x相反标签不产生可部署分区；解析source梯度草图与真实autograd一致；WIDE胜不冒称分裂；一次S读取；低秩不造观测；非对称SOURCE_SPLIT后CL teacher刷新；NONE前向确实不调用instrument。

初始全A的SOURCE_SPLIT实际新增1229参数、1490相关MAC；WIDE rank10新增1280参数/1280 MAC，差异4.15%/14.09%，满足预设匹配门。NEED与SOURCE单batchFP32保函数示例最大误差分别0和4.7e-10；正式测试同时覆盖FP64。

检查命令：

```text
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest experiments/mdmt_basp_v1/tests/test_model.py -q
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m py_compile experiments/mdmt_basp_v1/model.py experiments/mdmt_basp_v1/benefit.py experiments/mdmt_basp_v1/relation.py experiments/mdmt_basp_v1/losses.py experiments/mdmt_basp_v1/permissions.py experiments/mdmt_basp_v1/conflict.py experiments/mdmt_basp_v1/transaction.py
```

模型状态、teacher和Adam均未写入磁盘。未增加依赖或新摘要机制。

本阶段跳过未触发B/C及额外实验；真实自然数据性能与GPU运行由独立review放行后执行，当前报告不含研究结论。
