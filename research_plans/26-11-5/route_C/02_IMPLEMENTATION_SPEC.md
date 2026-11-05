# C2 实现规范、成本与验收

## 1. 模块与接口

待建包`experiments/aug_history_20261105/`新增：
- `drift_probe.py`：`input_drift(online,teacher,x_probe,fields)->u:[n,2,F,16]`；只对同坐标δ求导。
- `covariance.py`：`statistics(u)->A:[2,F,16,16]`；`covariance_root(A,gamma)->root:[2,F,16,16]`。
- `conditional_views.py`：`draw_conditional_views(model,x_target,root,base_view)->views[2]`。
- `conditional_objective.py`：两任务各自视图的监督替换，原2N分母。
- `ConditionalAugState`：固定教师φ、A/root、γ、K、last_refresh、固定旋转、probeRNG、selected/E10/E50与成本。

source `views.py`原draw_view生成base ξ；其字段顺序与逐字段RMS缩放保留。model_adapter需要支持embedding输出加δ和raw logits，不修改训练gate断图。

## 2. 输入探针的高效实现

源fixed1无BatchNorm/Dropout，样本间无前向混合。对一个任务求`grad(sum_i r_it), δ`即可得到各样本自己的输入导数；不用n次逐样本backward。两个模型使用同一个δ叶tensor但各自lookup输出为基点；对两任务分别求导，最多两个输入梯度遍历，保留必要图后释放。

probe只含train目标域128行。冻结模型参数，但不能在整个probe外包torch.no_grad（δ需要梯度）。使用requires_grad_(False)的独立/functional权重，确保online `.grad` 和Adam不变。probe结束恢复所有模式与hook；C的probe_input_full_graph=True只解除gate对embedding输入的detach，正式训练仍False。

每个字段u求FP64 norm，按C1标准化；小于数值阈值设0。构造A后对称化(A+Aᵀ)/2，再FP64 eigh；微小负特征值在数值容差内截0，明显负值/NaN记错误。γ范围本轮≤2，PSD的trace≤1，指数条件数受控；不需要临时扩大jitter或裁剪γ。

## 3. 训练视图实现

base_noise形状[N_target,F,16]，每任务noise_t=base_noise×root_tᵀ。传入原apply_view的noise字段，由原代码计算κ×detached RMS；不能又乘一遍RMS或κ。

原始tensor与base_noise不得原地修改。两任务root相同的情况复用一个增强forward；否则两个forward分别取对应任务输出。只为目标域样本做增强，非目标域clean损失保持完整。

view metadata保存seed/key、基础噪声hash、root hash、实际energy统计；实际energy用于核对实现，不按其高低调参数。θ、φ、A、root和views在使用时的step编号必须记录，防止用本步更新后teacher回写同一步增强。

## 4. 专门验收

- raw logit sigmoid与源预测一致，probe解除gate输入断图不改变前向值。
- 输入梯度用小型FP64网络有限差分确认；full graph应包括gate输入路径，正式训练仍保留detach。
- θ=φ时u=0、A=0、C=I。
- γ=0硬分支与原G-D0的loss/grad/Adam/EMA/RNG精确兼容；证明可复用普通Gaussian，而不是新增独立双视图控制。
- A PSD、C正定、trace C=16、root@rootᵀ=C，正/负γ方向趋势符合定义。
- Monte Carlo纯合成噪声检查期望能量与κ²s²16接近；这只是分布实现测试，不是推荐性能实验。
- t0视图不计t1监督，t1亦然；总loss与手工逐样本替换相同；尾batch分母正确。
- domain/dense/ID/labels未变；probe只读取train且不改loader顺序。
- R_f正交固定、旋转前后特征值一致；merge得到两任务相同root。
- 恢复包含teacher/A/root/last_refresh/R_f/RNG；连续运行和RAM续跑一致。
- 测试集冻结和四候选依赖的五seed闸门通过；不能在C1完成1seed后决定C3符号。

## 5. 资源与数据缓存

A/root各2×F×16²；KR每份FP64约132KiB，ML约24KiB，远小于模型。主要成本是目标域第二次增强forward与其backward、每K步两模型小probe及输入梯度，不是矩阵求逆。

正式模型更新仍每batch一次Adam/固定EMA；每候选是一条新online轨迹。不能把不同γ或teacher状态共用online。训练基础数组/编码器、索引、同seed初始参数定义和统计代码可共享；模型、Adam、teacher及A不能跨候选混用。

成本先以完整batch工程smoke测训练秒、probe秒、验证秒、GPU/RAM峰值，报告真实值。无需给未经测量的固定倍速承诺。OOM优先减少probe并行和临时图驻留，不改冻结probe样本数/训练batch/精度；确实无法运行则COST_BLOCKED，保留待解决项，不能写方法无效。
