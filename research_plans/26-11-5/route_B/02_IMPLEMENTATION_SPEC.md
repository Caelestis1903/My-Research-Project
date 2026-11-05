# B2 实现映射、状态隔离与验收

## 1. 新模块和公开接口

待建包`experiments/aug_history_20261105/`：

- `adam_counterfactual.py`：`predict_displacement(theta,grads,optimizer_snapshot,active_mask)->FlatTensor[P]`。只读，返回实际FP32参数位移；输出detach。
- `responses.py`：`extract_aug_grads(loss_terms,params)->dict[(d,t),GradientList]`；`response_basis(snapshot,g,a_by_cell)->ResponsePack`。
- `subspace.py`：`orthonormal_basis(columns:[P,m])->Q:[P,r]`，m≤2；`signed_basis(Q,signs)->Qtilde`。
- `directed_ema.py`：`update(phi,theta,Q,rho0,h)`；`DirectedEMAState(phi,buffers,h,K,last_refresh,Q,selection,snapshots)`。
- `worker.py`：沿原worker分离“loss/backward/step/shadow”，不能直接调用旧train_step后再尝试读取旧Adam状态。

ResponsePack包含D/M basis、各列norm、rank、orthogonality_error、refresh_step、state_hash_before/after、active_mask_hash、time/memory。日志保存标量/hash，不保存完整Q到磁盘。D和S可共享只读Q及固定符号；各h有独立φ。K10与K20的Q分别缓存。

## 2. 一次刷新步的精确顺序

1. optimizer.zero_grad(set_to_none=True)，保留θ_old和旧state的只读快照。
2. 原clean/aug前向，获得同图total与aug_by_cell。
3. 对两个aug标量autograd.grad，create_graph=False、retain_graph=True、allow_unused=True；None只在展平response时映射为零，但保留原active语义元数据。
4. total.backward()恰好一次，从p.grad读取正式g。任何额外grad调用不得累积到p.grad。
5. 用旧θ、旧m/v/step分别预测g、g−a0、g−a1、g−a0−a1。源FP32顺序与coupled L2完整匹配。参数快照允许通过顺序临时tensor复用节省内存。
6. 计算D/M和Q；对应时钟更新缓存。S不重抽符号。
7. 原optimizer.step()。预测全g后的θ必须与真实θ对齐，随后删除反事实临时tensor。
8. 保存每影子old innovation，按B1公式更新一次、复制buffers。

非刷新步只执行原损失一次backward、一次Adam和已有Q的影子递推。空目标域时aug损失为0且不额外前向；刷新置空Q，不出现旧方向永久残留。

## 3. Adam反事实实现策略

生产可以使用经过验收的纯函数公式，但必须与**独立克隆模型+独立torch.optim.Adam**的测试真值对齐。直接调用torch.optim的functional接口也会原地改写输入状态，只有传入深克隆tensor才合法；禁止把真实state引用传进去。

覆盖：第一步无state；已有长历史；grad=None vs zero；尾batch；weight_decay两数据值；非一致逐参数step；不同参数shape；negative/zero梯度；unused参数。state hash与逐tensor比对在调用前后不变。实现中不得把AdamW替换Adam，不得遗漏bias correction或把eps放到平方根内。

## 4. 参数展平与数值

布局使用source named_parameters完整顺序，记录shape、offset、dtype、alias，必须覆盖所有可训练参数，包括embedding和heads。Q在这个完整P空间形成；不得为了方便只用共享专家却仍称全参数方法。

Gram/特征分解FP64，Q存FP32前后都检查正交误差。norm用FP64，避免小响应下溢；只丢精确零列及明确数值秩阈值。两列共线变秩1、反向共线同样秩1；不能强凑两个单位方向。

## 5. 必需验收

- 梯度公式：toy FP64下g=g0+a0+a1；正式源g不被分解重组写回。
- 反事实纯性：真实θ、m、v、step、RNG不变。
- 原Adam等价：预测全g位移与克隆/真实一步匹配；包含None/zero差异。
- EMA：h=1、Q空退化F；常量θ不动；正交方向手工计算；一份共享φ只更新一次。
- Q：列重排/变号不改投影；同秩S的rank/support/absolute values相同，但投影通常不同。
- 刷新：k=1起、K10/K20各自正确；无未来泄漏；恢复后Q和last_refresh一致。
- 非反馈：添加整个15影子bank后，在线source轨迹hash保持一致；无法逐位一致需先定位算子/版本，不自动宣称“反事实修复了优化”。
- RAM恢复、选择、五seed完整性、test冻结与公共测试全部通过。

## 6. 成本与故障

每10步刷新最多增加两个autograd.grad遍历；总g来自必要backward，不新增正式optimizer step；四次Adam预测是O(P)向量计算，无新完整前向。平均额外梯度遍历约.2次/step，实际成本需测。15影子每epoch完整验证是不可忽略成本，不能只报训练步时间。

每时钟D最多2P、M最多P；两个时钟约6P个方向数，S通过符号应用无需再存一份大Q。连同15φ、selected/E10/E50及Adam总RAM预算见上级06。遇数值NaN/错误state hash即工程阻塞，不静默回F掩盖后把结果写成B失败。
