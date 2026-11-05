# A2 实现规范与验收

## 1. 复用与新增

复用上级09列出的fixed1、原Gaussian、标准Adam、评价/状态组件。新增：`group_registry.py`、`grouped_ema.py`、`ranking_probe.py`，全部位于待建包`experiments/aug_history_20261105/`。不得修改原vendor源码；新model_adapter只提供参数化只读前向。

### 接口合同

- `build_path_groups(model,bundle)->GroupRegistry`：12活动组+可选inactive；每记录`name,parameter_name,flat_ranges/row_indices,numel,source_cells`。
- `grouped_update(phi,theta,registry,rho:[12])->None`：只改φ；不改θ或Adam；内部no_grad。
- `sample_rank_probe(train_bundle,excluded_row_ids,seed,step)->ProbeBatch`：train行、cell标识、正负索引及hash，最多192个cell-row实例；重复原行跨任务可去重，但保存cell映射。
- `virtual_parameters(theta,phi,q:[12],K,registry)->OrderedDict`：仅q有图，返回与原named_parameters同名同形张量。
- `ranking_objective(params,probe,model_adapter)->scalar`：返回六格平均成对softplus，不创建模型参数.grad。
- `controller_step(state,theta,probe)->{q_before,q_after,grad_max,valid_cells}`。
- `GroupedEMAState`：`phi, buffers, q, eta,K,next_refresh,selection,snapshots,layout_hash,controller_updates,cost`。

q用FP32、数值归一化可FP64；模型权重FP32。每候选独立q/φ，不共享控制器结果。INACTIVE坐标用ρ0。各固定profile在初始化就展开为12值向量，禁止运行时根据域样本数重分配。

## 2. 只读函数调用

推荐functional_call或等价无副作用adapter。构造ψ时θ和φ全部detach，ψ仍通过q可微。q的梯度须走原**训练语义**计算图：gate输入detach保持原样；此探针优化的是参数混合对输出的变化，而不是C的完整输入导数。可以只读取tower raw logits；值必须与源sigmoid输出吻合。

不得直接load_state_dict把ψ灌入在线模型再忘记恢复；不得在临时模型中给q复制成常量导致无梯度。CPU单元测试验证q非零梯度时确实影响φ递推，而非只记录日志。

## 3. 每步执行伪代码

```text
loss = original_gaussian_loss(...)
optimizer.zero_grad(set_to_none=True)
loss.backward(); optimizer.step()
for A_estimator in bank:
    if step % K == 0 and adaptive:
        Qtrain = shared_probe_rows(step)
        psi = virtual_parameters(stop(theta), stop(phi_old), q, K)
        grad_q = autograd.grad(rank_loss(psi,Qtrain), q)
        q = bounded_normalized_step(q,grad_q,eta)
    phi = one_grouped_ema(phi_old,theta,rho(q))
    copy_buffers()
```

probe不得调用另一次optimizer.step。临时ψ生命周期限制在当前候选，顺序处理不同候选，避免保留15张完整图。评价顺序加载影子到独立scratch模型，减少GPU常驻副本，绝不覆盖online。

## 4. 参数路径验收

结构检查：组并集覆盖全部P坐标、交集为空、alias去重、domain表三个lookup行按真实映射分开、gate/tower索引3t+d正确。参数量与源相同。

梯度检查：对合成包含3域2任务的batch分别构造单cell数据损失，检查结构上禁止来源的路径梯度为零；允许来源因ReLU等出现零不判为“路径不存在”。使用固定多组合法输入确认主要允许路径至少有可响应实例。检查不包含coupled L2的data梯度，与真实Adam位移检查区分。

公式检查：q=0整模型精确F；固定profile手工小张量一致；常量θ不变；q最大变化≤η且边界[-1,1]；空probe无更新；probe不会改变θ、Adam、buffers、loader/Gaussian RNG；global adaptive的q数为1。

工程检查：连续与RAM恢复得到同q/φ/选点；新增A候选前后online hash不变；所有15候选均从θ0开始；独立seed单位正确；未满五seed的HPO拒绝执行。

## 5. 成本

每自适应候选每K步一次≤192 cell-row的只读前向及q反向，K=20/40。梯度沿网络传播但不存参数.grad；不是免费，单独计时。固定版仅O(P)逐步EMA。15影子参数约15×4P字节，ML44.26MiB、KR92.37MiB；另有online/Adam、snapshots、临时ψ。selected/E10/E50和控制器状态放RAM；具体并发与峰值按上级06，不给未经测量的墙钟承诺。

A主要算法推理成本与F相同，一个模型。故障优先减少并驻副本/顺序验证，不得靠删掉低分候选节省资源。
