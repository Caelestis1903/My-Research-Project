# 07｜资源预算、历史复用与故障恢复

## 1. 可复用内容

优先复用 26-11-07 分支的：
- 原始训练数据缓存、split与train-only预处理；
- M3oE fixed1模型构建、原始Gaussian loss与RNG；
- ParameterLayout、group_registry及共享参数S坐标；
- P_A/P_B与V_A/V_B面板身份、hash、行侧；
- 固定EMA、Fast/Slow pulse更新及指标读出；
- 11/07历史seed的逐窗pulse、预测与分析数据。

历史seed仅用于A0开发和指标设计；不作为A1新seed的独立确认。不得把历史Test重复访问当作新盲测。

## 2. 资源预算方法

11/07冻结 group_registry 中，S坐标数为MovieLens 483,152、KuaiRand 661,920；FP32单状态payload分别约1.93MB、2.65MB，B额外增加同等大小的一份S历史。该数值不包含临时pulse、scratch参数副本、CUDA allocator与进程开销。

在preflight动态计算：
- S参数数量、dtype、单状态字节；
- A新增状态字节；
- B新增第二份S状态字节；
- C新增状态字节（理论上只增加task scalar和日志）；
- 每个worker的模型、Adam、影子、panel和临时pulse峰值；
- Validation每epoch读出成本；
- P_A每epoch动作探针成本；
- 预测文件、checkpoint和日志磁盘上限。

A1计划10条唯一训练轨迹：KuaiRand Clean/G-D0 × 5 seeds；每条50epoch、49 steps/epoch、2450 Adam steps，总计24500次真实Adam。最多两个worker并行，资源门沿用11/07：主机RAM占用低于90%、GPU资源门通过后才启动；每个阶段动态报告显存、RSS、磁盘与墙钟。

不得假定GPU显存、S状态大小或磁盘需求，必须从实际代码与layout测量。发生资源阻塞时记RESOURCE_BLOCK，不得减少训练人口、seed或必要面板以伪造完成。

## 3. 数据与磁盘复用

训练bundle必须 `include_test=False`。Test数据文件hash可在preflight核验，但训练、A0/A1分析和controller不得访问Test标签或指标。面板索引、label、pair、row side及数据hash必须冻结。P_A/P_B属于训练人口的观测子集，不得改变loader。

训练参数和Adam状态优先RAM guardian + 原子事务；磁盘保存可恢复的最小状态与观测产物。若源实现已使用RAM-only参数生命周期，沿用其事务合同。每个epoch保存state hash与commit ledger。

## 4. 失败恢复

- worker死亡：只由guardian确认owner已退出后重试；不清理其他worker owner。
- 重复epoch提交：相同hash幂等；不同hash拒绝并保留先前已提交状态。
- manifest/source/data hash变化：停止并返回配置冲突，不允许继续旧run。
- GPU OOM/RAM/磁盘超限：停止新worker，保存已有有效状态，记录资源阻塞。
- 恢复后先检查online、Adam、EMA、controller、RNG、loader与panel身份全部一致，再继续下一epoch。
- 科学性失败、数据缺失、工程错误、资源阻塞和阶段未触发使用不同状态码，不互相替代。

## 5. 成本报告

最终输出每条轨迹Adam step数、训练/Validation/pulse/controller探针时间、GPU allocated/reserved峰值、worker与guardian RSS、磁盘写入量、恢复次数、推理吞吐与延迟。A/B/C增量成本分开报告。B双S状态应与A/C实际状态字节直接对比；B和参数插值C-P还需计入双forward成本。函数空间C-F复用online/F两路readout，不增加S状态或额外forward，但部署仍需同时保留online/F两套预测参数。若B收益不足以覆盖成本，优先C-F、C-P或固定F，按实际吞吐和收益决定。
