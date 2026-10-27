# Codex 实现与接口规范

状态：DESIGN_ONLY。此文件定义接口合同，不表示相关模块已存在。

## 1. 仓库与安全约束

目标仓库为 Caelestis1903/My-Research-Project。实现新包 experiments/mdmt_relation_basis_v2/；结果写入 artifacts/mdmt_relation_basis_v2/，报告写入 research_reports/26-10-27_v2/。不得覆盖 history_experiments、旧版研究计划、原始报告或现存工件。

开始前读取当前 main、仓库 AGENT.md/同名规则文件、v2 全部规划文件与现有数据/模型实现。核对源代码实际路径、Git blob SHA、依赖闭包和输入输出；旧版文档中的路径仅作线索。若当前 main 缺少旧 DFD 代码，固定来源于已核验历史提交的必要源码快照，并写入 SOURCE_MANIFEST.json；不得直接合并历史实验分支。

不得结束或修改他人的 GPU 进程，不得默认 GPU 空闲，不得改动用户主目录或数据。audit 必须先输出设备可见性、显存、环境版本、数据身份和预计预算。若资源不可用，保留审计结果并停止训练，不要声称实验完成。

## 2. 模块接口

- config.py：读取唯一权威 YAML，严格拒绝未知方法 ID、重复键、非法值和配置缺项。
- data.py：加载锁定的自然数据角色、任务 mask、domain、row_id；编码器只在 T0/fit 拟合。
- native.py：复用已核验底座，导出 h[B,256]、z0[B,2]；冻结参数并记录 checkpoint/hash。
- source_functions.py：拟合或加载每来源 φ_s，输出 [B,64]；记录来源训练数据、参数和标准化器身份。
- relation.py：输出请求条件 α[B, D−1, J] 及 NULL 权重；本域来源必须严格屏蔽。
- model.py：计算 z_hat[B,2]，支持任务 mask、批次不齐和仅推理输入。
- losses.py：稳定 BCE、可选来源保持风险；禁止把验证/测试标签用于梯度。
- trainer.py：按配置执行 warmup、联合训练、checkpoint 选择和恢复。
- diagnostics.py：输出逐域任务 AUC/LL、路由分布、基底差异、来源函数干预和成本。
- report.py：生成 JSON/CSV/Markdown，记录每个阶段状态及所有失败/未触发原因。
- cli.py：提供 audit、plan、train、evaluate、report 子命令。audit/plan 必须只读，不可偷偷初始化训练或写模型缓存。

## 3. 数据与角色

使用仓库当前锁定的自然 MovieLens-1M 与 KuaiRand 处理流水线及已登记 SHA。开始前重新核验 raw SHA、行数、schema、域计数、任务定义、mask、T0/F/S/Q/V/E 的实际行身份。旧版角色定义可借鉴，但本版必须在 config 中写明每个角色具体映射到仓库哪一份分割文件/行计划；不能只写“使用旧划分”。

T0/fit 用于底座、来源函数和标准化器。F 用于新模型目标训练。S 只用于预先定义的训练内约束/结构诊断，不用于选模型。Q 用于开发决策，不更新参数。V 仅选择完整 checkpoint。E/test 只在所有模型、阈值和报告口径冻结后统一评分；若历史 test 已暴露，必须标为历史复评，不称独立确认。

同一交互的两个任务必须共享划分和抽样计划。缺失标签由 mask 排除，BCE 分母为有效标签数。禁止因某域更大而改变训练采样比例，除非对照明确声明此为独立变量。

## 4. 模型张量合同

- h: float32 [B,256]
- z0: float32 [B,2]
- phi_s: float32 [B,64]
- f_sj: float32 [B,64]
- request r: float32 [B,32]
- relation α: float32 [B,3,2]，对每个目标域/任务仅保留两个 foreign source，各自两个基底；NULL 单独一项
- w_dts: float32 [3,2,3,64]，对角本域位置不参与计算
- residual head output: float32 [B,1] per target domain/task
- final logits: float32 [B,2]

实际实现可用 ParameterDict/批量张量，但需保留相同语义和维度映射。关系 softmax 的候选集合由 target domain 决定，不得将本域 masked logits 作为有限大负数后误参与统计。

## 5. 训练顺序与状态

1. audit：只读检查代码、数据、环境、GPU和预算。
2. build-prefix：仅在无可验证缓存时重建共同冻结前缀；记录完整来源。
3. train-baselines：先跑 DIRECT、B_REFERENCE、READ、TARGET_ONLY。
4. train-main：只有接口与小批次验收通过后，跑 LOWRANK_RELATION 与 FULL_RELATION。
5. select：只按 V macro LogLoss 选择完整状态；并列容差与最早 checkpoint 规则见配置。
6. diagnostics：只在锁定 checkpoint 上运行训练内/开发内预注册诊断；不得重新选择 checkpoint。
7. final-evaluate：配置和方法清单锁定后，统一运行 E/历史 test。
8. report：生成每域每任务和总体指标、成本、协议偏差与科学结论。

checkpoint 至少包含模型、optimizer、scheduler（如无则显式 null）、epoch/global step、Python/NumPy/Torch/CUDA RNG、row-plan hash、配置 hash、源码 commit、数据角色 hash、预处理器与来源函数身份。恢复训练必须逐字段验证；身份不匹配时禁止静默续训。

## 6. 计算成本与效率报告

每个方法同时报告：新增可训练参数、冻结来源函数参数、总部署参数、每批调用的来源函数数、实际 FLOPs 或可复现的算子计数、训练秒数、峰值显存、推理 P50/P95。关系低秩只减少关系参数，不代表减少来源函数执行。若要宣称稀疏执行，必须在调用昂贵函数前完成路由并实际跳过对应函数；本版不以执行全部函数后乘零宣称稀疏加速。

## 7. 失败处理

配置、数据身份、模型输入输出、标签泄漏、checkpoint 恢复、指标实现或安全审计失败属于实现阻塞；科学结果差属于有效阴性。任何未启动、未触发、资源不足或数据缺失均记录具体状态，不得重命名为失败实验。不得为了凑齐矩阵而改 seed、加预算或临时扩大超参搜索。
