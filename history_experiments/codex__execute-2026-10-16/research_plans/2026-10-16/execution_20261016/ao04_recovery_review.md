# Implementation report and SELF_REVIEW — AO04 recovery

主代理，2026-10-16。依据ao04_recovery_design.md和实际recover_ao04.py。py_compile通过；仅限零训练法证，不是独立PASS，不授权新训练。

实现逐一核对R2 config source SHA、数据hash、原setting/seed，strict load内嵌selected_cpu_state；CPU全validation重预测，与旧NPZ预测/label逐行核对后才生成恢复row IDs索引。旧NPZ不改。没有test调用、没有optimizer.step、没有改旧训练参数。

敏感度用原forward hooks捕获tower输入/logit、domain experts和gate。对原raw q autograd，与dlogit/dF点乘aD(1-bD)rho(1-rho)(Ej-Er)核对；再用h=.01有限差分核对概率导数。probe之后q/模型state hash不变，tau恢复。数值assert失败会停止该case，不标成功。

该脚本只覆盖五个实际resume及B1可读selected/last_observed，保留未完成seed2023状态。epoch/seed全覆盖缺口不变，不据first16/domain probe宣布总体collapse或identifiability。

Critical修正：原审计仅检查外置checkpoint文件名不足以推断权重不可恢复，使用内嵌state补证。公式修正明确是导数变量rho/q，不改模型或实验假设。Relevant：float32 CPU/GPU完整预测容差3e-6；有限差分2e-5、解析链式导数1e-6+1e-4相对误差；均仅用于绑定及数值验证。

首次RUN：两个B0完整validation绑定通过；B1的解析链式核对通过，但h=.01中心有限差分未满足2e-5绝对容差。未标该case成功，原日志/源码/partial产物保留。最小诊断修订：对ReLU模型同时记录h=.01/.003/.001三个固定尺度；每探针至少一个尺度仍须满足原2e-5容差，未放宽阈值。若仍失败，保存逐row差异并停机。此检查仅解决局部导数与有限邻域/float32舍入的数值可辨识性，不改变模型、评价协议或研究对照。重跑使用独立attempt2目录。

Attempt2：KR B0/2022、B0/2023、B1/2022全量validation均匹配，B1多尺度检查通过；随后B1/2023缺config导致入口失败。该缺失不能用默认配置补造。Attempt3按同source/state/NPZ哈希复用3项已通过的完整预测绑定；缺config项仅索引内嵌state结构、标INCOMPLETE并继续独立ML case。增量写出指标/probe以保留进度，不重复旧全量forward。ML的.downloading文件为0字节，不能当预测。

WP2追加：新bootstrap_validation.py在B0/B1 seed2022全量预测绑定成立后执行1000次逐cell/label配对bootstrap。weighted tied-AUC数值算法与已验证旧bootstrap实现一致；新增logloss使用相同行权重。不同任务单独分层，不合成macro区间；validation-selected模型的条件区间不称为post-selection有效区间。无新训练、无更改选点。
