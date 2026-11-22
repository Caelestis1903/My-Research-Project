# Codex 工程实现规范

## 1. 已核实的真实接口

下列为真实仓库接口；来源与固定版本见 [12](12_SOURCE_AND_REUSE_REGISTER.md)。

| 路径 | 已存在函数/类 | 本轮复用方式 |
|---|---|---|
| experiments/conditional_information_momentum_20261102_v3/protocol.py | load_protocol(), config(arm,dataset,lambda_pair=0.), load_dataset(), make_model, make_optimizer(), setup(), update_ema(), evaluate(), state_digest(), cpu_copy(), rng_state(), restore_rng() | 数据、模型、源配置、EMA、评价和状态身份 |
| 同包 boundary_sampler.py | CONDITIONS, boundaries(), sample(), forward_logits() | 原24条件、w/q、2048对、tower原logit |
| 同包 objectives.py | train_step(model,optimizer,ema,x,y,row_ids,config,context,diagnostic=False) | A00/A10/A20/A30保持原科学内核 |
| 同包 optimizers.py | shared_name(), PairOptimizer、state_dict()/load_state_dict() | 精确ECCM及collapsed内核；原PCOLL默认动态须由新适配器改静态配置 |
| 同包 run.py | job(), worker(), results()及manifest/schedule等CLI | 参考状态与选点schema；不复用旧两seed调度/旧driver |
| experiments/auxiliary_objective_attribution_20261115/protocol.py | optimizer(), dataset_cache(), source_manifest() | 普通原生Adam工厂、冻结数据身份、source closure范式 |
| experiments/completion_weight_attribution_20261117/core.py | train_step()内virtual_steps(), parameter_activity(), adam_step_counters() | 同真实prestate深拷贝+load_state_dict+原生step、None/private clock与实际/虚拟hash验收模式 |
| experiments/aug_history_20261105/group_registry.py | build_path_groups(), GroupRegistry | S/D/T/DT真实坐标划分；来源分支含依赖，不假装main可直接import |
| experiments/ema_causal_20261110/interventions.py | group_masks() | EMBED/STAR_SKIP/EXPERT细分的来源参考 |
| experiments/cell_mass_bce_20261117/runner.py | _complete()等状态完成逻辑 | 确认selected-only压缩限制；不可把旧complete后的状态当完整Adam |

旧 ModelAdapter 的 probe_input_full_graph=True 会改变 gate 的 embedding.detach 梯度语义，只适用于其原函数诊断。本轮训练梯度必须走真实 source forward，保持detach。

## 2. 新文件与接口合约

以下均为**拟新增**，本次规划不创建Python实现。新包为 experiments/conditional_history_shared_update_20261122/。

| 新模块 | 接口 | 输入/输出与调用位置 |
|---|---|---|
| protocol.py | resolve_protocol(plan_path)->FrozenProtocol；build_job(dataset,arm,seed)->RunIdentity | 校验manifest固定量；生成同源配置与唯一任务key |
| source_adapter.py | load_source_closure(source_register)->SourceAdapter；make_training_bundle(job)->(bundle,model,optimizer,ema) | 用已核源码，完整closure入source hash；源目录不齐时恢复指定文件，不修改科学公式 |
| pair_adapter.py | pair_step(...)->StepReceipt | 直接调用原P00/P01/P10内核；新增配置在本包命名，旧protocol不覆盖 |
| history_controls.py | make_history_optimizer(model,c)；step_history(...)->StepReceipt | A21、A22、λ=.025、两组作用域；公式以02/03为准 |
| groups.py | build_coordinate_manifest(model,bundle)->GroupManifest | 参数有序名称/shape/offset/row范围/组；严格覆盖且不重叠 |
| state.py | capture_full_state(...)->StatePacket；restore_full_state(packet,...); assert_same_state() | RAM内完整状态与身份；禁止序列化到磁盘 |
| probes.py | build_panels(bundle,seed,spec)->PanelManifest；decompose_cell_gradients(...)->GradientBank | 固定train/val面板、真实六格和伪分组；不污染live graph/RNG |
| virtual_adam.py | exact_step(pre,gradients,group,action)->VirtualResult | 从真实状态克隆，返回模型/optimizer/EMA后态、位移和hash，仅RAM |
| utility.py | evaluate_counterfactuals(results,panels)->UtilityRows；norm_matched_readouts() | 六格排序/LL、纠正破坏、方向/范数；禁止Validation挑最优源 |
| rollout.py | run_impulse(saved_branch,K,order_spec)->PulseReceipt；window_ensembles() | 复用B1首步；其余K−1步正常联合更新，保存面板score累计 |
| composition.py | compose_losses(...); compose_step(...)->StepReceipt | 输入冻结A/B/G配置、同一batch和prestate；按第11节返回自然BCE、PAIR、Gaussian残差通道及唯一更新receipt |
| controllers.py | select_source_on_train(pre,batch,panel,eta)->q_or_none；apply_source_attenuation() | B_ADAPT/B_STATIC/B_UNIFORM；仅EMBED |
| runner.py | run_registered_job(job,review_receipt,ram_store)->Result | 当前会话监督训练，完整50epoch和双读出selected |
| scheduler.py | admit(device,measurements,residency)->bool；next_ready_jobs(decisions)->list | 遵守当前资源门及DAG，只派发已触发任务 |
| reuse.py | audit_reuse(register)->ReuseManifest | 逐arm/seed/endpoint身份核验，区分数值/selected模型/完整state |
| analysis.py | summarize_stage(stage,rows)->Decision；freeze_selection()->FreezeManifest | 五seed输入完整后判断，Test隔离 |
| cli.py | audit/check/run/analyze/advance/freeze/evaluate/report | 与00命令一致；advance只展开已定义矩阵 |
| tests/test_scientific_contracts.py | 见第10节 | 仅会改变科学结论的必要检查 |

所有NEW接口必须通过明确参数传入本轮artifact root、source closure、RAM store，不依赖旧工作区绝对路径、旧guardian PID或旧报告HERE常量。

## 3. 数据、模型和坐标

环境：/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python，Python3.13.5，PyTorch2.9.0+cu126，依赖requirements-phase0-3.lock。首先核验现场GPU和软件；不使用base。

原始CSV：

- /home_new/xzf/datasets/ml-1m/ml-1m.csv；
- /home_new/xzf/datasets/kuairand/processed/kuai_rand.csv。

源 resolved_protocol.json 位于 v3/vendor/ema02/vendor/gradient_aug_20261031_v2/vendor_ga31/，包含raw_sha256、array_hashes、split_row_fingerprints、split_row_id_hashes、encoder_hash、scaler_hash、split_rows、train_domain_counts。逐项复用；路径迁移可改位置，不改内容身份。不要把缺失的旧prepared目录当作已经恢复的数据。

模型使用原fixed1构造：embedding维16、4个shared experts、源FCN维度[512,256,256,64]、融合常数1，不恢复DARTS/_weight_*。输入x形状[F,B]、y[B,2]，域在x最后一行。

| 坐标组 | 参数规则 |
|---|---|
| EMBED | embedding.*的共享坐标，排除domain_indicator embedding三行 |
| STAR_SKIP | shared_weight、shared_bias、skip_conn.*、star_mlp.* |
| EXPERT | expert.*，内层名字含domain_specific也按外层实际共享归属 |
| S_DENSE | STAR_SKIP∪EXPERT；必须等于v3 shared_name覆盖 |
| Dd | slot_weight.d/slot_bias.d/domain_expert.d，加domain embedding第d行 |
| Tt | task_expert.t |
| DTdt | gate.i/tower.i，d=i%3、t=i//3 |
| N | D/T/DT全部；它不是必然“完全不跨格”的零作用组 |

domain embedding名字由features.columns.index('domain_indicator')动态确定：embedding.embed_dict.<column>.weight。按行分坐标，不能按全tensor误归S。S总数ML483152/KR661920；D每域ML74960/KR295632；T每任务16576；DT每格5381；总参数ML773470/KR1614254。出现未映射名称即工程阻塞，不默认为共享。组布局和参数顺序写hash。

## 4. 源训练顺序与A控制

源epoch order由独立torch.Generator(seed+11000)的randperm生成；不drop_last，batch40960，尾批ML21927/KR40052；20/49步每epoch。训练种子只能2022–2026。

~~~text
restore/capture model mode, buffers, optimizer, EMA, RNG, loader
model.train()
optimizer.zero_grad(set_to_none=True)
probability, logits := source forward; logits via original tower hooks
pair := keyed source sampler(dataset,seed,epoch,global_step,row_ids)
L := mean BCE + lambda*(w/q)*mean softplus(-logit_margin)
if history mode: r := autograd.grad(pair_loss, S_dense, retain_graph=True)
L.backward()
optional read-only diagnostic clones from actual pre-step state
optimizer.step(pair,r)      # each parameter exactly once
update_ema(ema,model,rho)   # exactly once, after online update
update dynamic demand only after current sample/loss observed
append scalar receipt; preserve actual loader/RNG
~~~

源P00/P01/P10各自保留科学语义。PAIR没有额外forward view，辅助loss梯度覆盖真实全图；只是ECCM的历史替换在S_dense。2048pairs不算额外原始训练行；记录重复抽样、unique_pair_rows、pair_index_hash。

A21用source collapsed公式但dynamic=False；A22用两个native Adam parameter groups，S_dense β1=.99，其余.9；所有组β2=.999、eps、WD、lr一致。A22的长真实/WD历史属于其干预，不能只把差归于辅助pair。

λ=.025在本包显式新增config合法值；旧v3 config仅接受0/.05/.2，不修改旧注册表冒称已有配置。A50/A51只限制历史改造范围；其余参数用普通Adam处理同样完整pair loss，EMA始终全参数同ρ。

每个新形式同时实现λ=0源退化路径供数值检查；A22在λ=0时由工厂直接构造全参数β1=.9的A00 optimizer，不能仅关pair却保留S=.99；“λ=0”工程检查不计新性能配置或额外seed。

## 5. B 真实梯度分解与伪分组

使用一次原生probability forward和BCE reduction=none构造L_c/(2B)。六格autograd只读，最后主loss.backward保证主.grad与原源路径一致；g_J不能通过浮点求和六次autograd替代实际主backward来推进主轨迹。比较Σg_c≈g_J用于数值验收。

伪分组将2B个真实(sample,task)loss条目按标签0/1分层随机分给6个bucket，使每bucket正负条目数与真实cell相同。改变分组，不改任何标签、行、任务损失或总目标；两套独立确定性keyed partition不使用额外训练seed。两个分组各6grad，连同真实6grad，共18额外sweeps/锚点。不要只置换六个cell名字，那不会删除身份结构。

全模型有序梯度表保留None；坐标向量计算可以用0占位，但不能因此给原来None的参数制造.grad并推进Adam。空间mask按坐标切片应用，原始非None参数被消去某源后变为全零张量时仍保持非None。

## 6. B 真Adam反事实与RNG

每锚点保存真实pre-step模型（包含该forward后的buffer状态）、optimizer、EMA、RNG、row/order identity；model参数此时尚未step。所有分支使用相同prestate。对组G：

- JOINT：完整真实g_J，原生optimizer.step；
- LOO(q)：G内g_J−g_q，G外g_J；
- ONLY(q)：G内g_q，G外g_J；
- ZERO_DATA：G内0，G外g_J。

WD不分摊给源，不对每个q重复加；通过原生Adam各分支统一加一次。m、v和每参数step在各分支根据替代grad正常演化。模型forward/BN等buffer变化与source主步一致，主状态绝不能装入分支后态。

采用CWA中克隆、load_state_dict和step模式：同JOINT虚拟状态必须与实际source step一致；同实现同设备应hash相等。跨设备的容许数值差另记，不用容差掩盖本机同语义不等。

等范数控制将两分支在G的总applied update各缩放至min(norm_J,norm_LOO)，保持组外JOINT位移；只前向读出，不继续将这个构造状态当真实Adam训练。零范数时按04标NA，不能生成随机替代方向。

所有probe eval临时切eval并no_grad，随后恢复mode、buffers与RNG；clone forward hooks需解除，避免诊断递归进入观察器。每次诊断后比较live模型、optimizer、EMA、CPU/CUDA RNG、loader和源pair键的hash。

## 7. 脉冲、面板与有限窗口

40锚点=两数据×5seed×E5/E20/E35/E50末batch前。B1所有干预完成；EMBED上的可行动pulse按04/09触发。

训练P_A固定用于六选一+NO_ACTION；Validation只评价。留一seed的q_static由其余4seed早期训练面板冻结，不读当前评价seed的未来Validation。

E20/E35每锚点保留所选真/partition0伪源的首步poststate及pre-F预测起点。若阶段门需等五seed汇总，最多40份有界分支状态暂存RAM；不保存2080全部虚拟状态。受监督RAM需随时现场验收。

每段K=20/49个总更新，首步为已完成LOO，后K−1步重新正常forward/backward/Adam，使用同batch order和RNG流。JOINT同窗口来自父轨迹。最大1380逻辑段内step，40首步已在B1计费，新增1340次实际Adam；NO_ACTION分支也完整纳入统计。

窗口prediction ensemble从pre-F预测初始化，按真实分支每步online预测递推ρ；参数读出同一pre-F起点和ρ。probability与logit平均分别记录，只保留score banks。公式和适用边界见04，不创建历史θ磁盘快照。

## 8. B 完整候选的精确动作

本版作用组固定EMBED；N、S_DENSE的核心干预可形成机制结果，但未经其伪分组验证不能用于触发本版EMBED算法。

B_ADAPT在每epoch首个真实batch、实际prestate上，用该batch的源梯度和P_A面板，在live step之前从同一前态分别评估七个虚拟分支：JOINT与六个η减弱源贡献的真实Adam候选。每个clone都重置共同的模型、optimizer、EMA和RNG；live前态保持不变。选相对虚拟JOINT的Macro增益最大的q（严格>0才动作，平局用固定cell顺序；否则NO_ACTION），随后仅执行一次live所选Adam step和一次EMA。q在当前epoch固定；之后每batch重新计算当前g_q，不能使用冻结旧梯度。

$$
g'_{\rm EMBED}=g_{\rm EMBED}-\eta g_{q,\rm EMBED}.
$$
组外不改。训练面板只前向排序，无反向面板loss，不引入任意惩罚项。η=.25/.5各五seed完整训练。

B_STATIC使用留一seed训练诊断冻结的q_static，全程同η；不存在合格固定源时NO_ACTION，不凭Validation挑源。B_UNIFORM从对应seed/η的ADAPT JSON动作日志读取q_e与on/off时序：

$$
g'_{\rm EMBED}=(1-\eta\, n_{d(q_e)}/(2B))g_{\rm EMBED}
$$
有动作时使用当前真实batch计数，无动作时系数1；只缩data gradient，coupled WD原样。它匹配被去掉的名义数据loss质量，不匹配非线性Adam实际位移。先完成ADAPT再运行UNIFORM；日志携带source/action hash，禁止换用其他seed的时序。

active batch通常增加一次只对EMBED求g_q的autograd；epoch首batch已有6源梯度可复用。probe每η的JOINT+六候选共七次clone/eval计入实际成本；B_SEARCH中20条ADAPT轨迹×50epoch为7000次clone Adam。JOINT沿用已计算的自然联合梯度，不增加第七次源autograd，不能按纯clean吞吐估计。

## 9. 状态、输出与失败恢复

StatePacket必须含 model/optimizer/EMA、所有A history/active/selected/pair_clock/demand/moments、每参数step、grad None语义、Python/NumPy/Torch CPU/CUDA RNG、loader generator与epoch/batch cursor、selected模型/epoch/Validation、source/data/group/config hash和累计计数。

这些对象只通过RAM IPC交换，严禁torch.save/pickle/safetensors/npz等方式把训练状态写磁盘。已有数据cache是数据工件，不准把它当训练状态偷存。允许输出的JSON/CSV含数值和hash，不含可恢复参数/梯度数组。

主轨迹epoch原子提交：写标量 → RAM commit → 实际get/hash验收 → 更新progress。中断后有完整已验收RAM则从该提交继续；没有状态则原身份从头重放，所有无效旧attempt保留成本记录，不能拼接不连续的模型轨迹。

保留selected状态到最终freeze/Test结束；分析阶段不能先complete压缩掉后续B所需Adam。Test终评后才按本轮所有权协议释放RAM。

## 10. 必须通过的最小检查

| 检查 | 明确风险与验收 |
|---|---|
| source回归 | 相同seed/真实首批与尾批，A00/A10/A20/A30源实现与适配器loss、pair hash、update/EMA一致 |
| 空pair与活动时钟 | 缺正/负边界不产生pair；ECCM active计数与selected计数不混用；无pair时不推进辅助历史 |
| PCOLL静态 | dynamic=False，q/pair与P01相同；主损失不变；仅history处理变化 |
| 单长历史 | S=.99、其他=.9；WD/τ/v无额外step；每参数更新一次 |
| λ匹配 | A40/A41相同λ、pair、预算；λ=0返回源普通训练语义 |
| group映射 | 互斥覆盖，domain embedding按行归D，S_DENSE=shared_name；无虚构FUSION参数 |
| 梯度守恒 | ΣL_c=L0；Σg_c≈gJ；每伪分组loss/gradient总和恢复原目标 |
| 原生Adam | JOINT clone与actual同设备model/optimizer hash一致；None跳过和零tensor执行路径均覆盖 |
| 诊断不污染 | probe on/off同源短序列主状态、RNG、loader、EMA一致 |
| EMA与脉冲 | 一步δF=(1−ρ)δθ；固定update与单θ快照公式分别验收；真实脉冲后重新计算梯度 |
| RAM恢复 | 至少一个真实首批/尾批短序列暂停恢复与不中断一致；现场get/hash不只ping |
| 选择/统计 | 人工标量fixture确认最早strict-max、LL绑定同点、5seed配对和缺seed技术阻塞；不需额外模型训练 |
| Test封存 | freeze前evaluate拒绝Test；冻结后核对selected模型hash及主表行；不开放旧循环Test |

只有影响上述不变量的错误才阻塞；不扩展为通用代码整洁度、旧模块全覆盖测试或重复大规模模型验证。


## 11. AB 与 Gaussian 的唯一组合合约

本节定义已在矩阵中的条件组合，避免执行时重新选择梯度含义。没有新增搜索维度、完整训练臂或候选参数。

### 11.1 通道定义与同一步更新

令 g0=∇L0 为原始自然概率BCE梯度；gq=∇Lq为02的原始自然六格贡献；rP=(w_i/q_i)∇L_pair为本批选中条件的加权辅助梯度；λP为已经冻结的pair权重。无A组件时λP=0。

无Gaussian时gG=0；有Gaussian时定义gG=∇(L_Gaussian−L0)，见11.3。所有通道都在同一当前模型状态和原始batch上计算，先算完梯度，再统一执行一次optimizer/EMA；不得先更新EMBED再计算S_dense梯度。

AB_ADAPT/STATIC的实际data梯度为

$$
g_{\rm AB}=g_0+g_G+\lambda_P r_P-\eta P_Eg_q.
$$

B只减弱**原始自然BCE源贡献**，不减弱PAIR或Gaussian残差。组外保留完整mixed data梯度；coupled WD由optimizer原样加一次。

A20/A21在S_dense的普通通道输入为g0+gG+WD。A20按02/03保存未加权r_pair的条件历史，并用当前w重组；A21按02/03保存已乘w/q的单份辅助历史。二者都只接收PAIR，不接收Gaussian残差；各自λ、抽样概率修正和活动/PAIR时钟保持02/03的原定义。混合v使用本组完整实际data梯度+WD。A22采用同样路由后的单份混合m；A50/A51只改变历史作用域。EMBED本就不属于A条件历史，由普通Adam处理上述mixed梯度。全参数F仍在全部参数更新后只推进一次。

### 11.2 选择器、静态源与trace

AB中的B_ADAPT用**AB自身当前模型与全部实际optimizer状态**评候选。先克隆无B动作的A-only联合一步作为JOINT参照，再从相同prestate依次构造η减弱自然gq的六个候选，共七次虚拟step；live状态在选定前不变，选定后只step和EMA各一次；各候选的pair索引、rP、A的历史前态、λ、Gaussian视图（若有）完全相同。只用P_A比较当前Macro效用，最佳≤0为NO_ACTION。不能沿用clean模型的实时gq，也不能把P_A当额外反向loss。

B_STATIC沿用B阶段已冻结的q_static与η，不在AB或G背景重新搜源。B_UNIFORM使用**原独立B_ADAPT同dataset/seed/η已冻结的trace**，不在AB/G重新跑一个pilot；其作用仅为

$$
g_{\rm AB,uniform}=(I-P_E)g_0+
(1-\eta m_{q_e,b})P_Eg_0+g_G+\lambda_P r_P.
$$

这个“重放固定标量时序”是匹配控制，并非在新背景实时自适应。若它成为最终采用的简单策略，报告必须计入生成原trace的pilot成本，并限定为冻结时序方法；不能称为无先验pilot的通用控制器。

### 11.3 Gaussian 的源目标与实现

固定D0、λ_aug=.5、kappa ML.05/KR.1、源A1 Gaussian逐稀疏字段embedding RMS噪声，不扰动域字段。原始CSV/loader/标签不变。以同一个clean forward的逐元素BCE C及一份D0扰动forward的BCE A表示：

$$
L_{\rm Gaussian}=
\frac{\sum_{i,t}C_{it}+
 .5\left(\sum_{i:d_i=0,t}A_{it}-\sum_{i:d_i=0,t}C_{it}\right)}
 {2B}.
$$

这来自已核实的
v3/vendor/ema02/objectives.py:losses()
与其 vendor/gradient_aug_20261031_v2/objectives.py:losses()；
不是额外把一份Gaussian loss直接加在原完整BCE上。

新增compose_losses()应一次调用源Gaussian目标路径，并在**其原始完整batch forward**捕获tower logits；复用source返回的pred/real构造L0、Lq和PAIR。捕获钩子在该完整forward后解除，不能被随后D0 view覆盖；不能为PAIR再做一次原始forward，改变计算图、buffers或RNG。Gaussian随机键沿用原A1|gaussian及相同row IDs，PAIR继续condition/pair_indices独立命名键，互不消费对方全局随机流。

Gaussian残差gG全部进入A的普通/真实通道，不建立Gaussian专属H/Q；PAIR仍只由原始未扰动logits和原标签构造。B仍只减弱原始gq，不减弱gG或rP。D0的clean系数为.5−η（被选中且采用源减弱时），在本轮η∈{.25,.5}下非负；其他格按原系数处理。UNIFORM同样只缩g0，不缩gG/PAIR/WD。

selector在G的完整当前background上比较“Gaussian+A但无B”与“Gaussian+A+B”的真实一步，其余输入冻结。这是已冻结优化规则的背景检验，不重新搜索η、λ、scope或增强覆盖。

### 11.4 正面桥接门与最终解释

AB只在**至少同一个数据集**上A、B各自对F有独立正Validation增量，且以下桥接通过时触发：

- 使用已在RAM保留的获胜A轨迹E20/E35状态；缺失时使用既定A_STATE_REPLAY额度，不追加新科学seed。
- 在该状态比较A-only真实一步与按11.1/11.2加入B的一步；源选择只看P_A，Validation确认跨五seed平均ΔMacro>0，六格平均Δ≥−.001。NO_ACTION计零。
- 该正面边际证据与双方作用组/实际效用相符，才注册AB_FROZEN十条完整训练。无需从A状态虚构一个“已完整训练的B-only optimizer”。
- 只有A在ML有益而B在KR有益，不自动满足共同数据条件。不能把“没有看到抵消”当成正面桥接。
- AB完整训练后必须相对较强单方法有独立增量，才选择联合；局部桥接不替代该正式比较。

G_FROZEN只回答“冻结优化方案整体，在既有Gaussian背景下是否有新增AUC”。没有额外G+PAIR普通Adam臂时，不将这个差进一步拆成Gaussian背景下条件历史的纯独立作用；A在clean背景的独立归因继续保留。若需要这种更细主张，应另立任务，本轮不暗加矩阵。

组合最小检查：B η=0恢复对应A/G+A路径；λP=0恢复对应B/G+B路径；Gaussian λ_aug=0或kappa=0恢复对应clean组合。必须核对单步数、噪声/PAIR键、状态不污染及源Gaussian控制一致，避免额外forward和隐式双重WD。

