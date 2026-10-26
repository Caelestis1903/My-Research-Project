# REVIEW REPORT — D-V2 第一批

日期：2026-10-20。独立 reviewer；仅审 D-V2 16 配置 × 50 epoch。不覆盖 N17、AO04、N18，也不对未交付的全 core 签署 PASS。本轮未运行训练、optimizer step 或研究结果分析；仅阅读实际代码、核对 Git 对象/归档 SHA，以及无训练的 loader/import 基本检查。未修改 coder 文件。

## 1. Verdict

**最终 Verdict：PASS（2026-10-20 修复复查）。本批16条V2可RUN。**

初审为PASS WITH FIXES，阻塞项R1；以下保留初审发现。R1及复查发现的receipt SHA生命周期问题已修复，R2/R3也已补齐。最终关闭证据见第7节。没有发现需要改变算法或研究假设的问题。

## 2. Requirement-by-requirement verification

|设计要求|实际核对|结论|
|历史真实 Runner 与模型|`v2.py:10–16` 导入归档 `v2_20261016.Runner`；没有重写 step/fit；27 个 v2 归档文件逐一对 inventory SHA 及 `c31b531e210daab4cbfb2e53eaa44c22b913264b` 的 Git blob，全部相同|通过|
|固定 fusion、原 routing detach|归档 `v2_20261016.py:40–46` 将四个原始 sigmoid fusion 替换为 ConstantFusion buffer；不再属于 optimizer 参数。`stage_S1_models.StageS1M3OEAdapter.forward` 保留 gate 的 `emb.detach()`、task-major cell 索引、domain mask 与原融合公式|通过|
|完整配置|`v2.configurations()` 实际返回 Adam .01 / SGD .1 × SN/SB × 2022–2025，共16条，独立目录；传给 Runner 的 `epochs=50`|通过|
|单 optimizer、优化器细节|原 Runner 对全部可训练参数仅建一个 Adam 或 SGD；Adam betas(.9,.999)/eps1e-8；SGD momentum0；两者 WD1e-5；无 arch optimizer/scheduler/clipping/early stopping|通过|
|自然人口与统一接口|`bootstrap.population()` 调用核心 `data.load_population('movielens_natural')`。实测800167行，域计数155949/318599/325619；全局80/10/10，训练期拟合schema，人口seed固定2022；没有进入 role_reversal_sample|通过；源码追溯见R1|
|p/q与权重重算|原 Runner 从注入 bundle 的实际 `train_counts` 归一化 p；两臂 alpha=p；SN q=p、SB q=1/3；weight=alpha/q；未沿用人工系数|通过|
|IID采样与尾批|原 Stage2D Sampler 使用两个 uniform：按 q 抽域、域内均匀有放回；stream=17。`fit` 每epoch抽800167行，B40960，19个完整批+21927尾批，20main步/epoch，完整50epoch共1000main步|通过|
|loss/张量/梯度|输入 `[B,F]→[F,B]`，模型输出与标签均 `[B,2]`；先 mean task BCE，再逐行乘 p/q，再按实际B取均值；无自归一化。mask仅路由cell；backward/step保留原路径；固定fusion为buffer；shared路径和gate训练导数语义保持历史实现|通过|
|选模/末轮|E0不参与选模；E1..50 validation六cell宏AUC严格 `>`，平局保留最早；selected及E50均保留。COMMON也包括E10/20/30/40/50|通过|
|test隔离|实际 `fit:175–204` 仅val；205–206先落盘 `selection_frozen.json`；207–213才逐个加载selected/预定epoch state并在no_grad下test；没有每epoch test，没有test后训练|通过|
|日志及可复算指标|每epoch六cell AUC/LL/n/n_pos；batch row-ID SHA/loss/domain counts；曝光、unique/repeat；首两步C/P/R/E gradient/update/m/v摘要；完整model/optimizer/RNG/sampler/checkpoint；selected和共同epoch原始预测；最终artifact SHA。宏、按train/eval p加权cell、task/domain汇总可由完整cell表复算，无须重训|通过本批历史V2范围|
|输出隔离|`/home_new/xzf_natural_movielens_reassessment/V2/{optimizer}/{arm}/seed{seed}`；Runner `mkdir(exist_ok=False)`，不覆盖历史结果|通过|
|实际source完整冻结|归档V2校验存在，但核心population producer不在其source集合内|阻塞R1|

数据基本检查同时记录自然 row-ID hashes：train `a58f8ea1c284adad7bde05e3811c4055ba85db95613523048f613a220f93bcfe`；validation `3cf4c7b1d5dab7ef3ef8d6783faecde7f580337bcdbe4e9c3af37fe1813bbea4`；test `b9ff6de1e3a6f0f601d6843f1be66d40ee292681e3a91688c87ad3b818382ddf`。这些是数据身份检查，不是模型性能结果。

## 3. Bugs/deviations found

### R1 — MAJOR / BLOCKING：实际核心数据生产者没有进入source身份

位置：`experiments/natural_ml_extensions/v2.py:8–12`、`bootstrap.py:22–30,39–41`、`manifest.py:9–12`。

`population()` 实际先加载核心 `natural_ml_reassessment/data.py` 和其归档 `stage_F0_data.py`/`stage_S1_data.py`。无训练import实测 `stage_S1_data.__file__` 来自 **core**，而非 extensions/v2。随后 `old.sources=lambda:sources('v2')` 只收录V2归档和extensions目录的py文件，不校验或记录上述真实producer。因此自然数据生产代码可以改变而run/queue source身份和原 `fit:201` source drift断言均不变，不满足设计的source可追溯与训练前冻结要求。

修复范围应限于追溯机制：

1. 将核心 `__init__.py`、`bootstrap.py`、`data.py`，以及实际使用的 core归档 `experiments/stage_F0_data.py`、`stage_S1_data.py`、`stage_S1_protocol.py` 纳入带明确namespace的SHA集合；核心归档项同时按其inventory记录核对source_path/commit/SHA。也可校验全部已冻结core归档，但不能把未使用、仍在实现的core训练封装纳入V2运行依赖。
2. run/config与无训练manifest均记录同一实际producer集合。将核心与V2归档对应的source_path/commit/SHA记录写入可追溯manifest，或记录并冻结一份可解析的依赖inventory快照。
3. 保留原Runner每epoch `self.sources==sources()` 校验机制，让上面集合进入该比较；在数据装载前验证核心归档，且检查实际已加载模块的路径/身份，避免只验证一个未被执行的归档副本。
4. `bootstrap.sources` 当前还对 `HERE.glob('*.py')` 全目录取hash。N17/AO04/N18继续实现时新增或修改无关py文件会让V2在完整epoch后因source drift中断。应固定V2实际依赖列表（现有 `__init__.py/bootstrap.py/v2.py/manifest.py` 或其中明确依赖项），或在整批训练期间冻结整个目录；首选前者，避免把其它研究代码开发误当V2算法漂移。无需改动历史归档。

### R2 — MINOR / NONBLOCKING：测试访问标记保持false

归档 `v2_20261016.py:205–213` 写入 `test_access_started=False` 后未更新字段。代码时序满足隔离，但该文件不应被分析程序解释成“整个run从未访问test”；可由机械执行账本补记测试开始/结束时间，保持历史归档不变。

### R3 — MINOR / NONBLOCKING：研究范围标签

当前config没有显式 `exploratory_replication` 标签；最终分析/执行账本必须按用户要求注明历史test已有反复使用。独立review标记仍写“required before dispatch”，修复后可记录实际review文件与SHA，避免把布尔值当作放行证据。

## 4. Corrections made

无代码修正。遵照任务边界，仅新增此review报告；R1交coder修复后独立复查。

## 5. Remaining risks

本次未运行50epoch或性能诊断，因此不声明运行完成、数值稳定或任何算法效果。数据与import检查没有执行optimizer step。历史V2日志仅首两训练步提供分区摘要，不应声称已有全训练逐步机制记录。N17/AO04/N18及全core的未完成部分不在此签署范围。

## 6. Whether experiments are safe to run

**最终：是，仅D-V2 16配置×50epoch。** R1及后续receipt生命周期修复已独立复查；本批可由luna-runner按原完整50epoch机械执行。其余研究组必须各自IMPLEMENT/REVIEW，不能沿用本报告放行。

## 7. 修复复查与最终签署

重新读取actual `bootstrap.py/v2.py/manifest.py`及coder修复报告；再次执行无训练population/import身份基本检查，并直接核对三个核心归档对应Git blob。未创建Runner、optimizer或运行fit。

|发现|修复后的实际核对|状态|
|R1核心依赖遗漏|`dependency_inventory('v2')`显式收录V2四wrapper、27个V2归档、core三个wrapper及三个data producer归档；37文件SHA加canonical依赖记录SHA，共38项。三个core归档与各自source_commit的Git blob独立比对均相同；装载前校验、真实模块路径断言通过。run与queue使用同一inventory/paths记录；原fit每epoch equality仍在，实际source集合导入前后相同|关闭|
|R1无关文件造成漂移|固定 `V2_WRAPPERS/CORE_WRAPPERS/CORE_PRODUCERS`；没有glob其它study的新py文件；canonical摘要只含选中records|关闭|
|R2测试访问记录|`observed_evaluate`只包装test，先检查selection_frozen存在，前后读取参数/optimizer digest和step计数；验证与训练路径保持原实现；test调用记录UTC与freeze SHA|关闭|
|R3探索性与review身份|config/queue/receipt含探索性标签，config保存review绝对路径与SHA；CLI `--review-file` 经help检查可用|关闭|
|复查新增的receipt SHA生命周期问题|第一版修复在原fit生成artifact SHA后覆盖receipt，会使其SHA过期。最终改为fit内仅写 `test_access_progress.json`，fit后首次新建最终receipt及execution_manifest；原manifest列入的progress此后不再改变，execution_manifest引用原manifest与最终receipt，无循环或失效引用|关闭|

复查源码SHA：`bootstrap.py=4d633616735e312a9d6bb075407ab00c92b1842de6ce4747f95e31dbc8f371ff`；`v2.py=51f3b9046c8dfa975534a8536e5475982894a0628bf76a4715cb9372e8a0decd`；`manifest.py=7d37895cccbba606a34076b21d6746c50b7772c01164e0bd6e1cd4b1781879e3`；core `data.py=424e3ae9b2c7d895a4144404da4f026715fbe108b62f5793d2aa0569cd87f00a`。

最终无blocking issues。原历史V2归档及训练公式未改。本PASS不代表正式实验完成、数值稳定性确认或研究效果结论，也不覆盖AC-2D/N17/AO04/N18。
