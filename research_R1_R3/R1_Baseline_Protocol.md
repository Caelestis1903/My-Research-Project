# R1 Baseline Protocol

日期：2026-10-11（Asia/Shanghai）。仅R1。状态：**NOT READY for execution**；历史指标可读取，checkpoint严格复用不可用，数值重运行复现NOT PERFORMED。依据 work/R1_Code_Audit_Report.md 与 work/R1_Code_Audit_Inventory.json。本文不含新训练或评估结果。

## 1. Canonical baseline身份

B0固定为Stage S1 KuaiRand-Pure/natural/M3OE/joint/Sample：StageS1M3OEAdapter的设备安全forward，同原始构造器；D3/T2/N4、9个专家、6 gate/tower、4 global sigmoid架构scalar。不能混用旧examples trainer、S0权重尺度、Equal、SD reference、S6 T/HT/DH或S8 continuation。

历史S1源码hash与当前对应文件一致；根目录非git仓库，具体文件身份以Inventory为准。关键hash：

| 文件 | SHA256 |
|---|---|
| experiments/stage_S1_data.py | c050b5b168d4c0aedbbc37a190ab6e2b2339164198351fd56475673e53fa7afd |
| experiments/stage_S1_models.py | 1f6fa05358b047b048b115681cd62ac2f833ffd1d0892ad68b64b91216caf691 |
| experiments/stage_S1_protocol.py | 76ed4a851e6cdf15493be5628a08db4647a28eb3a2a11e4e1d7c1974a704218f |

其余runner/objective/model/config/manifest/parameter_manifest hash见完整Inventory，不以三文件hash代替整套版本身份。

## 2. Sample / Equal的实际定义

S1 Sample：L=(1/(B*T))Σ_iΣ_t BCE(p_it,y_it)，即torch binary_cross_entropy(reduction=mean)，不按域再乘p_d、不用balanced domain sampler。总体可写为Σ_d p_d·(1/T)Σ_t BCE_dt，p_d为train份额；batch中域计数自然波动。batch平均和全epoch样本加权日志需区分，最后小batch仍按原runner更新。

S2 Equal：固定p_d=n_d/N_train，w_d=1/(D*p_d)，L=(1/B)Σ_i w_di·mean_t BCE_it。全数据期望下每格1/(D*T)；batch中未出现的域不补项，也不把当前batch的权重重新归一。Sample为w=1。参考实现stage_S2_objectives.py:58–98,178–220，S3/S6继承同口径。

历史examples为六格batch内BCE之和，S0记录Sample以D*p_d系数sum域任务均值，数值尺度与S1不相同；论文Equal型无权重之和也不能直接视为同一个优化协议。S1 canonical修复四scalar错误进入两optimizer的legacy ~bool筛选。历史名称一致不等于loss实现、尺度、优化预算一致。

## 3. 数据版本与split

原数据字节hash已由审计Agent实际读取核对，未重新训练或预测：

| 数据 | 大小bytes | SHA256 |
|---|---:|---|
| KuaiRand CSV | 416,370,411 | a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f |
| MovieLens CSV | 75,514,352 | f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4 |

绝对位置在Inventory。KuaiRand hash与历史manifest一致；这证明字节身份，不等于本次独立复算split/encoder。未来load必须生成并匹配row/label/feature/split fingerprints。

KuaiRand：全局原始行序80/10/10 split-first/filter-second；processed数据按date/time_ms排序，不能在split之前shuffle。tab0/tab1/tab4→d0/d1/d2；标签使用stored_is_click/stored_long_view，不重新构造阈值。词表仅base train拟合，OOV=0。

下表为历史S1/E4记录，未在本次重新解析原CSV复算：

| 域 | Train | Validation | Test | Train份额 |
|---|---:|---:|---:|---:|
| d0=tab0 | 172,782 | 2,692 | 2,613 | 0.086126935 |
| d1=tab1 | 1,743,898 | 246,232 | 246,284 | 0.869283776 |
| d2=tab4 | 89,452 | 1,843 | 1,870 | 0.044589289 |
| 总计 | 2,006,132 | 250,767 | 250,767 | 1 |

每样本两个任务都有标签。d1 click train prior≈0.3778，validation≈0.2097，test≈0.2105；train→validation/test域mixture有明显时间shift，不能用train规模代替评估难度。小域validation/test只有约2千行，选择11档ρ会有选择噪声；严格验证选择、独立测试和完整曲线是必要报告内容。

MovieLens只审计，不进入本最小训练设计：age分域{1,18}/{25}/{35,45,50,56}；任务rating>3、rating>4；原始行序80/10/10，词表/scaler仅base train拟合。L/M/S=150000/34500/3000，以RandomState(10*seed+physical_domain)抽train子集；d2_large设置下d1最小、d0居中。S1/S6ML expert_num=4。不能把KuaiRand域大小或任务难度直接映射到MovieLens。

## 4. 训练、优化与随机协议

| 配置 | 锁定值 |
|---|---|
| embedding / FCN | 16 / [512,256,256,64] |
| batch / workers | 40960 / 0 |
| train loader | shuffle=True, drop_last=False |
| train epochs | 完整50 epochs，无实际early stop |
| shadow patience | 10，仅记录不break |
| main Adam | lr0.01, wd1e−5, betas(0.9,0.999), eps1e−8 |
| arch Adam | lr0.001, wd1e−5, betas(0.9,0.999), eps1e−8 |
| main更新 | 每普通训练batch一次 |
| 原四架构scalar更新 | 每epoch一批train_loader数据一次，非validation |
| scheduler/额外方法 | 无 |
| 原raw scalar初值 | 全1，sigmoid≈0.7310586 |

种子2022/2023/2024/2025；设置random/numpy/torch/cuda seed，cudnn deterministic=True、benchmark=False。没有torch.use_deterministic_algorithms强制保证；跨依赖版本不能承诺位级一致。未来各臂必须从同一B0初始化tensor复制并保同batch序列；新增参数不得消耗原RNG导致假配对。

新增ratio/temperature参数的optimizer归属不是历史baseline事实，由Design Specification与Charter单独预注册；原四scalar及所有已有参数的optimizer规则不能变化。任何新增actual patience10均形成不同baseline族，禁止直接复用上述50epoch历史结果。

## 5. Validation选择与test使用

每epoch完整验证集六格macro AUC最高选模；严格>替换，tie最早epoch。固定50epoch全部完成。test不参与训练、架构更新、选epoch或选ρ；selection冻结后评selected/final/common9/19/29/39/49。后续实验selected为主端点，common/final仅描述，不从test挑最有利checkpoint。

指标路由task-major；逐格AUC使用原预测排序，LogLoss固定clip epsilon=1e−7。六格mean与训练份额加权逐格AUC分别命名macro/W；不能直接称混合总体AUC。真实pooled AUC如报告必须从跨域原始预测重算并区分task。

## 6. 已有四seed模型工件与历史指标

本次逐目录审查16runs：S1 KuaiRand natural、S1 MovieLens d0_large、S6 KuaiRand F、S6ML MovieLens F各4seeds。每run声明7个checkpoint，共112个声明文件均未找到；selected可加载0/16。原optimizer/archoptimizer/RNG二进制工件也缺失。目录存在、manifest status=OK、metrics CSV和predictions_test.npz存在，不能证明checkpoint仍可用。

| 族 | selected epoch（2022/23/24/25） | 可加载selected |
|---|---|---:|
| S1 KuaiRand natural | 37/42/46/26 | 0/4 |
| S1 MovieLens d0_large | 6/6/7/6 | 0/4 |
| S6 KuaiRand F | 30/47/41/24 | 0/4 |
| S6ML MovieLens F | 7/8/8/12 | 0/4 |

可访问树 *.pth/*.pt/*.ckpt搜索无命中，但两处S3 archive拒绝访问；只断言所核查目录权重缺失，不断言全磁盘绝无checkpoint。

主Agent读取已保存S1 KuaiRand metrics_summary.csv并按六格简单平均核对下表（历史artifact算术摘要，不是新模型复现）：

| Seed | Epoch | Val macro AUC | Test六格macro AUC | Test六格macro LogLoss |
|---|---:|---:|---:|---:|
| 2022 | 37 | 0.684259 | 0.679009 | 0.437686 |
| 2023 | 42 | 0.690716 | 0.678227 | 0.433627 |
| 2024 | 46 | 0.681288 | 0.673248 | 0.435731 |
| 2025 | 26 | 0.697365 | 0.670475 | 0.438571 |

每seed六格齐全；这些数值仅作为历史参考和找回原权重后的核验目标，不用于选新臂或修改门槛。baseline性能重新前向复现 **NOT PERFORMED**，不能把CSV重读标为REPRODUCED。

## 7. 依赖与严格复用判定

S1历史Python3.8.20/Torch2.0.0/CUDA11.7/cuDNN8500（EDDAenv）；当前以及S6/S6ML为Python3.13.5/Torch2.9.0+cu126/CUDA12.6/cuDNN91002。当前numpy2.1.3、pandas2.2.3、sklearn1.6.1，cuda可用。requirements.lock含本机file://路径，不能视为可移植S1环境锁。

S1/S6ML已记录源码hash匹配；S6的stage_S6_report.py有drift，训练相关model/data/runner的已记录hash未检出变化。不能无git归因所有历史版本差异。

**严格历史复用：当前不允许。** 需要找回真正的权重、checkpoint hash、code/data/config/split/encoder指纹与环境，加载并重评同一selected，逐格AUC/LogLoss及selection身份与原记录一致；数值容差必须在复评前声明，不按差异临时放宽。建议原预测最大绝对差≤1e−6、逐格AUC/LogLoss差≤1e−6；不合格则作为当前runtime重建族重新定义baseline。

若找不回权重，未来任务仅能按固定协议匹配重训B0并让所有干预臂使用同一当前runtime。本次选定未来重建runtime为当前Python3.13.5/Torch2.9.0+cu126，完整版本登记于Inventory；即使源码与数据匹配，也称“当前runtime匹配重建”，不称严格复现S1 EDDAenv。历史数字只能作上下文，不能拼入paired B0。2个B0重训计入Charter固定10run预算，无本次执行授权。

## 8. 启动门禁与本次完成状态

| 检查项 | 状态 | 处理 |
|---|---|---|
| 六份R0正式报告 | COMPLETE | 已读 |
| 三份R0 work报告 | MISSING | 九份要求未满足，不虚构 |
| 当前结构/目标/选模审计 | COMPLETE | 代码证据+参数清点 |
| CSV字节身份与历史配置 | COMPLETE（限定范围） | 未复算split/encoder |
| 四seed selected checkpoint | NOT AVAILABLE | 不做冻结扫描 |
| baseline重新前向复现 | NOT PERFORMED | 不宣称复现 |
| 依赖严格历史恢复 | NOT ESTABLISHED | 未来匹配当前runtime重建 |
| 独立Charter审核 | 见最终Charter和Review | 通过设计不等于可训练 |

R1报告与预注册设计交付；资源准备度NOT READY。没有改模型、没有运行R2/R3、没有覆盖已有S0–S8文件。未来仅在另行授权且DESIGN→IMPLEMENT→REVIEW通过后由Runner开展已锁定预算，完成前均不得报告性能GO。
