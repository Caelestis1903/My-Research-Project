# 自然MovieLens数据协议审计

状态：基础数据构建及数组身份已核对；正式模型训练尚未完成。本报告只确认数据与解析权重，不宣称性能/机制已经复现。

|训练人口|总行数|d0|d1|d2|
|---|---:|---:|---:|---:|
|movielens_natural|800167|155949|318599|325619|
|movielens_artificial_d2_fixed2022|187500|34500|3000|150000|
|movielens_natural_matched_n187500|187500|36543|74656|76301|

原始来源 `/home/xzf/data/ml-1m/ml-1m.csv`，大小75,514,352字节，SHA256：`f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`。使用原始row顺序全局80/10/10划分：train 800167、validation 100021、test 100021；原始共1,000,209行。物理年龄域保持{1,18}/{25}/{35,45,50,56}，两个native标签保持rating>3/rating>4。

特征保持user_id/movie_id/gender/age/occupation/zip/cate_id/domain_indicator；cate_id为genres首类型，age归一化和类别词表仅从base-train拟合，OOV=0。元数据/选择训练划分属于允许训练数据；不能用validation/test拟合。

encoder SHA：`05ab58ef6b9baf0589a3a476789006d8cc1b5c10eba53f1934c0c42d755f9e6b`；scaler SHA：`b73c4a08a40b2059f892c72edc6f659baf2acdd430b6ca1a50068bcd0dde1c1e`。三个人口共用同一schema、val/test逐数组哈希；人工对照准确重现d2_large固定人口seed2022，自然等N采用预登记最大余数配额和逐域固定RandomState，不随优化seed变化。全部训练/验证/测试ID、标签、编码数组与NPZ哈希在服务器data/manifest.json，交付快照另存本报告data_manifest/。

完整自然训练人口是人工版本的4.267557倍；d1从3000变为318599，d2占比从80%变为40.694%。因此仅比较完整自然与旧人工性能不能识别单独的比例效应。自然等N人口d0/d1/d2为36543/74656/76301；支持数量和配比控制按设计分别执行，仍只有一个固定子集，不能声称跨子集普遍因果。

解析权重审计（STRONG，仅代数/固定人口）：SN/SB均采用Sample时，SN权重1，SB权重3p；Equal时EN权重1/(3p)，EB权重1。人工EN的d1单样本权重20.8333，而完整自然最高EN权重约1.7103。SB权重本身的名义ESS/B由人工约.4945变为自然约.9205；此量只描述权重离散度，不包含域内梯度差异、模型状态或训练路径，不能用它代替协方差实测或推断AUC。精确数值见OBJECTIVE_SAMPLING_WEIGHTS.csv。

N19/CM19还需按固定规则从base-train留5%meta；main与meta均保留row-ID哈希。main不等于800167并非人工域比例重采样，实际整数规模由各域floor规则确定并另行记录。其模拟稳定性、噪声或缺标条件须标记suite，不能把合成task1写成native rating>4。

历史记录存在两种row hash口径：部分后期报告对数组原始bytes求SHA，而stage_S1_data.row_ids_hash含其原函数编码口径。应比较同一算法的哈希或直接比较ID数组，不能仅因字符串不同断言人口不同。人工bundle某些legacy字段继承setting=natural；以独立dataset_config及明确selected/base ID身份为准，不能将其误报为完整自然人口。

测试标签率/数量仅为数据审计，尚未用于选模型或方案。本次复用历史测试总体，后续结果均属探索性复现。
