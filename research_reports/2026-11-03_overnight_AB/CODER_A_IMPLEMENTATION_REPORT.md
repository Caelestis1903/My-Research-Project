# CODER A IMPLEMENTATION REPORT

实现阶段状态：COMPLETE；家族为`A_HYPERGATE`和`A_AUTOIFS`。依据已批准的`DESIGN_SPECIFICATION.md`第4、9节及冻结02实施文件，未改变数学定义、科学矩阵或预算。

## 文件与来源

- `published_models.py`：直接映射冻结HG/AI配置至原adapter；不导入旧BASP/旧训练入口。为评价器提供`restore_model_meta`恢复AutoIFS阶段元数据。
- `published_training.py`：实现共同的`create_session / advance_epoch / export_session / restore_session`四函数。训练view无test数组，返回epoch记录仅含标量/指标，完整模型和优化状态留RAM。
- `vendor/sota2526/{hypergate,autoifs,sources}.py`：三个文件逐Git blob精确匹配baseline固定commit `d2fdd272a2cdf7da626e29b97f61555d25271f89`。HG blob=`0cc1365b4c72722684f344f2a4e43b8c49188b87`，AI blob=`7a805d713a4b8ac55d8f95192707efb1e79919dd`，sources blob=`bbfcb896611be55e74d16f505730a58d7b82d466`。
- `vendor/sota2526/SOURCE_MANIFEST.json`：保存上述文件SHA256/源路径/blob及实际导入的外部依赖源码身份。外部RecStudio和AutoIFS根分别为冻结commit `f08cca607e998e8be4c5e915a393346baffa71f6`、`65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e`；实际导入文件均核对工作树blob与对应HEAD相同。
- `tests/test_published_training.py`：四项小型真实adapter CPU检查。

## 训练与恢复行为

HG保留原生objective、辅助项、初始化、三个gate和一次Adam更新；完整50epoch，无早停。AI保留search两个互斥完整参数组，每批两次Adam调用，search温度按原乘法顺序推进；在进入零基search9及其温度递增之前捕获rewind。第10次search完成后导出最终selector，新建模型载入rewind、移植selector和gate embedding，启用hard ticket/temp100并冻结gate，再新建无历史状态的reuse Adam。reuse完整50epoch。

HG/AI均沿用PCG64 SeedSequence `[seed,dataset_index,method_index,formal_index=1,epoch_argument]`；HG参数1–50，AI search0–9、reuse101–150，自然无放回permutation、尾批保留。原生标签mask全True，不使用旧T0/F人口。每epoch记录真实更新次数、Adam调用、样本/标签暴露、各目标分量、行顺序hash和运行成本。

最终stage由公共selection严格按V六格macro AUC选点；search不进入候选。selected/E10/E50快照包括完整state_dict、buffers、model hash、validation及AI ticket/temp/gate冻结元数据。恢复先重建模型/参数组、载入全部optimizer状态，再还原RNG；无模型状态文件IO。

## 实际检查

使用项目Python `/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python`。

1. `python -m py_compile experiments/overnight_ab_20261103/published_models.py experiments/overnight_ab_20261103/published_training.py`：PASS。
2. `PYTHONPATH=/home/xzf/Code/phase0-3 python -m pytest experiments/overnight_ab_20261103/tests/test_published_training.py -q`：**4 passed in 2.96s**。运行目录为独立执行树；Git固定main取回尚在进行，PYTHONPATH只供已核对的旧history公共指标/状态函数依赖。新adapter、训练实现及测试均来自执行树。
3. CPU检查覆盖HG/AI的真实单batch backward和目标有限性；连续两epoch与一epoch→CPU RAM快照→恢复下一epoch的模型/optimizer/RNG/选点/数据顺序hash逐值一致；Auto search9→search10→reuse1两处恢复边界一致，rewind身份正确、selector移植相同、reuse optimizer初始状态为空、gate冻结；HG域token只加一次且logits/辅助目标与原生网络等价。

首次CPU命令因规划`protocol.json`尚未复制到执行树而4项FileNotFound失败；复制冻结规划后上述4项通过，未因此更改代码或科学配置。

两数据真实原batch的GPU smoke及独立review由后续阶段执行；本coder没有启动正式训练，也没有分析研究指标。当前A代码无已知实施阻塞。

