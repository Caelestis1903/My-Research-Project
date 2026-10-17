# N5-C实现与主agent自审

设计先于代码：N5_C_DESIGN_ADDENDUM.md。实现为n17_ml.py与n17_ml_control.py；训练step、fit、objective、原生state及逐epochmodel capture全部继承冻结底座。仅显式替换ML人口初始化，区分subset_seed和optimizer seed，config/split/cache/source字段据实登记。

必要核对preflight/ml_population_check/receipt.json：subset2022在optimizer seeds2030/2031及配对q下train/val/test row SHA完全相同；配对q初值相同，跨optimizer seed初值改变。训练N=187500、B40960、5更新、尾23660。仅一次192行CPU Adam更新与保存；不读取test预测/指标，初始化时的test行身份不等于test模型评估。并未运行正式ML实验。

主agent逐项核对与N1初始化：optimizer、loss、sampler seed流、固定fusion/detached routing、selection/diagnostic分半、source guards和resume state相同；ML数据和split文件指向subset2022。ML选择器要求全部pilot终态，共同eta需四条、own需两条完整轨迹，四点网格、平局及去重与N1一致。触发器只读已冻结N1资源门槛，非队列自动科研决策。延迟test脚本排除所有*_pilot，包含未来拓扑pilot，防止误将调参轨迹纳入独立确认。

Verdict: PASS_SELF_REVIEW，非独立review。仅在N1实际结果触发且主agent核对清单后可运行；N5-C当前NOT_YET_DECIDED。不据本工程检查宣称ML效应或跨环境普适性。
