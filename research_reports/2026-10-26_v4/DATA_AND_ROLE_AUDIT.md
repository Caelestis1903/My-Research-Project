# 数据、角色、源码与选点身份

两raw CSV完整SHA与规范一致：ML `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`，KR `a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f`。prepared x/y、fit-only词表/scaler/OOV、原始域/标签与固定整数角色协议均经独立完整核对，未改原始行或比例。ML8字段，KR34字段，完整schema/正负数/逐角色行hash在两个[evidence数据manifest](evidence/)。

自然fit域计数ML124759/254879/260495、KR138225/1395118/71561。角色如下；P_label是历史父池，不重复计入原子角色。

| 角色 | MovieLens行数 | KuaiRand行数 |
| --- | --- | --- |
| fit | 640133 | 1604904 |
| P_form | 24004 | 60183 |
| P_read | 24006 | 60184 |
| P_cal | 16003 | 40123 |
| P_update | 16004 | 40123 |
| Q_explore | 40008 | 100307 |
| validation | 100021 | 250767 |
| test_exposed | 100021 | 250767 |
| Q_legacy_confirm_exposed | 40009 | 100308 |

fit用于native/pilot/source，P_form用于η和reader预拟合，P_read拟合最终读取，P_cal选完整六头，validation只选native。P_update主算法不用。Q_explore开发，test/legacy已暴露；无新数据确认，也没有6101–6104正式训练。任务ML为rating>3、rating>4；KR保留实际is_click/long_view，不从名称假设嵌套。

ML原始用户/物品ID可恢复，KR processed表不含可信原始ID，禁止用UNK或profile编码替代。ML原始用户OOV比例在validation约76.16%、Q约0%、test约19.32%；支持分布是竞争解释，不是原因识别。完整预测的support分析与原始行索引保留在artifact中，KR相关分层为MISSING并用row-bootstrap退路说明限制。

## 冻结选择与状态

| 数据 | seed | selected epoch | native权重hash | epoch24权重hash |
| --- | --- | --- | --- | --- |
| movielens | 5101 | 13 | 9a7d8a18544dfba200132adfccb994a9628376bf915336e002a89346ba03ae19 | 3588a4a1cdcb6ef389890b6442a72468bcce119d489bbf5f0adb7f477e87d44b |
| movielens | 5102 | 13 | 159ad22eda369e5b38c7bf895281130f41691b0fee0a744ce99000cc5fdda988 | c892c92f767a2462022a7e00a43c2e82efee016373772b3907f3a813d23a0b74 |
| kuairand | 5101 | 14 | 921fd92949068ab6b9f10deea9decfb4f51f91c4dad066813ebc6e5dd97a21ea | da3af36dfb4b4f0790f26ff232b59a182e10080aeaeb1c1bf94e2ac729b4ef22 |
| kuairand | 5102 | 16 | 7380d8eae48fc3c4ea5f02e8cd56369ed77199826353de44a780227decd841ad | f5d1b83729e2020509a68e3ae80a3bb79b8a72cc74afc57399287a181b990c4a |

选择严格为validation macro LL的epoch12–24最小值/最早平局；24epoch全跑完，pilot/F固定pass12，reader/direct所选与pass10分别保存预测。选点、更新/库形成和终点不可混为同一阶段。prefix保留原FP32 sigmoid再FP64评价；P2保留原已锁FP64概率路径，未事后改变AUC ties。

fixed-fusion M3oE：exp_d/exp_t/bal_d/bal_t全1，4shared experts，fcn[512,256,256,64]。数据自然两任务Sample BCE，full40960物理/逻辑batch、短末批，coupled Adam .001/.9/.999/eps1e-8/wd1e-5；η/read wd0，无AMP/TF32，无复制/噪声/AutoML。h确为4shared专家逐个LN后拼接256，非STAR同维向量。canonical原始初始化后拷贝固定骨干，两schema真实前向/权限已审。

源码来自当前main独立分支；vendor7文件与pinned c5618e28 Git blob逐项相等，历史v3五报告与固定8a4b3b29逐项相等。前缀source1273530d…、P1需求77c80737…、P2 fc97ec1d…与最终all-source ca1ac6b7…分别准入，完整SHA在审查/冻结收据。未合并旧实验分支，未恢复有意移除的learned fusion/DARTS。

最终freeze `7163b07065bfdb7462dd05723b7c761acb8ea7cd41693e9fdb6d93a7dad401d4`：46实际variants、184角色导出、252临时完整状态身份。全行raw-X重建最大误差8.28890167393e-07≤1e-6，模型状态不变。模型与优化器最终按用户要求删除，identity/预测信息留存。Q/outer μ机制分组遗漏另列MAJOR，184完整性不能代替该分组完整性。
