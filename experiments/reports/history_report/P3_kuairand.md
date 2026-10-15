# P3 报告：KuaiRand-Pure 数据管线重建

> 一句话结论：官方 KuaiRand-Pure 数据已获取并重建预处理管线；按 tab∈{0,1,4} 切三域 + 全局时间排序 + 8:1:1 按行序切分，**三域 users/items/instances/占比与论文 Table 1 全部 12 个数字逐位一致（diff 全为 0）**；run_kuaishou.py 修复 2 处 bug 后端到端跑通冒烟训练。

## 1. 数据来源与许可

| 项 | 值 |
|---|---|
| 数据集 | KuaiRand-Pure（KuaiRand 官方三版本之一；7,583 候选池视频上的全量日志） |
| 官方发布 | Zenodo record [10439422](https://zenodo.org/records/10439422)（`KuaiRand-Pure.tar.gz`，md5 0820331067a3784d9691136f772b35a7） |
| 实际获取途径 | **Kaggle 镜像** [arashnic/an-unbiased-sequential-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/an-unbiased-sequential-recommendation-dataset)（匿名下载成功；Zenodo 在本环境 DNS 被屏蔽，USTC 官方镜像 rec.ustc.edu.cn 匿名登录当时处于"升级维护中"（API 返回 4101），重试 40 次（约 6.7 小时）未恢复） |
| 许可 | **CC BY-SA 4.0**（Kaggle 页面标注 + 压缩包内 LICENSE 文件均为官方 CC BY-SA 4.0 文本）。允许研究使用、署名 + 相同方式共享。本仓库为研究性复现，不再分发数据 |
| 完整性核对 | 文件结构与官方 README 的 KuaiRand-Pure 清单一致（3 个日志 + user_features + video_basic + video_statistic + load_data_pure.py）；日志 193MB / 特征 10.4MB 与官方"184MB logs + 10MB features"吻合；行数：1,186,059 + 1,141,112 + 295,497 = 2,622,668，与官方 Pure 版"15 policies 下 1,436,609 + random 1,186,059"合计一致 ✓ |

**信息来源声明**：本节的发布渠道、许可、版本统计来自 **KuaiRand 官方文档（GitHub README + 压缩包内文件）**；Table 1 的目标数字来自**论文陈述**（arXiv:2404.18465v3）；下采样/排序等实现决策属于**我们的推定**（论文未开源该预处理，官方文档亦未规定切分顺序），均在下文标明。

## 2. 重建的预处理管线（`experiments/p3_prepare.py`）

1. 三个日志按官方文件顺序读入并合并（2,622,668 行）。
2. 过滤 tab∈{0,1,4} → 2,507,666 行（95.62%）。**这三个 tab 与论文 Table 1 的三域逐位对应**（见 §4）。
3. join user_features_pure（user_id）+ video_features_basic_pure + video_features_statistic_pure（video_id，取 counts 列）：**inner join 零行丢失**（用户表覆盖 27,285 人、视频表覆盖 7,583 个，是日志的超集）。
4. 缺失值填 −1（onehot_feat4: 874 行、onehot_feat12–17: ~714 行、music_type: 203、tag: 96；均为特征类别缺失，非日志缺失）。
5. 全部 34 个特征列转为字符串后写 `D:/CODE/data/kuairand/processed/kuai_rand.csv`（416 MB），列 = run_kuaishou.py 的 x_used_cols + is_click + long_view。

## 3. 划分规则与候选方案

下游管线（与 MovieLens 同一套）按**行序** 8:1:1 切分。行序的选择即是划分规则。候选方案：

| 候选 | 方案 | 结果 |
|---|---|---|
| A | 文件顺序拼接（std 4/08–4/21 → std 4/22–5/08 → random 4/22–5/08）+ tab∈{0,1,4} | 聚合统计与论文一致，但 **8:1:1 切分损坏**：random 日志只含 tab=1（1,178,025 行全部是主域），val/test 段落在 random 日志内部 → 验证集只有域 1，域 0/2 为空，评估直接崩溃 |
| **B（采用）** | **全局按 (date, time_ms) 时间排序** + tab∈{0,1,4} + 8:1:1 按行序 | 三域在每个 split 内都存在（train: 172,782/1,743,898/89,452；val: 2,692/246,232/1,843；test: 2,613/246,284/1,870）；无时间泄漏；确定性可复现。**与论文 Table 1 全部数字逐位一致** |
| C | 不过滤 tab（15 个 tab 全保留） | 得到 15 个域而非 3 个，与论文的 3 域设置不符 → 否决 |

**采用 B 的理由**：(i) 是唯一同时满足"论文三域聚合统计逐位复现"+"8:1:1 行序切分可用"的无泄漏方案；(ii) 时间切分是推荐日志的自然语义（后续模型不使用时间特征，不影响特征空间）。**声明：论文未公开其切分顺序（E1 已核实 KuaiRand 预处理未开源），我们的时间排序属于推定重建；聚合统计（Table 1）不受该选择影响。**

## 4. 与论文 Table 1 的核对（采用方案 B）

| 域 (tab) | users | items | instances | 占比 | 论文 | diff |
|---|---|---|---|---|---|---|
| d0 (tab=0) | 15,398 | 6,233 | 178,087 | 7.10% | 15,398 / 6,233 / 178,087 / 7.10% | **全 0** |
| d1 (tab=1) | 27,049 | 7,580 | 2,236,414 | 89.18% | 27,049 / 7,580 / 2,236,414 / 89.18% | **全 0** |
| d2 (tab=4) | 11,809 | 4,633 | 93,165 | 3.72% | 11,809 / 4,633 / 93,165 / 3.72% | **全 0** |

（CSV：`experiments/results/P3_dataset_stats.csv`，含候选方案对照。）

## 5. run_kuaishou.py 调试记录（该脚本系另行获得、非官方仓库内容）

| # | 问题 | 修复 |
|---|---|---|
| 1 | 构造 CTRTrainer 时漏传 `optimizer_params_darts`（默认 None 被 `**` 展开 → TypeError） | 补传 `{"lr": args.learning_rate_darts, "weight_decay": args.weight_decay}`，并新增 `--learning_rate_darts`（默认 1e-3，与 run_movielens.py 一致） |
| 2 | Windows spawn 无法把 ~700MB 的数据集对象 pickle 给 8 个 DataLoader worker（`reduction.dump: OSError Errno 22` / pickle truncated） | `num_workers=0`（进程内加载；确定性；此规模下速度足够）。**该脚本此前从未成功运行，不存在需要保持一致的"原版轨迹"** |
| 3 | （潜在，未触发）tab 过滤后 `domain_indicator` 若不从 0 重编码会导致域 0 为空 | 脚本已有该修复（原注释所述），实测通过 |

## 6. 冒烟训练（1 seed × 1 epoch，as-published 超参）

`run_kuaishou.py --dataset_path D:/CODE/data/kuairand/processed --epoch 1 --seed 2022`：

- trainable params = 1,614,322；输入维度 34×16 = 544（34 个 sparse 特征，无 dense）。
- epoch 0 验证 6 项平均 AUC = 0.5601；test：d0t0 0.4576 / d0t1 0.4752 / d1t0 0.5568 / d1t1 0.5883 / d2t0 0.5950 / d2t1 0.6066。
- 1 epoch 的模型远未收敛（MovieLens 侧 1 epoch 也只有 ~0.72），**此结果仅证明管线可运行**，不作为任何方法的证据。主实验（P5）将给足 epoch 并配早停。
- 论文报告的 KuaiRand M3oE 平均 AUC ≈ 0.7460（Table 3，二手信息）——留给 P5 核对。

## 7. 与论文的偏差声明（汇总）

1. **切分顺序为推定**：论文只说 8:1:1，未说行序依据；我们采用全局时间排序（§3 方案 B）。聚合统计不受影响（逐位一致），但 train/val/test 的具体样本与论文可能不同——这会带来微小的、无法消除的复现不确定性。
2. **特征集合是脚本的子集**：论文说"21 user + 12 video + 17 common"共 50 个特征；run_kuaishou.py 实际使用 33 个原始特征列（7 个 user 离散 + 18 个 onehot + 7 个 video + counts + domain_indicator）。论文未给出特征清单，无法进一步对齐。
3. **KuaiRand-Pure 的 is_like 等其他反馈**未使用（管线只定义 click/long_view 双任务，与论文一致）。

## 8. EXP_CONTEXT 更新

已把 KuaiRand 重建方案、许可与来源、num_workers 适配写入 `experiments/EXP_CONTEXT.md` §7。P5 编排器的 `kuairand` 配置指向 `D:/CODE/data/kuairand/processed`。
