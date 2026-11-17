# C05 数据、源码、归档与模型可用性

已完成10条U新轨迹×50epochs、500ACK及20选中模型验收；30条C/W/S为已绑定数值控制。本页仅陈述已完成的实际文件与时间点可用性。

## 原始证据归档

- 实际压缩归档：677004657字节，原始成员总字节1263254172；4026成员，raw2381/source55/control1530/design7/receipt53。
- SHA256：`fa08b88dd6d488b48f1810a4e900e680ba3a35e577d2118f6b68accf92e44250`。
- 81个分卷，每卷上限8388608字节，最后一卷5916017字节。必须按manifest.parts顺序合并，再校验总长度与SHA；不依赖文件系统枚举顺序。
- [完整成员和分卷清单](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/attempt1/RAW_RUNS_MANIFEST.json)、[分卷索引](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/attempt1/RAW_RUNS_PART_INDEX.json)、[接续/恢复实证](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/ARCHIVE_CONTINUATION_04.json)。所有分卷位于同目录。
- 物理原始树`/tmp/phase0-3-c05-01a1e9a9/attempt1`；完整归档与暂存卷`/tmp/phase0-3-c05-01a1e9a9/publication`；逻辑发布树为上方attempt1。不存在跨根符号链接。

## 完整性与收尾偏差

首次标准pack已完成全部语义检查和成员内容回读，随后在真实3A+256MiB+1GiB容量门槛暂停。容量缺口281239635字节。接续仅复用同一份已验证归档，逐成员再比对当前文件与归档字节哈希，执行冻结pack的原样剩余语句；真实可用3441762304≥3373191251字节后才分卷。原归档inode/mtime/SHA不变，没有重建归档或修改55文件科学源码。临时转存43个本项目已发布C04分卷，最后43个全部原位恢复，原有89个分卷、C04manifest和publication receipt均核验不变。标准CLI没有再次运行；机械接续代码ARCHIVE_CONTINUATION.py与证明单独发布。

原始复审记录IMPLEMENTATION_REVIEW_CHECKS.json的类型描述被guard误识别为模型容器：原文件5167字节、SHA f9a949b78a9b912e35e53fd96258bccad3ccf5ee68b11d8ce9941ec7cb28e348保持不变并单独发布；归档REVIEW_SCHEMA_TEXT_PROVENANCE.json保存可逐字节还原的UTF8原文及SHA。

三份早期截断的数值遥测gzip原始压缩字节分别420/1266/49429字节，保存在同根.incomplete_bytes别名和PARTIAL_TELEMETRY_BYTE_PROVENANCE.json的明确hex表示中，可精确还原原文件与SHA。另保留可验证完整JSONL前缀1/15/1385条；未补造footer、缺失记录或未返回训练工作。目录别名和原路径由PARTIAL_TELEMETRY_CLASSIFICATION_CORRECTION.json解析。PID2982386的缺失terminal/supervisor、父子归属与未返回工作仍为未知，不由遥测文件名推断。

## 模型与Test

父智能体在2026-11-17T18:08:49.036397+00:00重新直接get核验10对/20个新U选中模型，共95508960字节，owner PID2926581、工具会话16855；见[时间点可用性凭据](../../artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_05/SELECTED_RAM_AVAILABILITY_UPDATE.json)。模型、EMA、Adam、RNG、loader和BytesIO仅存RAM，没有模型/optimizer/checkpoint文件进入归档。该凭据是特定时点的实际核验，不是持久化模型备份。

C04旧owner消失，C04新训练模型当前不可用；其数值、源码和已发布归档仍有效。C05只复用控制数值，未重建或冒称恢复这些模型。Test全程封闭，未选champion、未完成final freeze、不自动触发Test。本轮交付后按用户指令停止，不启动C06。

## 复现与判读

当前科学source55 SHA f716b4e3025621f20d10aae4547379b9adafa69f2824af769f4e01aaef1fe59d；冻结源码随归档提供。项目环境为phase0-3-env，Python3.13.5/PyTorch2.9.0+cu126，依赖锁requirements-phase0-3.lock；external Adam来源另有哈希。对数值复核不需要读取任何模型状态。源数据为自然ML/KR train/validation，Test未参与。

400终点、2400终点cell、4000曲线、24000曲线cell、28900配对seed与5780汇总，以及权重/质量/统计/诊断/阶段/物理成本表、schema和TABLE_RECEIPT均在归档attempt1/tables。报告分析和图表代码单独发布；全部逐seed绝对值见NUMERICAL_APPENDIX.md。原始表生成在父验收之前，RESOURCE_EXECUTION_AUDIT中的parent_storage_acceptance当时为null；后续PARENT_STORAGE_ACCEPTANCE独立绑定实测验收，不重写原表或TABLE_RECEIPT。
