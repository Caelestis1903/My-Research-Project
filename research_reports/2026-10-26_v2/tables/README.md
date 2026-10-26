# 原始数值与核算表

独立重复单位为训练seed。开发5201–5204、SCALE2022–2025分别成组；重放、OOF两折、任务、epoch、函数实例和bootstrap不增加n。每个最终效果条件须有四训练seed。`mean_delta`或配对表中的`mean`为候选减参照：LL负、AUC正更好。

- `S1_Q_*`：固定内容读取的四seed开发结果，含六格、域、任务、预定原生logit低/高群体与配对区间。
- `S2_Q_*`、`S3_Q_*`：实际三臂形成和触发归因，Q资源门槛决定不依赖暴露测试。
- `CORE_four_seed_*`：最终冻结原始预测复算出的所有实际方法、角色、selected/terminal、六格/域/任务、最差格、群体及配对结果；包含功能profile、预算、OOF/refit和选择身份。独立6201–6204若未触发则没有锁定cohort行，不能凭空填0。
- `SUPPORT_four_seed_*`：预先固定的fit-only编码OOV、ML全局/来源域原始user/item支持。KR原始身份缺失记MISSING；OOV不是原始user重叠的替代。
- `SCALE_*`：独立历史优化协议。B/Dconst01历史指标与本轮NO_AUG四seed匹配，旧raw p不可得，只作seed配对指标差。
- `COST_four_seed_*`：真实组件墙时、部署/活跃/训练阶段参数、候选政策、推理原始重复及四seed汇总。未测与0不同，失败seed不能从平均中删掉。

区间为描述性配对Student t95（df=3），不是多重比较修正后的确认性声明。within-seed bootstrap使用1000次原样保存的配对重抽样；ML按原始用户，KR按行。它是种子内不确定性，不与四seed t区间拼成新的大样本。最差cell先在每seed六格内求最差再汇总，区别于“均值最差的一个格”。

Macro为三个域×两个任务等权。`natural_logloss`、sample-weighted cell AUC、task pooled AUC是不同评价量；pooled AUC不是cell AUC加权平均。域LL低也可能来自标签基率，不直接称该域更容易。

较大的CSV在Git中以无损`.csv.gz`发布，服务器同时保留完整`.csv`；具体路径、原始与压缩SHA见上级`PUBLICATION_FILE_MANIFEST.json`。`gzip -dc table.csv.gz > table.csv`可恢复标准CSV，或用支持gzip的CSV阅读器直接读取。小表直接保留CSV。图由相同四seed表生成，PNG与独立PDF在上级`figures/`。

大规模逐行预测、φ、δ、请求、原始角色数据和完整逐prefix工件索引保留在服务器，不随Git上传；上级`ARTIFACT_INDEX.json`提供实际路径和SHA。所有已保存数组是输入导出/预测/诊断数据，没有模型、Adam或RNG检查点。
