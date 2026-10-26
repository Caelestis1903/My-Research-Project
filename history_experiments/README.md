# 历史实验分支原始文件归档（history_experiments）

本目录整理截至2026-10-26确认的**14条仍有独立文件内容未纳入归档前main**的研究/实验分支。各分支保留为独立子目录；不执行覆盖式合并、不修改原始实验源码、配置、数值、报告或结论。

## 归档规则与完整性

- 对比基准为归档前 `main` 提交 `139fb294b0dd1e024b2fa1e906a4bd5964bb0db0`（Git tree `63b0f5ad30d7e1b64c8c3b2800d2b2b530d6b6eb`）。
- 逐层比较原分支Git树与该main基准：**完整保留原分支中所有相对main缺失或内容/文件模式不同的原始文件**，保持分支内部的原始相对路径。与main完全一致的共同基础文件不重复纳入，否则大量基础项目文件会在14份归档中重复出现。
- 对完整新增目录，直接引用分支已有的Git tree；对单文件引用原始Git blob。Git对象哈希保持不变，不重新生成或修改文件内容；也不以最新实验结果覆盖历史版本。
- 原始分支名中的斜杠 `/` 在顶层归档目录名中转成 `__`，确保每条分支都作为 `history_experiments/` 下的单独一级子目录。完整映射和精确 SHA 见 [MANIFEST.json](MANIFEST.json)。
- 不含同一分支上原本就与main相同的文件，因此此处是**尚未纳入main的文件版本归档，不是可独立运行的完整Git工作树快照**。如需逐字节重建完整原始分支，请按清单记录的原提交SHA checkout；此处原文件内容均直接使用原Git对象。
- 祖先/后继分支**逐条保留各自版本**，即使同一文件存在于多个分支，其内容也不被择优合并。不同版本、消极结果、过程偏差和报告均保留。
- 归档不会自动删除源分支，亦不会将实验历史文件覆盖到现有 `experiments/`、`research_plans/` 等工作目录。

## 分支与目录

| 原始分支 | 当前目录 | 研究内容 | 原提交 |
|---|---|---|---|
| `10-24` | [`10-24/`](./10-24/) | 10月24日研究报告 | [`b64b29222f`](https://github.com/Caelestis1903/My-Research-Project/commit/b64b29222fcd36ca8ecbb323ccb127f7da1ccd57) |
| `26-10-25` | [`26-10-25/`](./26-10-25/) | MDMT-NEXT-v2 | [`619371d9c4`](https://github.com/Caelestis1903/My-Research-Project/commit/619371d9c4d57f6f0b3e8a71b686498f9e5f36d4) |
| `2026-10-25_15-56-04` | [`2026-10-25_15-56-04/`](./2026-10-25_15-56-04/) | MDMT-NEXT-v3 | [`8a4b3b296c`](https://github.com/Caelestis1903/My-Research-Project/commit/8a4b3b296ca5dc7349956fae9eca5e202aefe2c0) |
| `26-10-26` | [`26-10-26/`](./26-10-26/) | DFD-v4实验源码与结果 | [`dc32f60a6a`](https://github.com/Caelestis1903/My-Research-Project/commit/dc32f60a6aafce98a25838a485e02cead5425fc6) |
| `codex/execute-2026-10-15` | [`codex__execute-2026-10-15/`](./codex__execute-2026-10-15/) | 早期机制实验 | [`b17cb2707f`](https://github.com/Caelestis1903/My-Research-Project/commit/b17cb2707fc2448f09f21fd12279b2b3856c7756) |
| `codex/execute-2026-10-16` | [`codex__execute-2026-10-16/`](./codex__execute-2026-10-16/) | 早期机制实验续执行 | [`aa39e7b272`](https://github.com/Caelestis1903/My-Research-Project/commit/aa39e7b27285985bae9fa9a093f16c6d9634989b) |
| `codex/execute-2026-10-16-v2` | [`codex__execute-2026-10-16-v2/`](./codex__execute-2026-10-16-v2/) | B1重建与早期机制复核 | [`c31b531e21`](https://github.com/Caelestis1903/My-Research-Project/commit/c31b531e210daab4cbfb2e53eaa44c22b913264b) |
| `codex/execute-2026-10-16-v3` | [`codex__execute-2026-10-16-v3/`](./codex__execute-2026-10-16-v3/) | v3机制闭环 | [`2d04c6de6c`](https://github.com/Caelestis1903/My-Research-Project/commit/2d04c6de6c925eca8563f81b66af3f0e95f99183) |
| `codex/execute-2026-10-17-run` | [`codex__execute-2026-10-17-run/`](./codex__execute-2026-10-17-run/) | N17实验 | [`5550afbc57`](https://github.com/Caelestis1903/My-Research-Project/commit/5550afbc57f76e0b94c5783db16f4c1796933541) |
| `codex/execute-2026-10-18` | [`codex__execute-2026-10-18/`](./codex__execute-2026-10-18/) | N18六方向实验 | [`14cf09a386`](https://github.com/Caelestis1903/My-Research-Project/commit/14cf09a3865303f5e451636dc6407fd0e2cc1b8c) |
| `codex/execute-n19-2026-10-18` | [`codex__execute-n19-2026-10-18/`](./codex__execute-n19-2026-10-18/) | N19端到端实验 | [`c440350681`](https://github.com/Caelestis1903/My-Research-Project/commit/c440350681a7a3e41ab3f05eb1c4d31f419b989f) |
| `codex/conditional-paths-2026-10-19` | [`codex__conditional-paths-2026-10-19/`](./codex__conditional-paths-2026-10-19/) | 共享调制路径失活审计 | [`eab817b3de`](https://github.com/Caelestis1903/My-Research-Project/commit/eab817b3de626caee9ee9aeb1db4dd9aa3ef9e4c) |
| `codex/natural-movielens-reassessment-2026-10-20` | [`codex__natural-movielens-reassessment-2026-10-20/`](./codex__natural-movielens-reassessment-2026-10-20/) | MovieLens自然比例复核 | [`f0b1084b7b`](https://github.com/Caelestis1903/My-Research-Project/commit/f0b1084b7bcbde5a6c6cb6210c4ef7511bcfff8e) |
| `codex/m3oe-reproduction-2026-10-21` | [`codex__m3oe-reproduction-2026-10-21/`](./codex__m3oe-reproduction-2026-10-21/) | M3oE复现与生成/调幅 | [`c5618e2817`](https://github.com/Caelestis1903/My-Research-Project/commit/c5618e281785a4b7063ff9bf2a61cf38d4faf1be) |

## 与main已有研究方案的关系

2026-10-16 v2/v3、2026-10-17、2026-10-18、2026-10-19以及2026-10-26的主要*方案文档*已通过历史PR进入main；本归档主要补齐这些方案后续产生的代码、完整执行记录及结果。特别是 `26-10-26`：其代码及实验报告曾在PR #10中合入，但随后依用户要求从main文件树还原，因此本归档专门保存其目前不在main的原始文件。归档本身不恢复那些文件到根级执行目录。

## 后续使用

本目录是**保留历史证据**的归档，不是汇总报告；不同运行可能采用不同比例、采样/重加权设置、数据划分和选择准则。引用实验结果时，应从对应目录的原始协议和结果文件阅读，不要跨版本直接拼接数值。新增实验建议继续在独立工作分支进行，必要时更新本索引及归档。
