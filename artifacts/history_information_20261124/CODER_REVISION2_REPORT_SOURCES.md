# CODER IMPLEMENTATION REPORT · 第二轮来源修订

本轮修订对应独立审查 `341d00f9` 的 GA31-R3 来源阻断，依据设计 07 §7.2 和 11 §3.3。

## 改动

- `sources.py` 在共享依赖导入前，核对主干 `358d3a4` 和发表 `c8675ae` 各自 Git 路径的已知 blob。补件优先使用另一条工作区路径下的同 blob 字节，其次读取本地 Git 对象。安装后重新读取和校验目标路径。若目标已有不同字节，核验会失败并保留原文件。
- `objectives.py` 在导入主干模型和发表训练依赖前预检，并在导入后核对两套 GA31-R3 模块的实际路径。主干清单继续记录两处各自的 Git commit、blob、SHA256 和加载路径。
- Gaussian 专用闭包按公布的 Git commit 核验并恢复精确缺件。没有可用的对应字节，或现存内容不匹配时，Gaussian 仍为 `UNRESOLVED_IDENTITY`，主干来源审计继续。若 S0 已核验 Gaussian，`kernel_checks.py` 要求当前闭包与登记完全匹配。若 S0 记录为 `UNRESOLVED_IDENTITY`，恢复后可使用同一队列重试，机械检查收据绑定恢复后的实际闭包哈希。

## 检查

- 项目 Python 语法检查和 `python -m experiments.history_information_20261124.kernel_checks` 通过。
- `tests/test_frozen_source_recovery.py` 共 3 项通过。临时工作区分别移除发表闭包的 `views.py` 或 `protocol.py`，禁止从 Git 对象补件，以验证主干同 blob 副本足以在冷启动时恢复。两条路径原 commit、blob、实际模块位置及恢复字节核验通过。
- 独立临时工作区测试验证了 D0 coverage 的本地 Git 对象恢复。禁用唯一可用的精确对象后，Gaussian 标记 `UNRESOLVED_IDENTITY`，主干 175 条记录和来源哈希与恢复前一致。重新开放同一对象后，在 Train 输入和批次检查替身下验证了 S0 首次未解析的 Gaussian 收据可以绑定恢复闭包。另验证不同字节的已存专用文件未被覆盖。
- 真实工作区的 175 条主干记录及 7 条 Gaussian 专用记录均通过来源审计。Gaussian 闭包 SHA256 为 `fae0dc51d0ba7114045260cec3949547ea8627c869293f8add48354bd0da7483`。所有缺件注入和字节变更操作都在临时工作区执行。

## 待独立审查

S3d 派发和 clean 分析由并行 coder 的报告覆盖。本次回归用替身检查恢复后的收据绑定，真实双数据 Gaussian 首尾批机械验收随模块触发执行。正式训练和新 Test 均未启动。
