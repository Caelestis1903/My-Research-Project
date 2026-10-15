# REVIEW REPORT：marker-continuity successor checkpoint v1

1. **Verdict：FAIL，MAJOR regression blocker。**

首次capture guard与snapshot/resume digest连续性修复核心正确，独立普通调度unit检查通过；但新增regression重新引入可达partial-publication读取竞态，并缺少失败线程清理。本checkpoint不能作为修复后的qualification precheck PASS。需要最小修正后独立复审，不要求扩建算法或运行E2E来证明本缺陷。

本report绑定test SHA **`02d02e34ec811830411278fba3ffc4b09c0d4ec84763ad23afe26f79a7dc4fcb`**，普通源码副本 `/tmp/f0-v15-marker-review-20261014-a/archive/reviewed_test_source.py`；coder report SHA `188e88510e3c8393096b715cbc6abe8430442b125c9fa8d379a3b60868fc72a4`。Working copy之后的返工属于下一checkpoint，不包含在此判定。DESIGN v2 SHA `8af224ef00f25dab65e77433843e48d8bc5f47e039f4601f607fcfd0e467f316`；attempt4审核保持原绑定c2365231…。

2. **Requirement-by-requirement verification**

| 要求 | 实际代码/独立证据 | 结果 |
|---|---|---|
| first snapshot/resume不被later capture覆盖 | embedded wrapper:2218–2239 first-capture dict；每次仍计算current state，但已recorded时直接返回；记录resume之后set True | 普通连续调用unit PASS |
| snapshot bytes与digest连续 | first snapshot raw hash；later capture changed state但原snapshot/done/resume raw字节相同；integration:2853比resume与original snapshot SHA | unit PASS；integration未运行 |
| strict order | 原rendezvous helper仍检查snapshot<peer_wait<peer_claim<resume<child_defer；新unit也检查five timestamps | 普通调度PASS |
| regression有辨别力 | 提取实际embedded函数而非另写实现，changed real_capture state后验证三份marker不变 | 正向有效，但M1/M2使失败路径不可靠 |
| model/scientific/runtime不变 | actual diff仅test harness/新增unit helper与断言；successor runtime/runner保护SHA不变 | 符合；无公式/tensor/gradient变化 |
| production保护 | 19项v7 protected SHA独立一致，两historical audit冻结值保持 | 符合；B6没有关闭 |

3. **Bugs/deviations found**

**M1 MAJOR：新增unit regression的partial JSON读取竞态。** Reviewed test:275启动writer线程；:278–282只等待`is_file`并随即`json.loads(read_text())`。真实`Path.write_text`会先open创建文件，再写内容。独立probe使用Python线程trace在真实stdlib `pathlib/_abc.py:652`的open完成/`f.write`之前暂停，target存在且size0。执行未改写的实际test函数立即得到 **JSONDecodeError: Expecting value: line1 column1(char0)**。因此这是已复现的实际调度，非只靠理论推断。

**M2 MAJOR：失败遗留非daemon producer。** 同一test:275创建non-daemon thread，没有覆盖partial-read/assert/timeout失败的finally。M1发生时尚未发布peer-ready，producer继续等待generated wrapper的600秒deadline；probe记录`thread_alive_before_cleanup=true`。Reviewer仅在外置probe finally主动release调度、发布peer marker并join，才得到`threads_alive_after_reviewer_cleanup=false`。测试自身不能保证错误后及时退出/清理，可能把诊断失败变为长时间阻塞。

最窄修复边界：等待bounded complete/parseable expected JSON或已知producer publication完成，保留线程error检查；所有异常/timeout路径都无条件释放producer wait并有限join。两秒超时可保留其初衷，但不能只靠file existence。不得修改runtime/auth validator或放宽initial snapshot/hash/order检查。

核心first-capture修复在实际supervisor同步capture路径上未发现新错误；本finding针对新regression。不是科学模型错误或实际E2E运行已逆序的证据。

4. **Corrections made / independent checks**

Reviewer未改实现、测试或protected production state。普通copy源到外置archive，AST提取并执行实际unit/helper；不import训练模块，不跑pytest/E2E、main、模型、CUDA或样本。项目Python经任务槽wrapper，峰值一个项目Python进程。

Independent raw `/tmp/f0-v15-marker-review-20261014-a/archive/probe.stdout.txt`：普通first/later-capture regression **PASS**，embedded compile/order **PASS**；受控真实Path.write_text partial schedule **JSONDecodeError**、失败后thread存活。Probe脚本 `/tmp/f0-v15-marker-review-20261014-a/probe.py`、结构化 `/tmp/f0-v15-marker-review-20261014-a/archive/probe.json`绑定source SHA；没有monkeypatch被测函数、Path.write_text或JSON parser，仅控制真实线程调度并做reviewer外置cleanup。

5. **Coder evidence / remaining risks**

Coder raw **2 passed in0.13s**与早期static-order assertion失败输出保留，manifest `5d8da28cf7a62ea0a92d9269bd5655ddf04100c974cf3a0762bed75dc0a1317d`，两项payload独立SHA匹配；final stdout `05969f68db42c0d80b5f203b4c913fe621ff8643af5b714630e061a4653df4b3`。早期失败源于`index(return state)`选择新增early-return，最终`rindex`修正合理。普通schedule通过不能排除M1实测。

attempt4旧archive中original resume已被覆盖，本修复不能回填历史字节；后续需新qualification证据闭合Q1。受控>60秒PREPARED、完整fault/terminal contract与相关regression/policy矩阵Q2–Q4仍未闭合，本review不重复测试或减少其范围。B6历史生产log/pre-edit inventory缺项继续OPEN/FAIL。

6. **Whether experiments are safe to run**

**本checkpoint FAIL，先修M1/M2并独立review，再按原边界进入非训练qualification。** 不批准本次E2E、科学/生产RUN、staged graph GO或principal release。B6 OPEN/FAIL、生产hold保持；不提出算法变更。

Evidence `experiments/reports/stage_F0_v15_marker_continuity_review_v1_evidence.json`，SHA `07079b55d1de5b2201ff8c60cb2508b987a8b964f40356be5d5d590b7dfe3de7`。
