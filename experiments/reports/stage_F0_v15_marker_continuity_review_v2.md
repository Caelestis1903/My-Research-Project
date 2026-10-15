# REVIEW REPORT：marker-continuity implementation v2

1. **Verdict：PASS，仅限 fixture修复与qualification precheck。**

M1 partial marker parse与M2 failure thread leak已独立实测关闭，无剩余blocking finding。当前test SHA `49af0922b90493264a899feeb2a6256ba213b2a1e8a69e0c6ebb6e46ce5275c5`；coder report SHA `9b00f682bcac379b93ff8330bdfe6d74f9d41d86bac0e655c646c7aa2f72d022`；实际diff限新增regression的reader/cleanup，首次marker guard和digest/order检查保持。此PASS允许另行bounded isolated non-training qualification RUN，不意味着attempt4完整资格已通过。

2. **Requirement-by-requirement verification**

| Contract | actual code/independent evidence | Result |
|---|---|---|
| real open-before-write调度 | test:273–282调用原Path.open，opened event后暂停，主线程:292–294确认size0；独立trace再次暂停真实stdlib f.write前 | PASS |
| bounded完整marker等待 | :299–320两秒poll，JSON object/phase/hash/finite timestamp通过才接受，read/parse异常retry | PASS；独立17次retry后成功 |
| assertion路径清理 | :345–352 finally释放writer、必要时peer_ready、join/assert，无需等待600秒 | 独立故障probe PASS |
| timeout路径清理 | 同一finally覆盖完整JSON超时assert | 独立持续malformed-read超时probe PASS |
| initial marker immutable | 后续capture current state变化，原snapshot/done/resume byte一致，SHA连续与五阶段order仍强校验 | 实际提取函数unit PASS |
| runtime/scientific保持 | 比较归档226项Python源，仅test drift；runtime/runner不变 | PASS；无数学/tensor/gradient变化 |
| production保护 | 19项v7 protected SHA一致，两原始audit未写 | PASS；B6 OPEN/FAIL保留 |

3. **Bugs/deviations found**

未发现本checkpoint未关闭的blocking implementation问题。v1只看is_file并直接parse的实测竞态已由bounded complete-row retry纠正；v1异常后等待peer的non-daemon thread已由finally release/join纠正。Fixed test保留严格marker identity与order，而非放松断言。

4. **Corrections made / independent raw**

Reviewer未修改代码、测试或protected production state。普通copy source到 `/tmp/f0-v15-marker-v2-review-20261014-a/archive/reviewed_test_source.py`，从实际test AST提取helper/function进行CPU unit检查；没有pytest、E2E、训练、CUDA执行、样本读取或模型导入。项目Python任务槽wrapper，峰值一个Python进程。

独立probe `/tmp/f0-v15-marker-v2-review-20261014-a/probe.py`，raw `/tmp/f0-v15-marker-v2-review-20261014-a/archive/probe.stdout.txt`，JSON `/tmp/f0-v15-marker-v2-review-20261014-a/archive/probe.json`（源码SHA绑定49af…75c5）：

- **真实Path.write_text open/f.write调度**：size0后再暂停0.08秒，reader实际retry17次，最终约0.117秒PASS；没有改写被测函数或Path.write_text，coder test自身的Path.open pause保持。Original bytes/hash/order和later capture稳定检查实际执行。
- **早期assertion故障注入**：在writer仍暂停时触发AssertionError；实际test finally发布peer-release、完成resume并join，约0.046秒，无存活helper thread。
- **complete JSON timeout故障注入**：reviewer让snapshot read持续返回malformed JSON，约2.038秒产生预期bounded timeout AssertionError；实际test finally仍release/join、无存活helper thread。此项是显式read故障注入，非正常producer输出。
- Embedded wrapper compile/order检查PASS。

5. **Coder evidence / remaining risks**

Coder focused raw **2 passed in0.13s**、exit0；manifest `abcc43bc1f9992833111146f58233011211cb74eca42db6438d7b634e4aa1665` 的4项payload（source、runtime、report、stdout）独立hash吻合。Final stdout SHA `05969f68db42c0d80b5f203b4c913fe621ff8643af5b714630e061a4653df4b3`。原先report v1的失败与attempt4 raw archive保持不变，不能用v2修复重建其已覆盖marker。

静态同步supervisor路径的first-only guard不改变真实auth/deferral/claim行为；本unit不能替代新原始marker E2E证据。attempt4 Q1仍需新的immutable original-rendezvous archive；Q2受控>60秒、Q3完整syscall/terminal、Q4相关suite和策略矩阵继续待验。B6历史pre-edit inventory/source/生产日志incident不由此PASS解除。Cleanup对本次受控正常local filesystem producer与assert/parse timeout成立，不声称保证任意不可中断OS I/O的线程终止。

6. **Whether experiments are safe to run**

**Marker v2 code precheck PASS，可以按既定clone-only/fake-sentinel/非训练/独立basetemp和archive/最多5项目Python进程进入下一qualification。完整qualification仍未闭合；B6 OPEN/FAIL，staged GO、principal release、科学/生产RUN均NO-GO。**

Evidence `experiments/reports/stage_F0_v15_marker_continuity_review_v2_evidence.json`，SHA `82c7f0780c363083a32a0ae314ea6335917b994ab6fb6e48d9882f5510a06155`。
