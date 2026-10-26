# 独立审查：观察器代码身份守卫

Verdict：**PASS**。Blocking issues：无。仅修正代码身份的稳定表示和固定模块加载，不改变重建数学、训练、选点或1e-6容差。

已审读 CODER_IMPLEMENTATION_REPORT_OBSERVER_FINGERPRINT_GUARD.md、实际最终 observer/readers 代码、原源码快照及失败收据。首次正常RPC在任何替换前因旧 marshal(code) 指纹拒绝，记录必须保留；本审查不将其改成成功，不接触驻留 service。

独立离线复现：同一个原始函数 code 经单函数 marshal roundtrip 后，marshal SHA 可以变化，而 bytecode/filename/firstline/flags 相同。此事实证明该表示不适合作唯一代码语义身份；尚不据此断言真实首次拒绝的唯一原因。

CODE_SEMANTIC_V1 规范化包含实际co_code、异常表、参数计数/nlocals/stacksize/flags、name/qualname、有序names/varnames/freevars/cellvars及递归typed constants。float/complex保留精确位，tuple/frozenset区分类型且后者排序，未知类型拒绝。文件路径、行号及行表只作诊断metadata。独立检查原两个函数的compile、单函数marshal、module/pyc及FunctionType表示均得到同一语义身份；仅改变位置metadata仍一致。真正改变opcode、constant、global name、参数签名或异常表全部不一致并应拒绝。原函数体未执行，核心GPU数值fixture未重复。

旧身份仍从固定原baseline快照编译取得，新身份仍由精确审查源码SHA绑定。PID/PPID、八worker、typed正常RPC frame、op/config/FROZEN/PASS及仅两个固定目标等守卫全部保留；语义不符时在赋值前拒绝并记录actual/expected语义与marshal指纹、函数路径/行号等metadata，无权重归档。没有“任意旧函数均通过”或降低白名单。

首次失败已缓存旧observer模块。readers末端现在只刷新硬编码的这一个自身观察器模块再调用同一typed installer；不reload baselines或训练模块。独立实际测试确认缓存模块原对象刷新、安装函数更新且普通reviewer进程NO_OP；全部原reader数值定义AST仍相同。

证据：OBSERVER_SEMANTIC_FINGERPRINT_INDEPENDENT_CHECKS.json。本次刷新授权只允许下一次正常reload-readers RPC重试；若真实代码指令不符仍应拒绝，以拒绝metadata定位，不能绕过。原冻结SHA09def5e2…、136条训练完成证据及八个RAM模型必须保留。实际安装不变指纹和正常evaluate全角色概率≤1e-6仍待RUN，不预填修复完成。
