# D2 S0 smoke review

## 1. Verdict

**PASS.** The reviewed CUDA smoke and live RAM owner satisfy the final S0 gate for S1. Use `ram_pty_20261123T133555` with the bound implementation and evidence below. The implementation review remains **PASS WITH FIXES**, with an empty blocking list. [Implementation receipt](IMPLEMENTATION_REVIEW.json)

## 2. Requirement verification

| Requirement | Independent verification |
| --- | --- |
| Implementation and data identity | The read-only `cli.authority` check rehashed the reviewed implementation, donor sources and S0 artifacts and accepted their bindings. The GPU smoke binds the same implementation, implementation-review receipt and resource receipt. The Train/Validation data manifest carries `test_status=SEALED`. |
| CUDA source-native fidelity | Seed 2022 on `cuda:0`. MovieLens compared 40,960 first-batch and 21,927 tail rows at batch indices 1 and 20. KuaiRand compared 40,960 first-batch and 40,052 tail rows at indices 1 and 49. Ordered permutations and batch row-ID hashes match the independently reviewed CPU fixtures. In all four comparisons, BCE values matched, probability maximum absolute difference was 0, and candidate/donor hashes matched exactly for probabilities, online state θ, native Adam state and EMA state. Each update recorded one forward, one backward, one Adam step and one EMA tick. The fixture applies the first update and then the natural-tail update to both models. |
| Genuine IPC recovery | The reviewed code commits through authenticated Unix-socket IPC and issues a fresh get to hash the returned packet before writing `RAM_VERIFIED`. It issues a second get for restore and replays the next update. Both datasets report exact replay state, identical replay loss, and the same next permutation after restore. The live owner status, commit ledger and per-fixture scalar receipts agree on the two first-sequence commits, payload identities and subsequent releases. Read-only live gets confirmed the disposable states had been released. |
| Live storage identity | PID `191092` answered authenticated `status`, was non-zombie, and remained attached to the runner's `pts/2` session. Its recomputed generation was `b027c4c8cc666b9221efe49505ef24f6bb5f8bfc550f949ff9a2839597e93a84`. The checked stop controls were clear. |
| Resources | The preflight sample on physical GPU 0 was 44.87% memory and 83.52% power, with 48.24% host RAM. Both RTX 3090 cards expose 25,307,578,368 CUDA-usable bytes. In the recovery samples, NVIDIA process telemetry for smoke worker PID `191019` showed 1,388,314,624 bytes for MovieLens and 1,654,652,928 bytes for KuaiRand. The observed profiles across fidelity and recovery were 1,388,314,624 and 1,675,624,448 bytes, respectively. Both exceed the corresponding PyTorch reserved peaks. The environment receipt identifies project Python, PyTorch 2.9.0+cu126 and CUDA 12.6. |
| Logical scope | The accepted receipt records zero new scientific runs and zero Test evaluations. The live owner contained exactly the two released S0 fixture records at review. The design erratum fixes the S1 matrix at 19 named forks per anchor and 570 across 30 anchors. |

## 3. Bugs and deviations

No blocking issue in the accepted evidence. The archived first smoke had a numerical PASS and then returned `ConnectionRefusedError` on the live-owner check after the session closed. The accepted attempt binds a different owner and independently satisfies the live gate. [Archived attempt](../../artifacts/mdmt_shared_private_coordination_20261123/S0/attempts/smoke-20261123T133331/)

## 4. Corrections made

This review adds this receipt. Verification used the existing scalar records and read-only owner calls.

## 5. Evidence and remaining scope

| Artifact or identity | SHA-256 |
| --- | --- |
| [Approved design](DESIGN_SPECIFICATION.md) | `7c1df7c873468de90b17ff53e32deab0721001bdd53dab2600de3b924467a74b` |
| [Design erratum](DESIGN_ERRATUM.md) | `45b42e2d47dc10341ebe898e86d42a137861587fd4024a9a61627c1ecab792e5` |
| [Coder runtime report](CODER_RUNTIME.md) | `bd7c2a45921c1a476a9819178e62aa6ef916f29daa37227063239d65327eaa43` |
| Reviewed implementation | `f45b68ef687179f756f1397314b47d46359fbd8ecbdd9312ef6da3704d819202` |
| [Implementation receipt](IMPLEMENTATION_REVIEW.json) | `d3f39a943facbcd67ec59931641bdaac1c1eed130000c399f853e1214865569a` |
| [GPU smoke](../../artifacts/mdmt_shared_private_coordination_20261123/S0/GPU_SMOKE.json) | `d62e965b043b4f416f4d2f1f7a8369bf41c1a557f17a6b0ffa38377ece306543` |
| [Resource audit](../../artifacts/mdmt_shared_private_coordination_20261123/S0/RESOURCE_AUDIT.json) | `cce1df803e25b5a82dc869de28d36ed32353f56fa9b3d2a307c2326997721046` |
| [S0 owner commit ledger at review](../../artifacts/mdmt_shared_private_coordination_20261123/ram_pty_20261123T133555/epoch_commit_ledger.jsonl) | `6351616766f3bda0d9e86f851d8261165da3daccf7512679b729a357a985b45f` |
| [MovieLens IPC receipt](../../artifacts/mdmt_shared_private_coordination_20261123/ram_pty_20261123T133555/D2_S0_1ed905458e20421195136b94f0294d8f.json) | `eeef469295cfbda03c794a99086d2cece905af94f804c8ac8bc848e6ba63f618` |
| [KuaiRand IPC receipt](../../artifacts/mdmt_shared_private_coordination_20261123/ram_pty_20261123T133555/D2_S0_08e981f5a9ac4e64a790ada463d4bf8f.json) | `587eb61f1b12f7f37422f30edae500c425d9e6f8ddb5ac4aa46e3bcc1cca9659` |

The review validates the recorded replay and live release metadata. Released packets are no longer available for a new packet hash. The profiles are sampled smoke measurements. S1 worker admission uses fresh telemetry and, after any actual CUDA OOM, the specified physical-fit check.

## 6. S1 clearance

**S1 may proceed under the current-conversation runner with the accepted live owner and current admission checks.** Test stays sealed.
