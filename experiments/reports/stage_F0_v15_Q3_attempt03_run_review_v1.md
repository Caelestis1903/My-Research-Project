# REVIEW REPORT — Q3 attempt03 bounded raw RUN

## 1. Verdict

**FAIL for Q3 execution acceptance.** The archived record is coherent evidence of an unsuccessful/incomplete bounded attempt, not evidence that the fault/recovery qualification passed. The runner accurately discloses the observed first-node failure, interrupted siblings and invalid controller wait statuses. Q3 remains NOT CLOSED.

Two MAJOR orchestration defects explain why this attempt cannot supply the required acceptance record: overlapping wrappers share a scheduler whose release admission rejects any competing task, and command-substitution launch prevents the controller from collecting child exit codes. This review does not change the v6 implementation-readiness verdict or authorize any RUN.

## 2. Requirement-by-requirement verification

| Requirement | Independent verification |
|---|---|
| Bound input record | Runner report SHA is `e6fbfab50bc14bcfa7f41db1da3248145310593960f52520a4441a3dbb3ea61c`; final RUN artifact manifest SHA is `2a7b1be286a5c49fe5bd9af785d3abd6b956942a71cf406cdd356433c99eb8d2`; clone input manifest SHA is `913f6766127a7439e8d6ad73812402c4b6e0ae24e2ecf48030fd7647ca9e4ecc`. All match the supplied bindings. |
| Archive integrity/provenance chain | Independently ran SHA checks on all 52 RUN-manifest references and all 20 setup/preflight-manifest references: both exit 0, stderr empty. Nested setup manifest binds the B6 attempt03 manifest, attempt01 input manifest, builder/preflight scripts, raw command/output/status records and clone input manifest. This review verifies those archived references, not a new full rehash of every 1,493 clone payload or acceptance of B6. |
| Exact reviewed source graph | Current clone test SHA is `80a6362977f80735bc569793ec1324004f3596cf6bcb43216b33f16c691642f9`; runtime SHA is `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`. v6 review SHA is `cdd613c3124c151965e1d08d3a9f8986ab03b03cb64ec9d1a08db1386333bdd7`; its evidence SHA is `bc863621a092418ec0884deff52059b68cfede3212a1eccaf6083f0dce5b1322`. Run-start and before manifest-SHA records are identical. |
| Only three nodes | Read actual `run_nodes.sh` and each archived command. They select exactly the three v6-reviewed nodes. Each has a unique basetemp, required project Python/wrapper, empty CUDA visibility and an external scheduler. No additional pytest command or retry is present in this controller. This is a bounded archive observation, not universal process-history proof. |
| First-node raw failure | `fault_boundaries/stdout.raw` contains `1 failed in 165.80s` and the complete `PermissionError: another F0 task process/worker occupies scheduler slots` traceback. Its SHA is `7b565dca23fcb677dce8ba255e82ee98b71cde5a338c333f7c5983072a80f770`; stderr is empty. |
| First node is pre-matrix | Test line 1773 is the initial base-graph PREPARED seed call. The isolated test scheduler assignment and `boundaries` list/loop appear only after that call, beginning at lines 1780 onward. Runtime line 2251 calls resource admission, which raises at line 1163 before the requested PREPARED fault injection. Thus **first-node completed matrix phases = 0; selected fault/match/recovery/terminal cases = 0**. This does not establish zero progress for the two other nodes. |
| Interrupted siblings | Preimage-fsync and syscall-restart stdout and stderr are all empty (SHA `e3b0c442...b855`). Archived signal record names timeout wrapper PIDs `1628504` and `1628522`, SIGTERM timestamp `2026-10-13T23:12:32.573281582Z`. No pytest summary/real child return code/fault count is captured for either. Their results are **INTERRUPTED/UNKNOWN**, not PASS or an independently observed pytest FAIL. |
| Exit semantics | `controller.stderr.raw` reports that all three wait PIDs are not children of that shell. Controller output/status files record 127 for each and controller exit 0. These are shell `wait` statuses; they cannot be treated as pytest, wrapper or timeout return codes. Node `end_utc` values timestamp these failed waits, not independently observed child completion. |
| Timeout facts | Each command specifies `timeout --signal=TERM --kill-after=30s 3600s`. Group interval is `2026-10-13T23:08:46.720131490Z` through `2026-10-13T23:12:32.799038435Z` (about 226.079 seconds). Stop occurs far before the 3,600-second deadline, with an explicit SIGTERM record. No 124 or valid timeout wait result was captured. Evidence supports no observed deadline expiry; exact signal/timeout child RCs were lost. |
| Process cap | Independently parsed all 865 process samples: 602 samples at 3 project-Python processes, 262 at 2, 1 at 0; peak 3, no sample exceeds 5. Slot files name Python PIDs `1628499`, `1628511`, `1628529`, matching sampled PIDs. Baseline/after records are 0 and cap-violation marker is 0. Sampling cannot exclude arbitrarily short unsampled transients or prove no other executable was used. |
| Protected paths | Independently compared production before/after and clone before/after lists: both `cmp` exit 0. Independently checked current bytes against each nine-path after list: all 18 entries match. Hold, gates, pointers, state-lock bytes, queue and worker logs match the archived protected values. Hash equality is bounded byte evidence; it does not prove lock nonuse or absence of transient changes. |
| Scientific behavior | This review executes no Python, pytest, runtime imports, model/dataset action, algorithm, CUDA action, source copy or production mutation. No mathematics, tensors, masks, routing, weighting or gradients changed in this review. No research conclusion is inferred. |

## 3. Bugs/deviations found

**MAJOR M1 — parallel schedule conflicts with actual release admission.** The controller launches all three wrappers against the same external scheduler. `_assert_no_competing_task_slots()` rejects every active slot other than the caller's own. The first test builds its initial PREPARED seed before installing its private test scheduler, so the sibling wrappers trigger the guard. Staying below the five-process cap does not satisfy this stronger release precondition. This is an execution-scheduling failure; no evidence here requires changing the release guard or scientific hypothesis.

**MAJOR M2 — launcher loses child ownership and exit evidence.** Lines assigning `pid_a=$(start_node ...)` and siblings run the function in command-substitution subshells; background timeout processes are not children of the eventual controller shell. Actual stderr confirms the resulting `wait` errors. The controller ends with a successful printf and exits 0 despite the unsuccessful group. Child pytest/wrapper/timeout return codes and genuine child-end timestamps are unavailable.

No CRITICAL issue was identified within this bounded raw-record review. The interrupted siblings do not provide matrix acceptance evidence. Empty output alone does not locate their internal phase or prove they never entered cases.

## 4. Corrections made

None to the controller, test/runtime sources, clone, prior archive or protected paths. Created only independent review report/evidence and external SHA-check output files. Preserve attempt03 as-is, including its invalid waits and interrupted state.

## 5. Remaining risks

The first node exercised zero matrix cases; the two interrupted siblings have unknown case progress. There is no accepted fault-count, recovery, append-once or terminal-state result from this attempt. Static v6 review and clone preflight do not supply those missing execution facts. Sampling and protected SHA equality have the limitations stated above. B6 historical OPEN/FAIL, forward NOT_GRANTED and production execution NO_GO are not changed by this review.

## 6. Whether experiments are safe to run

**Recommendation only: a new bounded attempt of the same three v6 nodes, serially with one active wrapper at a time, is a reasonable next qualification step. This review explicitly does not authorize or start RUN.** The orchestrator must issue the separate bounded task after confirming its prechecks.

Use a new attempt ID, new external evidence/scheduler directories and unique basetemps. Bind the same reviewed source hashes and independently verify a fresh ordinary-copy clone of the approved input graph; preserve attempt03. Use the required project environment/wrapper, synthetic telemetry and administrative clone graphs only. Avoid overlapping these release tests with any competing wrapper in their admission scheduler; retain the process ceiling of five. The user's concurrency preference applies when tasks can safely run concurrently, and these recorded release-seeding calls are not compatible with shared-slot overlap.

Fix only controller child/status handling before retry: launch in the controller's own shell (or run each command synchronously), retain `$!` directly when needed, capture the actual direct-child `wait`/timeout status, and distinguish that status from inferred pytest results. Record true per-command start/end, stdout/stderr, timeout/signal events, process/slot samples and before/after protected hashes. Do not replace observed exit codes with controller 0, summaries or guessed pytest codes. Stop and preserve raw failure/timeout/interruption rather than silently retrying a node. No scheduler-guard bypass, harness/source edit, production claim/release, training or broader test scope is recommended.

Independent execution review must assess the new raw record before any Q3 closure. This recommendation grants no staged graph, scientific, training or production authority.

### Independent evidence

`experiments/reports/stage_F0_v15_Q3_attempt03_run_review_v1_evidence.json` records hashes, observed counts and exact limits. External verification outputs are under `/tmp/f0-v15-Q3-attempt03-run-review-20261014-a/`.
