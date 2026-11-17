REVIEW REPORT — Cycle04 attempt2 recovery02 preparation

1. Verdict: **BLOCKED**. **ready_for_runner=false**.

This is independent static/numeric preparation under the active CHANGE contract and hash=deny. It is not a recovery consumer receipt. No source/ACK/profile byte identity, derived endpoint identity, or live tensor identity is certified. No new consumer digest or native reviewer ID is invented.

2. Requirement-by-requirement verification

| Requirement | Completed finding | Limit |
|---|---|---|
| Policy and stop controls | Both workspaces ACTIVE, manual_stop_requested=false; all four stop sentinels absent. Technical PAUSE_TRAINING remains present. | No control was modified. |
| Continuation | Current W/AGENTS.md and CONTINUATION_SCOPE_20261117.json require finish Cycle04, exactly Cycle05, then stop; no Cycle06. | Earlier no-Cycle05 text is historical and superseded. |
| Scientific algorithm | Actual core.py implements ordered A=sum(w*pi²*class BCE), W=.75C+.25A, S=mC, m=1-sum(w*.25*2*pi0*pi1). No renormalization or applied ranking gradient. | Static semantic inspection; unchanged bytes/numerical proof not certified. |
| Shapes/masks/routing | [B,2] labels/probabilities, source [F,B], six domain/task cells, exact full/tails40960/21927/40052, absent-class nulls and identity delegates match design. Source gate uses emb.detach(). | No model forward or tensor inspection. |
| Gradients/Adam/EMA | Only selected W/S loss performs live backward; diagnostics use three other gradients and four sequential isolated source Adam clones. None/zero activity retained; one live Adam/EMA update; restore copies optimizer before RNG-last. | No fresh gradient, aliasing or replay execution. |
| Baselines | Source CLEAN delegate, numeric C/N references, Adam/EMA/loader/selection paths remain as specified in inspected code. | Historical proofs require identity verification before reuse. |
| Formal coverage | Exactly20 formal jobs: 2 datasets ×2 arms ×5 seeds2022–2026; four separate engineering jobs. | Job key digest recipe was read, not executed. |
| ACK schema and consistency | All889 formal ACK/epoch/transaction records agree by direct field equality, with contiguous prefixes, exact two-key counts, operation/branch clocks, recorded timestamp order and exact own journal rows. One ACK schema. Journal897=889formal+8engineering. Fifteen completed numeric results agree with final selected fields and counts. | Digest strings are compared as recorded labels only; referenced bytes/live states are unverified. |
| Quiescence | Procfs at 2026-11-17T10:00:52.345999+00:00: owner2476891 alive in S, command serves attempt2; controller2611647 and workers2611773/2611774/2612072/2612190/2612426/2619341/2620208 absent. No matching live formal Python runner command. | Read-only observation; owner untouched, no IPC. |
| Owner/report fields | Source/design labels agree across manifest/source receipt/reviews/signoff/owner/pause/runner report; owner PID and recorded endpoint fields agree, private files0600. Four coder IDs remain distinct from code/resource reviewers. | Endpoint derivation, server/source bytes and actual live storage status not verified. |
| Resource/pending predicates | Actual admission and run code preserve independent-card current memory<85%, power<85%, host<90%, no pending, and current used+remaining measured residency<=CUDA-usable100%. Pending release requires live matching child, job/device/PID, expected registration attempt, RUNNING and allocated=true, followed by resampling. No cap, projected85%, predicted power or both-card gate. | No current GPU measurement or admission execution. |
| Recorded resource evidence | All29 eligible numeric decisions satisfy their recorded predicate. Seven recovery generations each have matching STARTED/expected attempt/allocation/release fields. Engineering totals32/8/24/32 with zero scientific epochs. | Historical field evidence only; current resources must be sampled afresh. |
| Repeated NVML pause | Current failed key2df25390 again reports NVML code=3; four later NVML_RECHECK_02 samples report all return codes0. | Failing API remains unidentified; rechecks do not prove a permanent fix. |
| Mandatory recovery02 gate | Source/ACK/tensor checks and a new pause/parent/review binding remain outstanding. | BLOCKED under current constraints. |

The useful numeric frontier is **889/1000 epochs and32251/34500 updates**. Remaining **111 epochs /2249 updates**, with **15 completed and5 unfinished**:

| Dataset | Arm | Seed | Last ACK | Remaining epochs | Remaining updates | Exact key |
|---|---|---:|---:|---:|---:|---|
| movielens | COMPLETION-BCE-025 | 2026 | 20 | 30 | 600 | `2df253901fbd915c33bb3f5a2c6152666931bbd1bc093b16f9aee1828dd1932c` |
| movielens | COMPLETION-BCE-025 | 2025 | 26 | 24 | 480 | `67510a0c232f831ab36e16984ee074646cb59e69be1b16d94cba3a4fbac9a011` |
| movielens | MASS-SCALE-025 | 2025 | 26 | 24 | 480 | `c6481fe6067fe4a819b8e1249a1ccf9ef3c6491b61e70042af7d1d85c76636a0` |
| movielens | MASS-SCALE-025 | 2026 | 18 | 32 | 640 | `ea3f3584e6de6d4f328882f42055264e6306665c82c874d0245577eb1d29148d` |
| kuairand | MASS-SCALE-025 | 2025 | 49 | 1 | 49 | `f8df5a89010729c008e138d2f74668f9698d391c6fe32f93041a3537493c6649` |

All20 exact last-ACK values and paths are in the companion JSON. Every unfinished key has a nonzero ACK and requires its retained continuation; none is an epoch0 fresh-start case.

The recorded envelope must include live observation maxima. Reading all990 existing numeric resource observations with the four engineering profiles gives **MovieLens1709178880B** and **KuaiRand2027945984B**, supported by both arms and target cards on compatible recorded hardware. MovieLens exceeds its older engineering-only1675624448B value; recovery must not shrink back to that older value. The recorded usable capacity25307578368B is historical; production must remeasure current capacity.

3. Bugs/deviations found

- **MAJOR, blocking B1:** Required source/design/Adam/control/receipt byte identity, ACK/batch/transaction byte identity, derived owner binding and live continuation/selected tensor checks are unperformed. Direct string equality cannot replace them.
- **MAJOR, blocking B2:** RECOVERY_REVIEW_01 covers18 keys/771epochs and the previous pause. Two formal keys have since started and five old frontiers changed. The current failed key2df25390 and frontier require fresh binding. PARENT_RECOVERY_READY_01 cannot release this pause.
- **MAJOR operational risk R1:** NVML code=3 recurred. The worker traceback at runner.py:692 enters Profile.mark:116, which reports a stored sampler error; NVML._ok:52 omits the API name. No evidence identifies a particular NVML function or driver cause. No speculative repair is proposed.
- **MINOR R2:** Profile.close:123 raises again before worker terminal accounting/resource_profile writing at runner.py:737–743. Failed PID2619341 has supervisor exit1 and689.6625649929047 process-wall seconds, but no worker terminal record/resource_profile. Keep unknown terminal counts unknown.
- **MINOR R3:** Historical no-Cycle05 prose is superseded by the current explicit continuation. Preserve old receipts and use the current scope in new handoffs.

Parent-supplied, non-overlapping accounting evidence received during this review: E1..20 for failed key2df25390 contain400 returned batch records; E21 is empty and readable. Added to six other workers'2876 known returned calls, this establishes **at least3276 returned calls**, compared with **32251−29021=3230 new useful updates**. The **46** excess returned calls may only be classified as replay or returned work outside the new commit frontier. This reviewer did not recount the400 or independently reread the parent's updated R/EXECUTION_HISTORY.md. Raw failed-terminal null/unknown remains unchanged. Overlapping process-wall times are not elapsed controller time.

No additional scientific implementation deviation was found in the inspected code. No current byte-identity claim follows from that finding.

4. Corrections made

None. Only the two requested preparation reports were written. One read-only numeric reader initially assumed engineering admission rows carried formal key fields; the reader was corrected to handle that existing schema. No repository defect, production edit, broad test or generated fixture file resulted.

5. Remaining mandatory checks and exact consumer requirements

- **C1 — Frozen identity gate; protocol.source_manifest/design_bundle/validate_attempt:** Verify actual48 source paths and installed Adam against frozen receipt, seven-file design/research/control identities and required environment/code-review/engineering receipt bindings. Reuse prior numerical proofs only after their applicable identity is established. No broad retest or new training is requested. **Status:** UNPERFORMED; cryptographic verification forbidden now.
- **C2 — Owner binding and live storage gate; protocol.owner_binding/request:** Verify actual owner2476891/session67406 with private0600 bindings, derived endpoint and server/source/design identity; recheck procfs quiescence. Establish actual status of20 formal keys and released four engineering keys. Preserve owner and its states; no replacement, release of formal keys or old-owner IPC. **Status:** Recorded fields/procfs checked; endpoint/server identity and live status not verified.
- **C3 — Epoch identity and live continuation; protocol.epoch_files/validate_epoch_state; runner._verify_saved_ack:** Verify ACK-referenced epoch/transaction/batch bytes and current last ACK bindings; check actual journal/status/get. All five unfinished keys now have nonzero ACKs49/26/26/20/18 and require existing exact continuation: model,EMA,Adam,RNG,loader,source modes,selection,counts,operations,private steps/activity and scientific identity. Restore whole epoch/RNG last and replay uncommitted work; none qualifies for fresh epoch0 initialization. Numeric equality here is insufficient. **Status:** All889 numeric field checks completed; byte/live-tensor verification unperformed.
- **C4 — Completed state preservation and engineering/profile gate:** For15 completed formal keys verify actual compacted selected_models/selected_epochs/selected_validation and result binding, preserve them and skip re-registration/training. Verify four engineering case/profile receipts, parent release acceptance and actual state_released flags. Use compatible dataset/source/design/hardware envelopes grown by recorded maxima: ML1709178880B, KR2027945984B as currently recorded; never shrink back to older engineering-only ML peak. **Status:** Numeric results/cases/profiles inspected; live selected tensors/release flags and receipt bytes unverified.
- **C5 — Fresh pause02 parent acceptance and independent recovery review; runner.recover:765-792:** Create actual new parent pause/owner/ACK acceptance and fresh independent review tied to current TECHNICAL_PAUSE.json and NVML_RECHECK_02 evidence. Consumer requires verdict PASS or PASS WITH FIXES with no blockers, genuine fresh reviewer_id distinct from all coders and code reviewer, matching source_hash/design_bundle_sha256, test_accessed=false, actual current pause_sha256, actual owner_endpoint_hash, nonempty resource_profiles path/sha256 receipts verified by receipt_json, and last_acks matching every unfinished registered formal key (prefer full20-key map). Actual parent-policy pause/acceptance/NVML bindings must accompany it. No consumer digests are supplied by this preparation. **Status:** BLOCKED; new pause/review/parent binding cannot be certified under hash=deny.
- **C6 — Recovery mutation boundary and current-conversation runner dispatch:** Production recover first validates engineering/frozen identity and no live registered worker, verifies saved ACK states, recovers only FAILED/deadRUNNING keys, records an immutable review-digest receipt and only then clears PAUSE_TRAINING. PAUSED keys re-register; COMPLETE keys skip. These helpers hash and/or perform state IPC/mutations and must not run under current constraint. After all gates genuinely pass, parent alone dispatches fresh DEFAULT Luna/high runner with actual native model/reasoning receipt, no reuse of Fermat/Hypatia or coders. Fresh current per-card memory/power/host/physical fit and pending-generation checks remain mandatory at every admission. **Status:** Not authorized/released; ready_for_runner=false.
- **C7 — Scope, physical accounting and Test seal:** Finish remaining111 formal epochs/2249 useful updates without reducing budget; preserve replay/uncommitted/unknown costs and parent supplied3276-returned lower bound. FinishCycle04 reporting/review/delivery then exactlyCycle05 and stop; do not startCycle06. Test stays sealed; current technical pause and this preparation provide no Test eligibility. **Status:** Future parent work only; no RUN performed.

The production consumer is in protocol.py:337–373,500–578 and runner.py:744–792. In particular, recover hashes the review/pause, verifies receipt bytes, gets and verifies saved state, mutates recovery status and removes the pause latch. Calling it under the present hash=deny/no-state-IPC boundary is prohibited. This preparation deliberately supplies no substitute PASS, pause digest, review digest or fake live verification.

6. Whether experiments are safe to run

**No RUN release.** Keep ready_for_runner=false and the existing technical pause intact. Parent must integrate the completed static/numeric findings, resolve the mandatory verification boundary and obtain actual fresh acceptance before any recovery runner starts. No permission wait or further broad check is needed from this reviewer.

Coder identities preserved: Volta 01a1e54b-0376-79c0-b1a7-20a6893345c7; Chandrasekhar 01a1e54b-03a0-7613-a537-dba4cc0f0e80; Leibniz 01a1e6cd-54c0-7a11-80ca-fb7194be1f15; Kant 01a1e747-c8be-7a73-af93-5a9f2966d3b1. Closed Hypatia01a1e820-76e4-7851-85fe-f3e22a2045d6 and Fermat01a1e836-9e09-76c1-99d1-7d3551eae227 were not reused. No descendants or context inheritance were invoked.

Skipped cryptographic hashing, hash-verification helpers/IPC, live model-state IPC, training, Test, GPU queries and broad retests as required. Residual risk is unverified identity/live-state availability plus recurring telemetry failure. Context returned permanently; do not resume or reuse.
