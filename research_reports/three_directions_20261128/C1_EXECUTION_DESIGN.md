# DESIGN SPECIFICATION | C1 historical ranking memory

C1 adds 20 trajectories across MovieLens-1M and KuaiRand-Pure. It links the original verified A1.B0/P0 runs as C0/C1 controls, then trains C2 and C3 for seeds 2022–2026. The approved bank cap is 2,048 unique ordered pairs per active condition and the fixed dose is 2,048 draws per pair event. Candidate capacity is 4,096 per active condition. C-only preparation can proceed during B1 under the approved design. Shared integration follows completed B1 supervision. The independent C0 scientific review gates C1 roster freeze, and independent implementation review precedes C1 RUN. All ten registered stop paths were clear at design time. Test stays sealed. [C0 analysis](C0_ANALYSIS.md) · [C0 execution](C0_RUNNER_REPORT.md) · [Capacity ruling](../../artifacts/three_directions_20261128/C0/decisions.json)

## 1. Research question

Does replaying Train pairs that a fixed early teacher ranked correctly improve selected Validation ranking relative to replaying random Train pairs from the same candidate pools and relative to continued current macro6 pair training? If both practical comparisons pass, does distilling confidence add value beyond repeatedly enforcing the same hard-labeled relations?

## 2. Hypothesis

C3 improves six-cell Macro AUC over both C2 and C1 at the fixed exposure budget. The C0 audit recorded pre-selected reversals and sparse exact overlap on both datasets. These observations motivate the experiment. Large pair universes, general replay, repeated hard supervision and preservation of outdated teacher preferences remain competing explanations. A conditional C4 compares soft and hard supervision on identical relation identities. [C0 analysis](C0_ANALYSIS.md) · [Algorithm §3](../../research_plans/26-11-28_three_directions/02_ALGORITHM_DESIGN.md#3-c历史正确排序关系记忆)

## 3. Variables changed

| Logical C arm | Epochs 1–10 | Epochs 11–50 | Execution |
| --- | --- | --- | --- |
| C0 | Clean BCE and no-gradient current pair forward | Same | Link exact A1.B0 |
| C1 | Clean BCE plus current hard macro6 pair loss | Same | Link exact A1.P0 |
| C2 | C1 objective | Clean BCE plus soft EMA-E10 target on uniform random candidate bank | 10 new jobs |
| C3 | C1 objective | Clean BCE plus soft EMA-E10 target on historical-correct bank | 10 new jobs |
| C4 | C1 objective | Clean BCE plus hard labels on exact C3 draw identities | Conditional, 5 jobs per triggered dataset |

Here C0 denotes a training control. The completed C0 offline audit supplies candidate and teacher evidence.

## 4. Variables held constant

Use the existing Train/Validation caches, Train-fitted features, natural Sample objective, M3oE fixed1, native Adam with the exact A1 settings, fixed EMA, FP32, 50 epochs, batch size 40,960 with the natural tail, and seeds 2022–2026. For each dataset and seed share initialization, BCE-row shuffle, macro6 proposal and weight, endpoint count, loss coefficient .05, temperature 1 and current-sampler fallback. Preserve A1 P0's logical forward order, one gradient-bearing backward, diagnostic VJP, optimizer/EMA step counts and base BCE scaling. The C0 control retains A1.B0's matched no-gradient pair forward. Evaluate both online and fixed EMA on full Validation every epoch and select each independently by earliest strict maximum six-cell Macro AUC. Bind LogLoss and cell metrics to that same checkpoint. [A execution design](A_EXECUTION_DESIGN.md) · [Matrix §§7–8](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md#7-c0历史关系离线审计)

## 5. Mathematical and algorithmic definition

### Candidate and teacher identity

For each active condition \(i=(d,t,a,b)\), reuse C0's ordered unique candidates from the Train-only positive/negative joint-label pools, with at most 4,096 pairs. Regenerate the exact candidate positions with `c0_audit.candidate_set` and compare the ordered pair and condition hashes with the audited NPZ. This verifies stable IDs, order and label polarity with the current immutable Train offsets. Each matching seed uses the corresponding **A1.P0 fixed EMA E10**, with E10 global step equal to ten complete natural epochs. E5 and the selected checkpoints inform the audit. Bank membership uses the raw finite FP64 E10 margins from the audited FP32 routed teacher logits. Index the E10 slice by its recorded snapshot order, candidate SHA, condition hashes and teacher model hash. Use pickle-free numeric loads. Score reuse requires exact verified identity.

C2/C3 execute A1.P0's warmup directly for every epoch through E10, with no bank exposure or extra RNG draws. After the last E10 optimizer/EMA update, require exact matches to the registered P0 E10 fixed-EMA teacher and online model hashes. Verify the E10 step and the prefix row-order and current pair receipts against the original P0 trajectory. Install the frozen bank before the E10 RAM commit. Record E10 optimizer and loader/RNG digests to compare the new arms' prefixes. Validate the bank against the same teacher on recovery. A teacher or matched-prefix identity failure is a technical pause for the affected dataset/seed. Preserve the evidence for review of the discrepancy before resuming. The archived P0 margins avoid new teacher inference and remove a scoring-backend difference between arms.

### Deterministic banks and draw law

Let \(P_i\) be the C0 candidate sequence, \(H_i=\{j\in P_i:m_{E10,j}>0\}\), and \(b_i=\min(2048,|H_i|)\). Treat exact ties as ineligible for C3. Preserve canonical candidate index order after each selection.

- **C3:** If \(|H_i|>b_i\), choose \(b_i\) candidate indices uniformly without replacement from \(H_i\). Otherwise take all of \(H_i\).
- **C2:** Choose \(b_i\) unique indices uniformly without replacement from all of \(P_i\), including any historically incorrect, tied or correct margins selected by the draw. Incidental overlap with C3 is valid.
- **C4:** Take C3's exact ordered indices and use C3's draw stream.

Use v3 `boundary_sampler.rng` with dataset, seed, epoch 10 and that teacher's global step. Freeze stream names `c1_bank_correct|{d}|{t}|{a}|{b}` and `c1_bank_random|{d}|{t}|{a}|{b}`. Selection has no margin ranking, magnitude preference or response to observed flips. Freeze both banks independently of arm execution order. Keep condition, ordered stable Train IDs, offset mapping/fingerprint, teacher epoch/step/model hash, candidate digest and aligned signed target margins. Save target values as \(\bar m=\mathrm{clip}(m_{E10},-5,5)\), after the raw sign test. Hash ordered identities separately from the complete margin-bearing bank.

At each logical training step call the stock `sample_macro6(context)` exactly once. Preserve its selected \(i\), \(w_i/q_i\), and current draw hash. From E11, draw 2,048 indices uniformly **with replacement** from the selected nonempty bank using the same v3 event context and stream `c1_memory_draw|{d}|{t}|{a}|{b}` across all arms. Equal capacities yield common random positions for C2/C3, and C3/C4 also match actual pair identities and multiplicity. Keep the stock positive endpoint block followed by the negative block, including repeated rows. For \(b_i=0\), all memory arms reuse that exact current draw and its hard loss. If the stock sampler yields no event, omit the pair loss and pair forward for all arms on that step. The present audited macro6 sampler always returns an event on supported natural batches. An empty bank therefore has the same exposure and forward work as current P0. The stock sampler and the memory streams are independent of NumPy/Python/Torch global RNG state.

For memory current margin \(m_j=z^+_j-z^-_j\), use the selected task logit. For C2/C3 compute detached targets \(u_j=\sigma(\bar m_j/T)\) with \(T=1\) and apply stable FP32 BCE-with-logits to \(m_j/T\). C4 and the current objective use \(\mathrm{softplus}(-m_j)\), preserving A1's \(\tau=1\). One event contributes \(L=L_{BCE}+0.05(w_i/q_i)\operatorname{mean}_j\ell_j\). The one endpoint forward has 4,096 clean rows and is the only gradient-bearing pair forward. Expected draws per bank member through \(N_i\) matching events are \(2048N_i/b_i\) when \(b_i>0\). [C0 candidate contract](C0_EXECUTION_DESIGN.md#candidate-set-and-inference) · [Engineering §5](../../research_plans/26-11-28_three_directions/04_CODEX_IMPLEMENTATION_SPEC.md#5-c-memory-implementation)

## 6. Affected modules and coding ownership

| Owner and timing | Files and responsibility |
| --- | --- |
| C1 coder, C-only preparation | New `experiments/three_directions_20261128/memory_bank.py` for artifact verification, construction, stateless draws and bank state. New `c_step.py` for the C adapter, loss and E10 boundary. New `c1_protocol.py` for RunSpecs, exact A1 control links and the C1 roster. New `tests/test_c1_memory.py` and local `research_reports/three_directions_20261128/C1_CODER_IMPLEMENTATION_REPORT.md`. |
| C1 coder, after B1 terminal state and unlocked supervisor | `runner.py`: C1 roster/CLI, training parts and journal binding, and C-only E10 and restore validation hooks. `a_step.py`: narrow pair-selection and reduction hooks with stock defaults, including the C no-event case. Keep the same objective transaction. Prefer keeping C stage handling in the C-only protocol module. |
| Existing shared services | Use `protocol.py` for RunSpec, source-audit and Train/Validation verification, `macro_pair_sampler.py` for macro6, `c0_audit.py` for candidate identity, and the existing `scheduler.py` and `ram_state.py` for supervision and RAM. |

While B1 is active, limit writes to the C-only files. Before changing shared bytes, preserve the A1/B1 source snapshot in a local Git commit matching both frozen rosters, including the exact old `runner.py` and `a_step.py`, the frozen base commit, the audited GA31 callback amendment and the referenced design/source manifests. Record the commit and per-file SHA-256s in the coder report. A1 and B1 keep their original rosters, job keys, source hashes, results, journals and selected RAM models. A/B rerun source validation continues to require their original runtime bytes. C1 uses a separately frozen source identity including its actual shared files and C-only dependencies. The C-only control link is a read-only historical input against the recorded A1 source snapshot and never serves as an A/B launch identity. [Frozen A1 roster](../../artifacts/three_directions_20261128/A1/ROSTER.json) · [Frozen B1 roster](../../artifacts/three_directions_20261128/B1/ROSTER.json) · [Shared review](B_RUNTIME_FINAL_REVIEW.md)

## 7. Implementation requirements and commands

1. Before creating the C1 roster, verify C0's completed 20/20 input accounting, manifest SHA, decisions/capacity, and independent scientific review. Resolve all 20 exact A1 control links across both datasets and five seeds. For every link compare the expected original RunSpec, complete original config and numerical-source hash, dataset/split/scope binding, epoch-50 completion and recorded selected metrics with the frozen A1 roster and journal. Verify selected online and fixed-EMA RAM readbacks and the 10 P0 fixed-EMA E10 readbacks. Link the original receipt hashes, original model IDs, and audit inputs in C1 metadata. Preserve the original stage/arm in the links and expose logical C names for analysis. An invalid link pauses its comparison and does not create a replacement trajectory.
2. Freeze C1 source/config/data and gate hashes, candidate/margins file SHAs, teacher bindings and all 20 C2/C3 job records. Set each job's base config to the exact `a_config("P0", dataset)` dictionary and bind the memory mode and fixed settings in its C adapter configuration. Use a stable stage-specific journal binding whose algorithm hash includes the actual C numerical code and unchanged A warmup foundation. Validate C1 inputs before creating work and before E11. Preserve normal parser/worker, journal and scheduler usage. Store the sampler/bank state inside the verified complete-epoch packet and log bank metadata by condition separately. E1–E9 restore with a pending bank. E10–E50 restore with the immutable bank and its bound SHA. Verify every restored bank's raw identity, epoch, signed target/margin order and correspondence to the frozen inputs. Resume from the latest verified complete epoch or its verified predecessor and replay the same logical stream after rollback. If E10 itself must replay, bank reconstruction is deterministic and validates the original teacher identity again.
3. Honor the scoped stop markers and preserve existing current-telemetry and post-OOM physical-fit semantics. Verify the workload-specific measurements cover the bank transition and first memory step if either is a failing allocation boundary. Reuse existing resource, supervision, attempt and RAM checks. Budget/report the storage overhead of the compact bank in the existing epoch packets. Disk artifacts contain receipts and numeric diagnostics. Selected model, optimizer, EMA and bank state use the existing verified RAM path.

Run the focused implementation checks from the study worktree.

```bash
cd /home/xzf/Code/phase0-3_three_directions_20261128
export PYTHONDONTWRITEBYTECODE=1
/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q \
  experiments/three_directions_20261128/tests/test_c1_memory.py \
  experiments/three_directions_20261128/tests/test_c0_audit.py \
  experiments/three_directions_20261128/tests/test_direction_a.py \
  experiments/three_directions_20261128/tests/test_b_runtime.py \
  experiments/three_directions_20261128/tests/test_b_ram_state.py
```

After B1 exits, exact controls are linked, the scientific C0 review clears the stage, and the C1 implementation review passes, the foreground RUN child uses the existing entry point.

```bash
cd /home/xzf/Code/phase0-3_three_directions_20261128/m3oe/examples
export PYTHONPATH=/home/xzf/Code/phase0-3_three_directions_20261128
export PYTHONDONTWRITEBYTECODE=1
study_artifacts=/home/xzf/Code/phase0-3_three_directions_20261128/artifacts/three_directions_20261128
phase_python=/home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python
"$phase_python" -m experiments.three_directions_20261128.runner --stage C1 --artifacts "$study_artifacts" --freeze-roster
"$phase_python" -m experiments.three_directions_20261128.runner --stage C1 --artifacts "$study_artifacts"
```

Run the second command after successful roster freeze and a refreshed scoped stop check. [Execution runbook](../../research_plans/26-11-28_three_directions/05_EXECUTION_RUNBOOK.md)

## 8. Required ablations

C3−C2 tests correctness filtering with matched capacity, event and draw dose, target construction, teacher and candidate pool. C3−C1 tests the total historical memory intervention against ordinary continuing pair supervision. Report C2−C1 for the effect of generic replay/soft targets, and C1−C0 for the current pairwise anchor. Apply preregistered thresholds per dataset and like-for-like readout. Trigger C4 per dataset only if C3 clears both parent comparisons on the same readout under §11. C4 then adds five matching trajectories for that dataset, preserving the complete C3 bank and draw digests. Keep C4 marked `NOT_TRIGGERED` until the written Validation decision. Use one separate conditional roster if triggered so the initial 20-job roster stays immutable. [Matrix §8](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md#8-c1历史关系记忆训练)

## 9. Expected logging

- Record linked-control receipts and source, config, data, scope, initialization, warmup, teacher, candidate, bank and draw identities. Per condition publish candidate size, raw correct/tied/incorrect counts, selected bank size, unique counts, clip saturation, margin histogram, and C2/C3 overlap. Record phase, empty bank and no-event reasons explicitly.
- Per step log selected condition and \(w/q\), BCE row order, original current and applied pair hashes, draws and unique pairs/endpoints, endpoint forward rows, loss and gradients, current margins/signs, target summaries and memory source/miss. Include previous and current attempt provenance, timings and material allocation observations in existing trace/receipt formats. Aggregate draws and unique realized exposure per condition and stage by replaying frozen indices and checking committed hashes. Report overlap with the continued current sampler and the uniform-support expectation where defined.
- Publish standard endpoints, selected, six-cell, model verification, cost, status and analysis tables under `C1/`. Track both readouts' epoch AUC/LogLoss, the E10 and E50 endpoints and selected checkpoints, paired mean/SD, 95% paired t interval, positive seeds of five, and training/storage/replay cost ratios. Report domain size and B0/P0 cell AUC as distinct size and difficulty indicators. Examine pre-selected formation and selected/post-selected timing separately. For a direct retention comparison, the orchestrator can score the same frozen C3 subset at fixed-EMA selected C2/C3 using the existing Train-only scorer, and use C0's aligned P0 selected margins for C1. Keep the E10-to-selected horizon and ineligible selected-at-or-before-E10 windows visible. Test access count stays zero. [Decision rules](../../research_plans/26-11-28_three_directions/06_METRICS_AND_DECISION_RULES.md)

## 10. Reviewer acceptance

1. Verify source snapshot coverage and exact original roster/control links, sealed cache scope, original source and journal bindings, and C1's isolated config/algorithm identity. Check data or receipt tampering, wrong candidate order, wrong teacher and wrong warmup fail before memory exposure. Confirm the 20-job roster and dormant C4 status.
2. On a small enumerated pool, independently check polarity, strict raw signs and zero ties, clipping, capacity/uniqueness, deterministic sampling without replacement and C2's eligibility for negative margins. Cover small and empty positive pools, C2/C3 equal counts, C3/C4 exact bank/ordered repeated draws, correct \(w/q\), unchanged logical RNG and identical fallback. Check stable soft/hard losses and gradients against independent references and ensure the teacher target is detached.
3. Compare stock A1 P0 and C warmup on real seed-2022 first and natural-tail cases for both datasets and a reduced resumable fixture across the E10-to-E11 boundary. Check BCE forward order, endpoint ordering, gradient/Adam/EMA state, loader/RNG, first replay draw and bank hash. Require 2,048 draws and one clean pair forward for a valid event, and the stock clean BCE update on no event. Verify A1 B0/P0 defaults retain their meaning and B runtime regressions pass.
4. Exercise existing RAM commit/rollback with pre/post-E10 adapter state, verify restoration and uninterrupted/recovered hash equality and reject corrupt banks. Confirm diagnostics mode follows the same E10 transition before measuring E11 and produces no logical commit. Verify selected tie behavior, same-checkpoint LogLoss, independent scheduler admission and stop checks through the existing shared tests. Require independently reviewed PASS or PASS WITH FIXES with blocking fixes verified before formal RUN.

## 11. Falsification and disposition

For each dataset/readout require both C3−C2 and C3−C1 selected Validation paired mean Macro AUC ≥ +.001 and every paired cell mean ≥ −.001 for each comparison. A lower effect or failed cell protection defeats the practical claim in that scope. C3 beating C2 alone supports an effect of selecting historical correctness within replay, with continued current supervision still sufficient by the preregistered comparison. C3 and C2 improving similarly over C1 supports a generic replay/soft-target explanation. Greater Train relation preservation without the AUC gate separates retention from generalization. LogLoss-only improvement is calibration evidence. Report the five seed deltas, uncertainty and cost without introducing a new statistical gate.

When C4 is triggered, compare C3−C4 with the same +.001 practical and six-cell protection rules to assess an incremental soft-margin benefit. C4 parity or improvement supports the sufficiency of repeated hard labels at this dose. Retain the existing final authorized evaluation thresholds of +.002 versus original M3oE for a standalone method and +.001 versus the direct parent for a component. Technical failures leave the affected comparisons unevaluated. The fixed capacity and dose conclude this experiment without a tuning search. [Matrix §10](../../research_plans/26-11-28_three_directions/03_EXPERIMENT_MATRIX.md#10-阶段转移表) · [Decision rules §§2–6](../../research_plans/26-11-28_three_directions/06_METRICS_AND_DECISION_RULES.md#2-auc-准入门槛)
