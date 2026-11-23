# B_CORE implementation handoff

The approved design is research_reports/conditional_history_shared_update_20261122/DESIGN_SPECIFICATION.md. The scientific source is the original 2026-11-22 plan at 4a3e4f1b21f1aebb6daa2df6635c3d7ca547a983, especially 04, 06, 07 and 09. This delivery covers the clean observer, conditional one-pulse execution and the dataset-level evidence gates. Scientific RUN and interpretation belong to later stages.

## Implementation

| File under experiments/conditional_history_shared_update_20261122 | Scope |
|---|---|
| b_build_panels.py, b_panel_identity.py, b_panels.py | Reconstruct source P_A/P_B/disjoint V folds and verify the registered row/label/pair hashes before use. Capture logits by source tower hooks. Report fixed-pair and exact panel-population rank transitions separately, plus natural-row BCE. |
| b_pairs.py, b_pairs_kernel.cpp | Exact population negative/tie/positive to negative/tie/positive transitions. Only compiled code goes to the temporary directory. |
| b_virtual.py | Resolve names and domain embedding slices, require the published counts, split 2B loss entries by true source and two keyed class-capacity-matched pseudo assignments, calculate 18 retained-graph autograds and native Adam clones with private state/None semantics. |
| b_observer.py | At E5/E20/E35/E50 real tails, use the unchanged natural source loss for live backward. Perform one JOINT, three ZERO, 18 ONLY, 18 real LOO and 12 pseudo LOO. Observe raw, common applied radius, post-F, small online, identical small F displacement, full displacement on F, and true/pseudo common radius. Assert exact live and virtual JOINT post identities and neutral observer RNG/loader. Stream complete anchor scalars to disk. |
| b_observer.py, b_pulse.py | E20/35 maintain K parent score-bank steps from F_pre and save at most two selected conditional first posts per anchor. Save terminal joint parameters and the parent score banks for later readouts in guardian RAM. One admitted true or pseudo0 slot starts from its archived model/Adam/F and RNG/loader, uses the following epoch's same row hashes for K−1 ordinary live steps and checks the stochastic stream. Terminal matched radius compares full-model online and F displacement directions on JOINT online. NO_ACTION explicitly reuses parent with zero cost/differences. |
| b_tables.py, b_gate.py | Assemble anchor, raw/normalized effects, true/pseudo, and history interaction tables. Require 20 distinct complete anchors per dataset. Freeze leave-one-seed-out static sources from the eight early P_A profiles on other seeds. Export all choices inclusive of NO_ACTION. Search evaluation requires all 20 admitted pulse slots, averages epochs before seed statistics, and evaluates the five paired seed differences, t4 interval, six-cell protection, direct JOINT and matched pseudo comparisons. |
| b_checks.py | Tiny graph and real original-first/tail engineering checks. The real fixture uses the source input population, two actual optimizer steps with synthetic stage labels to exercise the production observer, independent no-observer hashes, and measured full-process residency. No outputs of this fixture are research endpoints. |

The narrower held-out V_B full-population AUC is the Validation diagnostic used for the search gate. P_A choices use the mean of the six full-population probability AUC gains on its fixed ranking rows. Fixed-pair gains are reported separately. A regression fixture exercises a positive population gain and a negative sampled-pair change on the same predictions. A selected exhibited self benefit and cross-cell damage on P_B or V_B establishes a local pulse trigger. The complete enumerated witness list remains in the output. The report can distinguish isolated observations from repeated evidence. The action-value threshold is applied to the five-seed mean terminal fixed-EMA validation gain against JOINT. The full posterior interpretation remains the orchestrator's responsibility.

Each branch and parent independently measures the per-step FP64 residual of its actual FP32 EMA recurrence. Their sum bounds the residual of the difference recurrence at each pulse step without storing all parent parameters across the window. Per-anchor direct one-step F residuals and terminal matched directions are also recorded. Hashes and output provenance are tied to the worker's committed attempt via b_stage.py and runner.py, owned by the separate runner implementer.

## Invocation and state

Create BObserver(bundle,dataset,seed,output_dir) once. Its step(model,optimizer,ema,x,y,row_ids,config,context,loader) replaces the clean v3 step and returns the same loss, predictions, diagnostics and counts. Context includes one-indexed epoch/global_step, dataset and seed. It uses a worker-owned A0 native Adam or clean v3 PairOptimizer and the worker-owned independent loader generator. The worker persists observer.state_dict() in the RAM packet, restores it with load_state_dict, and preserves deferred pulse states at the end of the replay.

b_gate.core_gate(dataset,anchor_paths,output_dir) accepts exactly twenty committed, hash-verified raw anchor files. b_pulse.run_pulse(observer,model_template,config,epoch,family,gate=gate) requires an affirmative dataset B_PULSE gate. The returned raw report can be supplied to b_gate.search_gate(gate,pulse_paths,output_dir). Formal pulse dispatch remains conditional on the actual gate and normal worker resource/review admission. ADAPT, STATIC, UNIFORM, AB and Gaussian full training have no current scientific authorization from a positive B gate.

## Reuse provenance

Frozen v3 numerical source is the local manifest verified by the common S0 audit and historical commit 875215c1e509ab55c1b6632d4f5d13e8b94782da. The observer imports the unmodified natural source losses, train_step, tower readout, shared_name, RNG and EMA helpers. Deep-copy/load-native-Adam follows the CWA virtual-step pattern in experiments/completion_weight_attribution_20261117/core.py. No retired training driver is imported.

The historical branch commit fb120946c50519ea3993bbd6a2532a2c457731fc supplied the following independently pinned inputs.

| Original file | Git blob | Local use |
|---|---|---|
| experiments/ema_mechanism_20261106/panels.py | 31f053df5e1070c401894bd6ddaf02da51fce002 | Panel construction prefix with adjusted import closure |
| artifacts/ema_causal_20261110/attempt1/PANEL_MANIFEST.json | 68b19a59358afa24deef45fbb078aa21f8db3ba5 | Fixed source panel specification and all 30 seed/panel identities |
| experiments/aug_history_20261105/group_registry.py | 856b7ef20631481ab25a615f93fdc62e4fda0a15 | Actual-name mapping and coordinate counts adapted to three scopes |
| experiments/ema_causal_20261110/pairs.py | 6b1780366bd1b5513679b169917d156c43661dff | Exact transition wrapper and metrics prefix |
| experiments/ema_causal_20261110/pairs_kernel.cpp | 7032fcc32d36a5696e9458b557c93bd5e0d787d7 | Unmodified kernel under local prefixed filename |

## Verification and pending evidence

The project Python compiled the package and ran b_checks.basic() successfully. The CPU numeric fixture covered exact virtual/actual JOINT and independent source post equivalence, native zero versus None and different private clocks, empty domains, rank transitions, both original fixed model coordinate registries, RNG isolation, a complete synthetic 52-intervention menu, K=2 parent/JOINT pulse equivalence, zero-action suffix reuse, and the two distinct EMA recurrences. PyTorch emitted its source TF32-settings deprecation warning.

The registered GPU0 original-first/tail check completed with PASS on both datasets for source 8d13cc6cf2fd730563720087ce8beef51d2055e2e6654b2db777ec8b3d800cad and guardian generation f2a6427b61b812f225f910e6991ede53507fd72383c1a03eaac50db9ea01d353. The receipt is artifacts/conditional_history_shared_update_20261122/B_CORE_CHECK.json, SHA256 7350e22b0194c461f5e8e1cab811e5b218e2a855b58548b027356e36e669a963. For MovieLens the first/tail batches were 40960/21927 rows and measured full process residency was 1,472,200,704 bytes. KuaiRand used 40960/40052 rows and 2,044,723,200 bytes. Each full tail fixture exercised 52 native Adam clones, 18 additional source gradients and 1114 panel forwards. Both independent source-vs-observer trajectories matched model/Adam/F, RNG/loader and loss exactly at both steps. Both real dataset panels matched their original seed-2022 hashes. GPU0 was released following successful completion.

The engineering work superseded two short, interrupted fixture invocations while fixing the JSON/RAM identity field and the definition of the P_A source score. Neither generated a completion receipt or consumed a scientific run slot. Engineering observations are distinct from the four-stage scientific anchors and ten clean trajectories. An independent review and matching receipt are required before scientific B replay.
