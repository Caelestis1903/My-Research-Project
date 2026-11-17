# Cycle04 attempt2 restart coder implementation report

Status: RESTART_REPAIR_COMPLETE_CPU_PASS_PENDING_DISTINCT_REVIEW. No known in-scope implementation/CPU blockers remain; this is not review or resource signoff.

Fresh native coder: 01a1e747-c8be-7a73-af93-5a9f2966d3b1. Original parent: 01a1dfb8-605b-7430-8462-5fd3da0b048f. fork_context=false; no children. Ponytail FULL and Stop That Shit CHANGE applied.
Read exact approved RESTART_DESIGN_SPECIFICATION.md/json and RESTART_DESIGN_ACCEPTANCE.json before edits. Verified all48 live files and48 preserved snapshot files against old receipt/RAM_STATE_LOSS before implementation; preserved snapshot verified again after repair. Both workspace stop controls remain clear. Stop after full current-cycle report/delivery persists; no C05/Astra.

Actual new48 source digest: d1930f26e2b73d04a2ed504112ca0161948468d5ce00b99b074d5214efdf3b3f
Old attempt1 source digest: 00ce7925c3589deb239f57e8374673e5afe8e3fe155e159dd34c140ae08cd553
Unchanged seven-file scientific design: 2acde7e9e6764f6243920cba3f538e2c4cd49b36b6ddfa2351addba3b9fc62b7
Unchanged six-file research bundle: a3648198e25f6929154b628880715c7b4eb7d9749fc8b178a1631722debba01a
Full exact diffs, all48 source hashes, all20 full new job identities and complete check outputs: RESTART_CPU_CHECKS.json, SHA256 c8d1d27ed846717cf4b9e3451e7fab4709870a1ff1744298b1d60aa1664bd3f6.
Parent owns subsequent actual source receipt freeze and reviews.

## Changes

- protocol.py changes only ATTEMPT=attempt2 and two attempt-specific canonical-write diagnostics. Source closure/digest/job recipes and review separation remain unchanged.
- run() captures prior owner registration count before Popen, then records pending key/PID/expected_attempt=prior+1 before further admission. An absent key expects registration1, independent of experiment attempt2.
- Pending release requires corresponding active live child, matching key/device, exact job, worker_pid, expected registration attempts, RUNNING and allocated is True. Frozen register resets allocated=false. Release takes a fresh NVML sample, followed by existing fresh admission sampling before another launch.
- Exit handling clears only the matching pending key/PID. A nonzero child exit PAUSES without allocation-release evidence.
- allocation_acks adds worker_pid. STARTUP_RESAMPLE binds PID/expected attempt; release audit binds expected and observed PID/attempt, source/design and current telemetry.
- README updates all current commands to attempt2/fresh restart review, real contributors, fresh owner/actual metadata gates, unchanged five-seed budget and report-then-USER_STOPPED instruction.

Current per-card memory<85%, power<85%, hostRAM<90%, pending exclusion and current used+measured full-process envelope<=CUDA-usable100% remain unchanged. No worker cap, projected85% ceiling, forecast power or cross-card fit requirement was added.

## File identities

| File | Added/removed lines | Old SHA256 | New SHA256 |
| --- | --- | --- | --- |
| experiments/completion_weight_attribution_20261117/README.md | +20/-16 | 99a6335d8fa190560e73727ba10a6eb2174989058b4ba58168ef11fc379df159 | 86724212cb2aa0352864c1d7f300514d55861f06be98974212b13b367cd2bfe5 |
| experiments/completion_weight_attribution_20261117/protocol.py | +3/-3 | 6bc39247c7bfd57ed6ace80b015629f949155225ca961183c3065cddf0d1a033 | 1ac986b4996808903a97dab431d29503add770e9273422ff06c888ab4499b53b |
| experiments/completion_weight_attribution_20261117/runner.py | +95/-9 | 06a2c55cad687e05ae069989e5799432d333c757f1386842fa748c7a4c8fd8b1 | b95d54a81210641824afe1596c6fd732743ec694ef528b1ffab0062d4c98f638 |

Only these three source files changed; the other45 closure files remain byte-identical. All48 attempt1/source_snapshot files remain exact. No old jobs/ACKs/results/reviews/source receipt/binding was written.
core.py: 790ae97d01a3ab3d39e42712580e333aa017b96d5869f36aed682181b03de683
checks.py: 7bbf10c3e4249d9a3af7e1ad80b7c93267a66f99ddc52e3050c709ba99605ae1
Installed Adam: b099132d262d6cb16eefa48ab4bae789a0655cafb8571e421ef4fe3d2ff7c34c
Numeric control index: e8d67c6b18ad9ddbfa19b752e35c6ab051e3170a3d28df635b86b27922441fdb; unchanged/read-only1020 numeric controls are validated by the bounded protocol check.
All20 new keys are unique and disjoint from old keys. All20 old on-disk jobs match the unchanged recipe; every new scientific job field excluding only source_hash/key matches its old counterpart. Full new jobs are in the JSON; no job file or registration was created.

## Verification

Both edited Python files compiled in memory using compile(source_text,filename,"exec"). No py_compile/compileall.
Exactly one bounded protocol/runner/tables finite entrypoint passed: 6/11/5 groups, exit0, 3.5701664419611916 seconds.
The actual production run() transcript with mocked launch/status/telemetry and in-memory files passed all eight cases:

1. Absent owner status retains pending.
2. Old PID/attempt with allocated=true retains pending.
3. Expected PID/wrong future attempt retains pending.
4. Wrong PID/expected attempt retains pending.
5. Exact generation allocated=false retains pending.
6. Exact live RUNNING job/PID/attempt/allocated=true releases once and resamples before next same-card launch; new key registration1.
7. Pending GPU0 permits fitting GPU1 admission; absent key registration1.
8. Matching allocated=true with child exit9 takes PAUSE/exit cleanup without ACK release or further launch.

Existing strict current85%/host90% and physical100% boundaries, populated CPU Adam isolation, real-core finite integration, commit/lost-ACK/recovery, engineering replay and cost fixtures passed in that same bounded entrypoint. Prefix recovery remains12 returned calls/3 diagnoses/9 sweeps/12 virtual steps with6 replay calls; suffix-receipt recovery remains8/2/6/8 with0 replay calls. These are CPU fixtures, not actual GPU engineering acceptance.
Reused unchanged core/checks proof by hashes:54 core calls,10 direct source calls,112 disposable Adam steps,4 actual-M3oE CPU comparisons. No math/core/old-QR broad retest.
The known installed PyTorch temporary distributed-template import was generated from its installed source in RAM before checks; stdlib IO/import hooks restored before checks. Production code/Adam/hash validation had no external overrides. Existing frozen TF32 setup emitted its installed deprecation warning.

## Genuine contributors and handoff

- core Volta: 01a1e54b-0376-79c0-b1a7-20a6893345c7; A4/CORE_IMPLEMENTATION_REPORT.md; report SHA256 377759b1679b3146d609a0cbd63135c7385d256e8bd1bc8fb6fb91727b11a440.
- runtime Chandrasekhar: 01a1e54b-03a0-7613-a537-dba4cc0f0e80; A4/RUNTIME_IMPLEMENTATION_REPORT.md; report SHA256 c9abe1fcc26f1f50150fda1075c029078394ecca2967ff3063bda467eadb40ab.
- R1 Leibniz: 01a1e6cd-54c0-7a11-80ca-fb7194be1f15; A4/R1_IMPLEMENTATION_REPORT.md; report SHA256 9a476322010f205e8a652780da4331fe3c82cbe3692fe4253fabbd59de245abb.
- restart coder: 01a1e747-c8be-7a73-af93-5a9f2966d3b1; A4/RESTART_CODER_IMPLEMENTATION_REPORT.md.

Use all four genuine contributor IDs in the fresh review. Code reviewer excludes every coder; resource reviewer also differs from code reviewer. Existing check_review is unchanged. No old source/review approval authorizes this candidate.
Parent next dispatches distinct code review, prepares attempt2 and fresh foreground RAM storage, then actual-metadata-verified DEFAULT Luna/high four-case GPU engineering, distinct resource signoff, parent actual RAM acceptance/four-key release and fresh supervised formal RUN.
All20 trajectories restart from source initialization for50 epochs each:1,000 useful ACK epochs/34,500 Adam+EMA updates. Historical attempt1 275 ACK epochs/7,733 updates retain separate numeric/physical provenance and zero restart credit.
No GPU query/training, real owner IPC, actual dataset-cache/Test access, tensor files, prepare/signoff/RUN, parent R edits, Git, global role edits, children or independent/background AI were performed.
Parent owns selected-RAM acceptance, analysis/report/evidence review/context/publication/delivery and final USER_STOPPED. No C05/Astra. Context returned permanently; do not resume/reuse.
