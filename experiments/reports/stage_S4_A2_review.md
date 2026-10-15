# REVIEW REPORT — Stage S4 Amendment A2

1. Verdict: **PASS**. Blocking issues: **0**. This independently reviews the binding A2 amendment (top of `stage_S4_design.md`), the coder's A2 report (first section of `stage_S4_implementation.md`), the actual current S4/S3 source and tests, the frozen A1 queue/state, and both retained A1 attempts. No A2 queue was frozen and no formal training was launched during review. All hashes below were recomputed by the reviewer from live files, not copied from the coder report.

2. Requirement-by-Requirement Verification (Amendment A2 §10)

| Requirement | Independent finding |
| --- | --- |
| Real S4 output path with disposable completed-run fixture | `test_disposable_complete_s4_writer_and_validator` copies the accepted S3 seed-2023 `target_only_shared` artifact tree into a temp dir, builds an `S4Runner` instance, and executes the **actual** inherited `S3Runner.write_outputs` plus the real `S4Runner.write_outputs` (only the already-accepted S2 base serializer is stubbed). It writes complete S4 config/manifest, re-stages S4 stage fields into the four inherited CSVs, writes and hashes `gradient_step.csv` and exactly 100 `probe.csv` rows into the manifest before validation, then passes the **real** `validate_run`. The S3 `formal_protocol_for` hook is not mocked: the crash role `target_only_shared` now dispatches through `S3Runner._output_formal_protocol` to the S4 override. Test passes. |
| Adversarial rejections | All demonstrated by tests run by the reviewer: S3-only role leak (`fresh_role="target_only_shared"` in manifest rejected — validator requires `fresh_s4`); missing `probe.csv` rejected; corrupt `probe.csv` rejected; observed-vs-expected protocol mismatch (`seed=9999`) rejected; partial manifest (`formal_protocol` removed) rejected; altered source hash (`code_hashes` entry zeroed) rejected; A1 promotion/collection into A2 rejected (mixed A1-raw/A2-queue collector call → "isolated", A1 canonical path present → freeze refuses "canonical"). The validator additionally writes config/manifest only after the protocol mismatch check raises, so a mismatch can never mark a manifest valid. |
| S3 non-regression (default hook) | `S3Runner._output_formal_protocol` at `stage_S3_run.py:414` defaults to the exact original `formal_protocol_for(self.fresh_role)`; `write_outputs` calls it polymorphically at line 435. `test_s3_writer_protocol_default_and_s4_dispatch` proves an ordinary S3 runner still returns the identical S3 protocol and that the S4 override returns the S4 arm protocol. `stage_S3_protocol.py` (role list) untouched — the A1 crash role still raises there, as designed ("do not add an S4 role to S3's allowed-role list"). |
| Unchanged-science gates re-run | Full suites run by reviewer in EDDAenv from `D:\CODE\phase0-3\experiments`: `python -m unittest discover -s tests -p "test_stage_S4_*.py" -v` → **21/21 OK in 40.3 s, zero skips**. Covers installed CUDA Adam clone parity (first/warmed, zero/`None` gradients, both LRs, coupled decay), the real-device CUDA M3OE test with `C` and `A_arch` on `cuda:0` verifying installed foreach dispatch, vector-relative tolerance accept/reject regressions, one-forward partition, F/T/N/U main and architecture steps with once-only Adam counters and private FULL deltas, undefined-direction fail-closed, probe support/content/math, N predicted-delta reference, process overlap, peer termination, stdout retention, promotion rollback/recovery, transient-errno retry classification, A1/A2 freeze isolation. CUDA confirmed live: `torch.cuda.is_available() == True`, CUDA 11.7, PyTorch 2.0.0; the CUDA test executed (unittest reported no skipped tests). |
| A2 matrix/schedule equality | Reviewer recomputed live: `_normalized_matrix(matrix(A2_RAW_ROOT)) == _normalized_matrix(A1 tasks)` → True (only `run_dir` removed); `waves(...) == A1 waves` → True, exact order `FT×2022–2025` then `NU×2022–2025`. `freeze_a2_queue` requires empty A2 raw root and empty A2 output root before initial freeze, freezes a complete ≥21-entry producer inventory (live: exactly 21 entries) including M3OE forward `a7702a64…`, and fails on any drift. Calling `freeze_a2_queue(allow_create=False)` traversed every predecessor check and then refused with exactly "A2 queue must be frozen after reviewer PASS before RUN". CLI `--a2 --freeze-only` without `--reviewer-pass` refuses before any creation; CLI `--a2` without `--reviewer-pass` refuses; no file was created by either. |
| A1 queue/state/attempts identity | Reviewer-recomputed live SHA-256: A1 queue `c6a8f1322213bd001f6994d42ff009e460e869e6d0624903de08e12088600d66`, A1 state `540aeb37729a749585dd018adb3e242caf05c83a5d88dfb647a51675a5f2d554` — both match A2 §5 exactly (no drift). State has `completed=[]`, one attempt, `retry_class=blocked`; 0/16 canonical run dirs exist; A1 output root absent. `_attempt_inventory` over `stage_S4_A1_raw_results/.attempts` finds exactly **56 files, no symlinks**, deterministic digest `3bb4e8859533c597790215f433876f89a40a88714cf23503731478c05d670c6a`. Both F and T `failure.json` record `ValueError: unknown Stage S3 fresh role` after ~9155 s of training, and the F console traceback ends in `stage_S3_protocol.formal_protocol_for` — byte-level confirmation of the A2 §5 failure audit. Neither attempt contains `probe.csv`. The embedded original-abort receipt is re-verified: old queue `693b4669…`, old state `8ebff1bc…`, old abort report `52b6d592…`, old attempt tree digest `4bae95e4…` (4 files) all match and match the A1 queue's embedded receipt. |
| Immutability / absence | `experiments/stage_S4_A2_raw_results`, `experiments/stage_S4_A2_outputs`, `experiments/stage_S4_queue_manifest_A2.json(.state.json)` do **not** exist. Old `.attempts` digest unchanged (`4bae95e4…`). S3 artifacts untouched: `stage_S3_summary_manifest.json` present (SHA `c86717df25c7e6b56d1daae2764d044e8ad889c1782e6f08fad0072bc42c07e4`, status `STRUCTURALLY_VALIDATED`). No A2 paths exist, so no old/A1 collector rows can be present there; the collector accepts only the matched isolated A2 (or A1) queue/raw/output tuple and rejects anything else. |
| Fail-closed finalization | A raised error in `write_outputs` leaves the child's `failure.json` and `process_interval.json` status `FAILED`; validation/promotion only follows a clean exit plus proven overlap plus full `validate_run`. `_promote_wave` journal (PREPARED→BOTH_PROMOTED→COMMITTED with reverse rollback) and `_recover_promotion` (crash-between-renames recovery to `ROLLED_BACK` with retained rollback paths) are intact from A1 and covered by passing fault-injection tests. Retry remains limited to one whole-wave retry on explicit transient `OSError` errnos only. |
| Scope | Only the A2-authorized files differ from the A1-reviewed state: S4 protocol/run/orchestrator/report, the two-line default-preserving hook in `stage_S3_run.py`, and the two S4 test modules. `stage_S4_gradient_controls.py` hash `e695bbfa…` is byte-identical to the A1 PASS state. M3OE forward `a7702a64…` unchanged. F/T/N/U update rules, 16 keys, seeds, eight waves, seven checkpoint labels, metrics, and tolerances (`1e-6`/`1e-4`, direction `0.99999`) untouched; grep confirms `0.003`/`0.005` appear only in the design document (interpretation stays with the Main Agent), coefficient `0.3834574423` only in the design/frozen S3 coefficient table, batch 40960 and lr 0.01/0.001 unchanged in code. |

3. Scientific & Algorithmic Integrity Audit

- **Mathematical and tensor correctness:** Unchanged from the A1 PASS. F applies FULL gradients; T applies target-only gradients to `C`/`A_arch`; N scales FULL shared gradients to the target-only norm before Adam; U commits one FULL-gradient Adam step and rescales only the shared delta to the same-state counterfactual target-only delta norm. Private main gradients remain FULL. Norm audits in float64; design tolerance `max(1e-6, 1e-4×reference_norm)` preserved.
- **Gradient flow and loss participation:** One stochastic forward per batch; probes reuse it without extra forwards, optimizer steps, or RNG/tau disturbance (architecture batch under `preserve_rng_state` with restored-digest evidence).
- **Controlled variables:** A2 changes only the output-finalization dispatch, validator, tests, and A2 provenance roots. Data/coefficient/initialization hashes remain bound to the frozen S3 values; pairing fields and block validation unchanged. No A1 attempt byte is read as a formal input: `_a2_predecessor_receipt` hashes the attempt tree and explicitly rejects any A1 canonical or collector product before freeze.
- **Fail-fast:** Undefined direction, nonfinite values, source drift, broken probe support/content, process-overlap failure, mixed collector roots, and predecessor byte drift all raise. `validate_run`'s broad catch converts defects into rejection reasons, never acceptance.

4. Over-Engineering & Code Bloat Audit

The hook is two lines in S3 plus a ~7-line S4 override — the minimum the amendment allows. The A2 freeze/receipt/collector routes mirror the already-reviewed A1 pattern without new abstraction. The `_read_json` empty-list workaround was replaced by a direct `json.load` for `nonfinite_inventory.json` with an exact `[]` requirement — no validator relaxation. No S1–S3 refactor, no speculative machinery, no new algorithm.

5. Bugs / Deviations Found

None. The coder's self-reported `_read_json` empty-list issue is confirmed fixed in current source (`stage_S4_orchestrator.py:376-379`). Minor cosmetic note (non-blocking): `AMENDMENT = "A1"` in `stage_S4_protocol.py` is still used by the A1 freeze path while the A2 path hardcodes `"A2"`; behavior is correct in both routes.

6. Corrections Made

Reviewer made no code changes. Reviewer independently: recomputed all module/evidence hashes; ran the full 21-test suites in EDDAenv with CUDA live; executed the A2 freeze preflight (`freeze_a2_queue(allow_create=False)` and both CLI refusal paths) confirming refusal without creating anything; recomputed normalized 16-task/eight-wave equality; recomputed old and A1 attempt-tree digests; spot-checked S3 summary integrity.

7. Remaining Risks

Two-process A2 CUDA training has not run; GPU capacity for two concurrent 40,960-batch processes and all scientific outcomes remain unknown. The fixture test exercises the writer/validator with synthetic step/probe values, so the first real wave remains the first end-to-end exercise of per-step logging volume; any new defect after A2 freeze stops for a fresh design/review decision. The two A1 attempts supply no valid AUC/loss evidence.

8. Whether Experiments Are Safe to Run

**Yes, for the isolated A2 route only.** Run from `D:\CODE\phase0-3`, in order:

```powershell
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_orchestrator.py --a2 --reviewer-pass --freeze-only
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_orchestrator.py --a2 --reviewer-pass
& 'C:\Users\86199\.conda\envs\EDDAenv\python.exe' experiments\stage_S4_report.py
```

The run command executes exactly two concurrent Python training processes per wave, `(F,T)` seeds 2022–2025 then `(N,U)` seeds 2022–2025, all 16 from epoch zero. The collector defaults to the matched A2 tuple and writes only under `experiments\stage_S4_A2_outputs`; collect only after 16/16 canonical runs and 4/4 valid four-arm blocks.

At freeze, verify the 21-entry producer inventory digest `8dc07f3df68fb3f8186c1c2585a736f6302edfc4a53e301d3dde04c3149f7e07` and these reviewer-recomputed post-review SHA-256 values; any mismatch stops the run:

| File | SHA-256 |
| --- | --- |
| `experiments/stage_S4_protocol.py` | `a40d62059410bbdf4054fb80cb786baf0e13ad9b6bf6f5bc6560f2474ddfffb2` |
| `experiments/stage_S4_gradient_controls.py` | `e695bbfafafb9b238d5cb691fde5113bc7e87f2b7699e766fb84d71167090645` |
| `experiments/stage_S4_run.py` | `99b171cba6a64e3adb4dab141bb9fb8f7c0ed091248c459ca586ce04914ca107` |
| `experiments/stage_S4_orchestrator.py` | `e44b44551c0dfc6136f13070227b885f41b8c0218cf3c3a8349837d9261c9801` |
| `experiments/stage_S4_report.py` | `97e1f97f600d9f5c99a7fbee7fe42b8b7157a22090b9f122172606e0364d9695` |
| `experiments/stage_S3_run.py` | `8db00e535f56160ae7888010bf917d275b1a88863605a918d0b24c4a9bf25e8d` |
| `m3oe/MDMTRec/models/multi_domain/model.py` (forward) | `a7702a64f8b5471f46209c9bfa1d9cb3108f11a89122933d7ae19942e9a12ce1` |

Predecessor evidence to be embedded at A2 freeze: A1 queue `c6a8f1322213bd001f6994d42ff009e460e869e6d0624903de08e12088600d66`, A1 state `540aeb37729a749585dd018adb3e242caf05c83a5d88dfb647a51675a5f2d554`, A1 attempt-tree digest `3bb4e8859533c597790215f433876f89a40a88714cf23503731478c05d670c6a` (56 files), plus the A1-embedded original-abort receipt (old queue `693b46697ba40bf6e9cd03c6e0d4688a3ee7aa80a5d6ce2a442bed5c846fb221`, old state `8ebff1bc35dd7fa5a182cfb3d56b88f7e336bedb66cd93fba588c5bc188c622e`, abort report `52b6d5924fb60c6453e669d6318a11cc064cf8858c770b77440ff30e33a5d30e`, old attempt digest `4bae95e4e35bb27fceaf3f70fe2ab21f57d24110f9471a1951a701f3ff1a0e18`).
