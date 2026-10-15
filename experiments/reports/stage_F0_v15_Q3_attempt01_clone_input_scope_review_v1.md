# REVIEW REPORT — Q3 attempt01 single frozen clone-input supplement

## 1. Scoped verdict

**PASS for the proposed copy-plan scope only.** A fresh ordinary-copy isolated qualification clone may include exactly `experiments/stage_F0_outputs/execution_isolation_gate_bootstrap_remediation_v1_no_go.json` from the live source after the specified identity/SHA safeguards. It is already a frozen dependency of the unchanged reviewed runtime, not a new control algorithm or production authorization artifact. No copy, test, retry or RUN was performed or authorized by this review.

The v6 implementation precheck remains unchanged. B6 historical OPEN/FAIL, forward NOT_GRANTED and execution NO_GO remain. Parent must separately authorize the next bounded qualification attempt after clone-input prechecks; no automatic retry follows this finding.

## 2. Independent verification

Runtime `stage_F0_successor_v15.py:43–44` declares this exact source path and SHA `7126c372c2382516d81d20c91c8de71aeec9175397e2fd164b7635b116f64cef`; `_assert_frozen_artifacts:176–249` requires its ordinary-file identity and exact digest. Its absence in a clone legitimately fails closed. No validator bypass or hash adjustment is warranted.

Independent read-only `lstat`/ancestor validation and `O_RDONLY|O_NOFOLLOW` FD hash verified a stable ordinary single-link source, mode0444, size11459, device2049/inode54136358. Before/open-FD/after stat projections match, including size/mtime/ctime; SHA exactly matches the frozen value. Every source ancestor was an ordinary directory, with no symlink. These current observations are newly bound, not reconstructed historical source evidence.

B6 attempt03 manifest has neither this exact regular source-path entry nor a raw regular payload with its digest. Therefore the file must not be described as copied from that baseline archive or retroactively inserted into its inventory. The original user read-only source/evidence scope and v6 qualification's exact frozen contract support adding this one necessary control input. This does not authorize reading training data, loading checkpoints, constructing models, GPU operations or production state changes.

## 3. Additional binding needed

A new fresh clone-input manifest should bind:

- Exact source path, current/frozen/after-copy SHA, stable before/FD/after identity, ordinary single-link/no-symlink ancestry and UTC.
- Exact destination relative path, exclusive ordinary-copy publication, distinct source/destination inode, nlink1/no symlink ancestry and destination readback SHA. No hardlink, symlink, source write, production flock, replacement of an existing protected path or validator bypass.
- Separate `LIVE_POSTINCIDENT_FROZEN_INPUT_SUPPLEMENT` provenance. Bind baseline03 and preserved attempt01 failure by their actual hashes, together with v6 test/runtime/coder/review and this scope review; preserve all existing manifests/reports/attempts.
- Full new-clone frozen dependency and source inventory checks before any rerun. Exact reviewed test SHA is `80a6362977f80735bc569793ec1324004f3596cf6bcb43216b33f16c691642f9`; runtime remains `0b8020e5d26a83d2acb4db1e9d3a79953554e35ae4289a59c17f0aaae58427c2`; coder report is `1667b761c43d68f8d33452b6557409a2a8caf15735a701ffb335f84e82f44723`.

A static comparison of all nineteen `_assert_frozen_artifacts` identities against baseline03 shows twelve exact regular rows with matching recorded SHA and seven without exact regular rows: the two v1.5 DESIGN/review documents, this remediation candidate, and four bootstrap incident failure/stdout/stderr/phases paths. This is a baseline raw-path coverage observation, not proof that all other inputs are absent from the attempt01 clone; those may already be separately sourced and bound. It prevents assuming baseline03 alone supplies the complete frozen graph. The one-file recommendation does not authorize adding those other inputs anew; their existing clone origins must be checked in the new input manifest, and any further additions reviewed separately.

## 4. Corrections and execution

None. Only new external audit metadata and this report/evidence were written. Source bytes were read and hashed but not copied. The initial audit wrongly assumed this was the sole missing baseline raw identity; corrected static mapping preserves the full nineteen-path coverage and discloses all seven omissions. No experiment, pytest, worker, claim, release or production file mutation occurred.

## 5. Limits

The stable current SHA supports an honest new supplemental snapshot; it does not establish baseline03 retention or restore B6 pre-incident inventory/preimage/shared-lock diagnostics. Baseline03's acceptance remains bounded to its actually inventoried paths and cannot be cited as complete qualification-input preservation. This addendum does not mutate or silently upgrade that baseline. Historical B6 remains OPEN/FAIL.

## 6. Recommendation for proceeding

Include this one file in the corrected clone-copy plan with the new provenance/readback binding. Stop before retry until the corrected fresh clone manifest, nineteen frozen dependencies and v6 reviewed-source identity checks are complete and parent separately instructs runner. Any later RUN remains only the previously listed three synthetic/administrative isolated Q3 nodes; no production/scientific authority is granted here.

Evidence `experiments/reports/stage_F0_v15_Q3_attempt01_clone_input_scope_review_v1_evidence.json` SHA `3a54b86f43e5767b2a07d686f4d2d6a38a8eda595399146bbf35e5e2bd593689`. Raw audit `/tmp/f0-v15-Q3-missing-frozen-artifact-scope-20261014-a/scope_audit.json` SHA `e4ce81c4ef5fbe2e570483f20167dbf50a7816ca8a976924aed2d15af4450ba1`.
