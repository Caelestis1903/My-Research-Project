# Data and execution availability

The completed cycle is rooted at
artifacts/26-11-16_augmentation_gradient_momentum/astra_cycles/cycle_01/attempt1.

Its ten CSV tables cover all50 new/reused trajectories:400 endpoints,
full validation curves and cell metrics, paired seed deltas/intervals,
support/weight ledgers, gradient diagnostics and physical attempt costs.
The numerical report files in this directory derive from these exports.

The raw archive contains4768 numerical/log/job files for30 new trajectories,
their39 physical attempts, and20 exact historical CLEAN/NDP-A controls.
Controls retain their original bytes and are mapped from original paths to
reused_controls/<key>/ by CONTROL_ARCHIVE_MAP.json. Source controls were
read without modification.

RAW_RUNS_MANIFEST.json records every member's SHA256, the source control
mapping and all11 archive parts. The full tar.gz has87002844bytes and SHA256:

    1a0013c1ae3e6b6ad2b3d1b26646e0c83441800ea3214663fe3ef5ca22a3e4ae

From the cycle's attempt1 directory, reconstruct the published archive with:

    cat RAW_RUNS.tar.gz.part-* > RAW_RUNS.tar.gz
    sha256sum RAW_RUNS.tar.gz
    tar -xzf RAW_RUNS.tar.gz

The parts preserve runs/, jobs/, launch_logs/ and reused_controls/. Top-level
journals, manifests, recovery/source/review receipts, policy history and CSV
tables are published directly. Local unarchived records remain available.
Socket credentials and guardian.json stay in the local runtime;
public manifests retain their fingerprint references.

Numerical reproduction from the repository root:

    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m research_reports.task_axis_even_views_20261116.ANALYSIS
    /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -B -m research_reports.task_axis_even_views_20261116.FIGURES

The project environment is phase0-3-env, Python3.13.5/PyTorch2.9.0+cu126 with
requirements-phase0-3.lock. Raw inputs are
/home_new/xzf/datasets/ml-1m/ml-1m.csv and
/home_new/xzf/datasets/kuairand/processed/kuai_rand.csv; their hashes and
prepared-array/split/preprocessing identities are in each data manifest.
Inputs and prepared caches remain outside the result archive.

Scientific79-file source aggregate:
6b16d5e6bd5fcb96d8824a4161ec102af25e5a23ed86ff21f70d21e05eee5ff1.
Final separately reviewed scheduler overlay:
b7eeba76d21727b596897eec84e0e5766c01e8a41fba832c1036d38bdf7c6f3f.
Seven-file design aggregate:
f682ad676b8a2a84b21ebb3fd63aef41348ee49f45d838a9cbca2af2b10e9754.
Original source/overlay archives and reviews preserve earlier versions.
CURRENT_UTILIZATION_EXECUTION_POLICY.jsonl records the two reviewed
current-utilization controller versions. The first pre-overlay scheduler,
policy-switch ACK continuation and physical-capacity recovery are recorded
in runner reports and recovery receipts.

All30 selected online/EMA states remain in the original parent's live RAM
owner, PID1570727/tool session59365,286526880tensor bytes. The final
RESOURCE_1_PARENT_STORAGE_ACCEPTANCE.json records the accepted same-owner
handoff. Model/optimizer/checkpoint files were never written by this cycle.
Test access remains false.
