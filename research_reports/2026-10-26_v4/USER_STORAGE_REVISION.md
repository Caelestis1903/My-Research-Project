# User storage revision

The user reported deleting their old `/home_new` files and explicitly requested that no other `/home_new` folders/files be touched, and that large model `.pth`/equivalent archives not remain in the final delivery. This direct instruction supersedes the locked plan's permanent model/optimizer retention requirement.

All new experiment writes stay under `/home/xzf/Code/phase0-3/artifacts/mdmt_dfd_v4/v4_0/`. Complete model/optimizer state is temporary solely for training continuation, legal checkpoint selection, dependent-stage execution and final prediction reconstruction. Native/pilot production keeps bounded rolling last/selected/endpoint state, not a per-step archive. Temporary model files and large h/z/intermediate caches are removed after their final authorized dependency. Final logs, metrics, raw row predictions, data/source/config identities, summaries and reconstruction/cleanup receipts remain.

No old data, report, checkpoint or other user's files are modified. Missing old RHD anchors are recorded MISSING_ARTIFACT; they are not retrained or replaced.
