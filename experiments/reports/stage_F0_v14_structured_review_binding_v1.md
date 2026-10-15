# F0 v1.4 structured independent review binding — v1

The separate immutable reviewer artifact is `experiments/reports/stage_F0_v14_implementation_review_structured.json`, SHA-256 `57a3024aea4a14dfab029ad3b670ca175e31a54379277082d042402c3c5fac44`.

It records the already completed implementation PASS and independently reviewed pre-migration source GO. It does not perform a new experiment or broaden either decision. It binds:

- Implementation PASS report v4 SHA `5540595752ad61ef519f15979fdb4b1cb0ddf5397595de9f7082122596652410`.
- Pre-migration source GO report SHA `f7fe005a16c3c7db955f3605b17781a85b50e5010f60b2a3aaff2500f62c1653`.
- Design SHA `c7204144394c734d84a61da2d0a702ccd8223308a35f3bd01119cf217840ebf9` and finalized coder report SHA `aa8c709dc9b349340f9613181e28d4e8c72b5301b9ae40cb8262ce0146c3b080`.
- Exact source snapshot file SHA `717ea4baeda17e89f684004ded3b230aff30826f6649b4ebb5d592c0d2984533` and embedded unsigned-snapshot SHA `3df82eb0ab75922503fe0b82a971e06c8a815959287a27046adcd879936e5b5f`.
- Actual complete control hashes/digest `fd4733a4eaa967398e50b21f728f1fcaa97cc00c199ee72826349b49f0f255c8`, immutable historical projection SHA `43d22e38332d301667bd1b6598044dd360597faec8557b47a9bbf3fb97490251`, and reviewed v1.4 module SHA `10d81444afa55c13c8aab09605da0ea9840354e0be1eff9d2b4cc7738bd3ae16`.

The required `source_snapshot_sha256` field uses the **file** SHA because the actual migration API passes `sha256_file(snapshot_file)` to `_validate_implementation_review`; the current gate also binds file bytes. The requested embedded digest is separately recorded as `source_snapshot_embedded_sha256`. Putting the embedded digest in the file-binding field would fail the unchanged approved API.

Before writing, the actual strict source validator and frozen-pointer validator passed again. The actual `_validate_implementation_review` passed both before immutable publication and after rereading the published JSON with the migration API's file-hash semantics. The artifact uses schema `F0-POSTABNORMAL-IMPLEMENTATION-REVIEW-v1.4`, verdict PASS, role independent reviewer, and reviewer identity `/root/f0_v14_review`.

Only the new reviewer JSON and this explanatory report were written. Production gate/hold/queue/source snapshot were unchanged. No migration, release, claim, training, checkpoint load or scientific-result analysis occurred. This structured decision authorizes only the already approved designed administrative migration; hold remains active until later separately approved postmigration/prelaunch/principal-release stages.
