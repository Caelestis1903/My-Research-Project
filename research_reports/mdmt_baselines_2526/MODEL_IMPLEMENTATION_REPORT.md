# Model coder implementation report

## Implemented

- Added exact pinned-source validation and loading in `experiments/mdmt_sota_baselines_2526/sources.py`. HyperGate loads the pinned RecStudio model/layers and initializer using a narrow in-memory runner shim; AutoIFS loads its official class from the external checkout.
- Added the HyperGate adapter in `hypergate.py`. It creates modules in schema order, follows the pinned `Recommender.fit()` initialization order with the official `xavier_normal_initialization`, maps source domain IDs to the upstream padded-token convention, emits raw `[B,2]` logits without labels, retains all three official gates and computes the predictive, domain-auxiliary and masked task-auxiliary losses in one forward pass.
- Added the AutoIFS adapter in `autoifs.py`. It preserves the external forward, four paths and selector; adds mixed sparse/dense input embeddings and output-width bias lists; exposes `optimizer_groups()`, `export_search_state()`, `import_selector_state()` and `freeze_gate()`; and uses the selector penalty during search but not ticket reuse. The final backbone embedding, including the dense-age vector, is transferred to the frozen ticket gate.
- Updated `models.py` with the common factory and exact mappings from the registered config fields to adapter settings. M3OE_FIXED1 uses the existing BASP direct-copy initialization helper and the native natural BCE mean over observed labels. All three wrappers validate sparse IDs in `[0,vocab_size-1]`, domains, labels and masks and return raw logits.
- Added model-level synthetic CPU checks in `tests/test_models.py` and source provenance in `SOURCE_MANIFEST.json` and `SOURCE_AUDIT.md`.

No dependency was added, and no existing research package or upstream checkout was edited.

## Verification

Command:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home_new/xzf/envs/miniconda3/envs/phase0-3-env/bin/python -m pytest -q experiments/mdmt_sota_baselines_2526/tests/test_models.py
```

Result after the initial independent review findings F1, F2 and F9 were corrected: **8 passed in 3.97s**.

The targeted regressions compare HyperGate parameters/logits with an independently constructed official core initialized through the pinned initializer, check zero Linear biases and padding rows, and require identical predictions from fresh processes with Python hash seeds 1 and 2 at Torch seed 2022. M3OE full-label loss and gradients match native mean BCE; an unequal task mask matches a hand-computed mean over the four observed labels. Each model accepts the largest legal sparse ID and OOV 0 and rejects ID equal to `vocab_size`. The existing checks also compare HyperGate total auxiliary objective and representative gradients against the pinned official `score()` path at `rtol=1e-6, atol=1e-6`, verify gradients through all HyperGate mechanisms and AutoIFS shared/LoRA/selector paths, corrected bias shapes, one-forward objectives, masked labels, empty/singleton inputs, and AutoIFS search-state-to-frozen-ticket transfer followed by a base update.

The initial independent review's **FAIL** remains recorded in `REVIEW_REPORT.md`. This report records only the model-owned fixes; the root agent will request the follow-up review. The full experiment remains outside RUN authorization.

## Limits

- HyperGate parameters remain an `UNVERIFIED_PROPOSAL` because no complete paper config was found in the pinned repository. This code does not claim published-configuration reproduction.
- The pinned AutoIFS checkout contains no `LICENSE` file. It remains an external runtime source and has not been copied into the deliverable.
- Verification used synthetic CPU tensors only. No real training, experimental metric collection or research-effect analysis was performed.
- Pipeline training/checkpoint and role-permission behavior belongs to the pipeline implementation report and independent review; this report does not mark the full deliverable reviewed or ready for RUN.
