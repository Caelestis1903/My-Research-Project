# Model source audit

## Pinned inputs

`SOURCE_MANIFEST.json` records the repository URLs, pinned commits, relevant paths and Git-provided blob IDs. Source identity is based on those Git objects and the existing direct-byte reference receipt; this work generated no new digest.

HyperGate is taken from the RecStudio submodule pinned by HyperGate parent commit `fdc82e3f3feee23694a8123f91092615f977900e`, at RecStudio commit `f08cca607e998e8be4c5e915a393346baffa71f6`. The adapter loads the upstream `HyperGate` class and its `ctr`/`layers` modules from `/tmp/mdmt_sota_sources_2526/RecStudio`. It provides an in-memory, narrow `_MultiTaskBase` lifecycle shim because the upstream runner imports optional NNI tooling. It does not copy RecStudio code into the experiment package. The pinned RecStudio checkout contains its MIT license.

The HyperGate adapter retains all three upstream gates, both augmentation networks, anchors, detach placement, expert biases, towers, and auxiliary objectives. It maps the external domain IDs `0,1,2` to RecStudio token IDs `1,2,3` because token `0` is padding. Its predictor computes logits without labels; its objective performs one predictor pass and adds the task and domain auxiliary losses with their registered coefficients. Input schema and sparse IDs are validated before casting. The official task order is fixed as `__mdmt_task_0`, then `__mdmt_task_1`.

The adapter now follows the pinned `Recommender.fit()` initialization point: after `_init_model`, its narrow base shim applies `recstudio/model/init.py:xavier_normal_initialization` to immediate children in their registered order, matching `Recommender._init_parameter()`. The pinned base YAML sets `train.init_method=xavier_normal`. `_OrderedFieldSet` preserves schema order through the upstream set subtraction before embeddings and other modules are created. The regression builds a second official core independently, applies the pinned initializer directly, compares its complete state and logits, checks zero Linear biases and zero padding rows, and compares predictions from fresh processes with `PYTHONHASHSEED=1` and `2` under Torch seed 2022.

The pinned HyperGate repository does not provide a complete paper configuration. The values in `default_config.json` are therefore labeled `UNVERIFIED_PROPOSAL`; the implementation does not describe them as published values or claim paper reproduction.

AutoIFS is loaded from `/tmp/mdmt_sota_sources_2526/AutoIFS-pinned` at commit `65fc27cdfad1eb6d7a4a2cc69cd31f61cc8e3d7e`. The pinned checkout has no `LICENSE` file. Its model and layer source remain external and are not copied or edited in the deliverable.

The AutoIFS adapter replaces the input embedding boundary to preserve the `[B,F,E]` interface while applying disjoint global offsets to sparse fields and `age × learned_vector` to the normalized dense age field. Each categorical block has exactly the declared vocabulary size, including OOV ID 0, and there is no extra row accepting an invalid ID. The official forward keeps its four cross-domain/task paths, LoRA initialization and selector algebra. After upstream construction, the adapter creates each of the four bias lists at the corresponding affine output width: `D_bias`, `T_bias`, `D_lora_bias`, and `T_lora_bias`. This is a local mechanical shape correction; it does not modify the pinned source.

Search uses the official selector and its temperature-scaled sigmoid penalty. Export records the final selector and the final search backbone embedding; import copies that embedding into the frozen gate embedding in the rewound model. In ticket mode the official `logit > 0` threshold and subtraction remain unchanged. `optimizer_groups()` separates trainable base and gate parameters during search and returns an empty gate group after ticket freeze.

M3OE_FIXED1 calls the existing `experiments/mdmt_basp_v1.prefix.make_native` helper. The parent reference receipt records that source as byte-equal to reference commit `97151c74544c3651feb805d1898bc9ce1b8d12d3` and identifies its existing Git blob. The helper constructs Original then fixed1, copies weights and checks direct tensor equality. The adapter does not call the legacy hash-based initializer.

M3OE_FIXED1's training objective is the natural BCE mean over all observed task labels, matching the native trainer's default mean reduction. With full labels, loss and parameter gradients match direct native BCE; with unequal task masks, the adapter divides by the total number of observed labels. HyperGate and AutoIFS keep their official sum-of-task-means objectives.

For the locked schema, ordinary sparse IDs are accepted on `[0,vocab_size-1]`; ID 0 remains OOV. RecStudio keeps its upstream zero padding row for ordinary fields and allocates one additional row only for its internally remapped domain token IDs 1–3. M3OE and AutoIFS use exactly `vocab_size` rows.

## Official trainer findings and protocol changes

The pinned RecStudio tree has no `hypergate.yaml`; the core's optional gate flags otherwise default to disabled. The registered adapter configuration explicitly enables all three mechanisms. Its predictive objective is the sum of task-mean BCE, with domain cross-entropy and task auxiliary BCE weighted separately; validation selection uses predictive macro LogLoss rather than the auxiliary training objective. The input-only predictor separates the upstream `score()` label-dependent auxiliary calculation from inference. Complete parameter provenance and the unverified configuration fields are listed in the design specification, section 4.

The upstream AutoIFS script includes search, a start-of-search-epoch-9 rewind, final selector/embedding transfer, hard ticket selection and reuse training. It also invokes test evaluation after search and in its training workflow. Those test calls are excluded from this integration: optimizer steps use only T0∪F, selection/stopping uses V macro LogLoss, and E requires a separate final evaluation after freeze. The original V macro-AUC selection is an explicitly changed evaluation rule. Search and reuse retain separate optimizer state and the official optimizer partition; complete settings are in the design specification. The official TFRecord preparation, splits and KuaiRand field set are not asserted identical to the repository's locked CSV protocol. These are documented data/selection adaptations, not method improvements.

The original source is retained unchanged in external checkouts; local bias corrections and input mappings live only in the adapters. Original dependencies are inventoried by the source manifest, while the actual project environment is recorded in `ENVIRONMENT.json`; no original TensorFlow/TFRecord stack was installed. The common adapter output is raw logits `[B,2]`, and all requested checkpoint, prediction and metric exports belong to the new pipeline rather than the original standalone runner.

## Verification and limits

`experiments/mdmt_sota_baselines_2526/tests/test_models.py` runs against the pinned official model code on synthetic CPU tensors. It checks independent HyperGate initialization/state/logit parity, zero biases and padding rows, cross-`PYTHONHASHSEED` reproducibility, HyperGate objective and representative gradient parity, gate and LoRA gradients, AutoIFS output-bias dimensions, one-forward objective and search-to-ticket state transfer, M3OE native-loss and gradient parity, sparse/domain boundaries, masked labels, and singleton/empty batches. The test command and result are recorded in `MODEL_IMPLEMENTATION_REPORT.md`.

These checks establish adapter correctness on synthetic inputs only. They do not establish model performance, paper reproduction, or real-data readiness. No real training was started.
