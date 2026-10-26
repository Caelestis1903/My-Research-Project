# 冻结协议、源码与父状态身份

本附录只含协议、聚合角色统计、配置及状态摘要，不含原始数据、逐行预测或checkpoint权重。它不读取正式模型质量结果。未来父状态没有预填hash：以下证书在实际父状态生成后核验，原配置保持不变。

## FROZEN_CONFIRMATION_SPEC.json

服务器来源 `artifacts/mdmt_next_v3/v3_0/FROZEN_CONFIRMATION_SPEC.json`；SHA256 `697c9f3250986d7687a26a21e1f4bc57b4a8077c62e1ff5208537d91d96c934b`。

```json
{
  "schema": "v3-post-P2-locked-reevaluation-v1",
  "status": "FROZEN",
  "frozen_utc": "2026-10-25T10:12:43.229970+00:00",
  "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
  "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
  "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
  "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
  "dataset": "kuairand",
  "seeds": [
    4101,
    4102,
    4103,
    4104
  ],
  "epochs": 50,
  "best_simple_kind": "R0_BASE",
  "methods": [
    "R6_PB_HB",
    "R0_BASE",
    "R5_PB_H1",
    "R7_PB_HISTORY",
    "STGU_DOMAIN_ADAM"
  ],
  "run_ids": [
    "FORMAL_kuairand_R6_PB_HB_4101",
    "FORMAL_kuairand_R0_BASE_4101",
    "FORMAL_kuairand_R5_PB_H1_4101",
    "FORMAL_kuairand_R7_PB_HISTORY_4101",
    "FORMAL_kuairand_STGU_DOMAIN_ADAM_4101",
    "FORMAL_kuairand_R6_PB_HB_4102",
    "FORMAL_kuairand_R0_BASE_4102",
    "FORMAL_kuairand_R5_PB_H1_4102",
    "FORMAL_kuairand_R7_PB_HISTORY_4102",
    "FORMAL_kuairand_STGU_DOMAIN_ADAM_4102",
    "FORMAL_kuairand_R6_PB_HB_4103",
    "FORMAL_kuairand_R0_BASE_4103",
    "FORMAL_kuairand_R5_PB_H1_4103",
    "FORMAL_kuairand_R7_PB_HISTORY_4103",
    "FORMAL_kuairand_STGU_DOMAIN_ADAM_4103",
    "FORMAL_kuairand_R6_PB_HB_4104",
    "FORMAL_kuairand_R0_BASE_4104",
    "FORMAL_kuairand_R5_PB_H1_4104",
    "FORMAL_kuairand_R7_PB_HISTORY_4104",
    "FORMAL_kuairand_STGU_DOMAIN_ADAM_4104"
  ],
  "physical_native_warmups": 4,
  "logical_trajectories": 20,
  "primary_endpoint_role": "benchmark_test_exposed",
  "primary_endpoint_metric": "macro_logloss",
  "primary_view": "selected",
  "companion_role": "Q_legacy_confirm_exposed",
  "development_diagnostic_role": "Q_explore",
  "selection": {
    "role": "validation",
    "metric": "macro_logloss",
    "direction": "min",
    "eligible_epoch": 12,
    "ties": "earliest epoch",
    "fixed_terminal_epoch": 50,
    "early_stopping": false
  },
  "comparisons": [
    {
      "candidate": "R6_PB_HB",
      "reference": "R0_BASE"
    },
    {
      "candidate": "R6_PB_HB",
      "reference": "R5_PB_H1"
    },
    {
      "candidate": "R6_PB_HB",
      "reference": "R7_PB_HISTORY"
    },
    {
      "candidate": "R6_PB_HB",
      "reference": "STGU_DOMAIN_ADAM"
    }
  ],
  "reporting": {
    "unit": "seed",
    "paired": true,
    "n": 4,
    "CI": "paired Student t95% df3",
    "secondary": [
      "macro_auc",
      "natural_logloss",
      "six cells AUC/LL",
      "task pooled AUC",
      "per domain/task",
      "worst cell",
      "endpoint50",
      "AUC/LL best validation secondary aliases"
    ],
    "no_development_pooling": true,
    "no_data_independence_claim": true
  },
  "exposure_claim": "Locked methods on new training seeds using previously exposed benchmarks; NOT unseen-data confirmation.",
  "eligibility": {
    "decision_utc": "2026-10-25T10:11:07.794363+00:00",
    "status": "ANALYZED_COMPLETE_P2",
    "complete_core_aux_p2_logical_trajectories": 69,
    "aggregation_receipt_sha256": "43168df377891f51dea5b6e4af9ea283898415b4a0803e5739c7f9895e350e66",
    "RHD_KR_formal_decision": "START_FIXED_PRINCIPAL_20_RUN_PACKAGE",
    "R7_selected_R6_minus_HISTORY_LL": -0.003774404430214484,
    "R7_evidence_strength": "WEAK_single_development_seed",
    "rationale": "OriginalKR principal comparisons passed bothseeds versusR0-R5. NecessaryR7 now complete; R6 has lower selected and24LL in3101, supporting testingcurrentfeedback without inventing a new threshold. Fournewseeds can falsify thisdevelopmentalpattern. NoMLformal.",
    "LPS_formal_decision": "NO_START",
    "LPS_reason": "Globalprincipal failsFEATURE; MLrestrictedfailsFEATURE; KRrestrictedfailsL2secondseed. P2variants neverreplaceL4 or complete missingsecondseed.",
    "P2_mechanism_readout": "CAL and OWNER selected AUC closely match L4. CAL LL is lower on both datasets; OWNER LL is lower on KR and near L4 on ML. GEOM beats L4 ML selected AUC. Cross-target learned formation necessity is not established; n1 attribution is WEAK.",
    "STATE": "NO_TRIGGER",
    "C14": "NO_TRIGGER",
    "PAIR": "NO_TRIGGER_AUC_BRANCH_ALREADY_ACTIVE",
    "no_additional_training_beyond_frozen_scope": true
  },
  "LPS_formal": "NO_START_FIXED_RULES",
  "STATE": "NO_TRIGGER",
  "C14": "NO_TRIGGER",
  "PAIR": "NO_TRIGGER",
  "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
  "protocol_manifest_sha256": "c9f2c1dcef104b3ab379e556e71155ce4c5540a0782881d021b6320a4d2d9844",
  "exact_constants": {
    "runtime": {
      "python": "3.13.5",
      "torch": "2.9.0+cu126",
      "cuda": "12.6",
      "cuda_available": true
    },
    "batch_size": 40960,
    "drop_last": false,
    "dtype": "float32",
    "AMP": false,
    "TF32": false,
    "adam": {
      "lr": 0.001,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08,
      "coupled_weight_decay": 1e-05
    },
    "static": {
      "movielens": "TRANSFER_1_TO_2",
      "kuairand": "BASE"
    },
    "RHD": {
      "panel_per_domain": 1024,
      "half_life": 8,
      "history_events": 32,
      "se_threshold": 1.0,
      "shared_private": "shared C8; private natural",
      "first_PB_coordinate": 10.0,
      "warmup_coordinates": [
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0
      ]
    },
    "rng": {
      "native": "seed",
      "main": "seed+11000",
      "bank": "seed+50000",
      "support": "seed+61000",
      "controller": "seed+71000",
      "projection": "seed+81000",
      "coverage": "seed+71000+300000000+1000003*global_decade",
      "meta": "seed+71000+100000000+1000003*refresh+9176*step+1009*domain+role_offset"
    },
    "gpu": {
      "gpu0_max_experiment_processes": 2,
      "gpu1_max_total_compute_processes": 1,
      "gpu1_external_occupied": true
    }
  },
  "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
  "data_identity_rule": "Exact saved arrays, fit-only encoder and roleIDs from unchanged protocol; no resampling/augmentation or role regeneration.",
  "parent_rule": "Prospective immutable source/data/seed/kind/boundary contracts; each actual parent SHA/full-state certificate resolved atomically only after dependency exists, no invented future digest, no config rewriting.",
  "fresh_STGU_rule": "Fresh native model/Adam/controller/rowplan seed epoch0; original audited STGU step object; full50 and LL selection; no3101 state import.",
  "warmup_rule": "Four fresh native10 shared physically; native BASE/PB standalone removes research-only warm probes, HISTORY independently charges shared8H initialization+firstblock, physical historysuffix237H.",
  "finite_scope": "Only exact20logical+4warm; no P2 winner replacement, no ML formal, no LPS formal, no negative-result tuning.",
  "seed_availability_audit_sha256": "991a281e0cc17501e0c8f8eb10ee87acf7e778551ca1384765856b3da6707423",
  "BEST_SIMPLE_lock_sha256": "27bd894c1ec73f25d6b508df40f1da11d216dbc222c296e7d99aae673bac37f6",
  "P2_complete_decision_sha256": "746dd07ea1b70277f719ab9af0bc22392b408de5a95400bf1e869312b53a2f91",
  "conditional_review_sha256": "84fc2c088f87ccf397c30aa78614e9a22ef3eaa8c70efad3a9d0f07a73217ddd",
  "budget": {
    "proposal": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/BUDGET_PROPOSAL_FORMAL_INITIAL.json",
    "proposal_sha256": "56ecf2d70c4665d2df5a612dbd1d92f9597f0b518c577c9a140b398f63bdf8d8",
    "independent_review_sha256": "2431610e7732016fed7948338fcf043238b0830ab5df6bbc7a3fca92f965d9eb",
    "reserved_seconds": 26591.01690750327,
    "scheduling_margin": 1.35,
    "not_timeout": true,
    "refinement_rule": "Separate retained-work measurement proposal; never silently alter methodlock or lower frozen reservation. Explicit positive revision if required."
  }
}
```

## FORMAL_NEW_SEED_AVAILABILITY_AUDIT.json

服务器来源 `artifacts/mdmt_next_v3/v3_0/FORMAL_NEW_SEED_AVAILABILITY_AUDIT.json`；SHA256 `991a281e0cc17501e0c8f8eb10ee87acf7e778551ca1384765856b3da6707423`。

```json
{
  "status": "NO_PRIOR_EXECUTION_FOUND_IN_AUDITED_V2_V3_REGISTRIES",
  "audit_utc": "2026-10-25T10:00:59.136222+00:00",
  "seeds": [
    4101,
    4102,
    4103,
    4104
  ],
  "metadata_files_checked": 320,
  "prior_matching_config_paths": [],
  "scope": "Accessible audited v2/v3 configs/resolved run identities; does not claim global never-use or new data independence.",
  "quality_metrics_read": false
}
```

## BEST_SIMPLE_LOCK.json

服务器来源 `artifacts/mdmt_next_v3/v3_0/BEST_SIMPLE_LOCK.json`；SHA256 `27bd894c1ec73f25d6b508df40f1da11d216dbc222c296e7d99aae673bac37f6`。

```json
{
  "locked_utc": "2026-10-25T09:40:20.189714+00:00",
  "dataset": "kuairand",
  "locked_method": "R0_BASE",
  "rule": "Mean two-seed validation-selected macroLL acrossR0-R4, lower frozen standalone cost estimate ties, then methodID. Q/test notused toselectBEST_SIMPLE.",
  "candidate_table": [
    {
      "method": "R0_BASE",
      "seed_validation_selected_LL": {
        "3101": 0.4349988674270548,
        "3102": 0.4319833433126837
      },
      "mean_validation_selected_LL": 0.43349110536986923,
      "mean_frozen_standalone_upper_seconds": 115.72529913371545,
      "selected_epochs": {
        "3101": 14,
        "3102": 17
      }
    },
    {
      "method": "R1_STATIC",
      "seed_validation_selected_LL": {
        "3101": 0.4349988674270548,
        "3102": 0.4319833433126837
      },
      "mean_validation_selected_LL": 0.43349110536986923,
      "mean_frozen_standalone_upper_seconds": 130.81740739318775,
      "selected_epochs": {
        "3101": 14,
        "3102": 17
      }
    },
    {
      "method": "R2_FREEZE",
      "seed_validation_selected_LL": {
        "3101": 0.4345972616341418,
        "3102": 0.4324847171488926
      },
      "mean_validation_selected_LL": 0.4335409893915172,
      "mean_frozen_standalone_upper_seconds": 137.84329119784525,
      "selected_epochs": {
        "3101": 14,
        "3102": 14
      }
    },
    {
      "method": "R3_CUM",
      "seed_validation_selected_LL": {
        "3101": 0.4345972616341418,
        "3102": 0.4324847171488926
      },
      "mean_validation_selected_LL": 0.4335409893915172,
      "mean_frozen_standalone_upper_seconds": 178.7189012702147,
      "selected_epochs": {
        "3101": 14,
        "3102": 14
      }
    },
    {
      "method": "R4_EMA",
      "seed_validation_selected_LL": {
        "3101": 0.4345972616341418,
        "3102": 0.4324847171488926
      },
      "mean_validation_selected_LL": 0.4335409893915172,
      "mean_frozen_standalone_upper_seconds": 183.0632836498553,
      "selected_epochs": {
        "3101": 14,
        "3102": 14
      }
    }
  ],
  "BASE_STATIC_identity_proof": [
    {
      "seed": 3101,
      "checkpoint": "primary_best",
      "model_equal": true,
      "optimizers_equal": true,
      "gradients_equal": true,
      "rng_equal": true
    },
    {
      "seed": 3101,
      "checkpoint": "epoch24",
      "model_equal": true,
      "optimizers_equal": true,
      "gradients_equal": true,
      "rng_equal": true
    },
    {
      "seed": 3102,
      "checkpoint": "primary_best",
      "model_equal": true,
      "optimizers_equal": true,
      "gradients_equal": true,
      "rng_equal": true
    },
    {
      "seed": 3102,
      "checkpoint": "epoch24",
      "model_equal": true,
      "optimizers_equal": true,
      "gradients_equal": true,
      "rng_equal": true
    }
  ],
  "validation_table_sha256": "537f72e97c3d3ebe66ef2689e0f01e0a1bf32f4ac4e28ffb1979f84fcf103e26",
  "cost_proposal_sha256": "218b3513ceb8378b9b86469be58f10472afd49015765631defbbbff54843e8f9",
  "formal_seeds": [
    4101,
    4102,
    4103,
    4104
  ]
}
```

## P2_COMPLETE_ANALYSIS_AND_FORMAL_DECISION.json

服务器来源 `artifacts/mdmt_next_v3/v3_0/P2_COMPLETE_ANALYSIS_AND_FORMAL_DECISION.json`；SHA256 `746dd07ea1b70277f719ab9af0bc22392b408de5a95400bf1e869312b53a2f91`。

```json
{
  "decision_utc": "2026-10-25T10:11:07.794363+00:00",
  "status": "ANALYZED_COMPLETE_P2",
  "complete_core_aux_p2_logical_trajectories": 69,
  "aggregation_receipt_sha256": "43168df377891f51dea5b6e4af9ea283898415b4a0803e5739c7f9895e350e66",
  "RHD_KR_formal_decision": "START_FIXED_PRINCIPAL_20_RUN_PACKAGE",
  "R7_selected_R6_minus_HISTORY_LL": -0.003774404430214484,
  "R7_evidence_strength": "WEAK_single_development_seed",
  "rationale": "OriginalKR principal comparisons passed bothseeds versusR0-R5. NecessaryR7 now complete; R6 has lower selected and24LL in3101, supporting testingcurrentfeedback without inventing a new threshold. Fournewseeds can falsify thisdevelopmentalpattern. NoMLformal.",
  "LPS_formal_decision": "NO_START",
  "LPS_reason": "Globalprincipal failsFEATURE; MLrestrictedfailsFEATURE; KRrestrictedfailsL2secondseed. P2variants neverreplaceL4 or complete missingsecondseed.",
  "P2_mechanism_readout": "CAL and OWNER selected AUC closely match L4. CAL LL is lower on both datasets; OWNER LL is lower on KR and near L4 on ML. GEOM beats L4 ML selected AUC. Cross-target learned formation necessity is not established; n1 attribution is WEAK.",
  "STATE": "NO_TRIGGER",
  "C14": "NO_TRIGGER",
  "PAIR": "NO_TRIGGER_AUC_BRANCH_ALREADY_ACTIVE",
  "no_additional_training_beyond_frozen_scope": true
}
```

## 精确源文件与数据/参数协议

`protocol_manifest.json` SHA256 `c9f2c1dcef104b3ab379e556e71155ce4c5540a0782881d021b6320a4d2d9844`。

```json
{
  "version": "MDMT-NEXT-v3.0",
  "source": {
    "files": {
      "experiments/data_generate/MDMTRec/basic/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/data_generate/MDMTRec/basic/activation.py": "152c5909e966055f83b7ec49de7634c853fbae8b029fbd37759371c14c921046",
      "experiments/data_generate/MDMTRec/basic/callback.py": "67ad8222b3c5d9ac720524d15276dcfbef400c7f2817d60b4ef2edc73bec3181",
      "experiments/data_generate/MDMTRec/basic/features.py": "116025a45517021bfe86447b8229e59cf04feb665549544ce30173d04421a2b0",
      "experiments/data_generate/MDMTRec/basic/initializers.py": "a9a6a2923eebeca7f541ffaddbccf2671a42e7f9b6a1ec8d38ba2872c53f6196",
      "experiments/data_generate/MDMTRec/basic/layers.py": "f1f20ef6f8d5cb39e125b74b25ea4cf27c964ca00c7f529587b0900d18934599",
      "experiments/data_generate/MDMTRec/models/multi_domain/fixed.py": "43d7591f28081d8e3ea6c247a877695f92232be87354d95deecbed9fe3e754be",
      "experiments/mdmt_next_20261024/baselines/__init__.py": "de2acfe168ac9eebb647ac08c69eebd165fac1c90a0610621919060522914b8e",
      "experiments/mdmt_next_20261024/baselines/autoifs.py": "29ac5baea08047f794fcceef7b98dc8997fc9c7b484827ae2439cafd37b8531d",
      "experiments/mdmt_next_20261024/baselines/common.py": "95da7a71a6837330f53e84c35734052878321a1a0f640945df6095f460746c37",
      "experiments/mdmt_next_20261024/baselines/gradcraft.py": "81371e3fec4911c02b3f5c285ab4c42b8df33837df9dfedfa42abce24564d6ee",
      "experiments/mdmt_next_20261024/baselines/official_gradcraft.py": "d9fa6d0eb421d2b8b627cbbf99d2986ba8a8b2c0037e459a0254f735619ab084",
      "experiments/mdmt_next_20261024/baselines/sources/AutoIFS_autoifs_trainer.py": "c63c257a8b6f47dc3e9c2b389d7136252aad63420809f414e298d48c9ba2270c",
      "experiments/mdmt_next_20261024/baselines/sources/AutoIFS_modules_autoifs.py": "c3b3b4446b5414166a106228797a189ee69e309a46549a1a03f942ac2ee0d933",
      "experiments/mdmt_next_20261024/baselines/sources/AutoIFS_modules_layers.py": "2ed020c04c0247d507cb95a829d0543895101ec484fae41b03d1d3b196582c72",
      "experiments/mdmt_next_20261024/baselines/sources/GradCraft_main.py": "e9d77d0fd685fd74287d475cacb3f56bdce0f5596e094e0b0f05be28d5f534b1",
      "experiments/mdmt_next_20261024/baselines/sources/GradCraft_models_basemodel.py": "0cdeb18292d82249e119a8433551a887f918a021bbafd8cc2acfc3743a011878",
      "experiments/mdmt_next_20261024/baselines/sources/GradCraft_solvers_global_pc_grad.py": "09483cb6e26ffa8a8885e03529dece1ccf8d22d7bf2d275c3a13791e4b5d06fb",
      "experiments/mdmt_next_20261024/baselines/sources/GradCraft_solvers_global_pc_grad_magnitude_positive.py": "d9fa6d0eb421d2b8b627cbbf99d2986ba8a8b2c0037e459a0254f735619ab084",
      "experiments/mdmt_next_20261024/baselines/sources/STEM_fuxictr_pytorch_layers_blocks_mlp_block.py": "c0447a79f6d147857861d3dfb37ea7ed4d82a12d53c1735e8f6a3f908e3c1a04",
      "experiments/mdmt_next_20261024/baselines/sources/STEM_fuxictr_pytorch_layers_embeddings_feature_embedding.py": "8efb1387241ee509b5dba8415fd067f4ae6d9e884d723ee82e15965d4fded9e9",
      "experiments/mdmt_next_20261024/baselines/sources/STEM_models_STEM.py": "b290d868eb056f4589faa36ccd288d4b058c5beed270de7329cce25f9a3cb854",
      "experiments/mdmt_next_20261024/baselines/stem.py": "636b3f396f7eed603d7deeadb272b83fe1720900ff2740d657fefa216a721f55",
      "experiments/mdmt_next_20261024/data_roles.py": "f8cb866a711330af2caed6ed696a134c2e78dd964213a00e0c9152262d4ae363",
      "experiments/mdmt_next_v2_20261025/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/actions/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/actions/a51.py": "f722b2008ab13c79089eba57f15bde0d4a890a832c3083bedb6f1d378a20e081",
      "experiments/mdmt_next_v2_20261025/actions/c8.py": "aa46bf5ee3eb0363fb7b4bc46b1120d414120d17d46abedbb4cfe7bc0025ddd0",
      "experiments/mdmt_next_v2_20261025/analyze.py": "fb64aa15d518a68440fc2c01625e10f355920155f31a7dadd04d37a500782939",
      "experiments/mdmt_next_v2_20261025/artifacts.py": "7504c12a986e545a1759844e8a0df92f0f86965acdda6f86c0e4bb31c12db243",
      "experiments/mdmt_next_v2_20261025/baselines/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/baselines/runner.py": "2cab8071c89179425303dd2877a3b66ac4fd36407c8f33f20da9449f63c8fc32",
      "experiments/mdmt_next_v2_20261025/baselines/stgu.py": "514b41072862ebd576067e49c039a5fbbf47ad9ad72626ae361df0079e3ae6a8",
      "experiments/mdmt_next_v2_20261025/configs/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/costs.py": "d566980a55747eaf0b6809d7c90a01f9eef6d550b17785d190df773f7186ea3b",
      "experiments/mdmt_next_v2_20261025/data/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/data/plans.py": "596ee6d348b7fbbe274f63391130f52bb69944706edf3d9acb737a3e6ab0e3ee",
      "experiments/mdmt_next_v2_20261025/data/roles.py": "f0c9c7fe88d734512d956d74e1e3aa32dbafc3919576262a4932bc437e002bb4",
      "experiments/mdmt_next_v2_20261025/designs/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/diagnostics.py": "4426ef50d6f754258529ec959dc572c8d492ef19d08855c959eb86e5f3ad1d76",
      "experiments/mdmt_next_v2_20261025/evaluate.py": "d1da2b76e27b276d2a804c5cc2009ff1323542e4838b550bba1aeadc71273d36",
      "experiments/mdmt_next_v2_20261025/models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/models/anchor.py": "175a08c21217ed3d6aedcb503de8a786505c1ef324ff373c155c07d698f33136",
      "experiments/mdmt_next_v2_20261025/models/fixed1_adapter.py": "628e3966938cbc586612acb5fff534369f15142bc665e33741b67ee5c4d32596",
      "experiments/mdmt_next_v2_20261025/models/ownership.py": "c57ba16086e04c0e29e9e2458edf99dc115d2055118b188720e9fd873b2f8484",
      "experiments/mdmt_next_v2_20261025/models/source_bank.py": "a62be5eb18c4ec5aa00cc444ac59f7d364cdc183d407563dd8f66ba0dd28bd4b",
      "experiments/mdmt_next_v2_20261025/policies/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/policies/calibration.py": "49242caf5335f38932b0139af7138ae025e402dce47b508e7727b4cd61c01686",
      "experiments/mdmt_next_v2_20261025/policies/direct.py": "a2c993dff649c935417e967d1c019d46880cf9afdc6e4321eae63e3c1de9226e",
      "experiments/mdmt_next_v2_20261025/policies/history.py": "6b7341f6d05c141a577fcb549d37ae56aacc63cc10beb02fcebd07a7b491afb7",
      "experiments/mdmt_next_v2_20261025/policies/local_tree.py": "4ed5f693d8afeac1ec28bf15dfa6bbc534a7d9eaa992f816e2f53e21d1b2ed79",
      "experiments/mdmt_next_v2_20261025/policies/state_residual.py": "e550010d8c9a49bf5124c8ae46123d30f6eecc98c3c74ca50df721ef1db075da",
      "experiments/mdmt_next_v2_20261025/run.py": "985a4177fcfc89be5ba8864bc1a3659cf908369f7c25a92c0be31e66b3bd93ea",
      "experiments/mdmt_next_v2_20261025/schedule.py": "c1fb7c96f9505b3ae6886f5a702fdb2fef361ac47a9556fa74b7a3e8eabd5ddf",
      "experiments/mdmt_next_v2_20261025/smoke.py": "dabb526b565bf5b495e5c8c0afbbb947a29871f2e855677d30b4f5e8c077b29f",
      "experiments/mdmt_next_v2_20261025/train.py": "ef6c4d9c4e9b826af39829f974aa7ce8f31c72090053e5a68b46d922dfb2a359",
      "experiments/mdmt_next_v2_20261025/utility/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "experiments/mdmt_next_v2_20261025/utility/forward.py": "9ed6e2b99ae3f296dd8879a577aa7b3c1b2f6745ec09da12c4439f9ebd953854",
      "experiments/mdmt_next_v2_20261025/utility/state.py": "67a2e1648c426452c3f743b218f8f27480781c60eccc32c608ed31cfe6b4586c",
      "experiments/mdmt_next_v2_20261025/utility/update.py": "a2c4395ee3b8c0ee03e44dc2708ea38b740dde57e3070ae603a2832fd461f19c",
      "experiments/mdmt_next_v3_20261025/__init__.py": "7f3bb6c5e198850382a7da25d61f619a30d7591509522a523e9c82b93e789a8e",
      "experiments/mdmt_next_v3_20261025/baseline_bridge.py": "968efcd4e7fddfdf3c4cdd3cb8c215c708cc1a912756b725543ee1fefe2700d5",
      "experiments/mdmt_next_v3_20261025/cli.py": "52b1b240e534af22d066312c4fa43ea2949e7b77895032fbb649da8fe285e55f",
      "experiments/mdmt_next_v3_20261025/common.py": "2b2738148c11a477078365b2c5d3b57af294daa3bfd85804fc07355c30441ff4",
      "experiments/mdmt_next_v3_20261025/data.py": "d4b07a16a115082ce3521adb8d66687079623a4eb761e62a6afd77a5be5df2bb",
      "experiments/mdmt_next_v3_20261025/lps.py": "b8ad3a0c78a75fd76af5cd0ccaaaba0dafe675dea3a63fe6f3295b969f8cb5c3",
      "experiments/mdmt_next_v3_20261025/p0_cached.py": "e30e1f0ed854cc01ac6b58a4bd27614a52199ae2a0dde9a1917be242685f9039",
      "experiments/mdmt_next_v3_20261025/plan.py": "9db01e2241ffbf7cf2b003f55b60a111339c52edf72e6f9f4f7e44122d307341",
      "experiments/mdmt_next_v3_20261025/rhd.py": "ca058ae4cb930b18b420be00f4c09173fe8f6c11becf00c16bacbae494aee110",
      "experiments/mdmt_next_v3_20261025/semantics.py": "6eaac4d4482290615f435a25e67b0aac55ce8a0a7a14cf1051b64c6fb82a88a8",
      "experiments/mdmt_next_v3_20261025/semantics_lps.py": "bcbc3ee200af3d1d586d118071af7f29c7873dfea7a9817941850c228ab2dd00"
    },
    "source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "git_head": null,
    "reference_commit": "619371d9c4d57f6f0b3e8a71b686498f9e5f36d4"
  },
  "data": {
    "movielens": {
      "historical": {
        "schema_version": "mdmt-next-v1",
        "protocol_id": "natural_roworder_fit80_probe10_query10_v1",
        "dataset": "movielens",
        "raw_path": "/home/xzf/data/ml-1m/ml-1m.csv",
        "raw_sha256": "f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4",
        "split_seed": 20261024,
        "preprocess_fit_role": "fit",
        "features": {
          "dataset": "movielens",
          "columns": [
            "user_id",
            "movie_id",
            "gender",
            "age",
            "occupation",
            "zip",
            "cate_id",
            "domain_indicator"
          ],
          "dense_columns": [
            "age"
          ],
          "sparse_columns": [
            "user_id",
            "movie_id",
            "gender",
            "occupation",
            "zip",
            "cate_id",
            "domain_indicator"
          ],
          "vocab_sizes": {
            "user_id": 5401,
            "movie_id": 3632,
            "gender": 3,
            "occupation": 22,
            "zip": 3201,
            "cate_id": 19,
            "domain_indicator": 3
          },
          "embedding_dim": 16,
          "domain_column": "domain_indicator",
          "encoder_hash": "cab74dbb13090a9339ae74310e24310d0145512d000ffc31f917965dfd7176d3",
          "scaler_hash": "b73c4a08a40b2059f892c72edc6f659baf2acdd430b6ca1a50068bcd0dde1c1e",
          "oov_index": 0
        },
        "roles": {
          "fit": {
            "n": 640133,
            "row_id_hash": "d5823cb852e44ac098c296e2c91f324d2c7141d5094a78f3943bd2ee25289f4c",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 124759,
                "positive": 69963,
                "negative": 54796
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 254879,
                "positive": 143910,
                "negative": 110969
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 260495,
                "positive": 156789,
                "negative": 103706
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 124759,
                "positive": 29257,
                "negative": 95502
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 254879,
                "positive": 56318,
                "negative": 198561
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 260495,
                "positive": 62389,
                "negative": 198106
              }
            ]
          },
          "P_label": {
            "n": 48010,
            "row_id_hash": "f536489d99daa3f9ee10231d1bfdd8865e50329c7348535e969707b89034ec7b",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 9357,
                "positive": 5223,
                "negative": 4134
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 19116,
                "positive": 10666,
                "negative": 8450
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 19537,
                "positive": 11734,
                "negative": 7803
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 9357,
                "positive": 2154,
                "negative": 7203
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 19116,
                "positive": 4116,
                "negative": 15000
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 19537,
                "positive": 4674,
                "negative": 14863
              }
            ]
          },
          "P_cal": {
            "n": 16003,
            "row_id_hash": "b4156cd40c4afbfdff06afc431f85eeb77fa8f2cb3cbbf4f0ac4fa08d24d88ce",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 3119,
                "positive": 1761,
                "negative": 1358
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 6372,
                "positive": 3488,
                "negative": 2884
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 6512,
                "positive": 3951,
                "negative": 2561
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 3119,
                "positive": 763,
                "negative": 2356
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 6372,
                "positive": 1374,
                "negative": 4998
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 6512,
                "positive": 1533,
                "negative": 4979
              }
            ]
          },
          "P_update": {
            "n": 16004,
            "row_id_hash": "dc21ab5390072a97e06c4f9a9314ced72e4e5bd04d801e42c2480887923c305b",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 3119,
                "positive": 1753,
                "negative": 1366
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 6372,
                "positive": 3510,
                "negative": 2862
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 6513,
                "positive": 3890,
                "negative": 2623
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 3119,
                "positive": 732,
                "negative": 2387
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 6372,
                "positive": 1341,
                "negative": 5031
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 6513,
                "positive": 1587,
                "negative": 4926
              }
            ]
          },
          "Q_explore": {
            "n": 40008,
            "row_id_hash": "545c60a22ecf4e54613472180edc1c0853537f5f9d6f94d0b7cf4ac58ac5c19f",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 7797,
                "positive": 4434,
                "negative": 3363
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 15930,
                "positive": 8981,
                "negative": 6949
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 16281,
                "positive": 9868,
                "negative": 6413
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 7797,
                "positive": 1864,
                "negative": 5933
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 15930,
                "positive": 3565,
                "negative": 12365
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 16281,
                "positive": 3897,
                "negative": 12384
              }
            ]
          },
          "Q_confirm": {
            "n": 40009,
            "row_id_hash": "fca485d1fea52f4031548126d0099a131d2b44fcd505c1444b166c0bc5ab4294",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 7798,
                "positive": 4360,
                "negative": 3438
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 15930,
                "positive": 8900,
                "negative": 7030
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 16281,
                "positive": 9839,
                "negative": 6442
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 7798,
                "positive": 1845,
                "negative": 5953
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 15930,
                "positive": 3444,
                "negative": 12486
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 16281,
                "positive": 3927,
                "negative": 12354
              }
            ]
          },
          "validation": {
            "n": 100021,
            "row_id_hash": "3cf4c7b1d5dab7ef3ef8d6783faecde7f580337bcdbe4e9c3af37fe1813bbea4",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 27111,
                "positive": 14634,
                "negative": 12477
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 35523,
                "positive": 21027,
                "negative": 14496
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 37387,
                "positive": 22148,
                "negative": 15239
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 27111,
                "positive": 5781,
                "negative": 21330
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 35523,
                "positive": 8541,
                "negative": 26982
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 37387,
                "positive": 8889,
                "negative": 28498
              }
            ]
          },
          "test": {
            "n": 100021,
            "row_id_hash": "b9ff6de1e3a6f0f601d6843f1be66d40ee292681e3a91688c87ad3b818382ddf",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 27687,
                "positive": 14281,
                "negative": 13406
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 41434,
                "positive": 22072,
                "negative": 19362
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 30900,
                "positive": 18099,
                "negative": 12801
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 27687,
                "positive": 4964,
                "negative": 22723
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 41434,
                "positive": 7031,
                "negative": 34403
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 30900,
                "positive": 6324,
                "negative": 24576
              }
            ]
          }
        },
        "array_hashes": {
          "x": "cb9518c9ed67378f9f503e489c7c1910cac465a6279a1f4282023f0b69c0161e",
          "y": "86a23b5684381e656e43390515ad9a2fed43577c0c78f412ca3c96ed1f15fb8a"
        },
        "label_definitions": [
          "rating>3",
          "rating>4"
        ],
        "provenance": "natural source row order; no verified user-disjoint or temporal claim",
        "data_hash": "01d840f45b891c51f7ba33c087a51ee25d19cd6eeb11fcdf93dee1b13163e7ba"
      },
      "roles": {
        "fit": {
          "n": 640133,
          "hash": "d5823cb852e44ac098c296e2c91f324d2c7141d5094a78f3943bd2ee25289f4c"
        },
        "P_label": {
          "n": 48010,
          "hash": "f536489d99daa3f9ee10231d1bfdd8865e50329c7348535e969707b89034ec7b"
        },
        "P_cal": {
          "n": 16003,
          "hash": "b4156cd40c4afbfdff06afc431f85eeb77fa8f2cb3cbbf4f0ac4fa08d24d88ce"
        },
        "P_update": {
          "n": 16004,
          "hash": "dc21ab5390072a97e06c4f9a9314ced72e4e5bd04d801e42c2480887923c305b"
        },
        "Q_explore": {
          "n": 40008,
          "hash": "545c60a22ecf4e54613472180edc1c0853537f5f9d6f94d0b7cf4ac58ac5c19f"
        },
        "validation": {
          "n": 100021,
          "hash": "3cf4c7b1d5dab7ef3ef8d6783faecde7f580337bcdbe4e9c3af37fe1813bbea4"
        },
        "benchmark_test_exposed": {
          "n": 100021,
          "hash": "b9ff6de1e3a6f0f601d6843f1be66d40ee292681e3a91688c87ad3b818382ddf"
        },
        "Q_legacy_confirm_exposed": {
          "n": 40009,
          "hash": "fca485d1fea52f4031548126d0099a131d2b44fcd505c1444b166c0bc5ab4294"
        },
        "P_structure": {
          "n": 24004,
          "hash": "a04fa99676ed28f48066c4c0cdf4cb6773c266ebe0eff542e6e7bae3ca640e5c"
        },
        "P_value": {
          "n": 24006,
          "hash": "47395d052f4f872e2028552a384a5cd5672739b35ccbefdedec8ba9cda61d4a6"
        }
      },
      "blocks": {
        "P_cal": "757c6f6aca23eb1804d375ceed8d7bbb787a7fd1e6fb48e4cd1421e70f3eda89",
        "P_value": "ab86e7045ddce3d77bc43489bfc7144101397438dc55cdc7381a2a06b73081d8"
      },
      "exposure": "EXPOSED_BENCHMARK_REASSESSMENT",
      "unseen": "UNSEEN_CONFIRM_PENDING_DATA",
      "version": "MDMT-NEXT-v3.0",
      "historical_role_hash": "7347b948841c61e47c181b3b6d8997035edd8d31448f71051afcc74f1a075ea9",
      "v3_aliases": {
        "P_form": {
          "alias_of": "P_structure",
          "n": 24004,
          "hash": "a04fa99676ed28f48066c4c0cdf4cb6773c266ebe0eff542e6e7bae3ca640e5c"
        },
        "P_read": {
          "alias_of": "P_value",
          "n": 24006,
          "hash": "47395d052f4f872e2028552a384a5cd5672739b35ccbefdedec8ba9cda61d4a6"
        }
      }
    },
    "kuairand": {
      "historical": {
        "schema_version": "mdmt-next-v1",
        "protocol_id": "natural_roworder_fit80_probe10_query10_v1",
        "dataset": "kuairand",
        "raw_path": "/home/xzf/data/kuairand/processed/kuai_rand.csv",
        "raw_sha256": "a1872911e49cb9324a89faf84df1dbed2077357496edd760ba0ab7b8f4fc681f",
        "split_seed": 20261024,
        "preprocess_fit_role": "fit",
        "features": {
          "dataset": "kuairand",
          "columns": [
            "user_active_degree",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "onehot_feat0",
            "onehot_feat1",
            "onehot_feat2",
            "onehot_feat3",
            "onehot_feat4",
            "onehot_feat5",
            "onehot_feat6",
            "onehot_feat7",
            "onehot_feat8",
            "onehot_feat9",
            "onehot_feat10",
            "onehot_feat11",
            "onehot_feat12",
            "onehot_feat13",
            "onehot_feat14",
            "onehot_feat15",
            "onehot_feat16",
            "onehot_feat17",
            "video_type",
            "upload_dt",
            "upload_type",
            "server_width",
            "server_height",
            "music_type",
            "tag",
            "counts",
            "domain_indicator"
          ],
          "dense_columns": [],
          "sparse_columns": [
            "user_active_degree",
            "is_live_streamer",
            "is_video_author",
            "follow_user_num_range",
            "fans_user_num_range",
            "friend_user_num_range",
            "register_days_range",
            "onehot_feat0",
            "onehot_feat1",
            "onehot_feat2",
            "onehot_feat3",
            "onehot_feat4",
            "onehot_feat5",
            "onehot_feat6",
            "onehot_feat7",
            "onehot_feat8",
            "onehot_feat9",
            "onehot_feat10",
            "onehot_feat11",
            "onehot_feat12",
            "onehot_feat13",
            "onehot_feat14",
            "onehot_feat15",
            "onehot_feat16",
            "onehot_feat17",
            "video_type",
            "upload_dt",
            "upload_type",
            "server_width",
            "server_height",
            "music_type",
            "tag",
            "counts",
            "domain_indicator"
          ],
          "vocab_sizes": {
            "user_active_degree": 10,
            "is_live_streamer": 3,
            "is_video_author": 3,
            "follow_user_num_range": 9,
            "fans_user_num_range": 10,
            "friend_user_num_range": 8,
            "register_days_range": 9,
            "onehot_feat0": 3,
            "onehot_feat1": 8,
            "onehot_feat2": 50,
            "onehot_feat3": 1461,
            "onehot_feat4": 17,
            "onehot_feat5": 35,
            "onehot_feat6": 4,
            "onehot_feat7": 119,
            "onehot_feat8": 455,
            "onehot_feat9": 8,
            "onehot_feat10": 6,
            "onehot_feat11": 6,
            "onehot_feat12": 4,
            "onehot_feat13": 4,
            "onehot_feat14": 4,
            "onehot_feat15": 4,
            "onehot_feat16": 4,
            "onehot_feat17": 4,
            "video_type": 4,
            "upload_dt": 4,
            "upload_type": 15,
            "server_width": 157,
            "server_height": 121,
            "music_type": 7,
            "tag": 112,
            "counts": 126,
            "domain_indicator": 3
          },
          "embedding_dim": 16,
          "domain_column": "domain_indicator",
          "encoder_hash": "58ff742e14eabc33ed001ad6ec2e4f74c6be375ab4b2a68028277b7b9bb3e33a",
          "scaler_hash": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
          "oov_index": 0
        },
        "roles": {
          "fit": {
            "n": 1604904,
            "row_id_hash": "1ad3f0d85d1ec28c3e55d7110d7f6682b3a2891e39ec70d75e2c3e1bb6bc419b",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 138225,
                "positive": 12544,
                "negative": 125681
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 1395118,
                "positive": 527229,
                "negative": 867889
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 71561,
                "positive": 44177,
                "negative": 27384
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 138225,
                "positive": 5687,
                "negative": 132538
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 1395118,
                "positive": 357295,
                "negative": 1037823
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 71561,
                "positive": 34681,
                "negative": 36880
              }
            ]
          },
          "P_label": {
            "n": 120367,
            "row_id_hash": "fccbf33b3374fa06b4f8bb2fc4eb18065dc97bbf4d093b468557ede8f54fa990",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 10366,
                "positive": 912,
                "negative": 9454
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 104634,
                "positive": 39679,
                "negative": 64955
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 5367,
                "positive": 3316,
                "negative": 2051
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 10366,
                "positive": 402,
                "negative": 9964
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 104634,
                "positive": 26913,
                "negative": 77721
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 5367,
                "positive": 2613,
                "negative": 2754
              }
            ]
          },
          "P_cal": {
            "n": 40123,
            "row_id_hash": "ca1500e61d337c3c8d180506fdb34aff86f1eaf63a65d30b27bb9560fb28cbc6",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 3456,
                "positive": 334,
                "negative": 3122
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 34878,
                "positive": 13037,
                "negative": 21841
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 1789,
                "positive": 1084,
                "negative": 705
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 3456,
                "positive": 145,
                "negative": 3311
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 34878,
                "positive": 8820,
                "negative": 26058
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 1789,
                "positive": 868,
                "negative": 921
              }
            ]
          },
          "P_update": {
            "n": 40123,
            "row_id_hash": "013ea3d38e7cf8a21cabd77a2593f5392a3994548fce6e6e67a7e7a1174e10b8",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 3456,
                "positive": 295,
                "negative": 3161
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 34878,
                "positive": 13216,
                "negative": 21662
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 1789,
                "positive": 1114,
                "negative": 675
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 3456,
                "positive": 129,
                "negative": 3327
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 34878,
                "positive": 8984,
                "negative": 25894
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 1789,
                "positive": 878,
                "negative": 911
              }
            ]
          },
          "Q_explore": {
            "n": 100307,
            "row_id_hash": "7e0f10824bb5966aed875738f4e23c2d9dc79c97e9782bde1c220628fb40eb0e",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 8639,
                "positive": 817,
                "negative": 7822
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 87195,
                "positive": 32928,
                "negative": 54267
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 4473,
                "positive": 2758,
                "negative": 1715
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 8639,
                "positive": 385,
                "negative": 8254
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 87195,
                "positive": 22257,
                "negative": 64938
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 4473,
                "positive": 2149,
                "negative": 2324
              }
            ]
          },
          "Q_confirm": {
            "n": 100308,
            "row_id_hash": "a6b322df9194fc62c378eec86076a5f2cbdca2a1db1cc747bbce2773f19959fc",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 8640,
                "positive": 802,
                "negative": 7838
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 87195,
                "positive": 32812,
                "negative": 54383
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 4473,
                "positive": 2731,
                "negative": 1742
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 8640,
                "positive": 373,
                "negative": 8267
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 87195,
                "positive": 22314,
                "negative": 64881
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 4473,
                "positive": 2178,
                "negative": 2295
              }
            ]
          },
          "validation": {
            "n": 250767,
            "row_id_hash": "02b8690eb5184db4e4cd60c41f6c5156df4c068b0dbdd03cf368d73f139bc8bd",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 2692,
                "positive": 245,
                "negative": 2447
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 246232,
                "positive": 51644,
                "negative": 194588
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 1843,
                "positive": 1063,
                "negative": 780
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 2692,
                "positive": 101,
                "negative": 2591
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 246232,
                "positive": 27460,
                "negative": 218772
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 1843,
                "positive": 829,
                "negative": 1014
              }
            ]
          },
          "test": {
            "n": 250767,
            "row_id_hash": "05985c4068faf12439837e3b4400ad7715049d5f074a721b380af414f1cfa58e",
            "cells": [
              {
                "cell": 0,
                "domain": 0,
                "task": 0,
                "n": 2613,
                "positive": 228,
                "negative": 2385
              },
              {
                "cell": 1,
                "domain": 1,
                "task": 0,
                "n": 246284,
                "positive": 51847,
                "negative": 194437
              },
              {
                "cell": 2,
                "domain": 2,
                "task": 0,
                "n": 1870,
                "positive": 1100,
                "negative": 770
              },
              {
                "cell": 3,
                "domain": 0,
                "task": 1,
                "n": 2613,
                "positive": 120,
                "negative": 2493
              },
              {
                "cell": 4,
                "domain": 1,
                "task": 1,
                "n": 246284,
                "positive": 27415,
                "negative": 218869
              },
              {
                "cell": 5,
                "domain": 2,
                "task": 1,
                "n": 1870,
                "positive": 827,
                "negative": 1043
              }
            ]
          }
        },
        "array_hashes": {
          "x": "5f9bd9e6d511713e63627dd353e3a703431ef85cd2417e6dcadb80f967919626",
          "y": "f53a00ec4f762d6724f3e7e92a0d78a11ab1d9c652f3aa05ad7c6682ed4ab08f"
        },
        "label_definitions": [
          "stored_is_click",
          "stored_long_view"
        ],
        "provenance": "natural source row order; no verified user-disjoint or temporal claim",
        "data_hash": "0edcd0ab32e03c2dfe7ab109be841dfba365fc1e570ac3a7ef563bd4e0db6dce"
      },
      "roles": {
        "fit": {
          "n": 1604904,
          "hash": "1ad3f0d85d1ec28c3e55d7110d7f6682b3a2891e39ec70d75e2c3e1bb6bc419b"
        },
        "P_label": {
          "n": 120367,
          "hash": "fccbf33b3374fa06b4f8bb2fc4eb18065dc97bbf4d093b468557ede8f54fa990"
        },
        "P_cal": {
          "n": 40123,
          "hash": "ca1500e61d337c3c8d180506fdb34aff86f1eaf63a65d30b27bb9560fb28cbc6"
        },
        "P_update": {
          "n": 40123,
          "hash": "013ea3d38e7cf8a21cabd77a2593f5392a3994548fce6e6e67a7e7a1174e10b8"
        },
        "Q_explore": {
          "n": 100307,
          "hash": "7e0f10824bb5966aed875738f4e23c2d9dc79c97e9782bde1c220628fb40eb0e"
        },
        "validation": {
          "n": 250767,
          "hash": "02b8690eb5184db4e4cd60c41f6c5156df4c068b0dbdd03cf368d73f139bc8bd"
        },
        "benchmark_test_exposed": {
          "n": 250767,
          "hash": "05985c4068faf12439837e3b4400ad7715049d5f074a721b380af414f1cfa58e"
        },
        "Q_legacy_confirm_exposed": {
          "n": 100308,
          "hash": "a6b322df9194fc62c378eec86076a5f2cbdca2a1db1cc747bbce2773f19959fc"
        },
        "P_structure": {
          "n": 60183,
          "hash": "1c40c654c13c84ae3f6484fa96caf921f14666b3692e7d8eba85efce78910385"
        },
        "P_value": {
          "n": 60184,
          "hash": "34f009c4c14b14fcf129519a276cf416924e638a23f25784ccb0c65429d4941f"
        }
      },
      "blocks": {
        "P_cal": "dd812a2e2f311fa50ae6da217e4771cf3b7764f9d481f5134bbef96ac8f7e7e6",
        "P_value": "0ec8a36c5ba3cb614b858ca9e059e02f0b99866d605667f414a3556d4bf987ea"
      },
      "exposure": "EXPOSED_BENCHMARK_REASSESSMENT",
      "unseen": "UNSEEN_CONFIRM_PENDING_DATA",
      "version": "MDMT-NEXT-v3.0",
      "historical_role_hash": "a252db1f2f9e576bdf04a4415cd384579f4fe0c98145a0f811550ae6792c1757",
      "v3_aliases": {
        "P_form": {
          "alias_of": "P_structure",
          "n": 60183,
          "hash": "1c40c654c13c84ae3f6484fa96caf921f14666b3692e7d8eba85efce78910385"
        },
        "P_read": {
          "alias_of": "P_value",
          "n": 60184,
          "hash": "34f009c4c14b14fcf129519a276cf416924e638a23f25784ccb0c65429d4941f"
        }
      }
    }
  },
  "authority": {
    "research_plans/26-10-25_v2/01_最新实验机制审查与算法自批判.md": "fb0da140ae0397585729be0dab5b37e1d1ecc7afbc34e4deeef40cc3aa1b071c",
    "research_plans/26-10-25_v2/04_Codex完整执行提示词.md": "9c6dbf8ba3d38f8b8bc6e885e22563868d9c74984964759ed7d04e38531d9878",
    "research_plans/26-10-25_v2/02_算法重新设计与潜在研究方向.md": "6888429694de15fd447299f6408a5026ac9fa2f842ef0eb28f2d9d2d60b3b9d7",
    "research_plans/26-10-25_v2/03_下一阶段实验矩阵与决策规范.md": "a4775be3264919eedcbe4b440442e1fef9c9fea6241929dcc513688a2d183cf6"
  },
  "design_sha256": "f4ce85f0b6cba652449a0b3339fbc0d2ef1913efbadd9c5e7183691e2be6adcf",
  "runtime": {
    "python": "3.13.5",
    "torch": "2.9.0+cu126",
    "cuda": "12.6",
    "cuda_available": true
  },
  "batch_size": 40960,
  "drop_last": false,
  "dtype": "float32",
  "AMP": false,
  "TF32": false,
  "adam": {
    "lr": 0.001,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "coupled_weight_decay": 1e-05
  },
  "seeds": [
    3101,
    3102
  ],
  "formal_seeds": [
    4101,
    4102,
    4103,
    4104
  ],
  "epochs": 24,
  "formal_epochs": 50,
  "selection": {
    "eligible_epoch": 12,
    "RHD": "validation_macro_logloss_min",
    "LPS": "validation_macro_auc_max",
    "ties": "earliest"
  },
  "static": {
    "movielens": "TRANSFER_1_TO_2",
    "kuairand": "BASE"
  },
  "RHD": {
    "panel_per_domain": 1024,
    "half_life": 8,
    "history_events": 32,
    "se_threshold": 1.0,
    "shared_private": "shared C8; private natural",
    "first_PB_coordinate": 10.0,
    "warmup_coordinates": [
      4.5,
      5.0,
      5.5,
      6.0,
      6.5,
      7.0,
      7.5,
      8.0,
      8.5,
      9.0,
      9.5,
      10.0
    ]
  },
  "LPS": {
    "bank_parameters": 15360,
    "reader_parameters": 1152,
    "gate_parameters": 4176,
    "allocator_parameters": 1992,
    "aux": 0.0,
    "coverage": "one whole logical batch per fixed consecutive ten-batch group starting epoch6",
    "meta_support": 512,
    "meta_query": 512,
    "meta_initial": 16,
    "meta_refresh": 8,
    "read_batch": 512,
    "read_initial_passes": 10,
    "read_refresh_passes": 2,
    "meta_backgrounds": [
      0.75,
      0.25
    ],
    "coverage_coefficient": 0.5
  },
  "rng": {
    "native": "seed",
    "main": "seed+11000",
    "bank": "seed+50000",
    "support": "seed+61000",
    "controller": "seed+71000",
    "projection": "seed+81000",
    "coverage": "seed+71000+300000000+1000003*global_decade",
    "meta": "seed+71000+100000000+1000003*refresh+9176*step+1009*domain+role_offset"
  },
  "gpu": {
    "gpu0_max_experiment_processes": 2,
    "gpu1_max_total_compute_processes": 1,
    "gpu1_external_occupied": true
  },
  "exposed_roles": [
    "Q_legacy_confirm_exposed",
    "benchmark_test_exposed"
  ],
  "development_role": "Q_explore",
  "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c"
}
```

## 正式队列及实际父证书

### FORMAL-WARMUP_kuairand_WARM_RHD_4101

```json
{
  "configuration": {
    "run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4101",
    "kind": "WARM_RHD",
    "dataset": "kuairand",
    "seed": 4101,
    "epochs": 10,
    "stage": "formal-warmup",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": []
  },
  "configuration_file_sha256": "55add18823ac6316afdb7aec0de6b364c8c025ed8b6fbf3308e203053baaaab1",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL_kuairand_R6_PB_HB_4101

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R6_PB_HB_4101",
    "kind": "R6_PB_HB",
    "dataset": "kuairand",
    "seed": 4101,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4101"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4101/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R6_PB_HB_4101.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4101,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "8d595a56e3426803c5121f6ef4a0131f2ba6ae1ee6b0ba6f96ddc047cb48d431",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4101/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "22886a1159eddadfe6fda50f44882738581a06d79bd00dfa7d4336b371707154",
    "parent_full_state_digest": "e8ec71875bebb95bf8787f277532684a5be05aef1de7ce0bb888e7376749b31e",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4101",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4101,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "ee9fdf9ef2ca29f77d638d011b1d5c0d63196cc7977cf5a4a26e2414218d9ef2"
}
```

### FORMAL_kuairand_R0_BASE_4101

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R0_BASE_4101",
    "kind": "R0_BASE",
    "dataset": "kuairand",
    "seed": 4101,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4101"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4101/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R0_BASE_4101.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4101,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "1030a47234895de6e8a840999b22e21f0cca1d868a6ed0bd561d4374bdee793b",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4101/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "22886a1159eddadfe6fda50f44882738581a06d79bd00dfa7d4336b371707154",
    "parent_full_state_digest": "e8ec71875bebb95bf8787f277532684a5be05aef1de7ce0bb888e7376749b31e",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4101",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4101,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "ee9fdf9ef2ca29f77d638d011b1d5c0d63196cc7977cf5a4a26e2414218d9ef2"
}
```

### FORMAL_kuairand_R5_PB_H1_4101

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R5_PB_H1_4101",
    "kind": "R5_PB_H1",
    "dataset": "kuairand",
    "seed": 4101,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4101"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4101/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R5_PB_H1_4101.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4101,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "c3683f3a0d3f6ae0f6d68984c495177a2c2456029ef9abb524765178ae524d7c",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4101/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "22886a1159eddadfe6fda50f44882738581a06d79bd00dfa7d4336b371707154",
    "parent_full_state_digest": "e8ec71875bebb95bf8787f277532684a5be05aef1de7ce0bb888e7376749b31e",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4101",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4101,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "ee9fdf9ef2ca29f77d638d011b1d5c0d63196cc7977cf5a4a26e2414218d9ef2"
}
```

### FORMAL_kuairand_R7_PB_HISTORY_4101

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R7_PB_HISTORY_4101",
    "kind": "R7_PB_HISTORY",
    "dataset": "kuairand",
    "seed": 4101,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL_kuairand_R6_PB_HB_4101"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4101/first_block.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R7_PB_HISTORY_4101.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4101,
      "dataset": "kuairand",
      "kind": "R6_PB_HB",
      "boundary": "first_block_10.5",
      "epoch": 10,
      "position": 20
    }
  },
  "configuration_file_sha256": "79cf6c8215b470b11316d7ff8a94aa792af95523cbda0d7229d278366313fa8c",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4101/first_block.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "3c1e6921b09aae3df639d5c149b3f263d2dfdb5e7fdaec258d43abaee303bfcb",
    "parent_full_state_digest": "d2762c1fd4ee8d6fe3f7dda7a4fd6ee1e0bf71dc5b09b7e017560bc9b2e6a532",
    "parent_kind": "R6_PB_HB",
    "parent_boundary": "first_block_10.5",
    "parent_run_id": "FORMAL_kuairand_R6_PB_HB_4101",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4101,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "baee9ecf95563e70b597c1ff285d1d784569439041bee4c8a2452798e28dac18"
}
```

### FORMAL_kuairand_STGU_DOMAIN_ADAM_4101

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_STGU_DOMAIN_ADAM_4101",
    "kind": "STGU_DOMAIN_ADAM",
    "dataset": "kuairand",
    "seed": 4101,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data"
  },
  "configuration_file_sha256": "cd0982f2b972da9fce2d7fe19652d707dd280cd5b47642b06eef88dd19444d48",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL-WARMUP_kuairand_WARM_RHD_4102

```json
{
  "configuration": {
    "run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4102",
    "kind": "WARM_RHD",
    "dataset": "kuairand",
    "seed": 4102,
    "epochs": 10,
    "stage": "formal-warmup",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": []
  },
  "configuration_file_sha256": "06aac83b8bcded5d71066d62ae251f108203aa6a3547d964b270fcef0f247a0c",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL_kuairand_R6_PB_HB_4102

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R6_PB_HB_4102",
    "kind": "R6_PB_HB",
    "dataset": "kuairand",
    "seed": 4102,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4102"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4102/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R6_PB_HB_4102.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4102,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "fcd1c2d9cd627b9b41c6dadc5cd6fdcf38c917f3046448efdfaecd42be881b3d",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4102/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "cd865ee58e77a0639c9b7b29b324cd6e656673fddbfc199ee111e55b8cfe7ad2",
    "parent_full_state_digest": "745a10a24eb329579cd6d75f7b135aed81b35dcb8083f15d42e130381bf8cc05",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4102",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4102,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "52b40c7b273c216cdcd6e79c12d55f3391584a9ea2386ec6c692780a0ec54bbc"
}
```

### FORMAL_kuairand_R0_BASE_4102

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R0_BASE_4102",
    "kind": "R0_BASE",
    "dataset": "kuairand",
    "seed": 4102,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4102"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4102/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R0_BASE_4102.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4102,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "df8115b263a007c5e81a4839884a592fd6a7744443a51f0b529e834f9b75e069",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4102/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "cd865ee58e77a0639c9b7b29b324cd6e656673fddbfc199ee111e55b8cfe7ad2",
    "parent_full_state_digest": "745a10a24eb329579cd6d75f7b135aed81b35dcb8083f15d42e130381bf8cc05",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4102",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4102,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "52b40c7b273c216cdcd6e79c12d55f3391584a9ea2386ec6c692780a0ec54bbc"
}
```

### FORMAL_kuairand_R5_PB_H1_4102

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R5_PB_H1_4102",
    "kind": "R5_PB_H1",
    "dataset": "kuairand",
    "seed": 4102,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4102"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4102/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R5_PB_H1_4102.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4102,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "e2aae1d0725b7b9a9036ca22c824782a9d16b19270aa182987a3f9d1a2a61798",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4102/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "cd865ee58e77a0639c9b7b29b324cd6e656673fddbfc199ee111e55b8cfe7ad2",
    "parent_full_state_digest": "745a10a24eb329579cd6d75f7b135aed81b35dcb8083f15d42e130381bf8cc05",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4102",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4102,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "52b40c7b273c216cdcd6e79c12d55f3391584a9ea2386ec6c692780a0ec54bbc"
}
```

### FORMAL_kuairand_R7_PB_HISTORY_4102

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R7_PB_HISTORY_4102",
    "kind": "R7_PB_HISTORY",
    "dataset": "kuairand",
    "seed": 4102,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL_kuairand_R6_PB_HB_4102"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4102/first_block.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R7_PB_HISTORY_4102.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4102,
      "dataset": "kuairand",
      "kind": "R6_PB_HB",
      "boundary": "first_block_10.5",
      "epoch": 10,
      "position": 20
    }
  },
  "configuration_file_sha256": "a56a82b4eb0c9acf6dd7442604f0ef82d6b712c22d9310a018084f00af6e3865",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4102/first_block.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "79c31e00342567ceb8ff5166df307f3cc89b5af6252838306d97baed9d824b82",
    "parent_full_state_digest": "92d783a1028df917b8c9c4b955423b134d8bcff94529f9698ef5697154afc540",
    "parent_kind": "R6_PB_HB",
    "parent_boundary": "first_block_10.5",
    "parent_run_id": "FORMAL_kuairand_R6_PB_HB_4102",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4102,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "14231c5d418d9a943222dd0dd97caf8d5536a7bd2f9a9e76f0549b5fd290e0ef"
}
```

### FORMAL_kuairand_STGU_DOMAIN_ADAM_4102

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_STGU_DOMAIN_ADAM_4102",
    "kind": "STGU_DOMAIN_ADAM",
    "dataset": "kuairand",
    "seed": 4102,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data"
  },
  "configuration_file_sha256": "b132f66c6b028dfaeb62dce93c1188fda463f036ce8b3437f1fdb85c9fea50be",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL-WARMUP_kuairand_WARM_RHD_4103

```json
{
  "configuration": {
    "run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4103",
    "kind": "WARM_RHD",
    "dataset": "kuairand",
    "seed": 4103,
    "epochs": 10,
    "stage": "formal-warmup",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": []
  },
  "configuration_file_sha256": "1cff68f5275b0caa52bdd1743294281c5ddfac9a4291f56281249aca2e6073ea",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL_kuairand_R6_PB_HB_4103

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R6_PB_HB_4103",
    "kind": "R6_PB_HB",
    "dataset": "kuairand",
    "seed": 4103,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4103"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4103/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R6_PB_HB_4103.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4103,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "e2459430b0f56d8db543de11f102618842e1b9cc6d90c3e102126e0846a44e4d",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4103/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "b5d8fcaf126033e6861985d2df8befde4d6aafa0ee4cfc5773ccbd5ce6f99953",
    "parent_full_state_digest": "5b9e10cf712ec23c26d8cabaec957435d1af27762b50a6a9a649823f32d74805",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4103",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4103,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "cee059a4ee56e969126e929993bf8760f382871c672ebd749bcf2fb3d10e1571"
}
```

### FORMAL_kuairand_R0_BASE_4103

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R0_BASE_4103",
    "kind": "R0_BASE",
    "dataset": "kuairand",
    "seed": 4103,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4103"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4103/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R0_BASE_4103.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4103,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "b492e3fa700cfe8993fe5b54670f96221eedad5d4b1575f20c07efaff1f7886f",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4103/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "b5d8fcaf126033e6861985d2df8befde4d6aafa0ee4cfc5773ccbd5ce6f99953",
    "parent_full_state_digest": "5b9e10cf712ec23c26d8cabaec957435d1af27762b50a6a9a649823f32d74805",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4103",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4103,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "cee059a4ee56e969126e929993bf8760f382871c672ebd749bcf2fb3d10e1571"
}
```

### FORMAL_kuairand_R5_PB_H1_4103

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R5_PB_H1_4103",
    "kind": "R5_PB_H1",
    "dataset": "kuairand",
    "seed": 4103,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4103"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4103/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R5_PB_H1_4103.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4103,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "b1a6a51bbf1b8aa8f8f4db08f0f076594b3cd94c6088f9094f1c575ec1c84073",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4103/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "b5d8fcaf126033e6861985d2df8befde4d6aafa0ee4cfc5773ccbd5ce6f99953",
    "parent_full_state_digest": "5b9e10cf712ec23c26d8cabaec957435d1af27762b50a6a9a649823f32d74805",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4103",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4103,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "cee059a4ee56e969126e929993bf8760f382871c672ebd749bcf2fb3d10e1571"
}
```

### FORMAL_kuairand_R7_PB_HISTORY_4103

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R7_PB_HISTORY_4103",
    "kind": "R7_PB_HISTORY",
    "dataset": "kuairand",
    "seed": 4103,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL_kuairand_R6_PB_HB_4103"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4103/first_block.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R7_PB_HISTORY_4103.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4103,
      "dataset": "kuairand",
      "kind": "R6_PB_HB",
      "boundary": "first_block_10.5",
      "epoch": 10,
      "position": 20
    }
  },
  "configuration_file_sha256": "4387918e2dec79b986d94cef8efafbfef0271564d7ce82842948ec91e79d8683",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4103/first_block.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "f647fbdc8a1b9d50dbe1b8f9dc18324d1b4a12ec7469c3661c813ccaa360b2d1",
    "parent_full_state_digest": "fd54989f381f3d4ccbbd669a894355a28bac6c0310477eedaa3cacfa12e87c2d",
    "parent_kind": "R6_PB_HB",
    "parent_boundary": "first_block_10.5",
    "parent_run_id": "FORMAL_kuairand_R6_PB_HB_4103",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4103,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "035d2142b6b74e702cb24da633fc6d0ed753e392682a43dd2180689030a81d7f"
}
```

### FORMAL_kuairand_STGU_DOMAIN_ADAM_4103

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_STGU_DOMAIN_ADAM_4103",
    "kind": "STGU_DOMAIN_ADAM",
    "dataset": "kuairand",
    "seed": 4103,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data"
  },
  "configuration_file_sha256": "52fb878901b1bd2684e14fab68235464da289a7dd49ac64ebd0606125f133ce1",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL-WARMUP_kuairand_WARM_RHD_4104

```json
{
  "configuration": {
    "run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4104",
    "kind": "WARM_RHD",
    "dataset": "kuairand",
    "seed": 4104,
    "epochs": 10,
    "stage": "formal-warmup",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": []
  },
  "configuration_file_sha256": "cc2eadc7532337369201eb2258fafc097f54631338b1df6e7671d9524504caa6",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

### FORMAL_kuairand_R6_PB_HB_4104

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R6_PB_HB_4104",
    "kind": "R6_PB_HB",
    "dataset": "kuairand",
    "seed": 4104,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4104"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4104/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R6_PB_HB_4104.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4104,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "c081140c7772f111b73f1d0485c8e5a6f93baa49d9b90c73e954859a56706b6d",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4104/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "ec3bf798722446056d8632839d082be9ef97ee34a2abe651d142592ee8006d5a",
    "parent_full_state_digest": "2973f8b6110bd7ef851f1b1ecccb2ab37c29e08ea7d1378daf61f17c7831184f",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4104",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4104,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "c69169b491c28de46543d1d83ffe1feaf25afc27b606e4937df73e72560c578f"
}
```

### FORMAL_kuairand_R0_BASE_4104

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R0_BASE_4104",
    "kind": "R0_BASE",
    "dataset": "kuairand",
    "seed": 4104,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4104"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4104/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R0_BASE_4104.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4104,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "b54f12cda02ee01124cdea1428d70bd5d669d1e86ddbeaa2364d5498647624c7",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4104/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "ec3bf798722446056d8632839d082be9ef97ee34a2abe651d142592ee8006d5a",
    "parent_full_state_digest": "2973f8b6110bd7ef851f1b1ecccb2ab37c29e08ea7d1378daf61f17c7831184f",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4104",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4104,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "c69169b491c28de46543d1d83ffe1feaf25afc27b606e4937df73e72560c578f"
}
```

### FORMAL_kuairand_R5_PB_H1_4104

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R5_PB_H1_4104",
    "kind": "R5_PB_H1",
    "dataset": "kuairand",
    "seed": 4104,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL-WARMUP_kuairand_WARM_RHD_4104"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4104/epoch10.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R5_PB_H1_4104.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4104,
      "dataset": "kuairand",
      "kind": "WARM_RHD",
      "boundary": "native_epoch10",
      "epoch": 10,
      "position": 0
    }
  },
  "configuration_file_sha256": "8f60948be9d10360edf0410f022f288457d7bbb7b66b98e167f854918d8ba6ed",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL-WARMUP_kuairand_WARM_RHD_4104/epoch10.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "ec3bf798722446056d8632839d082be9ef97ee34a2abe651d142592ee8006d5a",
    "parent_full_state_digest": "2973f8b6110bd7ef851f1b1ecccb2ab37c29e08ea7d1378daf61f17c7831184f",
    "parent_kind": "WARM_RHD",
    "parent_boundary": "native_epoch10",
    "parent_run_id": "FORMAL-WARMUP_kuairand_WARM_RHD_4104",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4104,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "c69169b491c28de46543d1d83ffe1feaf25afc27b606e4937df73e72560c578f"
}
```

### FORMAL_kuairand_R7_PB_HISTORY_4104

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_R7_PB_HISTORY_4104",
    "kind": "R7_PB_HISTORY",
    "dataset": "kuairand",
    "seed": 4104,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [
      "FORMAL_kuairand_R6_PB_HB_4104"
    ],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data",
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4104/first_block.pt",
    "parent_certificate": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/parent_certificates/FORMAL_kuairand_R7_PB_HISTORY_4104.json",
    "parent_contract": {
      "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
      "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
      "seed": 4104,
      "dataset": "kuairand",
      "kind": "R6_PB_HB",
      "boundary": "first_block_10.5",
      "epoch": 10,
      "position": 20
    }
  },
  "configuration_file_sha256": "00df9555c8b611a9f78cb73e6c0c01af70d25111c30736065e3e2e7949ae3268",
  "actual_parent_certificate": {
    "fork": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4104/first_block.pt",
    "parent_source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "parent_checkpoint_sha256": "cf8f7c3f8de8b1f86611378a420c3004e37f3e240e55c927f5288ad4527f6d53",
    "parent_full_state_digest": "5d5bdf64eeba33160a919194c27ceda402e1820cf2135319e3a9450d88098a77",
    "parent_kind": "R6_PB_HB",
    "parent_boundary": "first_block_10.5",
    "parent_run_id": "FORMAL_kuairand_R6_PB_HB_4104",
    "parent_data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "parent_seed": 4104,
    "parent_dataset": "kuairand"
  },
  "actual_parent_certificate_sha256": "3fcb863320119ff0a41d4f8cbf4d2af7ea0b53c9f65438500690d817465126f4"
}
```

### FORMAL_kuairand_STGU_DOMAIN_ADAM_4104

```json
{
  "configuration": {
    "run_id": "FORMAL_kuairand_STGU_DOMAIN_ADAM_4104",
    "kind": "STGU_DOMAIN_ADAM",
    "dataset": "kuairand",
    "seed": 4104,
    "epochs": 50,
    "stage": "formal",
    "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
    "conditional_design_sha256": "15baf107b2e4d4e08eaf2199b3b334a1e6bed9575480deb5b0c75c660a4746f8",
    "protocol_hash": "265fa10eeabb9ac190af4a5b36236ef2792bf2ad56b9972d71e5e67860128a4c",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "dependencies": [],
    "primary_endpoint_role": "benchmark_test_exposed",
    "primary_endpoint_metric": "macro_logloss",
    "companion_role": "Q_legacy_confirm_exposed",
    "development_diagnostic_role": "Q_explore",
    "checkpoint_selection": "validation_macro_logloss_min_eligible12_earliest",
    "exposure_claim": "locked reevaluation on previously exposed benchmarks; not unseen data"
  },
  "configuration_file_sha256": "2bc599eb371012775a8ec361365fba25e6a85f77981b337c8dce2f40a14dbad4",
  "initialization": "fresh seed epoch0; no inherited checkpoint"
}
```

