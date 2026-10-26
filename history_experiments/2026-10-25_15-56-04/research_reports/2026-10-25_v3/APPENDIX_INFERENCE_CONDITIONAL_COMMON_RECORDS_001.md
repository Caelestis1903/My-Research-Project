# CONDITIONAL_COMMON 完整推理records 1

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_conditional/benchmark_receipt.json；SHA256：459c19604423892d52529ab15d40fe2dd91f0e27c9503c7d9d7fd9238e2a21c3

原records位置：1–5。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "P2_movielens_LPS_CAL_PROB_CLIP_3101",
    "method": "LPS_CAL_PROB_CLIP",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "stage": "p2",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.005925293502514251,
    "p95_seconds": 0.013747130353294775,
    "mean_seconds": 0.007605994269639874,
    "rows_per_second_at_p50": 691273.7737399783,
    "rows_per_second_at_mean": 538522.6250234785,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.16677205098676495,
      0.0058461120061110705,
      0.0055519329907838255,
      0.005499228980625048,
      0.005476935999467969
    ],
    "timed_seconds": [
      0.005687320022843778,
      0.005636179004795849,
      0.0056327939964830875,
      0.005915587011259049,
      0.00670761099900119,
      0.012981620006030425,
      0.008260442002210766,
      0.006350379000650719,
      0.005583902006037533,
      0.005568507011048496,
      0.00665710202883929,
      0.005667071003699675,
      0.005646841978887096,
      0.005771611991804093,
      0.00881659600418061,
      0.018612999003380537,
      0.011345113016432151,
      0.007215476012788713,
      0.005688386008841917,
      0.005413556995335966,
      0.007562872982816771,
      0.005649625003570691,
      0.005496631987625733,
      0.005739426997024566,
      0.008310842997161672,
      0.014373457001056522,
      0.012948215997312218,
      0.0075028580031357706,
      0.005934999993769452,
      0.005501802021171898
    ],
    "loaded_inference_baseline_allocated_bytes": 3377152,
    "loaded_inference_baseline_reserved_bytes": 4194304,
    "peak_total_allocated_bytes": 53475328,
    "peak_total_reserved_bytes": 77594624,
    "peak_incremental_allocated_bytes": 50098176,
    "peak_incremental_reserved_bytes": 73400320,
    "memory_scope": "Only exact deployment modules and encoded input after CPU-only checkpoint extraction, GC and empty_cache; includes production forward temporaries and CAL minimum-norm solver, not a theoretical minimal footprint.",
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "all_policy_modules_eval": true,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:08:14.213458+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.518034+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "258"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.721441+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.767777+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.819648+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.873010+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.935075+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:14.988159+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.047349+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.062273+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.8847902840061579,
    "forward_latency_seconds_including_warmup": 0.4173260890529491,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P2_movielens_LPS_CAL_PROB_CLIP_3101/primary_best.pt",
    "checkpoint_sha256": "268847a3d6b819f39daef1c99f0eabecd4f1b3d8733462dcd169c76f7683e562",
    "checkpoint_selected_epoch": 12,
    "source_hash_at_training": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "manifest_sha256": "6eeff84d339a873782007f34c1faf3b0203a4d24d4c2d48dcd9eb2a4b6b8e3a9",
    "fit_row_id_hash": "9e9b225e6e09c1e13d2944ff1ddf1acd59f4cc7b9e9fcfb7b45424163181be8e",
    "input_feature_hash": "2097ab2423efc44351fee9645ab605c542f8bbbc8f27cd26855bd3346346b44c",
    "input_shape": [
      4096,
      8
    ],
    "input_dtype": "torch.float32",
    "domain_counts": [
      181,
      2178,
      1737
    ],
    "input_labels_accessed": false,
    "label_array_opened": false,
    "quality_tables_read": false,
    "serialized_training_parameter_inventory": {
      "model": {
        "parameters_total": 789486,
        "parameters_requires_grad_at_restore": 789486,
        "parameter_bytes": 3157944
      },
      "policy": {
        "parameters_total": 4110,
        "parameters_requires_grad_at_restore": 4110,
        "parameter_bytes": 16440
      },
      "training_only_allocator": {
        "parameters_total": 1992,
        "parameters_requires_grad_at_restore": 1992,
        "parameter_bytes": 7968
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 795588,
      "parameters_requires_grad_at_restore": 795588
    },
    "inference_parameter_inventory": {
      "parameters_total": 793596,
      "parameters_requires_grad_at_restore": 0,
      "parameter_bytes": 3174384,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "deployed_model",
        "deployed_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "production_entrypoint": "ExtensionLpsTrainer.predict_logits -> current-real packet -> production policy/projection",
    "CAL_probability_projection_included": true,
    "discarded_training_state": "No Data/Trainer constructor; no label array. Optimizer moments, saved gradients, phi/allocator/GEOM centers, controllers and RNG payloads never moved to GPU; CPU checkpoint payload released before memory baseline.",
    "training_only_modules_moved_to_GPU": false,
    "inference_parameter_definition": "Unique parameters resident in the exact deployed model/read-policy modules; no claim every subweight is active on every row."
  },
  {
    "status": "COMPLETE",
    "run_id": "P2_movielens_LPS_OWNER_META_3101",
    "method": "LPS_OWNER_META",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "stage": "p2",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.005413609993411228,
    "p95_seconds": 0.013474572014820297,
    "mean_seconds": 0.007045800698688254,
    "rows_per_second_at_p50": 756611.5780385253,
    "rows_per_second_at_mean": 581339.1799121667,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.006137618998764083,
      0.00438684897380881,
      0.004317395010730252,
      0.004528664023382589,
      0.005079683993244544
    ],
    "timed_seconds": [
      0.01221675900160335,
      0.013439911010209471,
      0.009013943985337391,
      0.008220788004109636,
      0.01350293101859279,
      0.014904145995387807,
      0.013362309982767329,
      0.008054089994402602,
      0.005158468993613496,
      0.004463201999897137,
      0.009181480010738596,
      0.006090867973398417,
      0.004869588010478765,
      0.004299013991840184,
      0.004201711999485269,
      0.0061034430109430104,
      0.00511515699326992,
      0.004469673993298784,
      0.004410911991726607,
      0.004197272006422281,
      0.005513512995094061,
      0.004822429997147992,
      0.004452398017747328,
      0.004771028005052358,
      0.004444586986210197,
      0.004718168987892568,
      0.005313706991728395,
      0.0070355980133172125,
      0.008344974019564688,
      0.006681946979369968
    ],
    "loaded_inference_baseline_allocated_bytes": 12942336,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 53472256,
    "peak_total_reserved_bytes": 77594624,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "Only exact deployment modules and encoded input after CPU-only checkpoint extraction, GC and empty_cache; includes production forward temporaries and CAL minimum-norm solver, not a theoretical minimal footprint.",
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "all_policy_modules_eval": true,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:08:15.128159+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.251099+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.313838+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.392214+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.452383+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.494618+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.532076+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.569717+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.621316+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.649251+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.535993072000565,
    "forward_latency_seconds_including_warmup": 0.2358242319605779,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P2_movielens_LPS_OWNER_META_3101/primary_best.pt",
    "checkpoint_sha256": "887cf6d2149d2a2a68ff0cf7a090ff304c3e0e670a6c05bd684e6853d44bab37",
    "checkpoint_selected_epoch": 12,
    "source_hash_at_training": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "manifest_sha256": "6eeff84d339a873782007f34c1faf3b0203a4d24d4c2d48dcd9eb2a4b6b8e3a9",
    "fit_row_id_hash": "9e9b225e6e09c1e13d2944ff1ddf1acd59f4cc7b9e9fcfb7b45424163181be8e",
    "input_feature_hash": "2097ab2423efc44351fee9645ab605c542f8bbbc8f27cd26855bd3346346b44c",
    "input_shape": [
      4096,
      8
    ],
    "input_dtype": "torch.float32",
    "domain_counts": [
      181,
      2178,
      1737
    ],
    "input_labels_accessed": false,
    "label_array_opened": false,
    "quality_tables_read": false,
    "serialized_training_parameter_inventory": {
      "model": {
        "parameters_total": 789486,
        "parameters_requires_grad_at_restore": 789486,
        "parameter_bytes": 3157944
      },
      "policy": {
        "parameters_total": 4176,
        "parameters_requires_grad_at_restore": 4176,
        "parameter_bytes": 16704
      },
      "training_only_allocator": {
        "parameters_total": 1992,
        "parameters_requires_grad_at_restore": 1992,
        "parameter_bytes": 7968
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 795654,
      "parameters_requires_grad_at_restore": 795654
    },
    "inference_parameter_inventory": {
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 0,
      "parameter_bytes": 3174648,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "deployed_model",
        "deployed_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "production_entrypoint": "ExtensionLpsTrainer.predict_logits -> current-real packet -> production policy/projection",
    "CAL_probability_projection_included": false,
    "discarded_training_state": "No Data/Trainer constructor; no label array. Optimizer moments, saved gradients, phi/allocator/GEOM centers, controllers and RNG payloads never moved to GPU; CPU checkpoint payload released before memory baseline.",
    "training_only_modules_moved_to_GPU": false,
    "inference_parameter_definition": "Unique parameters resident in the exact deployed model/read-policy modules; no claim every subweight is active on every row."
  },
  {
    "status": "COMPLETE",
    "run_id": "P2_movielens_LPS_GEOM_3101",
    "method": "LPS_GEOM",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "stage": "p2",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.004427111489349045,
    "p95_seconds": 0.008049799044965765,
    "mean_seconds": 0.005089412864375239,
    "rows_per_second_at_p50": 925208.2333716581,
    "rows_per_second_at_mean": 804807.9629520905,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.004796278022695333,
      0.004257255000993609,
      0.004290009004762396,
      0.004222675983328372,
      0.004285186994820833
    ],
    "timed_seconds": [
      0.0045999699796084315,
      0.004311261989641935,
      0.004202059004455805,
      0.004305098002078012,
      0.004215572989778593,
      0.004955205018632114,
      0.004345786001067609,
      0.004330470983404666,
      0.004206958983559161,
      0.004214193992083892,
      0.012204390979604796,
      0.008226887992350385,
      0.0059320869913790375,
      0.004849396995268762,
      0.00448792998213321,
      0.007833356998162344,
      0.005738248990382999,
      0.00493152198032476,
      0.00436629299656488,
      0.004210936021991074,
      0.005620212992653251,
      0.004785261990036815,
      0.004586520022712648,
      0.004201042000204325,
      0.004230544000165537,
      0.005221751023782417,
      0.004729903012048453,
      0.004347208014223725,
      0.004251098987879232,
      0.004241217015078291
    ],
    "loaded_inference_baseline_allocated_bytes": 12942336,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 53472256,
    "peak_total_reserved_bytes": 77594624,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "Only exact deployment modules and encoded input after CPU-only checkpoint extraction, GC and empty_cache; includes production forward temporaries and CAL minimum-norm solver, not a theoretical minimal footprint.",
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "all_policy_modules_eval": true,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:08:15.733917+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.857792+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.892583+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.926803+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:15.987200+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.036993+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.077195+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.113841+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.149646+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.166248+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.44716738400165923,
    "forward_latency_seconds_including_warmup": 0.1745337909378577,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P2_movielens_LPS_GEOM_3101/primary_best.pt",
    "checkpoint_sha256": "3cb74418c23e2323722bd3dca55b77a227fa27649472bff1beadea29503075a1",
    "checkpoint_selected_epoch": 14,
    "source_hash_at_training": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "manifest_sha256": "6eeff84d339a873782007f34c1faf3b0203a4d24d4c2d48dcd9eb2a4b6b8e3a9",
    "fit_row_id_hash": "9e9b225e6e09c1e13d2944ff1ddf1acd59f4cc7b9e9fcfb7b45424163181be8e",
    "input_feature_hash": "2097ab2423efc44351fee9645ab605c542f8bbbc8f27cd26855bd3346346b44c",
    "input_shape": [
      4096,
      8
    ],
    "input_dtype": "torch.float32",
    "domain_counts": [
      181,
      2178,
      1737
    ],
    "input_labels_accessed": false,
    "label_array_opened": false,
    "quality_tables_read": false,
    "serialized_training_parameter_inventory": {
      "model": {
        "parameters_total": 789486,
        "parameters_requires_grad_at_restore": 789486,
        "parameter_bytes": 3157944
      },
      "policy": {
        "parameters_total": 4176,
        "parameters_requires_grad_at_restore": 4176,
        "parameter_bytes": 16704
      },
      "training_only_allocator": {
        "parameters_total": 0,
        "parameters_requires_grad_at_restore": 0,
        "parameter_bytes": 0
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 793662
    },
    "inference_parameter_inventory": {
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 0,
      "parameter_bytes": 3174648,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "deployed_model",
        "deployed_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "production_entrypoint": "ExtensionLpsTrainer.predict_logits -> current-real packet -> production policy/projection",
    "CAL_probability_projection_included": false,
    "discarded_training_state": "No Data/Trainer constructor; no label array. Optimizer moments, saved gradients, phi/allocator/GEOM centers, controllers and RNG payloads never moved to GPU; CPU checkpoint payload released before memory baseline.",
    "training_only_modules_moved_to_GPU": false,
    "inference_parameter_definition": "Unique parameters resident in the exact deployed model/read-policy modules; no claim every subweight is active on every row."
  },
  {
    "status": "COMPLETE",
    "run_id": "P2_kuairand_LPS_CAL_PROB_CLIP_3101",
    "method": "LPS_CAL_PROB_CLIP",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "stage": "p2",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.006048973998986185,
    "p95_seconds": 0.010811994488176414,
    "mean_seconds": 0.006966372298969266,
    "rows_per_second_at_p50": 677139.6274287994,
    "rows_per_second_at_mean": 587967.4275527935,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.007284819992491975,
      0.006330992007860914,
      0.00612650701077655,
      0.006155287002911791,
      0.0061631129938177764
    ],
    "timed_seconds": [
      0.019619655009591952,
      0.012196459982078522,
      0.009119869995629415,
      0.007024575985269621,
      0.006185939011629671,
      0.008991847978904843,
      0.006963516003452241,
      0.0061927540227770805,
      0.006101482984377071,
      0.00599488100851886,
      0.006647062982665375,
      0.005995968997012824,
      0.0060309199907351285,
      0.006067028007237241,
      0.005997006024699658,
      0.006463523983256891,
      0.006149269000161439,
      0.006010949000483379,
      0.006008528027450666,
      0.005770717980340123,
      0.006327801005681977,
      0.00590812397422269,
      0.005792976007796824,
      0.005853783019119874,
      0.00589545100228861,
      0.00630412099417299,
      0.005828147986903787,
      0.0058323609991930425,
      0.005872416018974036,
      0.005844031984452158
    ],
    "loaded_inference_baseline_allocated_bytes": 16857088,
    "loaded_inference_baseline_reserved_bytes": 25165824,
    "peak_total_allocated_bytes": 63563776,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 73400320,
    "memory_scope": "Only exact deployment modules and encoded input after CPU-only checkpoint extraction, GC and empty_cache; includes production forward temporaries and CAL minimum-norm solver, not a theoretical minimal footprint.",
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "all_policy_modules_eval": true,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:08:16.229321+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.525601+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "330"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.600108+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.668317+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.715204+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.758445+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.801699+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.844326+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.886315+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:16.926618+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.7118063689849805,
    "forward_latency_seconds_including_warmup": 0.241051887976937,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P2_kuairand_LPS_CAL_PROB_CLIP_3101/primary_best.pt",
    "checkpoint_sha256": "bd76c67dbfdb29012c4b183f0d352ece21c1ab0edc51a31aeb85153b097633c0",
    "checkpoint_selected_epoch": 18,
    "source_hash_at_training": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "manifest_sha256": "231d6ad4c33a59e486f7dd9a08e2fb58a1df806889be7f9609cf5b7105c7b9b3",
    "fit_row_id_hash": "b9a743264dbafc8f950522b26b8316568dadc959d829ac95507366f4084f5bf5",
    "input_feature_hash": "156d6a78239e3857218f1bcd0b58364091dfc039c29ba8a2e645be7ce180dfe9",
    "input_shape": [
      4096,
      34
    ],
    "input_dtype": "torch.float32",
    "domain_counts": [
      1235,
      2710,
      151
    ],
    "input_labels_accessed": false,
    "label_array_opened": false,
    "quality_tables_read": false,
    "serialized_training_parameter_inventory": {
      "model": {
        "parameters_total": 1630766,
        "parameters_requires_grad_at_restore": 1630766,
        "parameter_bytes": 6523064
      },
      "policy": {
        "parameters_total": 4110,
        "parameters_requires_grad_at_restore": 4110,
        "parameter_bytes": 16440
      },
      "training_only_allocator": {
        "parameters_total": 1992,
        "parameters_requires_grad_at_restore": 1992,
        "parameter_bytes": 7968
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 1636868,
      "parameters_requires_grad_at_restore": 1636868
    },
    "inference_parameter_inventory": {
      "parameters_total": 1634876,
      "parameters_requires_grad_at_restore": 0,
      "parameter_bytes": 6539504,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "deployed_model",
        "deployed_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "production_entrypoint": "ExtensionLpsTrainer.predict_logits -> current-real packet -> production policy/projection",
    "CAL_probability_projection_included": true,
    "discarded_training_state": "No Data/Trainer constructor; no label array. Optimizer moments, saved gradients, phi/allocator/GEOM centers, controllers and RNG payloads never moved to GPU; CPU checkpoint payload released before memory baseline.",
    "training_only_modules_moved_to_GPU": false,
    "inference_parameter_definition": "Unique parameters resident in the exact deployed model/read-policy modules; no claim every subweight is active on every row."
  },
  {
    "status": "COMPLETE",
    "run_id": "P2_kuairand_LPS_OWNER_META_3101",
    "method": "LPS_OWNER_META",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "stage": "p2",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.005108641998958774,
    "p95_seconds": 0.013868824295059305,
    "mean_seconds": 0.006599706267782797,
    "rows_per_second_at_p50": 801778.6333109332,
    "rows_per_second_at_mean": 620633.68183446,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.008123989013256505,
      0.005635311012156308,
      0.0051751279970631,
      0.004673702991567552,
      0.0046725739957764745
    ],
    "timed_seconds": [
      0.006008178024785593,
      0.005175234982743859,
      0.004741657001432031,
      0.00465266898390837,
      0.004686241009039804,
      0.007234838005388156,
      0.004917946003843099,
      0.004785213008290157,
      0.004738816991448402,
      0.0047482000081799924,
      0.014814024994848296,
      0.009592421993147582,
      0.006436195981223136,
      0.005442302004666999,
      0.004752376000396907,
      0.008159231016179547,
      0.0047788110096007586,
      0.004759158007800579,
      0.0049510590033605695,
      0.007881166995503008,
      0.01641022300464101,
      0.01271357899531722,
      0.0074096609896514565,
      0.0054422249959316105,
      0.004788933991221711,
      0.00786248198710382,
      0.005704033013898879,
      0.0050420490151736885,
      0.004706895007984713,
      0.004655366006772965
    ],
    "loaded_inference_baseline_allocated_bytes": 16854016,
    "loaded_inference_baseline_reserved_bytes": 25165824,
    "peak_total_allocated_bytes": 63560704,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 73400320,
    "memory_scope": "Only exact deployment modules and encoded input after CPU-only checkpoint extraction, GC and empty_cache; includes production forward temporaries and CAL minimum-norm solver, not a theoretical minimal footprint.",
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "all_policy_modules_eval": true,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:08:17.002620+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.331601+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "330"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.373646+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.411425+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.477672+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.532424+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.588716+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.649410+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.721299+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:08:17.746449+00:00",
        "physical_gpu": 0,
        "self_pid": 1336423,
        "compute_processes": [
          {
            "pid": 1336423,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.7585573619871866,
    "forward_latency_seconds_including_warmup": 0.22627189304330386,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P2_kuairand_LPS_OWNER_META_3101/primary_best.pt",
    "checkpoint_sha256": "ae9e9306ed58d00d6a08c043e8a0c42c48ff72d8a1be32b083c5158c3acce2ce",
    "checkpoint_selected_epoch": 18,
    "source_hash_at_training": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
    "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "manifest_sha256": "231d6ad4c33a59e486f7dd9a08e2fb58a1df806889be7f9609cf5b7105c7b9b3",
    "fit_row_id_hash": "b9a743264dbafc8f950522b26b8316568dadc959d829ac95507366f4084f5bf5",
    "input_feature_hash": "156d6a78239e3857218f1bcd0b58364091dfc039c29ba8a2e645be7ce180dfe9",
    "input_shape": [
      4096,
      34
    ],
    "input_dtype": "torch.float32",
    "domain_counts": [
      1235,
      2710,
      151
    ],
    "input_labels_accessed": false,
    "label_array_opened": false,
    "quality_tables_read": false,
    "serialized_training_parameter_inventory": {
      "model": {
        "parameters_total": 1630766,
        "parameters_requires_grad_at_restore": 1630766,
        "parameter_bytes": 6523064
      },
      "policy": {
        "parameters_total": 4176,
        "parameters_requires_grad_at_restore": 4176,
        "parameter_bytes": 16704
      },
      "training_only_allocator": {
        "parameters_total": 1992,
        "parameters_requires_grad_at_restore": 1992,
        "parameter_bytes": 7968
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 1636934,
      "parameters_requires_grad_at_restore": 1636934
    },
    "inference_parameter_inventory": {
      "parameters_total": 1634942,
      "parameters_requires_grad_at_restore": 0,
      "parameter_bytes": 6539768,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "deployed_model",
        "deployed_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "production_entrypoint": "ExtensionLpsTrainer.predict_logits -> current-real packet -> production policy/projection",
    "CAL_probability_projection_included": false,
    "discarded_training_state": "No Data/Trainer constructor; no label array. Optimizer moments, saved gradients, phi/allocator/GEOM centers, controllers and RNG payloads never moved to GPU; CPU checkpoint payload released before memory baseline.",
    "training_only_modules_moved_to_GPU": false,
    "inference_parameter_definition": "Unique parameters resident in the exact deployed model/read-policy modules; no claim every subweight is active on every row."
  }
]
```
