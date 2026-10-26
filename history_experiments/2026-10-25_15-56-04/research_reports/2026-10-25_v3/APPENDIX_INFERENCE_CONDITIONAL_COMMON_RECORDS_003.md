# CONDITIONAL_COMMON 完整推理records 3

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_conditional/benchmark_receipt.json；SHA256：459c19604423892d52529ab15d40fe2dd91f0e27c9503c7d9d7fd9238e2a21c3

原records位置：11–11。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "FORMAL_kuairand_STGU_DOMAIN_ADAM_4101",
    "method": "STGU_DOMAIN_ADAM",
    "dataset": "kuairand",
    "seed": 4101,
    "predictor": "main",
    "stage": "formal",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.005737831495935097,
    "p95_seconds": 0.007025554552092217,
    "mean_seconds": 0.005444559865281917,
    "rows_per_second_at_p50": 713858.5374808873,
    "rows_per_second_at_mean": 752310.5818192543,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.004663578001782298,
      0.00337688100989908,
      0.0029270590166561306,
      0.002839780005160719,
      0.0027447199972812086
    ],
    "timed_seconds": [
      0.005407665012171492,
      0.005509889975655824,
      0.00359080100315623,
      0.0031058800232131034,
      0.0029455909971147776,
      0.006768843013560399,
      0.005107350007165223,
      0.004305554990423843,
      0.0033401959808543324,
      0.0030558390135411173,
      0.007009483000729233,
      0.006415440991986543,
      0.005704990995582193,
      0.004459911986486986,
      0.004238891007844359,
      0.006970627000555396,
      0.006389153975760564,
      0.006159109994769096,
      0.0057706719962880015,
      0.005158654996193945,
      0.007348853978328407,
      0.006455675000324845,
      0.006319596024695784,
      0.006119076977483928,
      0.00486241199541837,
      0.0070387040032073855,
      0.00640167499659583,
      0.006388200010405853,
      0.006120050995377824,
      0.004868006013566628
    ],
    "loaded_inference_baseline_allocated_bytes": 16723968,
    "loaded_inference_baseline_reserved_bytes": 25165824,
    "peak_total_allocated_bytes": 63430656,
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
        "time_utc": "2026-10-25T12:08:21.502314+00:00",
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
        "time_utc": "2026-10-25T12:08:21.804270+00:00",
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
        "time_utc": "2026-10-25T12:08:21.837843+00:00",
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
        "time_utc": "2026-10-25T12:08:21.878984+00:00",
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
        "time_utc": "2026-10-25T12:08:21.921461+00:00",
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
        "time_utc": "2026-10-25T12:08:21.972978+00:00",
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
        "time_utc": "2026-10-25T12:08:22.048466+00:00",
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
        "time_utc": "2026-10-25T12:08:22.118783+00:00",
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
        "time_utc": "2026-10-25T12:08:22.171398+00:00",
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
        "time_utc": "2026-10-25T12:08:22.193290+00:00",
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
    "total_benchmark_section_seconds": 0.7058263090148102,
    "forward_latency_seconds_including_warmup": 0.17988881398923695,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_STGU_DOMAIN_ADAM_4101/primary_best.pt",
    "checkpoint_sha256": "d8916b764f36de316940db0c39a1db2799651bdca2d31572f2a288a3cafe22cb",
    "checkpoint_selected_epoch": 16,
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
        "parameters_total": 1614254,
        "parameters_requires_grad_at_restore": 1614254,
        "parameter_bytes": 6457016
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254
    },
    "inference_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 0,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "deployed_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "production_entrypoint": "Fixed1Adapter.forward; same trainer.model path for formal RHD/STGU",
    "CAL_probability_projection_included": false,
    "discarded_training_state": "No Data/Trainer constructor; no label array. Optimizer moments, saved gradients, phi/allocator/GEOM centers, controllers and RNG payloads never moved to GPU; CPU checkpoint payload released before memory baseline.",
    "training_only_modules_moved_to_GPU": false,
    "inference_parameter_definition": "Unique parameters resident in the exact deployed model/read-policy modules; no claim every subweight is active on every row."
  }
]
```
