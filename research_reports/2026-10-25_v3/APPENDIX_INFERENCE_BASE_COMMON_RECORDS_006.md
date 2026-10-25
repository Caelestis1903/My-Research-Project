# BASE_COMMON 完整推理records 6

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json；SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

原records位置：26–30。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "P1_kuairand_L1_FULL_COARSE_3101",
    "method": "L1_FULL_COARSE",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005626837009913288,
    "p95_seconds": 0.006046831303683575,
    "mean_seconds": 0.005662528336203346,
    "rows_per_second_at_p50": 727940.0474518314,
    "rows_per_second_at_mean": 723351.7886016119,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.006431780988350511,
      0.005693185987183824,
      0.005562976002693176,
      0.005550478002987802,
      0.005630277009913698
    ],
    "timed_seconds": [
      0.006125556014012545,
      0.005701013986254111,
      0.005432649020804092,
      0.0055248440185096115,
      0.005447443982120603,
      0.00606243999209255,
      0.005671844002790749,
      0.005647831014357507,
      0.005549644993152469,
      0.005625933001283556,
      0.005975155014311895,
      0.005539393983781338,
      0.005658303998643532,
      0.005744266993133351,
      0.005468191026011482,
      0.0057045359862968326,
      0.005536103999475017,
      0.005615388014120981,
      0.005507686000782996,
      0.005472583987284452,
      0.005971031001536176,
      0.005627703998470679,
      0.005522980005480349,
      0.0056778190191835165,
      0.005549549998249859,
      0.006027754017850384,
      0.005682203016476706,
      0.005549265973968431,
      0.005625970021355897,
      0.00563076400430873
    ],
    "loaded_inference_baseline_allocated_bytes": 16854016,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63560704,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 71303168,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:48.810611+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.294582+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.337096+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.378986+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.420963+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.462403+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.504315+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.546589+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.588787+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:49.601731+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.8055369899957441,
    "forward_latency_seconds_including_warmup": 0.1987445480772294,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_L1_FULL_COARSE_3101/primary_best.pt",
    "checkpoint_sha256": "9bf7ed562d9cedd67bc04f5aea055815f1a92c63dcc5904907f407a0e0b190a0",
    "checkpoint_selected_epoch": 15,
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 1634330,
      "parameters_requires_grad_at_restore": 1634330,
      "parameter_bytes": 6537320,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1634330,
      "parameters_requires_grad_at_restore": 1634330,
      "parameter_bytes": 6537320,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "P1_kuairand_L2_BLOCK_UNIFORM_3101",
    "method": "L2_BLOCK_UNIFORM",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0055113730049924925,
    "p95_seconds": 0.006013206198986154,
    "mean_seconds": 0.0055832948002110545,
    "rows_per_second_at_p50": 743190.4892464413,
    "rows_per_second_at_mean": 733617.0033230499,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.006467556988354772,
      0.005596164992311969,
      0.005500883999047801,
      0.0055243730021175,
      0.005425298004411161
    ],
    "timed_seconds": [
      0.0057453340268693864,
      0.005504760018084198,
      0.005466460017487407,
      0.00544990599155426,
      0.005415471998276189,
      0.006015494000166655,
      0.00549495397717692,
      0.005488214024808258,
      0.005404966999776661,
      0.005377602006774396,
      0.005736921011703089,
      0.005462507018819451,
      0.005623004981316626,
      0.00545113900443539,
      0.005455152015201747,
      0.006040790001861751,
      0.005530536000151187,
      0.005539397010579705,
      0.005451707984320819,
      0.00541268999222666,
      0.006006755982525647,
      0.005614829016849399,
      0.005610061984043568,
      0.005495068005984649,
      0.0055482760071754456,
      0.00601040999754332,
      0.005517985991900787,
      0.005443246976938099,
      0.005619235977064818,
      0.005565965984715149
    ],
    "loaded_inference_baseline_allocated_bytes": 16854016,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63560704,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 71303168,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:49.694431+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.181404+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.223522+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.282521+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.324177+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.381759+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.423411+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.465645+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.506867+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:50.520295+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.839762494986644,
    "forward_latency_seconds_including_warmup": 0.19601312099257484,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_L2_BLOCK_UNIFORM_3101/primary_best.pt",
    "checkpoint_sha256": "a06cf1af19b9b9393f88db17dfc9a7190cb09da9c8683458aca003a942c8e883",
    "checkpoint_selected_epoch": 15,
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 1634942,
      "parameters_requires_grad_at_restore": 1634942,
      "parameter_bytes": 6539768,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1634942,
      "parameters_requires_grad_at_restore": 1634942,
      "parameter_bytes": 6539768,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "P1_kuairand_L3_SCALAR_META_3101",
    "method": "L3_SCALAR_META",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005457259991089813,
    "p95_seconds": 0.005677906206983607,
    "mean_seconds": 0.0054821502688961726,
    "rows_per_second_at_p50": 750559.805962631,
    "rows_per_second_at_mean": 747152.0843270732,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.006217551999725401,
      0.005532148003112525,
      0.005461592023493722,
      0.005405117000918835,
      0.005406784010119736
    ],
    "timed_seconds": [
      0.005668012017849833,
      0.005538814002647996,
      0.005413740000221878,
      0.0054559389827772975,
      0.005477265018271282,
      0.00568369502434507,
      0.005472085002111271,
      0.00542621299973689,
      0.005390025005908683,
      0.00539414698141627,
      0.005613185989204794,
      0.00550501499674283,
      0.005397517001256347,
      0.0053720180003438145,
      0.005335797002771869,
      0.005738523992476985,
      0.005422548012575135,
      0.005441245011752471,
      0.005458580999402329,
      0.005461566994199529,
      0.005581407021963969,
      0.005411677004303783,
      0.005411105987150222,
      0.005391178012359887,
      0.0054595669789705426,
      0.0056708309857640415,
      0.005484173016157001,
      0.005440573004307225,
      0.005493177013704553,
      0.005454886006191373
    ],
    "loaded_inference_baseline_allocated_bytes": 16854016,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63560704,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 71303168,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:50.613166+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.091073+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.132136+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.172523+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.212595+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.252655+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.292899+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.333602+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.374523+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.386870+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.7881354919809382,
    "forward_latency_seconds_including_warmup": 0.1924877011042554,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_L3_SCALAR_META_3101/primary_best.pt",
    "checkpoint_sha256": "93f958a3c7132786d3ae67398a904bce778288c9249e788a82c50d390d146185",
    "checkpoint_selected_epoch": 18,
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 1636943,
      "parameters_requires_grad_at_restore": 1636943,
      "parameter_bytes": 6547772,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "read_policy",
        "training_only_formation"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1634942,
      "parameters_requires_grad_at_restore": 1634942,
      "parameter_bytes": 6539768,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "P1_kuairand_L4_BLOCK_META_3101",
    "method": "L4_BLOCK_META",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005522837498574518,
    "p95_seconds": 0.005793564647319727,
    "mean_seconds": 0.005562812901916914,
    "rows_per_second_at_p50": 741647.749197257,
    "rows_per_second_at_mean": 736318.1311721883,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0063438900106120855,
      0.005583105026744306,
      0.005524075007997453,
      0.005466011993121356,
      0.005473129014717415
    ],
    "timed_seconds": [
      0.005788965005194768,
      0.005681249022018164,
      0.005527233995962888,
      0.0055657589982729405,
      0.005731468991143629,
      0.006073055992601439,
      0.005563607002841309,
      0.005502544023329392,
      0.005464675981784239,
      0.005400970025220886,
      0.005797327990876511,
      0.00553386501269415,
      0.005398739012889564,
      0.005443235015263781,
      0.005436145991552621,
      0.005725691007683054,
      0.005421303998446092,
      0.005457418999867514,
      0.005426174000604078,
      0.005518441001186147,
      0.005681096983607858,
      0.005456749990116805,
      0.005454212019685656,
      0.005436665000161156,
      0.005433390993857756,
      0.005749271978856996,
      0.005658281006617472,
      0.005534068011911586,
      0.00560660200426355,
      0.005416177998995408
    ],
    "loaded_inference_baseline_allocated_bytes": 16854016,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63560704,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 71303168,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:51.487324+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:51.982571+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.024117+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.090238+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.132057+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.172940+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.213823+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.254353+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.314554+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.326781+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.8529876419925131,
    "forward_latency_seconds_including_warmup": 0.19527459811070003,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_L4_BLOCK_META_3101/primary_best.pt",
    "checkpoint_sha256": "58234240ad0032b7f6eca0d61722d0e72ba9d1e63d249600e11601e2e3ab0820",
    "checkpoint_selected_epoch": 18,
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 1636934,
      "parameters_requires_grad_at_restore": 1636934,
      "parameter_bytes": 6547736,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "read_policy",
        "training_only_formation"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1634942,
      "parameters_requires_grad_at_restore": 1634942,
      "parameter_bytes": 6539768,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "P1_kuairand_L5_BLOCK_CONSTANT_3101",
    "method": "L5_BLOCK_CONSTANT",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0049555805017007515,
    "p95_seconds": 0.005260590347461402,
    "mean_seconds": 0.004991939664857152,
    "rows_per_second_at_p50": 826542.9244049725,
    "rows_per_second_at_mean": 820522.7376515598,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.005678916990291327,
      0.004897849023109302,
      0.004871754004852846,
      0.0049392550135962665,
      0.004844137001782656
    ],
    "timed_seconds": [
      0.005202136992011219,
      0.00499405397567898,
      0.0048731160059105605,
      0.0049228080024477094,
      0.004873284022323787,
      0.005275748990243301,
      0.005009633983718231,
      0.0049608479894232005,
      0.004934289987431839,
      0.005027056991821155,
      0.005244839994702488,
      0.0049923080077860504,
      0.004979585995897651,
      0.0049731449980754405,
      0.005011076980736107,
      0.005235721997451037,
      0.00500931401620619,
      0.004879123996943235,
      0.004906767018837854,
      0.004825521027669311,
      0.005208596994634718,
      0.0049503130139783025,
      0.0049425800098106265,
      0.004889000003458932,
      0.004872016987064853,
      0.005273476999718696,
      0.004890665004495531,
      0.0048444139829371125,
      0.004861547990003601,
      0.0048951979842968285
    ],
    "loaded_inference_baseline_allocated_bytes": 16829440,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63536128,
    "peak_total_reserved_bytes": 98566144,
    "peak_incremental_allocated_bytes": 46706688,
    "peak_incremental_reserved_bytes": 71303168,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:52.415946+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "328"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.885452+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "332"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.924123+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:52.962375+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.001109+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.039028+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.077123+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.115042+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.153334+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.186014+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "400"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.7838760879822075,
    "forward_latency_seconds_including_warmup": 0.17499010197934695,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_L5_BLOCK_CONSTANT_3101/primary_best.pt",
    "checkpoint_sha256": "d90475fffc86c30eb7d87582847880d5c061b9090041b0e97730fe8a3d2ef9fa",
    "checkpoint_selected_epoch": 15,
    "data_hash": "8b5fded108a53c040ab95d939a267419dae100e7bb9546bc1ea1a13f29beb520",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 1632758,
      "parameters_requires_grad_at_restore": 1632758,
      "parameter_bytes": 6531032,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "read_policy",
        "training_only_formation"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1630766,
      "parameters_requires_grad_at_restore": 1630766,
      "parameter_bytes": 6523064,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_read_policy"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  }
]
```
