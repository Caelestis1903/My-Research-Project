# BASE_COMMON 完整推理records 4

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json；SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

原records位置：16–20。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "P1_kuairand_R0_BASE_3101",
    "method": "R0_BASE",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0023806285025784746,
    "p95_seconds": 0.002666267441236414,
    "mean_seconds": 0.0024292888653387004,
    "rows_per_second_at_p50": 1720554.0451034654,
    "rows_per_second_at_mean": 1686090.1387406313,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0032032899907790124,
      0.0024329750158358365,
      0.002347393980016932,
      0.0023678849975112826,
      0.0023640319996047765
    ],
    "timed_seconds": [
      0.002667328983079642,
      0.0023833480081520975,
      0.0023888559953775257,
      0.002367270994000137,
      0.00234968401491642,
      0.002591430995380506,
      0.0024134599952958524,
      0.0023563030117657036,
      0.0023779089970048517,
      0.00234465001267381,
      0.002664970001205802,
      0.0023898000072222203,
      0.0024267669941764325,
      0.002377819997491315,
      0.0023655480181332678,
      0.0025789979845285416,
      0.00240888400003314,
      0.0023414450115524232,
      0.0023496270005125552,
      0.0023622649896424264,
      0.002615710982354358,
      0.0023860629880800843,
      0.0023663339961785823,
      0.0023474550107493997,
      0.0024064199824351817,
      0.002771636995021254,
      0.002437740011373535,
      0.0023465769772883505,
      0.002349007991142571,
      0.0023453560133930296
    ],
    "loaded_inference_baseline_allocated_bytes": 16723968,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63430656,
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
        "time_utc": "2026-10-25T12:07:40.593705+00:00",
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
        "time_utc": "2026-10-25T12:07:41.081991+00:00",
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
        "time_utc": "2026-10-25T12:07:41.108568+00:00",
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
        "time_utc": "2026-10-25T12:07:41.134252+00:00",
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
        "time_utc": "2026-10-25T12:07:41.159801+00:00",
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
        "time_utc": "2026-10-25T12:07:41.185940+00:00",
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
        "time_utc": "2026-10-25T12:07:41.211828+00:00",
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
        "time_utc": "2026-10-25T12:07:41.254885+00:00",
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
        "time_utc": "2026-10-25T12:07:41.281016+00:00",
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
        "time_utc": "2026-10-25T12:07:41.294864+00:00",
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
    "total_benchmark_section_seconds": 0.7156241060001776,
    "forward_latency_seconds_including_warmup": 0.08559424194390886,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_R0_BASE_3101/primary_best.pt",
    "checkpoint_sha256": "fb6465197e16a72bb4030c0d9bef3dc787a4f3232f0f182fed532af08d7ffe3a",
    "checkpoint_selected_epoch": 14,
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
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
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
    "run_id": "P1_kuairand_R1_STATIC_3101",
    "method": "R1_STATIC",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0023889504955150187,
    "p95_seconds": 0.0026756970008136706,
    "mean_seconds": 0.0024278076026045406,
    "rows_per_second_at_p50": 1714560.4346719496,
    "rows_per_second_at_mean": 1687118.8621395824,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.00295537497731857,
      0.0024739769869484007,
      0.0024426549789495766,
      0.0023510100145358592,
      0.002375998010393232
    ],
    "timed_seconds": [
      0.0026687890058383346,
      0.002461766009218991,
      0.00237337700673379,
      0.002350610011490062,
      0.0023991800262592733,
      0.0026916539936792105,
      0.0023817470064386725,
      0.0023880899825599045,
      0.0024092870007734746,
      0.0023564410221297294,
      0.0025675659999251366,
      0.0023923490080051124,
      0.0023740240139886737,
      0.002467156999045983,
      0.0023356849851552397,
      0.0026813489967025816,
      0.0023963679850567132,
      0.002389811008470133,
      0.0023222479794640094,
      0.002321795007446781,
      0.0025963630178011954,
      0.0023937700025271624,
      0.0023698789882473648,
      0.0023234070104081184,
      0.0023946250148583204,
      0.002620493993163109,
      0.0023867209965828806,
      0.0023443379905074835,
      0.00234789599198848,
      0.002327442023670301
    ],
    "loaded_inference_baseline_allocated_bytes": 16723968,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63430656,
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
        "time_utc": "2026-10-25T12:07:41.386370+00:00",
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
        "time_utc": "2026-10-25T12:07:41.880149+00:00",
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
        "time_utc": "2026-10-25T12:07:41.906711+00:00",
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
        "time_utc": "2026-10-25T12:07:41.932472+00:00",
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
        "time_utc": "2026-10-25T12:07:41.958396+00:00",
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
        "time_utc": "2026-10-25T12:07:42.002704+00:00",
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
        "time_utc": "2026-10-25T12:07:42.028494+00:00",
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
        "time_utc": "2026-10-25T12:07:42.053821+00:00",
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
        "time_utc": "2026-10-25T12:07:42.098904+00:00",
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
        "time_utc": "2026-10-25T12:07:42.130442+00:00",
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
    "total_benchmark_section_seconds": 0.7582038120017387,
    "forward_latency_seconds_including_warmup": 0.08543324304628186,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_R1_STATIC_3101/primary_best.pt",
    "checkpoint_sha256": "77f60917df53d660cbc69690be1e86c8f4155372edfa0a58df7f51ba51b946a3",
    "checkpoint_selected_epoch": 14,
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
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
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
    "run_id": "P1_kuairand_R2_FREEZE_3101",
    "method": "R2_FREEZE",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0024955169938039035,
    "p95_seconds": 0.002787993747915607,
    "mean_seconds": 0.0025444875325774776,
    "rows_per_second_at_p50": 1641343.2608032408,
    "rows_per_second_at_mean": 1609754.3994844786,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.003110379009740427,
      0.002541440015193075,
      0.002466864010784775,
      0.00245995499426499,
      0.0024695929896552116
    ],
    "timed_seconds": [
      0.0026798140024766326,
      0.002527472999645397,
      0.0024875849776435643,
      0.002520398993510753,
      0.0024567700165789574,
      0.0027714909811038524,
      0.0025169349974021316,
      0.0024777539947535843,
      0.0024896449758671224,
      0.002454575995216146,
      0.002801496011670679,
      0.0025614170008338988,
      0.0024716990010347217,
      0.00246472901199013,
      0.002486443001544103,
      0.0027225500089116395,
      0.0024912779917940497,
      0.002476599009241909,
      0.002499755995813757,
      0.0025147160049527884,
      0.002681518002646044,
      0.0025019879976753145,
      0.0024906430044211447,
      0.0025099449849221855,
      0.002485778008121997,
      0.0028724269941449165,
      0.0025043820205610245,
      0.0024711489968467504,
      0.0024615550064481795,
      0.002482115989550948
    ],
    "loaded_inference_baseline_allocated_bytes": 16723968,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63430656,
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
        "time_utc": "2026-10-25T12:07:42.238613+00:00",
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
        "time_utc": "2026-10-25T12:07:42.712389+00:00",
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
        "time_utc": "2026-10-25T12:07:42.738892+00:00",
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
        "time_utc": "2026-10-25T12:07:42.765180+00:00",
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
        "time_utc": "2026-10-25T12:07:42.810156+00:00",
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
        "time_utc": "2026-10-25T12:07:42.836986+00:00",
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
        "time_utc": "2026-10-25T12:07:42.863132+00:00",
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
        "time_utc": "2026-10-25T12:07:42.889591+00:00",
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
        "time_utc": "2026-10-25T12:07:42.916341+00:00",
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
        "time_utc": "2026-10-25T12:07:42.929854+00:00",
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
    "total_benchmark_section_seconds": 0.7247565709985793,
    "forward_latency_seconds_including_warmup": 0.0893828569969628,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_R2_FREEZE_3101/primary_best.pt",
    "checkpoint_sha256": "7d7201e8dfb8a45e77ec70ff8e32cab4ebbf70a297fd77699d201c5da8315dcd",
    "checkpoint_selected_epoch": 14,
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
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
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
    "run_id": "P1_kuairand_R3_CUM_3101",
    "method": "R3_CUM",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.002490857004886493,
    "p95_seconds": 0.002868850200320594,
    "mean_seconds": 0.0025524061357524866,
    "rows_per_second_at_p50": 1644413.947474537,
    "rows_per_second_at_mean": 1604760.285843945,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0032577600213699043,
      0.002560094988439232,
      0.002532377024181187,
      0.00249466099194251,
      0.0024619530013296753
    ],
    "timed_seconds": [
      0.002782483003102243,
      0.002491774008376524,
      0.0024669370031915605,
      0.002487974998075515,
      0.00248495300184004,
      0.0028888770029880106,
      0.0025297870161011815,
      0.0024980930029414594,
      0.002492769999662414,
      0.0024642999924253672,
      0.002789649006444961,
      0.002491328021278605,
      0.00249283600714989,
      0.0024817080120556056,
      0.002439624979160726,
      0.0029086360009387136,
      0.0025140359939541668,
      0.0024759480147622526,
      0.0024580339959356934,
      0.002461844007484615,
      0.002844372997060418,
      0.002539378998335451,
      0.002476627007126808,
      0.002468552003847435,
      0.0024698190100025386,
      0.002746117999777198,
      0.0024903859884943813,
      0.002498015994206071,
      0.002479054994182661,
      0.0024582660116720945
    ],
    "loaded_inference_baseline_allocated_bytes": 16723968,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63430656,
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
        "time_utc": "2026-10-25T12:07:43.019124+00:00",
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
        "time_utc": "2026-10-25T12:07:43.513510+00:00",
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
        "time_utc": "2026-10-25T12:07:43.559202+00:00",
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
        "time_utc": "2026-10-25T12:07:43.586421+00:00",
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
        "time_utc": "2026-10-25T12:07:43.613527+00:00",
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
        "time_utc": "2026-10-25T12:07:43.640264+00:00",
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
        "time_utc": "2026-10-25T12:07:43.685851+00:00",
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
        "time_utc": "2026-10-25T12:07:43.712555+00:00",
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
        "time_utc": "2026-10-25T12:07:43.739077+00:00",
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
        "time_utc": "2026-10-25T12:07:43.790347+00:00",
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
    "total_benchmark_section_seconds": 0.7853084369853605,
    "forward_latency_seconds_including_warmup": 0.08987903009983711,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_R3_CUM_3101/primary_best.pt",
    "checkpoint_sha256": "ff8ff8cae422c3df56d55155c746d0a1d29de9286387911a328abe1747865761",
    "checkpoint_selected_epoch": 14,
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
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
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
    "run_id": "P1_kuairand_R4_EMA_3101",
    "method": "R4_EMA",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.002494112981366925,
    "p95_seconds": 0.0028872726979898285,
    "mean_seconds": 0.002607033799480026,
    "rows_per_second_at_p50": 1642267.2230971446,
    "rows_per_second_at_mean": 1571134.21422344,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.003194240009179339,
      0.0025638630031608045,
      0.0024731589946895838,
      0.0024710000143386424,
      0.0024830609909258783
    ],
    "timed_seconds": [
      0.0029410540009848773,
      0.002530892990762368,
      0.0024701250076759607,
      0.0024595640134066343,
      0.0024583670019637793,
      0.0027337759966030717,
      0.004228403995512053,
      0.002583118010079488,
      0.0024675250169821084,
      0.00248427499900572,
      0.002709413005504757,
      0.0025393139803782105,
      0.0024820340040605515,
      0.0024836939992383122,
      0.0024457469990011305,
      0.0027385700086597353,
      0.0025115619937423617,
      0.002478690992575139,
      0.002510627993615344,
      0.002444991987431422,
      0.002821539994329214,
      0.002526551019400358,
      0.002476377005223185,
      0.002461821015458554,
      0.002498314977856353,
      0.002803223003866151,
      0.0025092000141739845,
      0.002466477977577597,
      0.0024558519944548607,
      0.002489910984877497
    ],
    "loaded_inference_baseline_allocated_bytes": 16723968,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63430656,
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
        "time_utc": "2026-10-25T12:07:43.882306+00:00",
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
        "time_utc": "2026-10-25T12:07:44.369579+00:00",
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
        "time_utc": "2026-10-25T12:07:44.397100+00:00",
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
        "time_utc": "2026-10-25T12:07:44.423674+00:00",
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
        "time_utc": "2026-10-25T12:07:44.452144+00:00",
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
        "time_utc": "2026-10-25T12:07:44.478579+00:00",
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
        "time_utc": "2026-10-25T12:07:44.505266+00:00",
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
        "time_utc": "2026-10-25T12:07:44.531336+00:00",
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
        "time_utc": "2026-10-25T12:07:44.557771+00:00",
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
        "time_utc": "2026-10-25T12:07:44.589461+00:00",
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
    "total_benchmark_section_seconds": 0.7213597879745066,
    "forward_latency_seconds_including_warmup": 0.09139633699669503,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_kuairand_R4_EMA_3101/primary_best.pt",
    "checkpoint_sha256": "f16b575a2b00b3c37217c1237908489bccecba7cf0cdde969b1c0067a7091a30",
    "checkpoint_selected_epoch": 14,
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
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1614254,
      "parameters_requires_grad_at_restore": 1614254,
      "parameter_bytes": 6457016,
      "buffer_values": 0,
      "buffer_bytes": 0,
      "modules": [
        "main_model"
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
