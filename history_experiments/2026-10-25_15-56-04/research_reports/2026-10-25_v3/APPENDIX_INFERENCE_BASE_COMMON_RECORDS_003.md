# BASE_COMMON 完整推理records 3

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json；SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

原records位置：11–15。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "P1_movielens_L1_FULL_COARSE_3101",
    "method": "L1_FULL_COARSE",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005269927496556193,
    "p95_seconds": 0.00572793824103428,
    "mean_seconds": 0.005338972266569423,
    "rows_per_second_at_p50": 777240.294610631,
    "rows_per_second_at_mean": 767188.8512415707,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.007217525009764358,
      0.005214150995016098,
      0.005237822013441473,
      0.005144644994288683,
      0.0052028299833182245
    ],
    "timed_seconds": [
      0.00571954800398089,
      0.005236863013124093,
      0.005295993003528565,
      0.005329467006959021,
      0.005193993012653664,
      0.0057348029804416,
      0.005188421986531466,
      0.005207863025134429,
      0.005222954001510516,
      0.005207698995945975,
      0.00562162499409169,
      0.005312736990163103,
      0.005300458986312151,
      0.005449165008030832,
      0.005245334003120661,
      0.005635843001073226,
      0.005185056972550228,
      0.0052977720042690635,
      0.005183005996514112,
      0.005184564972296357,
      0.005756723025115207,
      0.005372598010580987,
      0.005288406013278291,
      0.005261090002022684,
      0.005244739993941039,
      0.005678693996742368,
      0.005144474998814985,
      0.005278764991089702,
      0.005222400999628007,
      0.005168108007637784
    ],
    "loaded_inference_baseline_allocated_bytes": 12942336,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53472256,
    "peak_total_reserved_bytes": 79691776,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:37.439018+00:00",
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
        "time_utc": "2026-10-25T12:07:37.659184+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "334"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.700922+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.741514+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.781843+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.822765+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.863018+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.903175+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.942952+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:37.955833+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.5310875079885591,
    "forward_latency_seconds_including_warmup": 0.18818614099291153,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L1_FULL_COARSE_3101/primary_best.pt",
    "checkpoint_sha256": "295bf3f5c02d204d905180fc5799a5f438fc553c8fdcd7c7dc0c9040d64d69dd",
    "checkpoint_selected_epoch": 12,
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 793050,
      "parameters_requires_grad_at_restore": 793050,
      "parameter_bytes": 3172200,
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
      "parameters_total": 793050,
      "parameters_requires_grad_at_restore": 793050,
      "parameter_bytes": 3172200,
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
    "run_id": "P1_movielens_L2_BLOCK_UNIFORM_3101",
    "method": "L2_BLOCK_UNIFORM",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005229140500887297,
    "p95_seconds": 0.0055894393386552105,
    "mean_seconds": 0.005288323796897506,
    "rows_per_second_at_p50": 783302.7242823133,
    "rows_per_second_at_mean": 774536.5369652659,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.005730868986574933,
      0.0052694140176754445,
      0.005160871980478987,
      0.0051235339778941125,
      0.005105382006149739
    ],
    "timed_seconds": [
      0.00554453898803331,
      0.00538273699930869,
      0.005267630011076108,
      0.005363149975892156,
      0.005386403005104512,
      0.0054296349990181625,
      0.005182958993827924,
      0.005149200995219871,
      0.005147797986865044,
      0.005112971004564315,
      0.005648379010381177,
      0.005234039999777451,
      0.005279285978758708,
      0.005224241001997143,
      0.005331876978743821,
      0.005393759987782687,
      0.0051863490079995245,
      0.005074127984698862,
      0.005205213004956022,
      0.005161737994058058,
      0.005494192999321967,
      0.005221111001446843,
      0.0051945309969596565,
      0.005147453979589045,
      0.005125079012941569,
      0.0056261759891640395,
      0.005294547008816153,
      0.005146737006725743,
      0.00551437700050883,
      0.005179475003387779
    ],
    "loaded_inference_baseline_allocated_bytes": 12942336,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53472256,
    "peak_total_reserved_bytes": 79691776,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:38.045965+00:00",
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
        "time_utc": "2026-10-25T12:07:38.274509+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "334"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.315150+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.356000+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.396185+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.436692+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.476566+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.517254+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.557698+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.570599+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.5384886819811072,
    "forward_latency_seconds_including_warmup": 0.1850397848756984,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L2_BLOCK_UNIFORM_3101/primary_best.pt",
    "checkpoint_sha256": "45c88e438c2dbc06a093a4cda08cded878c779e454e9c504208adfa4584b765a",
    "checkpoint_selected_epoch": 12,
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 793662,
      "parameter_bytes": 3174648,
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
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 793662,
      "parameter_bytes": 3174648,
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
    "run_id": "P1_movielens_L3_SCALAR_META_3101",
    "method": "L3_SCALAR_META",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.00519509450532496,
    "p95_seconds": 0.005669942189706489,
    "mean_seconds": 0.005265142534820673,
    "rows_per_second_at_p50": 788436.0902004015,
    "rows_per_second_at_mean": 777946.6506198786,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.005796291021397337,
      0.005341144977137446,
      0.005102593015180901,
      0.005223792017204687,
      0.005146188021171838
    ],
    "timed_seconds": [
      0.005438650987343863,
      0.005196787999011576,
      0.005154384998604655,
      0.0051597069832496345,
      0.00511954800458625,
      0.005675280990544707,
      0.005245877982815728,
      0.005175525991944596,
      0.005220088001806289,
      0.005143592017702758,
      0.0058165930095128715,
      0.005166888993699104,
      0.0052017190027982,
      0.00510024800314568,
      0.005205200024647638,
      0.005663416988682002,
      0.005193401011638343,
      0.005167057999642566,
      0.0050803349877242,
      0.005282964004436508,
      0.0055450969957746565,
      0.005191638018004596,
      0.005234481010120362,
      0.005123334005475044,
      0.005236833996605128,
      0.0055505089985672385,
      0.005140236025908962,
      0.005217225989326835,
      0.005138859007274732,
      0.0051687940140254796
    ],
    "loaded_inference_baseline_allocated_bytes": 12942336,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53472256,
    "peak_total_reserved_bytes": 79691776,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:38.657081+00:00",
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
        "time_utc": "2026-10-25T12:07:38.869035+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "334"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.909477+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:38.949778+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.007080+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.047705+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.087551+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.146677+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.206484+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.219284+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.5764277229900472,
    "forward_latency_seconds_including_warmup": 0.1845642850967124,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L3_SCALAR_META_3101/primary_best.pt",
    "checkpoint_sha256": "1b1a68352a75482781a18405430038aa5c1c5a0cf165ed2d75293acea120b72f",
    "checkpoint_selected_epoch": 12,
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 795663,
      "parameters_requires_grad_at_restore": 795663,
      "parameter_bytes": 3182652,
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
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 793662,
      "parameter_bytes": 3174648,
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
    "run_id": "P1_movielens_L4_BLOCK_META_3101",
    "method": "L4_BLOCK_META",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005239522011834197,
    "p95_seconds": 0.005661584857443813,
    "mean_seconds": 0.005297824666680147,
    "rows_per_second_at_p50": 781750.6999204523,
    "rows_per_second_at_mean": 773147.5195397012,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0058682730013970286,
      0.005261602986138314,
      0.005261996993795037,
      0.005358332971809432,
      0.005166453978745267
    ],
    "timed_seconds": [
      0.005718506989069283,
      0.005280956014757976,
      0.005268509994493797,
      0.0052744229906238616,
      0.005261612008325756,
      0.005580614990321919,
      0.005326561018591747,
      0.005386887991335243,
      0.005127769982209429,
      0.0051352349983062595,
      0.005600800999673083,
      0.005164388014236465,
      0.0051427099970169365,
      0.005188050010474399,
      0.005170788004761562,
      0.005694860999938101,
      0.005173417011974379,
      0.005169293988728896,
      0.005151238990947604,
      0.0051149019855074584,
      0.005620914016617462,
      0.005279788980260491,
      0.005132410995429382,
      0.0052015919936820865,
      0.005107823002617806,
      0.00560486302128993,
      0.005294538015732542,
      0.005217432015342638,
      0.005343262979295105,
      0.0052005879988428205
    ],
    "loaded_inference_baseline_allocated_bytes": 12942336,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53472256,
    "peak_total_reserved_bytes": 79691776,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:39.308143+00:00",
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
        "time_utc": "2026-10-25T12:07:39.520936+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "334"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.581559+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.622270+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.662919+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.722965+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.781766+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.822627+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.863433+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:39.876574+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.5828904680092819,
    "forward_latency_seconds_including_warmup": 0.1858513999322895,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L4_BLOCK_META_3101/primary_best.pt",
    "checkpoint_sha256": "4611351fe1adaaf6e0a60f15bc8ba12248db835f0984b03d4b30d64610060099",
    "checkpoint_selected_epoch": 12,
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 795654,
      "parameters_requires_grad_at_restore": 795654,
      "parameter_bytes": 3182616,
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
      "parameters_total": 793662,
      "parameters_requires_grad_at_restore": 793662,
      "parameter_bytes": 3174648,
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
    "run_id": "P1_movielens_L5_BLOCK_CONSTANT_3101",
    "method": "L5_BLOCK_CONSTANT",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.004671078495448455,
    "p95_seconds": 0.004980321000039111,
    "mean_seconds": 0.004686043565743603,
    "rows_per_second_at_p50": 876885.2854841088,
    "rows_per_second_at_mean": 874084.9167393576,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.005260735022602603,
      0.0047540320083498955,
      0.004735866008559242,
      0.0047178609820548445,
      0.0046824780001770705
    ],
    "timed_seconds": [
      0.0051591139927040786,
      0.0047273520031012595,
      0.004679721023421735,
      0.004669554997235537,
      0.004675766977015883,
      0.004880329972365871,
      0.004685856023570523,
      0.00468302599620074,
      0.004666861001169309,
      0.004755745001602918,
      0.005019876000005752,
      0.004588951007463038,
      0.004609067982528359,
      0.004672889015637338,
      0.004638194979634136,
      0.004922685009660199,
      0.004680664016632363,
      0.004672601993661374,
      0.004650521994335577,
      0.004662618011934683,
      0.004931976000079885,
      0.004612591990735382,
      0.004528544988716021,
      0.0045394059852696955,
      0.004557353007839993,
      0.004820728994673118,
      0.004509567021159455,
      0.004499335977016017,
      0.004451188986422494,
      0.0044292170205153525
    ],
    "loaded_inference_baseline_allocated_bytes": 12917760,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53447680,
    "peak_total_reserved_bytes": 79691776,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:39.979255+00:00",
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
        "time_utc": "2026-10-25T12:07:40.191269+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "334"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.246535+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.284779+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.322428+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.360188+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.397891+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.434441+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.491359+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:40.505114+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "382"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.5567200080258772,
    "forward_latency_seconds_including_warmup": 0.16473227899405174,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L5_BLOCK_CONSTANT_3101/primary_best.pt",
    "checkpoint_sha256": "53b55c8f7778b41bb529cb6ddee40a605f16ae97d23791cccf039020926e3485",
    "checkpoint_selected_epoch": 14,
    "data_hash": "8e2a30a9b8b352a2450210baa621fef6d55688d2c02aba70b8237ace2f49b093",
    "source_hash_at_training": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
    "fit_input_rows": 4096,
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
    "restored_training_parameter_inventory": {
      "parameters_total": 791478,
      "parameters_requires_grad_at_restore": 791478,
      "parameter_bytes": 3165912,
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
      "parameters_total": 789486,
      "parameters_requires_grad_at_restore": 789486,
      "parameter_bytes": 3157944,
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
