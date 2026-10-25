# BASE_COMMON 完整推理records 8

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json；SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

原records位置：36–40。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "V3_RESUME_BASELINE_PILOT_movielens_STGU_DOMAIN_ADAM_3101",
    "method": "STGU_DOMAIN_ADAM",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0017510585021227598,
    "p95_seconds": 0.002048768902022857,
    "mean_seconds": 0.0018009329030367856,
    "rows_per_second_at_p50": 2339156.5701742875,
    "rows_per_second_at_mean": 2274376.792768462,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0022228300222195685,
      0.0017854470061138272,
      0.00176092199399136,
      0.0017364969826303422,
      0.0017310729890596122
    ],
    "timed_seconds": [
      0.002137189992936328,
      0.001797677978174761,
      0.0017469610029365867,
      0.0017563970177434385,
      0.0017399410135112703,
      0.002097538992529735,
      0.0017782330105546862,
      0.001749741000821814,
      0.001733113022055477,
      0.0017523760034237057,
      0.00194741299492307,
      0.0017416159971617162,
      0.001733338984195143,
      0.0017365549865644425,
      0.001726015005260706,
      0.001989161013625562,
      0.0017545139999128878,
      0.0017424150137230754,
      0.0017331349954474717,
      0.0017279050080105662,
      0.0019339160062372684,
      0.0017611600051168352,
      0.001734195015160367,
      0.0017415289767086506,
      0.0017313920252490789,
      0.0019560999935492873,
      0.0017743100179359317,
      0.0017557440150994807,
      0.001746836001984775,
      0.0017715680005494505
    ],
    "loaded_inference_baseline_allocated_bytes": 12812288,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 53342208,
    "peak_total_reserved_bytes": 77594624,
    "peak_incremental_allocated_bytes": 40529920,
    "peak_incremental_reserved_bytes": 50331648,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:56.793089+00:00",
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
        "time_utc": "2026-10-25T12:07:56.976726+00:00",
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
        "time_utc": "2026-10-25T12:07:57.019174+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.042442+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.064648+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.086655+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.108593+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.130394+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.152294+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:57.165106+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "380"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.38585812199744396,
    "forward_latency_seconds_including_warmup": 0.06326475608511828,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_movielens_STGU_DOMAIN_ADAM_3101/primary_best.pt",
    "checkpoint_sha256": "398f182aff8ca7cdbafff0356f9a69f708ed6177dcae68b92a5a9ce710e20a62",
    "checkpoint_selected_epoch": 14,
    "data_hash": "7347b948841c61e47c181b3b6d8997035edd8d31448f71051afcc74f1a075ea9",
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
      "parameters_total": 772974,
      "parameters_requires_grad_at_restore": 772974,
      "parameter_bytes": 3091896,
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
      "parameters_total": 772974,
      "parameters_requires_grad_at_restore": 772974,
      "parameter_bytes": 3091896,
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
    "run_id": "V3_RESUME_S2_AUX0_kuairand_DIRECT51_3101",
    "method": "DIRECT51",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.007814875003532507,
    "p95_seconds": 0.008390503255941439,
    "mean_seconds": 0.007922729900262009,
    "rows_per_second_at_p50": 524128.66464895627,
    "rows_per_second_at_mean": 516993.51758344594,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.02032845199573785,
      0.007880362012656406,
      0.007882296980824322,
      0.00778437900589779,
      0.007733950013061985
    ],
    "timed_seconds": [
      0.008599492983194068,
      0.00783865901757963,
      0.0077225609857123345,
      0.007801423984346911,
      0.007731721998425201,
      0.008391545008635148,
      0.007976227003382519,
      0.007874673989135772,
      0.00786428598803468,
      0.007768536015646532,
      0.008286812022561207,
      0.007868593005696312,
      0.007713012979365885,
      0.007781825988786295,
      0.007772932993248105,
      0.0083446670032572,
      0.007812905998434871,
      0.0077707470045425,
      0.007915706024505198,
      0.00792852698941715,
      0.00818620398058556,
      0.007799137005349621,
      0.007747708004899323,
      0.007816844008630142,
      0.007773015007842332,
      0.008389230002649128,
      0.008000410016393289,
      0.007762541004922241,
      0.007765704998746514,
      0.007676245993934572
    ],
    "loaded_inference_baseline_allocated_bytes": 17053184,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63759872,
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
        "time_utc": "2026-10-25T12:07:57.250588+00:00",
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
        "time_utc": "2026-10-25T12:07:57.692531+00:00",
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
        "time_utc": "2026-10-25T12:07:57.774826+00:00",
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
        "time_utc": "2026-10-25T12:07:57.828599+00:00",
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
        "time_utc": "2026-10-25T12:07:57.881994+00:00",
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
        "time_utc": "2026-10-25T12:07:57.954022+00:00",
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
        "time_utc": "2026-10-25T12:07:58.007233+00:00",
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
        "time_utc": "2026-10-25T12:07:58.078659+00:00",
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
        "time_utc": "2026-10-25T12:07:58.131437+00:00",
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
        "time_utc": "2026-10-25T12:07:58.162695+00:00",
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
    "total_benchmark_section_seconds": 0.9261333969770931,
    "forward_latency_seconds_including_warmup": 0.2892913370160386,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_S2_AUX0_kuairand_DIRECT51_3101/primary_best.pt",
    "checkpoint_sha256": "743466cbd2d80e75a1b6f3ec222ac5ea7a707003562da10d45db917da2bd2ba7",
    "checkpoint_selected_epoch": 15,
    "data_hash": "a252db1f2f9e576bdf04a4415cd384579f4fe0c98145a0f811550ae6792c1757",
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
      "parameters_total": 1684848,
      "parameters_requires_grad_at_restore": 1640096,
      "parameter_bytes": 6739392,
      "buffer_values": 17795,
      "buffer_bytes": 71180,
      "modules": [
        "main_model",
        "legacy_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1684848,
      "parameters_requires_grad_at_restore": 1640096,
      "parameter_bytes": 6739392,
      "buffer_values": 17795,
      "buffer_bytes": 71180,
      "modules": [
        "main_model",
        "original_CPU_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "V3_RESUME_S2_AUX0_kuairand_K1_DIRECT7_3101",
    "method": "K1_DIRECT7",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.00617532100295648,
    "p95_seconds": 0.0067020922666415576,
    "mean_seconds": 0.006267483469370442,
    "rows_per_second_at_p50": 663285.3576419769,
    "rows_per_second_at_mean": 653531.8393765842,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.009372972999699414,
      0.006300731998635456,
      0.0061768699961248785,
      0.006141851015854627,
      0.006115785014117137
    ],
    "timed_seconds": [
      0.0067022250150330365,
      0.006195190013386309,
      0.006130168010713533,
      0.006105807988205925,
      0.006176934984978288,
      0.0067718009813688695,
      0.006218466005520895,
      0.006145505991298705,
      0.006106102024205029,
      0.006225496996194124,
      0.0066888119908981025,
      0.006170525011839345,
      0.006176052003866062,
      0.006125405983766541,
      0.00610699100070633,
      0.006701930018607527,
      0.0063223320175893605,
      0.006260681024286896,
      0.006116420001490042,
      0.006111276976298541,
      0.006494019005913287,
      0.006204250996233895,
      0.006174590002046898,
      0.006079726008465514,
      0.006165505008539185,
      0.006634794001001865,
      0.006171137007186189,
      0.0061540009919553995,
      0.006173853995278478,
      0.006214503024239093
    ],
    "loaded_inference_baseline_allocated_bytes": 17053184,
    "loaded_inference_baseline_reserved_bytes": 27262976,
    "peak_total_allocated_bytes": 63759872,
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
        "time_utc": "2026-10-25T12:07:58.254178+00:00",
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
        "time_utc": "2026-10-25T12:07:58.694676+00:00",
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
        "time_utc": "2026-10-25T12:07:58.742909+00:00",
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
        "time_utc": "2026-10-25T12:07:58.806400+00:00",
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
        "time_utc": "2026-10-25T12:07:58.851452+00:00",
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
        "time_utc": "2026-10-25T12:07:58.914954+00:00",
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
        "time_utc": "2026-10-25T12:07:58.960193+00:00",
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
        "time_utc": "2026-10-25T12:07:59.004856+00:00",
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
        "time_utc": "2026-10-25T12:07:59.050145+00:00",
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
        "time_utc": "2026-10-25T12:07:59.063065+00:00",
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
    "total_benchmark_section_seconds": 0.8227169479941949,
    "forward_latency_seconds_including_warmup": 0.22213271510554478,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_S2_AUX0_kuairand_K1_DIRECT7_3101/primary_best.pt",
    "checkpoint_sha256": "9492f2b9297d47012775c0b8d2be2bb0d9c5f32f9b5199dafa281e640820f8b1",
    "checkpoint_selected_epoch": 12,
    "data_hash": "a252db1f2f9e576bdf04a4415cd384579f4fe0c98145a0f811550ae6792c1757",
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
      "parameters_total": 1680360,
      "parameters_requires_grad_at_restore": 1635608,
      "parameter_bytes": 6721440,
      "buffer_values": 17795,
      "buffer_bytes": 71180,
      "modules": [
        "main_model",
        "legacy_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1680360,
      "parameters_requires_grad_at_restore": 1635608,
      "parameter_bytes": 6721440,
      "buffer_values": 17795,
      "buffer_bytes": 71180,
      "modules": [
        "main_model",
        "original_CPU_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "V3_RESUME_S2_AUX0_movielens_DIRECT51_3101",
    "method": "DIRECT51",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0072122069977922365,
    "p95_seconds": 0.007859977884800173,
    "mean_seconds": 0.007306068030690464,
    "rows_per_second_at_p50": 567926.0178269767,
    "rows_per_second_at_mean": 560629.874070979,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.008036765997530892,
      0.007308267988264561,
      0.007135299005312845,
      0.007243767002364621,
      0.0071837569994386286
    ],
    "timed_seconds": [
      0.007805903005646542,
      0.0072181329887826,
      0.007247998990351334,
      0.007101671013515443,
      0.00718362900079228,
      0.00789092798368074,
      0.007235999975819141,
      0.007163217989727855,
      0.007218038983410224,
      0.0071205730200745165,
      0.00782214998616837,
      0.007253100018715486,
      0.007266909989994019,
      0.0070879470149520785,
      0.00716220898902975,
      0.007924637000542134,
      0.007290234992979094,
      0.007215343997813761,
      0.007113667001249269,
      0.007152303005568683,
      0.00753580400487408,
      0.007209069997770712,
      0.007437374006258324,
      0.007183251989772543,
      0.00717772901407443,
      0.007726014999207109,
      0.007186787988757715,
      0.007105441007297486,
      0.007028342981357127,
      0.007117629982531071
    ],
    "loaded_inference_baseline_allocated_bytes": 13687808,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 54217728,
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
        "time_utc": "2026-10-25T12:07:59.170999+00:00",
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
        "time_utc": "2026-10-25T12:07:59.366840+00:00",
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
        "time_utc": "2026-10-25T12:07:59.417492+00:00",
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
        "time_utc": "2026-10-25T12:07:59.467248+00:00",
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
        "time_utc": "2026-10-25T12:07:59.517855+00:00",
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
        "time_utc": "2026-10-25T12:07:59.568380+00:00",
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
        "time_utc": "2026-10-25T12:07:59.618803+00:00",
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
        "time_utc": "2026-10-25T12:07:59.669289+00:00",
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
        "time_utc": "2026-10-25T12:07:59.719045+00:00",
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
        "time_utc": "2026-10-25T12:07:59.732333+00:00",
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
    "total_benchmark_section_seconds": 0.5945782580238301,
    "forward_latency_seconds_including_warmup": 0.25608989791362546,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_S2_AUX0_movielens_DIRECT51_3101/primary_best.pt",
    "checkpoint_sha256": "1827aa12db6335d426b6f7a28b32acfd815e3143e60285371cddfc448daeb296",
    "checkpoint_selected_epoch": 14,
    "data_hash": "7347b948841c61e47c181b3b6d8997035edd8d31448f71051afcc74f1a075ea9",
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
      "parameters_total": 995312,
      "parameters_requires_grad_at_restore": 798816,
      "parameter_bytes": 3981248,
      "buffer_values": 4003,
      "buffer_bytes": 16012,
      "modules": [
        "main_model",
        "legacy_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 995312,
      "parameters_requires_grad_at_restore": 798816,
      "parameter_bytes": 3981248,
      "buffer_values": 4003,
      "buffer_bytes": 16012,
      "modules": [
        "main_model",
        "original_CPU_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  },
  {
    "status": "COMPLETE",
    "run_id": "V3_RESUME_S2_AUX0_movielens_K1_DIRECT7_3101",
    "method": "K1_DIRECT7",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005292704998282716,
    "p95_seconds": 0.005804199805425014,
    "mean_seconds": 0.005362499664382388,
    "rows_per_second_at_p50": 773895.3902265474,
    "rows_per_second_at_mean": 763822.8916274899,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.00603074100217782,
      0.005319499992765486,
      0.005277028976706788,
      0.005251169000985101,
      0.005222770996624604
    ],
    "timed_seconds": [
      0.005813341005705297,
      0.005367796984501183,
      0.005317162984283641,
      0.005365365999750793,
      0.005204518995014951,
      0.005567933985730633,
      0.00529492599889636,
      0.005232919997069985,
      0.005230415990808979,
      0.0052098919986747205,
      0.005804927001008764,
      0.00533113197889179,
      0.0052532359841279685,
      0.005180459993425757,
      0.005246347980573773,
      0.005651454994222149,
      0.005365079006878659,
      0.005205671972362325,
      0.00522976802312769,
      0.005257620010524988,
      0.005803311010822654,
      0.005396559019573033,
      0.0053025610104668885,
      0.0053370420064311475,
      0.0052757280063815415,
      0.005629666004097089,
      0.005290483997669071,
      0.005286389001412317,
      0.005216306977672502,
      0.005206972011364996
    ],
    "loaded_inference_baseline_allocated_bytes": 13687808,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 54217728,
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
        "time_utc": "2026-10-25T12:07:59.842941+00:00",
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
        "time_utc": "2026-10-25T12:08:00.054229+00:00",
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
        "time_utc": "2026-10-25T12:08:00.095001+00:00",
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
        "time_utc": "2026-10-25T12:08:00.135621+00:00",
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
        "time_utc": "2026-10-25T12:08:00.176060+00:00",
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
        "time_utc": "2026-10-25T12:08:00.216664+00:00",
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
        "time_utc": "2026-10-25T12:08:00.257075+00:00",
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
        "time_utc": "2026-10-25T12:08:00.298239+00:00",
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
        "time_utc": "2026-10-25T12:08:00.338459+00:00",
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
        "time_utc": "2026-10-25T12:08:00.351326+00:00",
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
    "total_benchmark_section_seconds": 0.5408091880090069,
    "forward_latency_seconds_including_warmup": 0.18797619990073144,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_S2_AUX0_movielens_K1_DIRECT7_3101/primary_best.pt",
    "checkpoint_sha256": "0641214534b54046fdd2381f98f5339c755d3f1cdf6e0b0a4f57b24c60361a59",
    "checkpoint_selected_epoch": 14,
    "data_hash": "7347b948841c61e47c181b3b6d8997035edd8d31448f71051afcc74f1a075ea9",
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
      "parameters_total": 990824,
      "parameters_requires_grad_at_restore": 794328,
      "parameter_bytes": 3963296,
      "buffer_values": 4003,
      "buffer_bytes": 16012,
      "modules": [
        "main_model",
        "legacy_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 990824,
      "parameters_requires_grad_at_restore": 794328,
      "parameter_bytes": 3963296,
      "buffer_values": 4003,
      "buffer_bytes": 16012,
      "modules": [
        "main_model",
        "original_CPU_selector_heads"
      ],
      "parameter_devices": [
        "cpu",
        "cuda:0"
      ]
    },
    "inference_parameter_definition": "Unique parameters of stored modules needed to execute this exact restored deployed pipeline, including resident unused subweights in those modules; not a claim that every weight is arithmetically active for every input.",
    "discarded_training_state": "Trainer, all optimizers/moments, saved gradients, inactive local heads, formation allocator, controllers, AutoIFS rewind/workflow states are released before timed inference; frozen AutoIFS selector embedding remains in the main model."
  }
]
```
