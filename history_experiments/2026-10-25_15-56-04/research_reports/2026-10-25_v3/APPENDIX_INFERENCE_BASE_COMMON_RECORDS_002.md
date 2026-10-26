# BASE_COMMON 完整推理records 2

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json；SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

原records位置：6–10。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "P1_movielens_R5_PB_H1_3101",
    "method": "R5_PB_H1",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.001919061498483643,
    "p95_seconds": 0.0021707460429752244,
    "mean_seconds": 0.001961091733149563,
    "rows_per_second_at_p50": 2134376.622758821,
    "rows_per_second_at_mean": 2088632.5360322234,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0024047399929258972,
      0.001972847996512428,
      0.001903055002912879,
      0.0018959229928441346,
      0.0018992320110555738
    ],
    "timed_seconds": [
      0.0022215030039660633,
      0.0019408199877943844,
      0.0019093680020887405,
      0.0019004750065505505,
      0.0019194819906260818,
      0.0020988209871575236,
      0.0019192259933333844,
      0.0019041230261791497,
      0.0019275519880466163,
      0.0018983699847012758,
      0.002153547015041113,
      0.0019188970036339015,
      0.001902798016089946,
      0.0019120379874948412,
      0.001916203007567674,
      0.002184817974921316,
      0.0019231089972890913,
      0.0019125689868815243,
      0.0018965369963552803,
      0.0018923169991467148,
      0.002114728995366022,
      0.001912214996991679,
      0.001895761990454048,
      0.0019234940118622035,
      0.0019199600210413337,
      0.0021412539936136454,
      0.0019377600110601634,
      0.0019235330109950155,
      0.001900822011521086,
      0.0019106499967165291
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
        "time_utc": "2026-10-25T12:07:34.748742+00:00",
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
        "time_utc": "2026-10-25T12:07:34.959185+00:00",
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
        "time_utc": "2026-10-25T12:07:34.982993+00:00",
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
        "time_utc": "2026-10-25T12:07:35.006550+00:00",
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
        "time_utc": "2026-10-25T12:07:35.029975+00:00",
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
        "time_utc": "2026-10-25T12:07:35.053197+00:00",
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
        "time_utc": "2026-10-25T12:07:35.076421+00:00",
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
        "time_utc": "2026-10-25T12:07:35.099443+00:00",
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
        "time_utc": "2026-10-25T12:07:35.123010+00:00",
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
        "time_utc": "2026-10-25T12:07:35.154674+00:00",
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
    "total_benchmark_section_seconds": 0.4201649319729768,
    "forward_latency_seconds_including_warmup": 0.06890854999073781,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_R5_PB_H1_3101/primary_best.pt",
    "checkpoint_sha256": "7c2e5e44958b2f1afa1f6edc64332844446ddab3a0de1259f97446f2a00eb5c6",
    "checkpoint_selected_epoch": 13,
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
    "run_id": "P1_movielens_R6_PB_HB_3101",
    "method": "R6_PB_HB",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0019229629979236051,
    "p95_seconds": 0.002198236997355707,
    "mean_seconds": 0.0019699433663239083,
    "rows_per_second_at_p50": 2130046.1862359373,
    "rows_per_second_at_mean": 2079247.5915911759,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.002417326992144808,
      0.0019460819894447923,
      0.0018934410181827843,
      0.0018927350174635649,
      0.0018909559876192361
    ],
    "timed_seconds": [
      0.0021840870031155646,
      0.0019186160061508417,
      0.001889715000288561,
      0.0019401200115680695,
      0.001887180987978354,
      0.0022053199936635792,
      0.0019409120141062886,
      0.0019227929878979921,
      0.0019284859881736338,
      0.0019220359972678125,
      0.0023182749864645302,
      0.0019231330079492182,
      0.001908095000544563,
      0.0019235789950471371,
      0.001886159006971866,
      0.0021895800018683076,
      0.0019237409869674593,
      0.0019003539928235114,
      0.001965749019291252,
      0.0018874089873861521,
      0.002111189009156078,
      0.0019648259913083166,
      0.001894507004180923,
      0.0019145830010529608,
      0.0018858340044971555,
      0.0021615379955619574,
      0.0019290170166641474,
      0.0018920449947472662,
      0.0018928369972854853,
      0.001886584999738261
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
        "time_utc": "2026-10-25T12:07:35.239390+00:00",
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
        "time_utc": "2026-10-25T12:07:35.436951+00:00",
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
        "time_utc": "2026-10-25T12:07:35.460997+00:00",
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
        "time_utc": "2026-10-25T12:07:35.484617+00:00",
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
        "time_utc": "2026-10-25T12:07:35.508425+00:00",
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
        "time_utc": "2026-10-25T12:07:35.532459+00:00",
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
        "time_utc": "2026-10-25T12:07:35.555881+00:00",
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
        "time_utc": "2026-10-25T12:07:35.579072+00:00",
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
        "time_utc": "2026-10-25T12:07:35.618896+00:00",
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
        "time_utc": "2026-10-25T12:07:35.632249+00:00",
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
    "total_benchmark_section_seconds": 0.40682866098359227,
    "forward_latency_seconds_including_warmup": 0.06913884199457243,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_R6_PB_HB_3101/primary_best.pt",
    "checkpoint_sha256": "0a86c1e99d855f68d78c200fc964190c78d9567e47f7494f9a078cafda31e21f",
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
    "run_id": "P1_movielens_L0_OWN_3101",
    "method": "L0_OWN",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0031023324991110712,
    "p95_seconds": 0.0035075195075478403,
    "mean_seconds": 0.003178129370401924,
    "rows_per_second_at_p50": 1320296.906013024,
    "rows_per_second_at_mean": 1288808.453849063,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.015452097984962165,
      0.003312350978376344,
      0.0033380390086676925,
      0.0030371560133062303,
      0.0031017120054457337
    ],
    "timed_seconds": [
      0.003457174025243148,
      0.0031604150135535747,
      0.0030660420015919954,
      0.0030522260058205575,
      0.003096440021181479,
      0.0035037409979850054,
      0.003204085020115599,
      0.0030861080158501863,
      0.003067637007916346,
      0.003082676004851237,
      0.0035106110153719783,
      0.003138201020192355,
      0.0030790370074100792,
      0.0030853099888190627,
      0.003105315990978852,
      0.0034705860016401857,
      0.003180976986186579,
      0.003087309974944219,
      0.0030993490072432905,
      0.0031117960170377046,
      0.003326644975459203,
      0.0030918149859644473,
      0.0031629780132789165,
      0.003090173995587975,
      0.003074536012718454,
      0.003532253991579637,
      0.0031280989933293313,
      0.0031308270117733628,
      0.003067328012548387,
      0.0030941879958845675
    ],
    "loaded_inference_baseline_allocated_bytes": 12915712,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53445632,
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
        "time_utc": "2026-10-25T12:07:35.722250+00:00",
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
        "time_utc": "2026-10-25T12:07:35.936501+00:00",
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
        "time_utc": "2026-10-25T12:07:35.978909+00:00",
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
        "time_utc": "2026-10-25T12:07:36.009236+00:00",
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
        "time_utc": "2026-10-25T12:07:36.039078+00:00",
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
        "time_utc": "2026-10-25T12:07:36.069105+00:00",
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
        "time_utc": "2026-10-25T12:07:36.098723+00:00",
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
        "time_utc": "2026-10-25T12:07:36.146954+00:00",
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
        "time_utc": "2026-10-25T12:07:36.176812+00:00",
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
        "time_utc": "2026-10-25T12:07:36.206600+00:00",
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
    "total_benchmark_section_seconds": 0.49853757899836637,
    "forward_latency_seconds_including_warmup": 0.12358523710281588,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L0_OWN_3101/primary_best.pt",
    "checkpoint_sha256": "fa3b7152acf17d38dd5b428a5b7864a012928977b07d6cc9bde3d220623e2509",
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
      "parameters_total": 793608,
      "parameters_requires_grad_at_restore": 793608,
      "parameter_bytes": 3174432,
      "buffer_values": 8951,
      "buffer_bytes": 35795,
      "modules": [
        "main_model",
        "read_policy",
        "local_mono",
        "local_feature"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 789486,
      "parameters_requires_grad_at_restore": 789486,
      "parameter_bytes": 3157944,
      "buffer_values": 8192,
      "buffer_bytes": 32768,
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
    "run_id": "P1_movielens_L0_OWN_3101",
    "method": "L0_OWN",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "mono",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0035346550139365718,
    "p95_seconds": 0.00395717330975458,
    "mean_seconds": 0.003600581201802318,
    "rows_per_second_at_p50": 1158811.817235384,
    "rows_per_second_at_mean": 1137594.1189577098,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.004932426993036643,
      0.003634783992310986,
      0.0035244160098955035,
      0.0035670140059664845,
      0.0035254810063634068
    ],
    "timed_seconds": [
      0.0040047059883363545,
      0.00350508198607713,
      0.003509275004034862,
      0.003476011974271387,
      0.0034280859981663525,
      0.0037638999929185957,
      0.003486191009869799,
      0.0034685170103330165,
      0.0034844900073949248,
      0.0034464250202290714,
      0.003961010021157563,
      0.003580115007935092,
      0.0035423070075921714,
      0.003519046993460506,
      0.0035350240068510175,
      0.003935854998417199,
      0.0035411800199653953,
      0.003521393024129793,
      0.003520385973388329,
      0.0035069490259047598,
      0.003952483995817602,
      0.003538210003171116,
      0.003528112982166931,
      0.003565596998669207,
      0.0035372629936318845,
      0.003902626020135358,
      0.0036140279844403267,
      0.003534286021022126,
      0.0035200669954065233,
      0.0035888119891751558
    ],
    "loaded_inference_baseline_allocated_bytes": 12920832,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53450752,
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
        "time_utc": "2026-10-25T12:07:36.293230+00:00",
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
        "time_utc": "2026-10-25T12:07:36.538306+00:00",
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
        "time_utc": "2026-10-25T12:07:36.570907+00:00",
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
        "time_utc": "2026-10-25T12:07:36.602281+00:00",
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
        "time_utc": "2026-10-25T12:07:36.633934+00:00",
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
        "time_utc": "2026-10-25T12:07:36.665648+00:00",
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
        "time_utc": "2026-10-25T12:07:36.697374+00:00",
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
        "time_utc": "2026-10-25T12:07:36.746593+00:00",
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
        "time_utc": "2026-10-25T12:07:36.778020+00:00",
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
        "time_utc": "2026-10-25T12:07:36.790862+00:00",
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
    "total_benchmark_section_seconds": 0.5125241409987211,
    "forward_latency_seconds_including_warmup": 0.12720155806164257,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L0_OWN_3101/mono_primary_best.pt",
    "checkpoint_sha256": "fa3b7152acf17d38dd5b428a5b7864a012928977b07d6cc9bde3d220623e2509",
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
      "parameters_total": 793608,
      "parameters_requires_grad_at_restore": 793608,
      "parameter_bytes": 3174432,
      "buffer_values": 8951,
      "buffer_bytes": 35795,
      "modules": [
        "main_model",
        "read_policy",
        "local_mono",
        "local_feature"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 789498,
      "parameters_requires_grad_at_restore": 789498,
      "parameter_bytes": 3157992,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_local_head"
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
    "run_id": "P1_movielens_L0_OWN_3101",
    "method": "L0_OWN",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "feature",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0036997769930167124,
    "p95_seconds": 0.004025110011571087,
    "mean_seconds": 0.0037449370633112268,
    "rows_per_second_at_p50": 1107093.7539562937,
    "rows_per_second_at_mean": 1093743.3475526469,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.005883934005396441,
      0.003734249999979511,
      0.0036472869978751987,
      0.0036339339858386666,
      0.003633308981079608
    ],
    "timed_seconds": [
      0.00399863300845027,
      0.0037314670043997467,
      0.0036545780021697283,
      0.003712920006364584,
      0.0037169389834161848,
      0.003976688982220367,
      0.0037048549856990576,
      0.0037029259838163853,
      0.0036636220174841583,
      0.00374364698654972,
      0.004124741011764854,
      0.0036228119861334562,
      0.0036603619810193777,
      0.0037122289941180497,
      0.003622391988756135,
      0.004046773014124483,
      0.0036534119863063097,
      0.0036927270120941103,
      0.003750509989913553,
      0.0036413189955055714,
      0.0039016430091578513,
      0.0036966280022170395,
      0.003622485004598275,
      0.0036909119808115065,
      0.003650330996606499,
      0.003973206999944523,
      0.0038939680089242756,
      0.0036066390166524798,
      0.003579816984711215,
      0.003598928975407034
    ],
    "loaded_inference_baseline_allocated_bytes": 12945408,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 53475328,
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
        "time_utc": "2026-10-25T12:07:36.875046+00:00",
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
        "time_utc": "2026-10-25T12:07:37.087938+00:00",
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
        "time_utc": "2026-10-25T12:07:37.122296+00:00",
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
        "time_utc": "2026-10-25T12:07:37.154644+00:00",
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
        "time_utc": "2026-10-25T12:07:37.186972+00:00",
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
        "time_utc": "2026-10-25T12:07:37.219398+00:00",
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
        "time_utc": "2026-10-25T12:07:37.252194+00:00",
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
        "time_utc": "2026-10-25T12:07:37.302598+00:00",
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
        "time_utc": "2026-10-25T12:07:37.335024+00:00",
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
        "time_utc": "2026-10-25T12:07:37.348078+00:00",
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
    "total_benchmark_section_seconds": 0.48703302498324774,
    "forward_latency_seconds_including_warmup": 0.13288082586950622,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P1_movielens_L0_OWN_3101/feature_primary_best.pt",
    "checkpoint_sha256": "fa3b7152acf17d38dd5b428a5b7864a012928977b07d6cc9bde3d220623e2509",
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
      "parameters_total": 793608,
      "parameters_requires_grad_at_restore": 793608,
      "parameter_bytes": 3174432,
      "buffer_values": 8951,
      "buffer_bytes": 35795,
      "modules": [
        "main_model",
        "read_policy",
        "local_mono",
        "local_feature"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 793596,
      "parameters_requires_grad_at_restore": 793596,
      "parameter_bytes": 3174384,
      "buffer_values": 8445,
      "buffer_bytes": 33777,
      "modules": [
        "main_model",
        "deployed_local_head"
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
