# BASE_COMMON 完整推理records 7

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json；SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

原records位置：31–35。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "V3_RESUME_BASELINE_PILOT_kuairand_AUTOIFS_3101",
    "method": "AUTOIFS",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.002424308506306261,
    "p95_seconds": 0.0027566547534661366,
    "mean_seconds": 0.00248748846934177,
    "rows_per_second_at_p50": 1689553.9447002031,
    "rows_per_second_at_mean": 1646640.7987345841,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0033310010039713234,
      0.0024632200074847788,
      0.002430096996249631,
      0.002443198987748474,
      0.002385864994721487
    ],
    "timed_seconds": [
      0.0027567200013436377,
      0.0024477679980918765,
      0.0023838529887143523,
      0.002384634019108489,
      0.0023983110149856657,
      0.002738621027674526,
      0.0024444159935228527,
      0.0024335019988939166,
      0.002412872010609135,
      0.002377572003751993,
      0.002737759001320228,
      0.002475817978847772,
      0.002412767003988847,
      0.00241670900140889,
      0.002407380990916863,
      0.002842686983058229,
      0.00248177201137878,
      0.002396318013779819,
      0.0023841949878260493,
      0.0024072180094663054,
      0.0027565750060603023,
      0.0024590689863543957,
      0.002389137982390821,
      0.002409286011243239,
      0.0024114970001392066,
      0.0027315150073263794,
      0.0024319080112036318,
      0.002432111999951303,
      0.00246112301829271,
      0.002401538018602878
    ],
    "loaded_inference_baseline_allocated_bytes": 19734016,
    "loaded_inference_baseline_reserved_bytes": 48234496,
    "peak_total_allocated_bytes": 92495360,
    "peak_total_reserved_bytes": 121634816,
    "peak_incremental_allocated_bytes": 72761344,
    "peak_incremental_reserved_bytes": 73400320,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:53.270349+00:00",
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
        "time_utc": "2026-10-25T12:07:53.754577+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "352"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.781031+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.806878+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.832638+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.878598+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.922878+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.948643+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.974517+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:53.987506+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "422"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.7313578269968275,
    "forward_latency_seconds_including_warmup": 0.08767803607042879,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_kuairand_AUTOIFS_3101/primary_best.pt",
    "checkpoint_sha256": "895fdb54f70a9042c230ea1795e5554e95018559e90d2bfee2816fc8cad3ea33",
    "checkpoint_selected_epoch": 23,
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
      "parameters_total": 2397359,
      "parameters_requires_grad_at_restore": 1265623,
      "parameter_bytes": 9589436,
      "buffer_values": 2,
      "buffer_bytes": 5,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 2397359,
      "parameters_requires_grad_at_restore": 1265623,
      "parameter_bytes": 9589436,
      "buffer_values": 2,
      "buffer_bytes": 5,
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
    "run_id": "V3_RESUME_BASELINE_PILOT_kuairand_STEM_DOMAIN_TASK_3101",
    "method": "STEM_DOMAIN_TASK",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.005098167501273565,
    "p95_seconds": 0.00529436903889291,
    "mean_seconds": 0.005069759199977852,
    "rows_per_second_at_p50": 803425.9366677899,
    "rows_per_second_at_mean": 807927.9189468987,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.006152415997348726,
      0.005167942988919094,
      0.0050881179922726005,
      0.0050770160160027444,
      0.005091208993690088
    ],
    "timed_seconds": [
      0.005265376006718725,
      0.005099597998196259,
      0.005075909983133897,
      0.0050777620053850114,
      0.005074646993307397,
      0.005290854984195903,
      0.0051202890172135085,
      0.005101360991830006,
      0.005087805009679869,
      0.005090549995657057,
      0.005298319010762498,
      0.005116910004289821,
      0.005106266005896032,
      0.005090446007670835,
      0.005264219973469153,
      0.005291667999699712,
      0.005113616003654897,
      0.005105505988467485,
      0.005096737004350871,
      0.005106446013087407,
      0.005296578980050981,
      0.0051450599858071655,
      0.005051212996477261,
      0.004870700009632856,
      0.00479110999731347,
      0.004993630020180717,
      0.0048168430221267045,
      0.0047511939774267375,
      0.004737884009955451,
      0.004764276003697887
    ],
    "loaded_inference_baseline_allocated_bytes": 26685440,
    "loaded_inference_baseline_reserved_bytes": 35651584,
    "peak_total_allocated_bytes": 222212096,
    "peak_total_reserved_bytes": 241172480,
    "peak_incremental_allocated_bytes": 195526656,
    "peak_incremental_reserved_bytes": 205520896,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:54.078854+00:00",
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
        "time_utc": "2026-10-25T12:07:54.578836+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "340"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.618989+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.657941+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.697035+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.736221+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.775464+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.830738+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.868257+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:54.881005+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "536"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.8164991369994823,
    "forward_latency_seconds_including_warmup": 0.17866947798756883,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_kuairand_STEM_DOMAIN_TASK_3101/primary_best.pt",
    "checkpoint_sha256": "0f9d873fb83c024a760011e18eddefe837be974dfb33f96367e8e20d9bd04424",
    "checkpoint_selected_epoch": 24,
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
      "parameters_total": 4114407,
      "parameters_requires_grad_at_restore": 4114407,
      "parameter_bytes": 16457628,
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
      "parameters_total": 4114407,
      "parameters_requires_grad_at_restore": 4114407,
      "parameter_bytes": 16457628,
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
    "run_id": "V3_RESUME_BASELINE_PILOT_kuairand_STGU_DOMAIN_ADAM_3101",
    "method": "STGU_DOMAIN_ADAM",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0022753925004508346,
    "p95_seconds": 0.002555816894164309,
    "mean_seconds": 0.0023221806351405878,
    "rows_per_second_at_p50": 1800128.9883782421,
    "rows_per_second_at_mean": 1763859.3389407122,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0029610799974761903,
      0.002362553001148626,
      0.002286523988004774,
      0.0022704350121784955,
      0.0022453409910667688
    ],
    "timed_seconds": [
      0.0025032589910551906,
      0.0022784770117141306,
      0.0023093170020729303,
      0.0022752319928258657,
      0.0022362860036082566,
      0.0024601080222055316,
      0.0023464819823857397,
      0.0022458469902630895,
      0.0022586330014746636,
      0.002241993002826348,
      0.002596659993287176,
      0.002272335026646033,
      0.002250278979772702,
      0.0022755530080758035,
      0.0022607459977734834,
      0.002533213992137462,
      0.002283687994349748,
      0.00229200700414367,
      0.0022670640028081834,
      0.0022454020217992365,
      0.002562151988968253,
      0.0022759870043955743,
      0.0022809160000178963,
      0.0022469570103567094,
      0.0022682929993607104,
      0.0025480740005150437,
      0.0022663139970973134,
      0.0022886230144649744,
      0.002261243003886193,
      0.0022342780139297247
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
        "time_utc": "2026-10-25T12:07:54.973099+00:00",
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
        "time_utc": "2026-10-25T12:07:55.401103+00:00",
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
        "time_utc": "2026-10-25T12:07:55.426362+00:00",
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
        "time_utc": "2026-10-25T12:07:55.450814+00:00",
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
        "time_utc": "2026-10-25T12:07:55.475709+00:00",
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
        "time_utc": "2026-10-25T12:07:55.500721+00:00",
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
        "time_utc": "2026-10-25T12:07:55.545879+00:00",
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
        "time_utc": "2026-10-25T12:07:55.570700+00:00",
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
        "time_utc": "2026-10-25T12:07:55.614212+00:00",
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
        "time_utc": "2026-10-25T12:07:55.627121+00:00",
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
    "total_benchmark_section_seconds": 0.6679309039900545,
    "forward_latency_seconds_including_warmup": 0.08179135204409249,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_kuairand_STGU_DOMAIN_ADAM_3101/primary_best.pt",
    "checkpoint_sha256": "3115a02e9a33c105459a407fbb94c352fb5b89d4ccd87282ae734ad818cbefb3",
    "checkpoint_selected_epoch": 14,
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
    "run_id": "V3_RESUME_BASELINE_PILOT_movielens_AUTOIFS_3101",
    "method": "AUTOIFS",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0015752354956930503,
    "p95_seconds": 0.0018973779486259445,
    "mean_seconds": 0.0016361507364005472,
    "rows_per_second_at_p50": 2600246.1290385653,
    "rows_per_second_at_mean": 2503436.822093179,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.0028013180126436055,
      0.001658576977206394,
      0.0015530109812971205,
      0.0015470969956368208,
      0.0015472840168513358
    ],
    "timed_seconds": [
      0.0018512550159357488,
      0.0015908039931673557,
      0.0015430340135935694,
      0.0015800190158188343,
      0.0015446289908140898,
      0.001898543006973341,
      0.0015704660036135465,
      0.0015860910061746836,
      0.0015364220016635954,
      0.001557612995384261,
      0.001966166018974036,
      0.0016429730167146772,
      0.0015518440050072968,
      0.001555441995151341,
      0.0015725450066383928,
      0.001895953988423571,
      0.0015859049744904041,
      0.001584486017236486,
      0.0015594260185025632,
      0.0015699420182500035,
      0.0018955979903694242,
      0.0015704919933341444,
      0.0015629839908797294,
      0.0016929999983403832,
      0.0016080259811133146,
      0.0017920109967235476,
      0.0015779259847477078,
      0.0015615920128766447,
      0.0015430440253112465,
      0.0015362900157924742
    ],
    "loaded_inference_baseline_allocated_bytes": 17480192,
    "loaded_inference_baseline_reserved_bytes": 29360128,
    "peak_total_allocated_bytes": 76233216,
    "peak_total_reserved_bytes": 96468992,
    "peak_incremental_allocated_bytes": 58753024,
    "peak_incremental_reserved_bytes": 67108864,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:55.714779+00:00",
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
        "time_utc": "2026-10-25T12:07:55.917373+00:00",
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
        "time_utc": "2026-10-25T12:07:55.939895+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:55.961523+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.002267+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.042473+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.063928+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.085457+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.106387+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.118810+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "398"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.4179477779834997,
    "forward_latency_seconds_including_warmup": 0.05819180907565169,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_movielens_AUTOIFS_3101/primary_best.pt",
    "checkpoint_sha256": "6d38f9265ca1edab26f9f6c10679699180914ccb12d6b336686f681578173fef",
    "checkpoint_selected_epoch": 23,
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
      "parameters_total": 1848911,
      "parameters_requires_grad_at_restore": 991399,
      "parameter_bytes": 7395644,
      "buffer_values": 2,
      "buffer_bytes": 5,
      "modules": [
        "main_model"
      ],
      "parameter_devices": [
        "cuda:0"
      ]
    },
    "inference_parameter_inventory": {
      "parameters_total": 1848911,
      "parameters_requires_grad_at_restore": 991399,
      "parameter_bytes": 7395644,
      "buffer_values": 2,
      "buffer_bytes": 5,
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
    "run_id": "V3_RESUME_BASELINE_PILOT_movielens_STEM_DOMAIN_TASK_3101",
    "method": "STEM_DOMAIN_TASK",
    "dataset": "movielens",
    "seed": 3101,
    "predictor": "main",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "output": "two Bernoulli pre-sigmoid logits",
    "p50_seconds": 0.0034486280055716634,
    "p95_seconds": 0.0036453029999393037,
    "mean_seconds": 0.0034861849698548516,
    "rows_per_second_at_p50": 1187718.708246419,
    "rows_per_second_at_mean": 1174923.3145740223,
    "percentile_method": "linear interpolation between sorted timing observations",
    "warmup_seconds": [
      0.003923046984709799,
      0.003456478996668011,
      0.003479913022601977,
      0.0034307579917367548,
      0.003430223005125299
    ],
    "timed_seconds": [
      0.0036200320173520595,
      0.0034827759955078363,
      0.0034582900116220117,
      0.0034346650063525885,
      0.0034377170086372644,
      0.003645374992629513,
      0.003446859016548842,
      0.003484659013338387,
      0.0034387010091450065,
      0.0034286919981241226,
      0.003642798023065552,
      0.0034836980048567057,
      0.0034413970133755356,
      0.0034627510176505893,
      0.0034380290016997606,
      0.003610796993598342,
      0.0034503969945944846,
      0.0034320999984629452,
      0.0034527439856901765,
      0.003439969994360581,
      0.0036452150088734925,
      0.0034655010094866157,
      0.0034396889968775213,
      0.003437075996771455,
      0.0034382989979349077,
      0.003669411991722882,
      0.0034569610143080354,
      0.0034377809788566083,
      0.0034277710074093193,
      0.003435396996792406
    ],
    "loaded_inference_baseline_allocated_bytes": 23490048,
    "loaded_inference_baseline_reserved_bytes": 35651584,
    "peak_total_allocated_bytes": 111840256,
    "peak_total_reserved_bytes": 127926272,
    "peak_incremental_allocated_bytes": 88350208,
    "peak_incremental_reserved_bytes": 92274688,
    "memory_scope": "After full identity restore, discarded training state, gradient clearing, GC and empty_cache; peaks include loaded pipeline weights/buffers/input and forward temporaries. Not a theoretical minimum deployment footprint.",
    "all_policy_modules_eval": true,
    "no_grad": true,
    "AMP": false,
    "TF32": false,
    "isolation_checks": [
      {
        "time_utc": "2026-10-25T12:07:56.206278+00:00",
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
        "time_utc": "2026-10-25T12:07:56.438211+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "340"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.469228+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.500083+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.530667+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.561111+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.591726+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.622605+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.670646+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      },
      {
        "time_utc": "2026-10-25T12:07:56.702288+00:00",
        "physical_gpu": 0,
        "self_pid": 1335512,
        "compute_processes": [
          {
            "pid": 1335512,
            "process_name": "/home/xzf/miniconda3/envs/phase0-3-env/bin/python",
            "used_gpu_memory_mib": "428"
          }
        ],
        "other_gpu0_compute_processes": [],
        "isolated": true
      }
    ],
    "total_benchmark_section_seconds": 0.5100771620054729,
    "forward_latency_seconds_including_warmup": 0.12230596909648739,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/V3_RESUME_BASELINE_PILOT_movielens_STEM_DOMAIN_TASK_3101/primary_best.pt",
    "checkpoint_sha256": "8a8e4cf128bed73498bc39fcfdbbaa61bfb33718972bad1a2b2af22871f86db2",
    "checkpoint_selected_epoch": 15,
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
      "parameters_total": 3313047,
      "parameters_requires_grad_at_restore": 3313047,
      "parameter_bytes": 13252188,
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
      "parameters_total": 3313047,
      "parameters_requires_grad_at_restore": 3313047,
      "parameter_bytes": 13252188,
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
