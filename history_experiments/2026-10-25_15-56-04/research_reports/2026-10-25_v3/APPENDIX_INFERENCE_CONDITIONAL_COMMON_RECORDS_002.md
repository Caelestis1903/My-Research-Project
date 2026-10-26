# CONDITIONAL_COMMON 完整推理records 2

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

源收据：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_conditional/benchmark_receipt.json；SHA256：459c19604423892d52529ab15d40fe2dd91f0e27c9503c7d9d7fd9238e2a21c3

原records位置：6–10。完整保留5次warmup、30次计时、P50/P95、模块/参数、checkpoint/source和GPU0隔离检查。

```json
[
  {
    "status": "COMPLETE",
    "run_id": "P2_kuairand_LPS_GEOM_3101",
    "method": "LPS_GEOM",
    "dataset": "kuairand",
    "seed": 3101,
    "predictor": "main",
    "stage": "p2",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.004724922982859425,
    "p95_seconds": 0.0055378377976012414,
    "mean_seconds": 0.0048873497299306715,
    "rows_per_second_at_p50": 866892.437159089,
    "rows_per_second_at_mean": 838082.0334824091,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.009478360996581614,
      0.006011342979036272,
      0.005259933997876942,
      0.004782433999935165,
      0.004677365999668837
    ],
    "timed_seconds": [
      0.006261248985538259,
      0.005310342996381223,
      0.004880616994341835,
      0.004647308000130579,
      0.004738024988910183,
      0.005601087992545217,
      0.005049831001088023,
      0.00470151798799634,
      0.004719825985375792,
      0.004673268005717546,
      0.005460532003780827,
      0.004919249011436477,
      0.0046581509814132005,
      0.004658404999645427,
      0.0046625300019513816,
      0.005399921996286139,
      0.0048636240244377404,
      0.00473993198829703,
      0.0046294899948406965,
      0.004639327002223581,
      0.005306620994815603,
      0.004806245007785037,
      0.004664045001845807,
      0.004681821999838576,
      0.004616889986209571,
      0.004961425001965836,
      0.004730019980343059,
      0.004550197976641357,
      0.004544840019661933,
      0.00454415098647587
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
        "time_utc": "2026-10-25T12:08:17.831563+00:00",
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
        "time_utc": "2026-10-25T12:08:18.151566+00:00",
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
        "time_utc": "2026-10-25T12:08:18.195152+00:00",
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
        "time_utc": "2026-10-25T12:08:18.233868+00:00",
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
        "time_utc": "2026-10-25T12:08:18.270920+00:00",
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
        "time_utc": "2026-10-25T12:08:18.307833+00:00",
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
        "time_utc": "2026-10-25T12:08:18.344499+00:00",
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
        "time_utc": "2026-10-25T12:08:18.381155+00:00",
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
        "time_utc": "2026-10-25T12:08:18.416735+00:00",
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
        "time_utc": "2026-10-25T12:08:18.457047+00:00",
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
    "total_benchmark_section_seconds": 0.6401641810080037,
    "forward_latency_seconds_including_warmup": 0.17682992987101898,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/P2_kuairand_LPS_GEOM_3101/primary_best.pt",
    "checkpoint_sha256": "c236bb2b9574aef7f900205ac9adb7d48025c052d09b5a9fce2eb6043cd3b2de",
    "checkpoint_selected_epoch": 19,
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
        "parameters_total": 0,
        "parameters_requires_grad_at_restore": 0,
        "parameter_bytes": 0
      }
    },
    "restored_training_parameter_inventory": {
      "parameters_total": 1634942,
      "parameters_requires_grad_at_restore": 1634942
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
  },
  {
    "status": "COMPLETE",
    "run_id": "FORMAL_kuairand_R6_PB_HB_4101",
    "method": "R6_PB_HB",
    "dataset": "kuairand",
    "seed": 4101,
    "predictor": "main",
    "stage": "formal",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.005267123007797636,
    "p95_seconds": 0.00699633800832089,
    "mean_seconds": 0.004740798599474753,
    "rows_per_second_at_p50": 777654.1375502596,
    "rows_per_second_at_mean": 863989.455374419,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.00281924000591971,
      0.0022668720048386604,
      0.002219170011812821,
      0.002273714984767139,
      0.0022034169815015048
    ],
    "timed_seconds": [
      0.002479823015164584,
      0.0024077599809970707,
      0.002490983984898776,
      0.0026782319764606655,
      0.002993732981849462,
      0.006994732015300542,
      0.006147918000351638,
      0.005567830987274647,
      0.004657992016291246,
      0.0035406039969529957,
      0.006962723011383787,
      0.006373584008542821,
      0.00593943300191313,
      0.004540144989732653,
      0.0037612020096275955,
      0.005748941010097042,
      0.0026704019983299077,
      0.0022380310110747814,
      0.0022034540015738457,
      0.002204638993134722,
      0.006997652002610266,
      0.006333296012599021,
      0.005584155995165929,
      0.005300871009239927,
      0.003744527988601476,
      0.007024321996141225,
      0.0063959780090954155,
      0.00632041398785077,
      0.006687202985631302,
      0.005233375006355345
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
        "time_utc": "2026-10-25T12:08:18.532195+00:00",
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
        "time_utc": "2026-10-25T12:08:18.818706+00:00",
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
        "time_utc": "2026-10-25T12:08:18.843093+00:00",
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
        "time_utc": "2026-10-25T12:08:18.876364+00:00",
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
        "time_utc": "2026-10-25T12:08:18.925629+00:00",
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
        "time_utc": "2026-10-25T12:08:18.974383+00:00",
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
        "time_utc": "2026-10-25T12:08:19.010898+00:00",
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
        "time_utc": "2026-10-25T12:08:19.081331+00:00",
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
        "time_utc": "2026-10-25T12:08:19.139264+00:00",
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
        "time_utc": "2026-10-25T12:08:19.177898+00:00",
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
    "total_benchmark_section_seconds": 0.6603945920069236,
    "forward_latency_seconds_including_warmup": 0.15400637197308242,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R6_PB_HB_4101/primary_best.pt",
    "checkpoint_sha256": "cdd423b42557a0956ddf9f554e6582c88bd8614a16b0b7f29647a1735afa29e1",
    "checkpoint_selected_epoch": 13,
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
  },
  {
    "status": "COMPLETE",
    "run_id": "FORMAL_kuairand_R0_BASE_4101",
    "method": "R0_BASE",
    "dataset": "kuairand",
    "seed": 4101,
    "predictor": "main",
    "stage": "formal",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.003222841987735592,
    "p95_seconds": 0.006721556407865135,
    "mean_seconds": 0.0036943083667817216,
    "rows_per_second_at_p50": 1270927.962210738,
    "rows_per_second_at_mean": 1108732.5673271313,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.0028099229966755956,
      0.0023024719848763198,
      0.002229164994787425,
      0.00226051002391614,
      0.0022482629865407944
    ],
    "timed_seconds": [
      0.0025275530060753226,
      0.0023844910028856248,
      0.0023235899861902,
      0.0022759440180379897,
      0.002237982989754528,
      0.0028700680122710764,
      0.002643442014232278,
      0.0025336490070912987,
      0.002462926000589505,
      0.002388814988080412,
      0.0039138339925557375,
      0.003379553003469482,
      0.0028435050044208765,
      0.0026975609944202006,
      0.002578535000793636,
      0.0060250950045883656,
      0.004490024002734572,
      0.0038981410034466535,
      0.0031867349753156304,
      0.002784395997878164,
      0.007092549989465624,
      0.005512914009159431,
      0.004443326004547998,
      0.003564996994100511,
      0.003322856005979702,
      0.006975090014748275,
      0.006411681999452412,
      0.005555447976803407,
      0.004245599004207179,
      0.0032589490001555532
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
        "time_utc": "2026-10-25T12:08:19.257746+00:00",
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
        "time_utc": "2026-10-25T12:08:19.537967+00:00",
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
        "time_utc": "2026-10-25T12:08:19.562726+00:00",
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
        "time_utc": "2026-10-25T12:08:19.587265+00:00",
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
        "time_utc": "2026-10-25T12:08:19.614594+00:00",
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
        "time_utc": "2026-10-25T12:08:19.647173+00:00",
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
        "time_utc": "2026-10-25T12:08:19.687003+00:00",
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
        "time_utc": "2026-10-25T12:08:19.731767+00:00",
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
        "time_utc": "2026-10-25T12:08:19.780316+00:00",
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
        "time_utc": "2026-10-25T12:08:19.807806+00:00",
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
    "total_benchmark_section_seconds": 0.5647078540059738,
    "forward_latency_seconds_including_warmup": 0.12267958399024792,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R0_BASE_4101/primary_best.pt",
    "checkpoint_sha256": "5b2fbf272b9b69d4b9aa774d222c4bc00ed56fffc66992c41195ff89e88659f8",
    "checkpoint_selected_epoch": 15,
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
  },
  {
    "status": "COMPLETE",
    "run_id": "FORMAL_kuairand_R5_PB_H1_4101",
    "method": "R5_PB_H1",
    "dataset": "kuairand",
    "seed": 4101,
    "predictor": "main",
    "stage": "formal",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.0049180869973497465,
    "p95_seconds": 0.007042630240903236,
    "mean_seconds": 0.00475178716199783,
    "rows_per_second_at_p50": 832844.1530634265,
    "rows_per_second_at_mean": 861991.4698110948,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.004765486984979361,
      0.003117504995316267,
      0.002969310007756576,
      0.002827402000548318,
      0.0027967220230493695
    ],
    "timed_seconds": [
      0.005191880016354844,
      0.00409791199490428,
      0.003359467984410003,
      0.0030680499912705272,
      0.0031437809811905026,
      0.007040366996079683,
      0.005967734003206715,
      0.004744356992887333,
      0.0038730119995307177,
      0.0031536169990431517,
      0.007008986023720354,
      0.006371890020091087,
      0.005220172984991223,
      0.004213831998640671,
      0.0034058159799315035,
      0.00697987197781913,
      0.003388128010556102,
      0.002417323994450271,
      0.002409617998637259,
      0.0024063389864750206,
      0.007056064991047606,
      0.00640261298394762,
      0.006398152007022873,
      0.006201206997502595,
      0.00509181700181216,
      0.007044481986667961,
      0.00640619097976014,
      0.005597230978310108,
      0.002457455004332587,
      0.002436245995340869
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
        "time_utc": "2026-10-25T12:08:19.901069+00:00",
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
        "time_utc": "2026-10-25T12:08:20.237174+00:00",
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
        "time_utc": "2026-10-25T12:08:20.270607+00:00",
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
        "time_utc": "2026-10-25T12:08:20.309014+00:00",
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
        "time_utc": "2026-10-25T12:08:20.354995+00:00",
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
        "time_utc": "2026-10-25T12:08:20.427684+00:00",
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
        "time_utc": "2026-10-25T12:08:20.486644+00:00",
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
        "time_utc": "2026-10-25T12:08:20.557350+00:00",
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
        "time_utc": "2026-10-25T12:08:20.628455+00:00",
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
        "time_utc": "2026-10-25T12:08:20.670821+00:00",
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
    "total_benchmark_section_seconds": 0.7997889450052753,
    "forward_latency_seconds_including_warmup": 0.1590300408715848,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R5_PB_H1_4101/primary_best.pt",
    "checkpoint_sha256": "ca1853323f8fc7dfcf875df421081f01ac97723beee6843da2e710b5c5984aed",
    "checkpoint_selected_epoch": 13,
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
  },
  {
    "status": "COMPLETE",
    "run_id": "FORMAL_kuairand_R7_PB_HISTORY_4101",
    "method": "R7_PB_HISTORY",
    "dataset": "kuairand",
    "seed": 4101,
    "predictor": "main",
    "stage": "formal",
    "physical_gpu": 0,
    "torch_device": "cuda:0",
    "batch_size": 4096,
    "warmup_repetitions": 5,
    "timed_repetitions": 30,
    "p50_seconds": 0.00492483600100968,
    "p95_seconds": 0.007046395253564697,
    "mean_seconds": 0.005225582965067588,
    "rows_per_second_at_p50": 831702.8220148338,
    "rows_per_second_at_mean": 783835.9906217703,
    "percentile_method": "linear interpolation of sorted30 synchronized timings",
    "warmup_seconds": [
      0.003208729001926258,
      0.0027037390100304037,
      0.0026193509984295815,
      0.002413619018625468,
      0.0024077340203803033
    ],
    "timed_seconds": [
      0.0038730230007786304,
      0.004363010986708105,
      0.004905457986751571,
      0.004134359012823552,
      0.0035468740097712725,
      0.007054610003251582,
      0.006298988009802997,
      0.0056142959801945835,
      0.004472950007766485,
      0.004116488998988643,
      0.007098579982994124,
      0.006261936010560021,
      0.004944214015267789,
      0.003966582007706165,
      0.0032348849927075207,
      0.007036355003947392,
      0.006387553992681205,
      0.005393725004978478,
      0.004250678000971675,
      0.003683507995447144,
      0.007030746026430279,
      0.006406563974451274,
      0.0060809089918620884,
      0.004686981992563233,
      0.0037815559771843255,
      0.006990261987084523,
      0.006394791998900473,
      0.005854560004081577,
      0.00468122199526988,
      0.004221821000101045
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
        "time_utc": "2026-10-25T12:08:20.755816+00:00",
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
        "time_utc": "2026-10-25T12:08:21.079547+00:00",
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
        "time_utc": "2026-10-25T12:08:21.109849+00:00",
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
        "time_utc": "2026-10-25T12:08:21.152582+00:00",
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
        "time_utc": "2026-10-25T12:08:21.200175+00:00",
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
        "time_utc": "2026-10-25T12:08:21.247459+00:00",
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
        "time_utc": "2026-10-25T12:08:21.296998+00:00",
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
        "time_utc": "2026-10-25T12:08:21.348635+00:00",
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
        "time_utc": "2026-10-25T12:08:21.400398+00:00",
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
        "time_utc": "2026-10-25T12:08:21.421423+00:00",
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
    "total_benchmark_section_seconds": 0.7024870349850971,
    "forward_latency_seconds_including_warmup": 0.17012066100141965,
    "checkpoint_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/checkpoints/FORMAL_kuairand_R7_PB_HISTORY_4101/primary_best.pt",
    "checkpoint_sha256": "57afbcd2544649bc76eac13a1ba36e529d5c2a30efd888baea8b5637ed9558c5",
    "checkpoint_selected_epoch": 12,
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
