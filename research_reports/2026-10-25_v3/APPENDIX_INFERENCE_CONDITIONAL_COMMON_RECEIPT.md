# CONDITIONAL_COMMON 推理收据头

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

服务器原始文件：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_conditional/benchmark_receipt.json

SHA256：459c19604423892d52529ab15d40fe2dd91f0e27c9503c7d9d7fd9238e2a21c3

完整records按原顺序另见同scope的RECORDS分页；本收据头其余字段全部保留。

```json
{
  "schema": "conditional-final-common-inference-v1",
  "status": "COMPLETE",
  "started_utc": "2026-10-25T12:08:14.138522+00:00",
  "gate": {
    "final_method_freeze_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/FINAL_METHOD_FREEZE.json",
    "final_method_freeze_sha256": "8a0698db50ed4d129909b9decdd400e1ee8db89cfa142591f942282f9405e382",
    "completed_core_count": 52,
    "completed_aux_external_count": 10,
    "complete_P2_logical_runs": 7,
    "formal_seed_for_cost_measurement": 4101,
    "expected_measurements": 11,
    "P2_manifest_sha256": "4ca6f2f6cb5ba48fc6e5fc4ed4c4199f0a7b89f54250a3247be61185cd06120c",
    "formal_manifest_sha256": "701f27b64744baf7d6f24bc67aefd3bed1c6ebaecf45412625a4999d28ce2cca"
  },
  "source_hash": "31216926246c5eafa5850891ed34203d04a01b4556fe54afa5107ccffdfc2276",
  "base_source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
  "extension_hash": "d9c76c76f083cf34499c303f3a58c85b8f28c45ed720e386a36c81d6dd318910",
  "helper_sha256": "4d9675314c762ab33210fc3adaff6fd9617afc55821acde5ef2de66a677d693e",
  "base_benchmark_helper_sha256": "f50a14226efbd2d0f679c2c5bcaa7d17af257171682057444797dcf2368cf57b",
  "conditional_review_sha256": "84fc2c088f87ccf397c30aa78614e9a22ef3eaa8c70efad3a9d0f07a73217ddd",
  "physical_gpu": 0,
  "hardware_rows": [
    [
      "0",
      "GPU-4ac4b341-3045-1093-fec3-492516863e5a",
      "NVIDIA GeForce RTX 3090",
      "570.144",
      "24576",
      "350.00"
    ]
  ],
  "torch_version": "2.9.0+cu126",
  "CUDA_VISIBLE_DEVICES": "0",
  "batch_size": 4096,
  "warmup_repetitions": 5,
  "timed_repetitions": 30,
  "expected_measurements": 11,
  "input": "first4096 sortedfit IDs; x.npy features only; y.npy never opened",
  "selection": "existing primary_best aliases only, no validation/quality/latency selection",
  "separate_from_base_benchmark": true,
  "quality_tables_read": false,
  "labels_read": false,
  "ended_utc": "2026-10-25T12:08:22.263183+00:00"
}
```
