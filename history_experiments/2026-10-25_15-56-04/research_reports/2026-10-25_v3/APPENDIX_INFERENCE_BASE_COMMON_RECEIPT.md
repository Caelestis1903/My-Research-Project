# BASE_COMMON 推理收据头

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

服务器原始文件：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/inference_common/benchmark_receipt.json

SHA256：fc9cf2241ea4104b87b72cd05903e2876163de6c5b45680c54d9b5978054bd16

完整records按原顺序另见同scope的RECORDS分页；本收据头其余字段全部保留。

```json
{
  "schema": "mdmt-next-v3-final-common-inference-v1",
  "status": "COMPLETE",
  "started_utc": "2026-10-25T12:07:31.629690+00:00",
  "gate": {
    "final_method_freeze_path": "/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/FINAL_METHOD_FREEZE.json",
    "final_method_freeze_sha256": "8a0698db50ed4d129909b9decdd400e1ee8db89cfa142591f942282f9405e382",
    "completed_core_count": 52,
    "completed_aux_external_count": 10
  },
  "source_hash": "3be008befd51aaa7ac775c34a484e8a1dabc31578620383260bc2e1878af89d2",
  "review_approval_sha256": "15be36953149a075e436ae095e82110d21e18b01cdc42c6acde8db56dc29df4d",
  "helper_sha256": "f50a14226efbd2d0f679c2c5bcaa7d17af257171682057444797dcf2368cf57b",
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
  "CUDA_VISIBLE_DEVICES": "0",
  "torch_version": "2.9.0+cu126",
  "batch_size": 4096,
  "warmup_repetitions": 5,
  "timed_repetitions": 30,
  "expected_measurements": 40,
  "include_old_aux": true,
  "input": "first4096 sorted fit row IDs; encoded features only; labels are never loaded by the benchmark forward",
  "selection": "preexisting primary selected checkpoint; no score or latency-based model selection",
  "parameter_scope": "restored training graph compared with inference-resident modules; allocator/optimizers/rewind/controller discarded before memory/timing",
  "caveats": [
    "Timing is a fixed-batch hardware cost measurement, not research quality evidence.",
    "No energy, GPU-hours, sparse speedup or all-weight activity claim.",
    "GPU0 compute process isolation is checked before/after load and every five repetitions; unrelated processes are never terminated."
  ],
  "ended_utc": "2026-10-25T12:08:00.430961+00:00"
}
```
