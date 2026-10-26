# RESEARCH_ACTUAL 完整成本/来源收据

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

服务器原始文件：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/ledgers/research_actual.json

SHA256：e8f2138cb639e51263290efffe34a5d656cde410f7055439d366bfca9ffac6d3

```json
{
  "schema": "v3-cost-information-ledgers-v1",
  "created_utc": "2026-10-25T12:10:10.041683+00:00",
  "helper_sha256": "0f04f1d9191333b33b45ed677c2ea6485f1cac71748e880b79ee4ffdc021f1f3",
  "csv_reference": "research_actual.csv",
  "ledger": {
    "additive_recorded_component_seconds": 18715.76654259939,
    "physical_warmups_counted_once": true,
    "reused_historical_prefix_cost_included": false,
    "known_missing_costs": [
      "Dataset audit/hash/reporting operations not individually journaled.",
      "Semantic check wall cost may not have component journals.",
      "Diagnostic and selection-correction wrapper wall/restore/IO may lack independent canonical process intervals; additive recorded leaves do not make those complete totals."
    ],
    "caveat": "Component sum is measured overlapping process/CPU/device work, not exclusive GPU-hours or energy."
  },
  "not_additive_with_other_ledger_views": true
}
```
