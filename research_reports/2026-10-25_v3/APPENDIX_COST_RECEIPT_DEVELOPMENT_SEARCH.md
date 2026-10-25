# DEVELOPMENT_SEARCH 完整成本/来源收据

本附录仅完整转录已完成的成本、信息访问和推理测量记录，不作研究效果判断。三本账是不同会计视图，不能相加；并发进程wall、组件累计wall、活动区间并集也不能混称GPU小时或能耗。`UNKNOWN`、`null`、未记录项和无法分离费用均保留，不视为0。原CSV字符串和嵌套收据保留原始记录值；汇总推理秒数显示10位小数。

信息访问的raw materialization与cached objective/selection重复分别列出；表内EXACT仅限所注明的已记录集合/并集，不表示未记账的CAL raw_head_cal_risks等额外复用也是精确已知。训练seed、epoch、task、cell、probe和重复标签访问不混作独立重复数。

服务器原始文件：/home/xzf/Code/phase0-3/artifacts/mdmt_next_v3/v3_0/ledgers/development_search.json

SHA256：984bdc364f997438ace07246a3c566dd5786ad35ecf8b86cf8fcb8f11d7f827c

```json
{
  "schema": "v3-cost-information-ledgers-v1",
  "created_utc": "2026-10-25T12:10:10.041683+00:00",
  "helper_sha256": "0f04f1d9191333b33b45ed677c2ea6485f1cac71748e880b79ee4ffdc021f1f3",
  "csv_reference": "development_search.csv",
  "ledger": {
    "current_view_seconds": 18715.76654259939,
    "reused_historical_recorded_seconds": 1124.0350542269298,
    "note": "Alternative accounting view overlaps research actual; do not sum ledgers. Historical STATIC/aux/external prefixes are not current new compute or free. Legacy cache/actor/probe accounting is incomplete."
  },
  "not_additive_with_other_ledger_views": true
}
```
