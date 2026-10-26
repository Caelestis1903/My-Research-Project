# Split evidence

原CSV哈希、split类别计数和rowhash已由 `dataset_preflight_*.json`、`split_cell_counts.csv` 与 `dataset_prediction_binding.json` 实测。保留global row-order80/10/10，未读取更早原始事件bytes重新证明时间排序。train→val域/label shift明显，val→test域mix近似一致；粗域mix不足以解释AUC反号。
