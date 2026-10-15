# F0-B1-ML MovieLens 只读数据审计

状态：**DATA_GATE_PASS**。未训练，未计算模型测试表现。旧 test 已查看，因此新训练仍为探索性。

- 源文件：`/home/xzf/data/ml-1m/ml-1m.csv`；75,514,352 bytes；SHA-256 `f060ecb8fb7745f1e24998837ca5a1ec3e18b610f288e73549c0203a9822b5e4`；与 S1 冻结 hash 匹配：True。
- MovieLens S1 canonical summary 中144/144个 MovieLens run manifests 的文件hash匹配；另有48个 KuaiRand manifests，使用不同数据文件hash。
- 原始字段：user_id, movie_id, rating, timestamp, gender, age, occupation, zip, title, genres。F0 输入特征：user_id, movie_id, gender, age, occupation, zip, cate_id, domain_indicator；rating、timestamp、title 均不作为输入。
- 标签：task0=`rating>3`；task1=`rating>4`，嵌套关系成立。encoder 和 age scaler 由全局 base-train 前缀拟合。
- Split：全局原始行序 80/10/10 前缀；原始timestamp为非递减序列，不是严格递增；train/validation边界的timestamp=975768738跨界出现（train有3行、validation有1行）。validation/test边界时间严格先后。该固定row-order切分在此源文件中总体按时间排序，但有同一时间戳跨train/validation，不能描述为严格时间隔离。train 使用完整自然前缀，不经历史 rotation 抽样。

| Split | 行数 | row ID 区间 | 用户数 | age counts | domain counts | 正例数 t0/t1 |
|---|---:|---|---:|---|---|---:|
| train | 800167 | [0, 800167) | 5400 | `{'1': 20574, '18': 135375, '25': 318599, '35': 163489, '45': 69411, '50': 59498, '56': 33221}` | `{'0': 155949, '1': 318599, '2': 325619}` | 463020/184780 |
| validation | 100021 | [800167, 900188) | 1134 | `{'1': 1705, '18': 25406, '25': 35523, '35': 19060, '45': 8215, '50': 6955, '56': 3157}` | `{'0': 27111, '1': 35523, '2': 37387}` | 57809/23211 |
| test | 100021 | [900188, 1000209) | 1209 | `{'1': 4932, '18': 22755, '25': 41434, '35': 16454, '45': 6007, '50': 6037, '56': 2402}` | `{'0': 27687, '1': 41434, '2': 30900}` | 54452/18319 |

- Duplicate `(user_id,movie_id)` pairs: 0; exact duplicate rows: 0.
- Cross-split user intersections: `{'train_validation': 523, 'train_test': 1009, 'validation_test': 560}`.
- Timestamp diagnostics: non-decreasing (541,754 adjacent equal pairs; 458,454 strict increases). Ranges are train `[956703932, 975768738]`, validation `[975768738, 978133367]`, test `[978133414, 1046454590]`. Timestamp `975768738` crosses the train/validation boundary (3 train rows, 1 validation row); validation/test boundary is strictly ordered. timestamp is excluded from features.
- Original 0-based row IDs exclude header; split ID hashes: `{'train': 'a58f8ea1c284adad7bde05e3811c4055ba85db95613523048f613a220f93bcfe', 'validation': '3cf4c7b1d5dab7ef3ef8d6783faecde7f580337bcdbe4e9c3af37fe1813bbea4', 'test': 'b9ff6de1e3a6f0f601d6843f1be66d40ee292681e3a91688c87ad3b818382ddf'}`.
- No model test metrics computed.


Encoder reconstruction sizes (base-train only): `{"cate_id": {"loader_embedding_rows": 19, "match": true, "train_vocab_plus_oov": 19}, "domain_indicator": {"loader_embedding_rows": 3, "match": true, "train_domain_coverage": [0, 1, 2]}, "gender": {"loader_embedding_rows": 3, "match": true, "train_vocab_plus_oov": 3}, "movie_id": {"loader_embedding_rows": 3663, "match": true, "train_vocab_plus_oov": 3663}, "occupation": {"loader_embedding_rows": 22, "match": true, "train_vocab_plus_oov": 22}, "user_id": {"loader_embedding_rows": 5401, "match": true, "train_vocab_plus_oov": 5401}, "zip": {"loader_embedding_rows": 3201, "match": true, "train_vocab_plus_oov": 3201}}`.

- Encoder/scaler fit scope: global train prefix only; age min/max are 1/56. Base-train vocabulary sizes were independently reconstructed for each sparse feature and match the loader schema.
- User IDs overlap between splits, so this remains a warm-user setup and does not establish cold-start performance.

- Source-order caveat: the global row-order rule is temporally ordered in this CSV, but timestamp equality crosses the train/validation boundary; this is not strict timestamp isolation.
