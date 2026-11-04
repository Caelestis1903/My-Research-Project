# G-D0双数据集对比

每行n=5，seed2022–2026；固定EMA按validation六格Macro AUC最高、并列最早选点后的test结果。
每格为五seed均值AUC / LogLoss；逐格SD及Traffic/Pooled见CSV。MovieLens5条Gaussian轨迹复用，新增训练0；KuaiRand沿用原报告。

| 数据集 | n | d0t0 AUC / LL | d0t1 AUC / LL | d1t0 AUC / LL | d1t1 AUC / LL | d2t0 AUC / LL | d2t1 AUC / LL | Macro AUC ± SD | Macro LL ± SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movielens | 5 | 0.774415 / 0.585145 | 0.789658 / 0.407081 | 0.782567 / 0.573874 | 0.801390 / 0.390412 | 0.756629 / 0.580689 | 0.771482 / 0.437731 | 0.779357 ± 0.000393 | 0.495822 ± 0.001550 |
| kuairand | 5 | 0.643921 / 0.287652 | 0.662809 / 0.181037 | 0.702451 / 0.504622 | 0.730723 / 0.345700 | 0.655604 / 0.649660 | 0.679574 / 0.644436 | 0.679180 ± 0.003475 | 0.435518 ± 0.002752 |

| 数据集 | 增强域 | batch | epoch | Adam lr | coupled wd | Gaussian κ | λ | EMA ρ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| movielens | d0 | 40960 | 50 | 0.01 | 0.0001 | 0.05 | 0.5 | 0.9828205985452511 |
| kuairand | d0 | 40960 | 50 | 0.01 | 1e-05 | 0.1 | 0.5 | 0.995 |

两数据集均为fixed1 M3oE、FP32、原始自然人口、Sample目标、train-only预处理。MovieLens领域按固定年龄划分，任务为rating>3/rating>4；KuaiRand域为tab0/1/4，任务为is_click/long_view。
原KuaiRand270端点和10条新增训练的统计保持原归档；本补充复用MovieLens G-D0五条及发表模型十五条历史轨迹。

[逐seed端点](MOVIELENS_ENDPOINTS.csv) · [来源审计](MOVIELENS_SOURCE_AUDIT.json) · [中文补充分析](MOVIELENS_SUPPLEMENT.md)
