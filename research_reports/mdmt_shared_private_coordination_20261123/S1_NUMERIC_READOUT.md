# D2 S1 numerical readout

**COMPLETE_NO_GO_S1.** V_A froze P_AXIS, with KuaiRand as its only qualifying dataset. V_B confirmed zero datasets. All quantities below come from Validation or the execution records. [Scope decision](../../artifacts/mdmt_shared_private_coordination_20261123/SCOPE_DECISION.json)

## Registered selection and confirmation

The unit is the paired seed (2022–2026). Sequential arms average AB and BA within each anchor, then E10/E25/E40 within seed for the aggregate. Equal-dataset values average the two same-seed deltas before inference. AUC and LogLoss contrasts are candidate minus reference, in units of 10⁻⁶ unless stated. Intervals are paired t 95% CIs with df=4. Parentheses give positive/negative/zero seed counts.

Online V_A requires all three aggregate Macro AUC contrasts >0 and every six-cell mean contrast ≥−1,000 within a qualifying dataset. The score is the minimum of three equal-dataset means. The exact tie order is P_CELL, P_AXIS, P_FULL. The table follows descending score, with eligibility shown separately.

| Scope | V_A qualifying dataset | Equal Δ vs U_ONLY | Equal Δ vs reverse | Equal Δ vs J_ACC | Score |
| --- | --- | --- | --- | --- | --- |
| P_AXIS | KuaiRand | -12.473 | +9.500 | +15.174 | -12.473 |
| P_FULL | None | -18.616 | +17.013 | +9.031 | -18.616 |
| P_CELL | None | -24.078 | -9.428 | +3.569 | -24.078 |

| V_A aggregate | Scope | Δ vs U_ONLY | Δ vs reverse | Δ vs J_ACC | Worst cell means, same order |
| --- | --- | --- | --- | --- | --- |
| Equal-dataset | P_CELL | -24.078 [-81.716, +33.560] (1/4/0) | -9.428 [-74.580, +55.724] (3/2/0) | +3.569 [-120.860, +127.998] (4/1/0) | d0t1 -38.600, d1t0 -25.976, d0t1 -86.270 |
| Equal-dataset | P_AXIS | -12.473 [-101.916, +76.969] (2/3/0) | +9.500 [-12.691, +31.691] (3/2/0) | +15.174 [-102.299, +132.646] (3/2/0) | d2t0 -86.035, d2t1 -4.219, d2t0 -17.812 |
| Equal-dataset | P_FULL | -18.616 [-149.631, +112.398] (2/3/0) | +17.013 [-45.349, +79.375] (3/2/0) | +9.031 [-89.244, +107.306] (2/3/0) | d2t0 -67.278, d1t0 -14.438, d0t1 -22.708 |
| MovieLens | P_CELL | -30.319 [-45.474, -15.163] (0/5/0) | -4.234 [-77.106, +68.637] (4/1/0) | +3.731 [-65.644, +73.106] (3/2/0) | d0t1 -44.892, d1t0 -37.440, d0t1 -21.258 |
| MovieLens | P_AXIS | -34.565 [-81.845, +12.715] (1/4/0) | +11.690 [-15.217, +38.598] (4/1/0) | -0.515 [-98.143, +97.113] (2/3/0) | d0t1 -60.847, d1t0 -4.255, d0t1 -37.213 |
| MovieLens | P_FULL | -35.015 [-91.362, +21.332] (1/4/0) | +26.164 [-55.334, +107.662] (4/1/0) | -0.965 [-107.006, +105.075] (2/3/0) | d0t1 -78.794, d1t0 -29.012, d0t1 -55.159 |
| KuaiRand | P_CELL | -17.837 [-129.269, +93.594] (1/4/0) | -14.621 [-72.859, +43.616] (2/3/0) | +3.407 [-216.534, +223.348] (4/1/0) | d0t1 -32.308, d0t1 -63.077, d0t1 -151.282 |
| KuaiRand | P_AXIS | +9.618 [-156.843, +176.079] (2/3/0) | +7.309 [-34.625, +49.243] (3/2/0) | +30.862 [-142.601, +204.326] (4/1/0) | d2t0 -152.003, d2t1 -13.993, d2t0 -35.794 |
| KuaiRand | P_FULL | -2.217 [-247.147, +242.712] (2/3/0) | +7.862 [-52.833, +68.558] (2/3/0) | +19.027 [-91.130, +129.183] (4/1/0) | d2t0 -116.610, d0t1 -52.308, d1t0 -26.661 |

For frozen P_AXIS on online V_B, confirmation requires the same dataset to have qualified in V_A, both aggregate Macro AUC means >0, each of their six cell means ≥−1,000, and both contrasts positive together at ≥2 anchors. Dataset-conditional GO also requires no confirmation contrast on the other dataset with an upper 95% CI <0. All intervals are available. Both datasets pass their V_B six-cell guards. KuaiRand has one jointly positive anchor (E25), MovieLens has zero, and both aggregate J_ACC contrasts are negative. No other-dataset upper bound is negative. P_AXIS keeps P_CELL unchanged whereas full J_ACC advances its clock.

| V_B | Anchor | Δ vs reverse | Δ vs J_ACC | Both means >0 |
| --- | --- | --- | --- | --- |
| Equal-dataset | Aggregate | +32.006 [+9.704, +54.307] (5/0/0) | -41.576 [-137.970, +54.818] (1/4/0) | No |
| Equal-dataset | E10 | +16.810 [-16.442, +50.061] (3/2/0) | -25.860 [-129.634, +77.914] (3/2/0) | No |
| Equal-dataset | E25 | +26.262 [+6.186, +46.338] (5/0/0) | -6.433 [-50.526, +37.659] (3/2/0) | No |
| Equal-dataset | E40 | +52.945 [+3.435, +102.455] (4/1/0) | -92.435 [-325.542, +140.673] (2/3/0) | No |
| MovieLens | Aggregate | +11.265 [-19.429, +41.960] (4/1/0) | -1.147 [-103.616, +101.322] (2/3/0) | No |
| MovieLens | E10 | -29.941 [-98.407, +38.525] (2/3/0) | +74.869 [-205.730, +355.467] (2/3/0) | No |
| MovieLens | E25 | +26.588 [-7.494, +60.669] (4/1/0) | -37.404 [-74.364, -0.443] (0/5/0) | No |
| MovieLens | E40 | +37.149 [+14.340, +59.958] (5/0/0) | -40.907 [-74.573, -7.241] (0/5/0) | No |
| KuaiRand | Aggregate | +52.746 [+3.875, +101.616] (4/1/0) | -82.005 [-251.364, +87.354] (1/4/0) | No |
| KuaiRand | E10 | +63.560 [-14.500, +141.619] (4/1/0) | -126.589 [-350.395, +97.217] (1/4/0) | No |
| KuaiRand | E25 | +25.936 [-17.742, +69.614] (4/1/0) | +24.537 [-80.403, +129.476] (3/2/0) | Yes |
| KuaiRand | E40 | +68.741 [-43.681, +181.164] (4/1/0) | -143.963 [-586.309, +298.383] (2/3/0) | No |

| V_B aggregate | Reference | Five paired Macro AUC deltas in seed order |
| --- | --- | --- |
| MovieLens | P→U | +32.506, +13.991, +23.748, -31.101, +17.184 |
| MovieLens | J_ACC | +23.741, -35.894, -89.616, +127.873, -31.841 |
| KuaiRand | P→U | +41.852, -5.428, +48.657, +88.608, +90.040 |
| KuaiRand | J_ACC | -32.037, -311.229, +52.101, -73.067, -45.793 |

Complete paired estimates, per-seed deltas, and leave-one-seed-out means. [PAIRED_SEEDS.csv](../../artifacts/mdmt_shared_private_coordination_20261123/PAIRED_SEEDS.csv)

## Endpoint AUC and LogLoss

P_AXIS endpoint contrasts by panel and readout. Cells show ΔMacro AUC / ΔMacro LogLoss. Each cell aggregates five seeds and all three anchors.

| Dataset | Panel | Readout | U_ONLY | P→U | J_ACC | J2 |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | FULL | online | -33.211 / +665.343 | +11.333 / +6.304 | -0.560 / -2490.762 | +501.766 / -9560.827 |
| MovieLens | FULL | fixed_ema | +1.756 / +44.653 | +2.134 / +14.210 | +1.752 / +6.916 | +0.367 / -159.454 |
| MovieLens | V_A | online | -34.565 / +680.891 | +11.690 / +4.479 | -0.515 / -2552.083 | +517.717 / -9731.240 |
| MovieLens | V_A | fixed_ema | +1.584 / +46.121 | +1.974 / +14.644 | +1.588 / +7.082 | +0.347 / -163.375 |
| MovieLens | V_B | online | -32.506 / +649.909 | +11.265 / +8.063 | -1.147 / -2430.545 | +488.593 / -9392.360 |
| MovieLens | V_B | fixed_ema | +1.805 / +43.206 | +2.237 / +13.784 | +1.789 / +6.750 | +0.519 / -155.591 |
| KuaiRand | FULL | online | +7.051 / +97.747 | +31.014 / +31.422 | -29.211 / -205.010 | +350.148 / -1835.493 |
| KuaiRand | FULL | fixed_ema | -0.170 / -1.420 | +0.895 / -1.071 | +0.549 / -0.769 | +0.714 / -7.678 |
| KuaiRand | V_A | online | +9.618 / +80.065 | +7.309 / +25.216 | +30.862 / -170.713 | +318.146 / -2010.254 |
| KuaiRand | V_A | fixed_ema | -0.778 / -1.715 | +1.162 / -1.067 | +0.965 / -0.741 | -1.675 / -8.006 |
| KuaiRand | V_B | online | +7.787 / +115.149 | +52.746 / +37.558 | -82.005 / -238.764 | +404.678 / -1661.617 |
| KuaiRand | V_B | fixed_ema | +0.056 / -1.128 | +1.320 / -1.073 | +0.812 / -0.796 | +5.679 / -7.352 |

Online V_B P_AXIS at the cell, domain and task levels. Each entry is ΔAUC / ΔLogLoss. Within each vector, cells follow d0t0, d0t1, d1t0, d1t1, d2t0, d2t1, domains follow d0–d2, and tasks follow t0–t1.

| Dataset | Reference | Six cells | Three domains | Two tasks | Traffic AUC / LL | Pooled AUC / LL |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | U_ONLY | -54.036/+682.015, -37.551/+423.873, -34.880/+1174.748, -30.368/+1012.301, -19.663/+226.174, -18.538/+380.343 | -45.793/+552.944, -32.624/+1093.524, -19.101/+303.258 | -36.193/+694.312, -28.819/+605.506 | -31.195/+652.093 | +15.773/+652.093 |
| MovieLens | P→U | +12.474/+113.372, +14.577/-105.305, -6.568/+186.287, +1.898/-31.401, +13.953/+15.119, +31.258/-129.696 | +13.525/+4.034, -2.335/+77.443, +22.606/-57.289 | +6.620/+104.926, +15.911/-88.801 | +11.269/+7.307 | -3.711/+7.307 |
| MovieLens | J_ACC | -35.739/-2888.475, +6.930/-1868.985, +45.523/-2237.408, +0.474/-2197.878, +5.778/-2990.881, -29.849/-2399.642 | -14.404/-2378.730, +22.998/-2217.643, -12.036/-2695.261 | +5.187/-2705.588, -7.482/-2155.501 | -0.242/-2439.184 | +56.951/-2439.184 |
| MovieLens | J2 | +450.057/-9434.477, +527.570/-7221.483, +465.615/-10293.523, +520.093/-10610.512, +547.691/-10201.853, +420.530/-8592.311 | +488.814/-8327.980, +492.854/-10452.017, +484.110/-9397.082 | +487.788/-9976.617, +489.398/-8808.102 | +488.500/-9479.609 | +494.980/-9479.609 |
| KuaiRand | U_ONLY | +198.709/-76.728, +290.093/-82.170, -40.776/+260.937, -28.684/+190.915, -140.393/+98.744, -232.226/+299.196 | +244.401/-79.449, -34.730/+225.926, -186.310/+198.970 | +5.847/+94.317, +9.727/+135.980 | -32.863/+222.460 | -46.584/+222.460 |
| KuaiRand | P→U | +119.780/-1.846, +157.450/-26.986, -3.899/+78.630, +11.764/+9.135, -8.572/+97.814, +39.953/+68.600 | +138.615/-14.416, +3.932/+43.883, +15.690/+83.207 | +35.769/+58.199, +69.722/+16.917 | +5.460/+43.549 | +0.701/+43.549 |
| KuaiRand | J_ACC | -210.941/+101.569, -265.792/+70.233, -19.048/-77.576, -15.550/-204.601, -1.456/-294.481, +20.757/-1027.729 | -238.367/+85.901, -17.299/-141.088, +9.651/-661.105 | -77.148/-90.162, -86.862/-387.366 | -19.465/-142.499 | +1.665/-142.499 |
| KuaiRand | J2 | +363.262/+50.444, +475.894/-134.241, +200.541/-4465.099, +123.713/-3676.749, +586.319/-587.761, +678.340/-1156.298 | +419.578/-41.899, +162.127/-4070.924, +632.330/-872.029 | +383.374/-1667.472, +425.982/-1655.763 | +168.352/-4004.203 | +226.341/-4004.203 |

Matched AB and BA Macro AUC endpoint contrasts for P_AXIS, averaged over anchors within each seed. Each row gives AB and BA separately. U_ONLY and J_ACC use the same accumulated reference for both mappings.

| Dataset | Panel | Readout | Reverse AB / BA | J_ACC AB / BA | U_ONLY AB / BA | J2 AB / BA |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | FULL | online | +33.666 / -11.001 | +10.654 / -11.775 | -21.997 / -44.425 | +506.302 / +497.231 |
| MovieLens | FULL | fixed_ema | +2.551 / +1.716 | +2.080 / +1.424 | +2.084 / +1.428 | +0.277 / +0.457 |
| MovieLens | V_A | online | +14.699 / +8.682 | +1.436 / -2.467 | -32.613 / -36.516 | +518.538 / +516.896 |
| MovieLens | V_A | fixed_ema | +1.947 / +2.001 | +1.550 / +1.626 | +1.545 / +1.622 | +0.046 / +0.648 |
| MovieLens | V_B | online | +53.109 / -30.578 | +19.521 / -21.816 | -11.838 / -53.174 | +496.619 / +480.566 |
| MovieLens | V_B | fixed_ema | +2.992 / +1.482 | +2.370 / +1.207 | +2.387 / +1.224 | +0.480 / +0.559 |
| KuaiRand | FULL | online | +24.035 / +37.993 | -24.370 / -34.052 | +11.892 / +2.210 | +271.638 / +428.658 |
| KuaiRand | FULL | fixed_ema | +0.729 / +1.062 | +0.439 / +0.658 | -0.280 / -0.060 | +1.261 / +0.168 |
| KuaiRand | V_A | online | -95.320 / +109.938 | -21.370 / +83.094 | -42.614 / +61.850 | +249.077 / +387.216 |
| KuaiRand | V_A | fixed_ema | +0.985 / +1.340 | +1.540 / +0.390 | -0.203 / -1.353 | -0.481 / -2.869 |
| KuaiRand | V_B | online | +125.284 / -19.793 | -34.824 / -129.186 | +54.968 / -39.394 | +318.620 / +490.737 |
| KuaiRand | V_B | fixed_ema | -0.418 / +3.058 | -1.321 / +2.945 | -2.077 / +2.189 | +3.780 / +7.578 |

Other scope endpoint AUCs are descriptive on V_B. Values are aggregate online Macro AUC means.

| Dataset | Scope | U_ONLY | P→U | J_ACC | J2 |
| --- | --- | --- | --- | --- | --- |
| MovieLens | P_CELL | -28.892 | -5.939 | +2.467 | +492.207 |
| MovieLens | P_FULL | -30.693 | +22.288 | +0.666 | +490.406 |
| KuaiRand | P_CELL | +22.087 | +19.767 | -67.705 | +418.978 |
| KuaiRand | P_FULL | +67.957 | +86.992 | -21.835 | +464.848 |

Endpoint provenance and paired summaries. [FORK_ENDPOINTS.csv](../../artifacts/mdmt_shared_private_coordination_20261123/FORK_ENDPOINTS.csv) [PAIRED_SEEDS.csv](../../artifacts/mdmt_shared_private_coordination_20261123/PAIRED_SEEDS.csv)

## Literal online post-U response and after-P change

A shared update is post-U minus the same unstepped anchor. An after-P change is U→P endpoint minus its own post-U state. Both tables use P_AXIS AB and BA, with each seed-anchor-mapping included once per panel. The post-U endpoint metrics agree exactly across the three private scopes for every matched dataset, seed, anchor, panel and mapping. FULL uses all Validation rows and overlaps V_A and V_B.

| Dataset | Panel | Anchor | Post-U ΔMacro AUC | After-P ΔMacro AUC | Post-U ΔMacro LL | After-P ΔMacro LL |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | FULL | Aggregate | -313.073 [-520.057, -106.088] (0/5/0) | -2.644 [-45.934, +40.646] (1/4/0) | +1473.631 | +620.693 |
| MovieLens | FULL | E10 | -420.277 [-860.012, +19.458] (0/5/0) | +28.399 [-90.700, +147.499] (2/3/0) | -1125.190 | +2.068 |
| MovieLens | FULL | E25 | -352.966 [-680.556, -25.377] (0/5/0) | -19.279 [-32.797, -5.760] (0/5/0) | +2325.874 | +623.886 |
| MovieLens | FULL | E40 | -165.976 [-459.954, +128.002] (2/3/0) | -17.053 [-36.427, +2.322] (0/5/0) | +3220.211 | +1236.126 |
| MovieLens | V_A | Aggregate | -324.478 [-496.471, -152.485] (0/5/0) | -4.366 [-46.190, +37.458] (1/4/0) | +1548.578 | +636.529 |
| MovieLens | V_B | Aggregate | -307.900 [-578.507, -37.294] (1/4/0) | -0.991 [-46.397, +44.415] (2/3/0) | +1399.766 | +604.973 |
| MovieLens | V_B | E10 | -373.735 [-835.178, +87.707] (0/5/0) | +31.796 [-91.482, +155.073] (2/3/0) | -1241.805 | +7.331 |
| MovieLens | V_B | E25 | -324.050 [-665.074, +16.975] (0/5/0) | -22.545 [-33.569, -11.522] (0/5/0) | +2221.143 | +610.886 |
| MovieLens | V_B | E40 | -225.916 [-597.983, +146.151] (1/4/0) | -12.224 [-35.909, +11.461] (1/4/0) | +3219.962 | +1196.702 |
| KuaiRand | FULL | Aggregate | -74.728 [-622.369, +472.913] (2/3/0) | +34.533 [-113.536, +182.602] (2/3/0) | -787.858 | +73.180 |
| KuaiRand | FULL | E10 | -44.672 [-779.180, +689.836] (3/2/0) | -17.619 [-141.660, +106.422] (3/2/0) | -1299.028 | -83.923 |
| KuaiRand | FULL | E25 | -200.027 [-741.870, +341.815] (1/4/0) | +58.953 [-100.898, +218.804] (4/1/0) | +586.019 | -17.014 |
| KuaiRand | FULL | E40 | +20.516 [-668.566, +709.597] (2/3/0) | +62.265 [-400.614, +525.144] (1/4/0) | -1650.564 | +320.479 |
| KuaiRand | V_A | Aggregate | -67.122 [-582.466, +448.221] (1/4/0) | +23.436 [-129.965, +176.837] (2/3/0) | -875.890 | +56.604 |
| KuaiRand | V_B | Aggregate | -116.406 [-814.358, +581.546] (2/3/0) | +47.249 [-136.900, +231.399] (3/2/0) | -700.711 | +89.490 |
| KuaiRand | V_B | E10 | -149.178 [-1357.935, +1059.578] (2/3/0) | -53.639 [-208.429, +101.150] (3/2/0) | -1110.364 | -58.462 |
| KuaiRand | V_B | E25 | -406.711 [-1420.460, +607.038] (2/3/0) | +16.594 [-132.897, +166.085] (3/2/0) | +538.283 | +26.783 |
| KuaiRand | V_B | E40 | +206.671 [-1036.598, +1449.940] (3/2/0) | +178.793 [-450.247, +807.834] (2/3/0) | -1530.052 | +300.149 |

Actual FULL cell shifts. ΔAUC and ΔLogLoss use 10⁻⁶ units. Logit and positive-minus-negative pair-margin shifts give signed mean / mean absolute / mean row-level RMS in natural logit units. Shift summaries average the per-comparison statistics over both batch mappings, three anchors and five seeds. Grade counts sum over those same 30 comparisons. Grades are 0, ½ and 1, with up/down counting each directional transition once. Ties show the before → after count at grade ½, including unchanged ties. The same frozen pairs recur across comparisons.

| Dataset | Change | Cell | ΔAUC / ΔLL | Logit shift | Pair-margin shift | Grade up / down / unchanged | Grade ½ before → after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MovieLens | Post-U | d0t0 | -187.174/+841.237 | +0.02070/+0.10817/+0.13290 | +0.04249/+0.11432/+0.14597 | 939/957/120,984 | 0 → 0 |
| MovieLens | Post-U | d0t1 | -280.674/+1675.174 | +0.00289/+0.09910/+0.12581 | +0.03704/+0.11060/+0.14165 | 915/948/121,017 | 0 → 0 |
| MovieLens | Post-U | d1t0 | -373.172/+725.704 | +0.02804/+0.12131/+0.14641 | +0.04376/+0.11980/+0.15297 | 936/966/120,978 | 0 → 0 |
| MovieLens | Post-U | d1t1 | -464.845/+1503.608 | +0.00181/+0.11370/+0.14078 | +0.03953/+0.11674/+0.14963 | 974/949/120,957 | 0 → 0 |
| MovieLens | Post-U | d2t0 | -285.893/+2575.226 | +0.02751/+0.11623/+0.14064 | +0.04488/+0.11603/+0.14750 | 920/957/121,003 | 0 → 0 |
| MovieLens | Post-U | d2t1 | -286.679/+1520.840 | +0.02012/+0.10707/+0.13146 | +0.03760/+0.10705/+0.13725 | 929/882/121,069 | 0 → 0 |
| MovieLens | After-P | d0t0 | -14.061/+684.321 | -0.00755/+0.03162/+0.03595 | +0.01254/+0.02355/+0.02940 | 71/100/122,709 | 0 → 0 |
| MovieLens | After-P | d0t1 | -20.487/+387.822 | -0.01219/+0.03003/+0.03427 | +0.01168/+0.02289/+0.02872 | 66/73/122,741 | 0 → 0 |
| MovieLens | After-P | d1t0 | -2.983/+1136.833 | -0.00241/+0.03488/+0.03971 | +0.01249/+0.02479/+0.03108 | 82/64/122,734 | 0 → 0 |
| MovieLens | After-P | d1t1 | -2.338/+952.295 | -0.01020/+0.02863/+0.03269 | +0.01205/+0.02334/+0.02936 | 54/65/122,761 | 0 → 0 |
| MovieLens | After-P | d2t0 | +9.445/+183.038 | -0.01142/+0.03045/+0.03428 | +0.00885/+0.02076/+0.02625 | 59/75/122,746 | 0 → 0 |
| MovieLens | After-P | d2t1 | +14.560/+379.850 | -0.01243/+0.02927/+0.03333 | +0.00741/+0.02065/+0.02596 | 60/74/122,746 | 0 → 0 |
| KuaiRand | Post-U | d0t0 | -309.750/-91.630 | -0.00441/+0.03860/+0.04820 | +0.00128/+0.03889/+0.04994 | 1,082/1,138/120,660 | 0 → 0 |
| KuaiRand | Post-U | d0t1 | -329.523/-45.555 | -0.00684/+0.04610/+0.05686 | +0.00180/+0.04606/+0.05890 | 1,091/1,212/120,577 | 0 → 0 |
| KuaiRand | Post-U | d1t0 | +68.266/-1563.777 | -0.02225/+0.08679/+0.10531 | +0.00657/+0.07739/+0.10012 | 1,270/1,203/120,407 | 0 → 0 |
| KuaiRand | Post-U | d1t1 | +93.935/-1298.708 | -0.02425/+0.09420/+0.11506 | +0.00820/+0.08355/+0.10900 | 1,136/1,120/120,624 | 0 → 0 |
| KuaiRand | Post-U | d2t0 | +47.680/-1101.632 | -0.01299/+0.05781/+0.07057 | +0.00243/+0.05405/+0.06904 | 1,210/1,105/120,565 | 0 → 0 |
| KuaiRand | Post-U | d2t1 | -18.974/-625.844 | -0.01452/+0.06004/+0.07275 | +0.00224/+0.05507/+0.07034 | 1,105/1,091/120,684 | 0 → 0 |
| KuaiRand | After-P | d0t0 | +124.100/-45.231 | -0.00725/+0.01391/+0.01703 | +0.00193/+0.01361/+0.01727 | 293/275/122,312 | 0 → 0 |
| KuaiRand | After-P | d0t1 | +227.176/-92.905 | -0.00732/+0.01904/+0.02284 | +0.00185/+0.01636/+0.02106 | 364/327/122,189 | 0 → 0 |
| KuaiRand | After-P | d1t0 | -7.601/+225.867 | -0.00241/+0.01856/+0.02195 | +0.00504/+0.01523/+0.01952 | 140/164/122,576 | 0 → 0 |
| KuaiRand | After-P | d1t1 | +5.987/+159.497 | -0.00374/+0.02011/+0.02413 | +0.00544/+0.01661/+0.02147 | 141/146/122,593 | 0 → 0 |
| KuaiRand | After-P | d2t0 | -84.887/+40.034 | -0.00497/+0.02087/+0.02490 | +0.00130/+0.01761/+0.02257 | 261/370/122,249 | 0 → 0 |
| KuaiRand | After-P | d2t1 | -57.578/+151.820 | -0.00507/+0.02259/+0.02654 | +0.00102/+0.01758/+0.02243 | 275/285/122,320 | 0 → 0 |

Raw shifts and nine grade transitions for each anchor, mapping and panel. [CELL_RESPONSE.csv](../../artifacts/mdmt_shared_private_coordination_20261123/CELL_RESPONSE.csv) [PAIR_TRANSITIONS.csv](../../artifacts/mdmt_shared_private_coordination_20261123/PAIR_TRANSITIONS.csv)

## Selected J_ORIG Validation endpoints and support

Each online or fixed-EMA entry gives the five-seed mean selected AUC / LogLoss. Selection uses the earliest strict maximum of six-cell Macro AUC in that readout over 50 epochs. Epoch vectors use seed order. Train and Validation rows are raw populations. Positive counts are Validation counts.

| Dataset | Group | Train rows | Validation rows | Validation positives | Online AUC / LL | Fixed-EMA AUC / LL |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | Macro | 800,167 | 100,021 | — | 0.756665/0.526497 | 0.759512/0.525966 |
| MovieLens | d0 | 155,949 | 27,111 | 14,634, 5,781 | 0.765203/0.515263 | 0.767988/0.514784 |
| MovieLens | d1 | 318,599 | 35,523 | 21,027, 8,541 | 0.753927/0.531014 | 0.756619/0.529324 |
| MovieLens | d2 | 325,619 | 37,387 | 22,148, 8,889 | 0.750865/0.533215 | 0.753927/0.533789 |
| MovieLens | t0 | 800,167 | 100,021 | 57,809 | 0.755586/0.582712 | 0.759054/0.581922 |
| MovieLens | t1 | 800,167 | 100,021 | 23,211 | 0.757744/0.470283 | 0.759969/0.470010 |
| MovieLens | d0t0 | 155,949 | 27,111 | 14,634 | 0.761241/0.585904 | 0.764915/0.584495 |
| MovieLens | d0t1 | 155,949 | 27,111 | 5,781 | 0.769166/0.444622 | 0.771062/0.445074 |
| MovieLens | d1t0 | 318,599 | 35,523 | 21,027 | 0.754492/0.577618 | 0.757770/0.575937 |
| MovieLens | d1t1 | 318,599 | 35,523 | 8,541 | 0.753362/0.484410 | 0.755469/0.482711 |
| MovieLens | d2t0 | 325,619 | 37,387 | 22,148 | 0.751026/0.584613 | 0.754476/0.585333 |
| MovieLens | d2t1 | 325,619 | 37,387 | 8,889 | 0.750704/0.481817 | 0.753378/0.482245 |
| MovieLens | Traffic | — | — | — | 0.755839/0.527567 | 0.758695/0.527052 |
| MovieLens | Pooled | — | — | — | 0.755892/0.527567 | 0.758988/0.527052 |
| MovieLens | Selected epochs | — | — | — | 29, 17, 24, 26, 17 | 26, 18, 20, 28, 22 |
| KuaiRand | Macro | 2,006,132 | 250,767 | — | 0.686278/0.430564 | 0.688454/0.429661 |
| KuaiRand | d0 | 172,782 | 2,692 | 245, 101 | 0.679539/0.221480 | 0.677939/0.221509 |
| KuaiRand | d1 | 1,743,898 | 246,232 | 51,644, 27,460 | 0.710149/0.423667 | 0.717428/0.419843 |
| KuaiRand | d2 | 89,452 | 1,843 | 1,063, 829 | 0.669147/0.646546 | 0.669996/0.647629 |
| KuaiRand | t0 | 2,006,132 | 250,767 | 52,952 | 0.678822/0.478882 | 0.681365/0.477338 |
| KuaiRand | t1 | 2,006,132 | 250,767 | 28,390 | 0.693734/0.382247 | 0.695543/0.381983 |
| KuaiRand | d0t0 | 172,782 | 2,692 | 245 | 0.671254/0.290488 | 0.669598/0.290204 |
| KuaiRand | d0t1 | 172,782 | 2,692 | 101 | 0.687825/0.152473 | 0.686280/0.152814 |
| KuaiRand | d1t0 | 1,743,898 | 246,232 | 51,644 | 0.696740/0.502353 | 0.703774/0.497872 |
| KuaiRand | d1t1 | 1,743,898 | 246,232 | 27,460 | 0.723557/0.344981 | 0.731082/0.341815 |
| KuaiRand | d2t0 | 89,452 | 1,843 | 1,063 | 0.668474/0.643804 | 0.670724/0.643938 |
| KuaiRand | d2t1 | 89,452 | 1,843 | 829 | 0.669820/0.649287 | 0.669268/0.651321 |
| KuaiRand | Traffic | — | — | — | 0.709519/0.423135 | 0.716656/0.419388 |
| KuaiRand | Pooled | — | — | — | 0.714721/0.423135 | 0.721720/0.419388 |
| KuaiRand | Selected epochs | — | — | — | 44, 18, 40, 50, 43 | 33, 50, 50, 50, 47 |

MovieLens t0 is rating >3 and t1 is rating >4. KuaiRand t0 is is_click and t1 is long_view. V_A/V_B row counts are 49,922/50,099 for MovieLens and 125,332/125,435 for KuaiRand. Each cell is evaluable in each panel. [BASELINE_ENDPOINTS.csv](../../artifacts/mdmt_shared_private_coordination_20261123/BASELINE_ENDPOINTS.csv) [PANELS_movielens.json](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_movielens.json) [PANELS_kuairand.json](../../artifacts/mdmt_shared_private_coordination_20261123/PANELS_kuairand.json)

## Completion, physical counts and RAM evidence

Ten J_ORIG trajectories finished 50 epochs each, with 30 distinct anchor receipts and 19 forks per anchor (570). The five analysis outputs contain 4,140 endpoint rows, 52,920 grade-transition rows, 52,920 cell-response rows, 178,200 paired-seed metric rows, and one scope decision. Baseline endpoints contain 1,000 Validation rows (20 selected). The run and every trajectory recorded zero Test evaluations.

| Dataset | Baseline attempts | Train fwd / backward / Adam / EMA | Validation fwd | Attempt wall sum / per-attempt range (s) | Recorded GPU intervals sum (s) | Peak process GPU / RSS (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| MovieLens | 5 | 5,000/5,000/5,000/5,000 | 1,500 | 3,764.2 / 708.0–792.2 | 2,075.9 | 1.316 / 1.621 |
| KuaiRand | 5 | 12,250/12,250/12,250/12,250 | 3,500 | 8,073.3 / 1,515.0–1,689.6 | 4,855.5 | 1.721 / 2.205 |

| Dataset | Fork train fwd / backward / Adam / EMA | Validation fwd | Clone / update / evaluation row-time sum (s) | Anchor bytes each |
| --- | --- | --- | --- | --- |
| MovieLens | 570/570/495/570 | 2,070 | 34.8 / 208.0 / 700.2 | 18,787,412 |
| KuaiRand | 570/570/495/570 | 4,830 | 36.1 / 210.5 / 1467.5 | 41,887,784 |

Supervisor status time is 2026-11-23T14:11:39+00:00. Attempt wall sums reflect concurrent workers. Reported fork timings sum recorded clone, optimizer-update and evaluation event durations. Admissions show ten starts and ten exits, with zero resumed epochs and no OOM in the physical-capacity record. The anchor index records 30 RAM_VERIFIED handles at E10/E25/E40, and the ten completion records each contain an E50 RAM_VERIFIED baseline checkpoint. Receipt status describes the recorded verification.
[S1_RUN.json](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN.json) [S1_RUN_MANIFEST.json](../../artifacts/mdmt_shared_private_coordination_20261123/S1_RUN_MANIFEST.json) [S1_ADMISSIONS.jsonl](../../artifacts/mdmt_shared_private_coordination_20261123/S1_ADMISSIONS.jsonl) [S1_PHYSICAL_CAPACITY.json](../../artifacts/mdmt_shared_private_coordination_20261123/S1_PHYSICAL_CAPACITY.json) [ANCHOR_INDEX.json](../../artifacts/mdmt_shared_private_coordination_20261123/ANCHOR_INDEX.json) [COSTS.csv](../../artifacts/mdmt_shared_private_coordination_20261123/COSTS.csv)
