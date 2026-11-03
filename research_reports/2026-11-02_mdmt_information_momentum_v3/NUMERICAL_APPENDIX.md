# 完整数值附录

seed为统计单位。AUC/LogLoss均来自未扰动真实测试输入；选点只使用validation。

## EMA selected逐seed

| dataset | method | seed | epoch | macro_auc | macro_logloss | worst_cell_auc | sample_weighted_cell_auc | pooled_task_mean_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kuairand | A0 | 2022 | 33 | 0.6734721 | 0.4362044 | 0.6322403 | 0.7162516 | 0.7214985 |
| kuairand | A0 | 2024 | 50 | 0.6848714 | 0.4345008 | 0.6482364 | 0.7180252 | 0.7230996 |
| kuairand | A1 | 2022 | 33 | 0.6804473 | 0.4360394 | 0.6463864 | 0.7151126 | 0.7202332 |
| kuairand | A1 | 2024 | 39 | 0.6774416 | 0.4352078 | 0.6351723 | 0.7172837 | 0.7225327 |
| kuairand | P00 λ=0.05 | 2022 | 34 | 0.6764533 | 0.4350595 | 0.6482199 | 0.7143531 | 0.7196011 |
| kuairand | P00 λ=0.05 | 2024 | 37 | 0.6704084 | 0.4378288 | 0.6245265 | 0.7146943 | 0.7200759 |
| kuairand | P00 λ=0.2 | 2022 | 50 | 0.6738855 | 0.4358496 | 0.6342639 | 0.7109992 | 0.7164672 |
| kuairand | P00 λ=0.2 | 2024 | 48 | 0.6721701 | 0.4364123 | 0.6326548 | 0.7106611 | 0.7161996 |
| kuairand | P11 λ=0.05 | 2022 | 35 | 0.6828313 | 0.4336474 | 0.647293 | 0.716827 | 0.7219301 |
| kuairand | P11 λ=0.05 | 2024 | 28 | 0.6783196 | 0.4365394 | 0.627535 | 0.7121365 | 0.7174668 |
| kuairand | P11 λ=0.2 | 2022 | 45 | 0.6803987 | 0.4364027 | 0.6416933 | 0.7134904 | 0.7188002 |
| kuairand | P11 λ=0.2 | 2024 | 42 | 0.6707786 | 0.4403104 | 0.6331182 | 0.7139688 | 0.7191122 |
| movielens | A0 | 2022 | 26 | 0.7795384 | 0.4949422 | 0.7580916 | 0.7806821 | 0.7810897 |
| movielens | A0 | 2024 | 20 | 0.7790776 | 0.49505 | 0.7562826 | 0.7803414 | 0.7808026 |
| movielens | A1 | 2022 | 20 | 0.7782094 | 0.4956851 | 0.7549843 | 0.7793365 | 0.7796966 |
| movielens | A1 | 2024 | 22 | 0.7794142 | 0.4947866 | 0.7568739 | 0.7806562 | 0.7810216 |
| movielens | P00 λ=0.05 | 2022 | 23 | 0.7782234 | 0.4923124 | 0.755961 | 0.7793868 | 0.7796725 |
| movielens | P00 λ=0.05 | 2024 | 21 | 0.7780118 | 0.4929313 | 0.7557069 | 0.7792408 | 0.779457 |
| movielens | P00 λ=0.2 | 2022 | 42 | 0.7777543 | 0.4908026 | 0.7558245 | 0.7788902 | 0.7793125 |
| movielens | P00 λ=0.2 | 2024 | 40 | 0.7768467 | 0.4899066 | 0.7549416 | 0.7779457 | 0.7783576 |
| movielens | P11 λ=0.05 | 2022 | 23 | 0.7789351 | 0.4919056 | 0.7567127 | 0.7800271 | 0.7803094 |
| movielens | P11 λ=0.05 | 2024 | 22 | 0.7778577 | 0.4946842 | 0.7560721 | 0.7790516 | 0.7791904 |
| movielens | P11 λ=0.2 | 2022 | 31 | 0.7771842 | 0.4912974 | 0.7555322 | 0.7783939 | 0.7787407 |
| movielens | P11 λ=0.2 | 2024 | 40 | 0.7777223 | 0.4930988 | 0.75583 | 0.7789429 | 0.7793197 |

## EMA selected六格均值与样本SD

| dataset | method | cell | n | auc_mean | auc_sd | logloss_mean | logloss_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kuairand | A0 | d0t0 | 2 | 0.6406267 | 0.0107617 | 0.2883975 | 0.0008459 |
| kuairand | A0 | d0t1 | 2 | 0.6609774 | 0.0406404 | 0.179542 | 0.0030781 |
| kuairand | A0 | d1t0 | 2 | 0.7040184 | 0.0009844 | 0.5044871 | 0.0082136 |
| kuairand | A0 | d1t1 | 2 | 0.7324046 | 0.0010628 | 0.3453729 | 0.0071851 |
| kuairand | A0 | d2t0 | 2 | 0.6568229 | 0.0054406 | 0.6497337 | 0.0074821 |
| kuairand | A0 | d2t1 | 2 | 0.6801806 | 0.0003546 | 0.6445826 | 0.0029209 |
| kuairand | A1 | d0t0 | 2 | 0.6407794 | 0.0079296 | 0.2883801 | 0.0026552 |
| kuairand | A1 | d0t1 | 2 | 0.6640193 | 0.0068735 | 0.1803635 | 0.0007203 |
| kuairand | A1 | d1t0 | 2 | 0.7031694 | 0.0015763 | 0.5045018 | 0.0076729 |
| kuairand | A1 | d1t1 | 2 | 0.731325 | 0.0016746 | 0.3459128 | 0.0082375 |
| kuairand | A1 | d2t0 | 2 | 0.6551068 | 0.0048003 | 0.6499212 | 0.0038637 |
| kuairand | A1 | d2t1 | 2 | 0.6792667 | 0.0101464 | 0.644662 | 0.0065836 |
| kuairand | P00 λ=0.05 | d0t0 | 2 | 0.6363732 | 0.0167538 | 0.2881677 | 0.0024039 |
| kuairand | P00 λ=0.05 | d0t1 | 2 | 0.6471905 | 0.0037275 | 0.1815875 | 0.0003851 |
| kuairand | P00 λ=0.05 | d1t0 | 2 | 0.7012186 | 0.0006192 | 0.504405 | 0.0031664 |
| kuairand | P00 λ=0.05 | d1t1 | 2 | 0.7301574 | 0.0001344 | 0.3449252 | 0.0034957 |
| kuairand | P00 λ=0.05 | d2t0 | 2 | 0.6524705 | 0.005732 | 0.65067 | 0.0024922 |
| kuairand | P00 λ=0.05 | d2t1 | 2 | 0.673175 | 0.0001869 | 0.6489097 | 0.0005761 |
| kuairand | P00 λ=0.2 | d0t0 | 2 | 0.6407499 | 0.0091727 | 0.2872438 | 0.001727 |
| kuairand | P00 λ=0.2 | d0t1 | 2 | 0.6455191 | 0.0181929 | 0.1814042 | 0.0018501 |
| kuairand | P00 λ=0.2 | d1t0 | 2 | 0.6979156 | 0.0006193 | 0.5075792 | 0.0031689 |
| kuairand | P00 λ=0.2 | d1t1 | 2 | 0.7258916 | 0.0002118 | 0.3485618 | 0.00205 |
| kuairand | P00 λ=0.2 | d2t0 | 2 | 0.6516706 | 0.0041784 | 0.6485576 | 0.0053038 |
| kuairand | P00 λ=0.2 | d2t1 | 2 | 0.67642 | 0.0020281 | 0.6434388 | 0.0023493 |
| kuairand | P11 λ=0.05 | d0t0 | 2 | 0.637414 | 0.013971 | 0.2889536 | 0.0030759 |
| kuairand | P11 λ=0.05 | d0t1 | 2 | 0.6779917 | 0.0105182 | 0.178649 | 0.0012266 |
| kuairand | P11 λ=0.05 | d1t0 | 2 | 0.701208 | 0.0035718 | 0.5066047 | 0.0085954 |
| kuairand | P11 λ=0.05 | d1t1 | 2 | 0.7296572 | 0.0030131 | 0.3467477 | 0.0069753 |
| kuairand | P11 λ=0.05 | d2t0 | 2 | 0.6574829 | 0.0046993 | 0.6470349 | 0.003285 |
| kuairand | P11 λ=0.05 | d2t1 | 2 | 0.6796989 | 0.0072333 | 0.6425706 | 0.0043182 |
| kuairand | P11 λ=0.2 | d0t0 | 2 | 0.6374058 | 0.0060636 | 0.2894493 | 0.0009687 |
| kuairand | P11 λ=0.2 | d0t1 | 2 | 0.6472089 | 0.0175477 | 0.1827227 | 0.0032366 |
| kuairand | P11 λ=0.2 | d1t0 | 2 | 0.7006642 | 5.09e-05 | 0.5127155 | 0.0015037 |
| kuairand | P11 λ=0.2 | d1t1 | 2 | 0.7289802 | 0.0010273 | 0.3512647 | 0.0003976 |
| kuairand | P11 λ=0.2 | d2t0 | 2 | 0.6590956 | 0.0103662 | 0.6485584 | 0.0052331 |
| kuairand | P11 λ=0.2 | d2t1 | 2 | 0.6801774 | 0.0079158 | 0.6454288 | 0.0060348 |
| movielens | A0 | d0t0 | 2 | 0.7747428 | 0.0009781 | 0.5849287 | 0.000664 |
| movielens | A0 | d0t1 | 2 | 0.7895993 | 0.0003417 | 0.4052498 | 0.0005897 |
| movielens | A0 | d1t0 | 2 | 0.7824043 | 0.0006757 | 0.5736327 | 0.0001231 |
| movielens | A0 | d1t1 | 2 | 0.8008168 | 0.0002902 | 0.3884322 | 0.0007034 |
| movielens | A0 | d2t0 | 2 | 0.7571871 | 0.0012791 | 0.5801537 | 0.0001765 |
| movielens | A0 | d2t1 | 2 | 0.7710979 | 0.0010054 | 0.4375795 | 0.001033 |
| movielens | A1 | d0t0 | 2 | 0.7742824 | 0.0001338 | 0.5850813 | 0.0004148 |
| movielens | A1 | d0t1 | 2 | 0.7894907 | 2.41e-05 | 0.4047703 | 0.0004526 |
| movielens | A1 | d1t0 | 2 | 0.7816816 | 0.0012744 | 0.5748955 | 0.0003443 |
| movielens | A1 | d1t1 | 2 | 0.800313 | 0.0014861 | 0.3890272 | 0.0022476 |
| movielens | A1 | d2t0 | 2 | 0.7559291 | 0.0013361 | 0.5806488 | 0.0017851 |
| movielens | A1 | d2t1 | 2 | 0.7711739 | 0.0009054 | 0.4369918 | 0.0003026 |
| movielens | P00 λ=0.05 | d0t0 | 2 | 0.7730911 | 0.0005242 | 0.5821467 | 0.0004808 |
| movielens | P00 λ=0.05 | d0t1 | 2 | 0.7882577 | 0.0015711 | 0.4036807 | 0.0023591 |
| movielens | P00 λ=0.05 | d1t0 | 2 | 0.7805031 | 0.0002766 | 0.571618 | 0.000784 |
| movielens | P00 λ=0.05 | d1t1 | 2 | 0.8000071 | 0.0006299 | 0.3854026 | 0.0002919 |
| movielens | P00 λ=0.05 | d2t0 | 2 | 0.755834 | 0.0001797 | 0.5787286 | 0.0002877 |
| movielens | P00 λ=0.05 | d2t1 | 2 | 0.7710128 | 2.45e-05 | 0.4341546 | 0.0005432 |
| movielens | P00 λ=0.2 | d0t0 | 2 | 0.7721723 | 0.0002651 | 0.5807671 | 0.0005969 |
| movielens | P00 λ=0.2 | d0t1 | 2 | 0.7889036 | 0.0011309 | 0.40017 | 0.0002449 |
| movielens | P00 λ=0.2 | d1t0 | 2 | 0.7795258 | 0.0009162 | 0.5695883 | 0.0035064 |
| movielens | P00 λ=0.2 | d1t1 | 2 | 0.7982711 | 0.0027293 | 0.3818013 | 0.0016749 |
| movielens | P00 λ=0.2 | d2t0 | 2 | 0.755383 | 0.0006243 | 0.5769881 | 0.0001759 |
| movielens | P00 λ=0.2 | d2t1 | 2 | 0.769547 | 1.71e-05 | 0.4328126 | 0.000952 |
| movielens | P11 λ=0.05 | d0t0 | 2 | 0.7738742 | 0.0011221 | 0.5833316 | 0.004542 |
| movielens | P11 λ=0.05 | d0t1 | 2 | 0.7888621 | 0.0021466 | 0.4044295 | 0.0024604 |
| movielens | P11 λ=0.05 | d1t0 | 2 | 0.7805709 | 0.0006309 | 0.5722959 | 0.0024477 |
| movielens | P11 λ=0.05 | d1t1 | 2 | 0.7997397 | 5.69e-05 | 0.3859618 | 0.000871 |
| movielens | P11 λ=0.05 | d2t0 | 2 | 0.7563924 | 0.000453 | 0.5785054 | 0.0003072 |
| movielens | P11 λ=0.05 | d2t1 | 2 | 0.770939 | 0.0001615 | 0.4352452 | 0.0011605 |
| movielens | P11 λ=0.2 | d0t0 | 2 | 0.7713027 | 0.0016633 | 0.5852674 | 0.0007766 |
| movielens | P11 λ=0.2 | d0t1 | 2 | 0.7870004 | 0.0002267 | 0.4036806 | 0.0025665 |
| movielens | P11 λ=0.2 | d1t0 | 2 | 0.7812088 | 0.0012248 | 0.5697349 | 0.0002757 |
| movielens | P11 λ=0.2 | d1t1 | 2 | 0.79781 | 2.65e-05 | 0.3849411 | 0.0001341 |
| movielens | P11 λ=0.2 | d2t0 | 2 | 0.7556811 | 0.0002105 | 0.5773598 | 0.0027084 |
| movielens | P11 λ=0.2 | d2t1 | 2 | 0.7717163 | 0.0010688 | 0.4322047 | 0.0017328 |

完整online/EMA、selected/E10/E50、逐域逐任务及原始计数见tables/endpoints.csv；全部方法两两配对、逐seed差和95% t区间见tables/paired_seed.csv.gz与paired_summary.csv.gz。n=2的t区间自由度为1。

## 运行成本

| dataset | arm | lambda_pair | seed | steps | worker_seconds | peak_allocated_gib | peak_reserved_gib |
| --- | --- | --- | --- | --- | --- | --- | --- |
| kuairand | A0 | 0.0 | 2022 | 2450 | 1145.711 | 1.004 | 1.227 |
| kuairand | A0 | 0.0 | 2024 | 2450 | 1046.76 | 1.004 | 1.227 |
| kuairand | A1 | 0.0 | 2022 | 2450 | 1636.485 | 1.018 | 1.34 |
| kuairand | A1 | 0.0 | 2024 | 2450 | 694.966 | 1.018 | 1.34 |
| kuairand | P00 | 0.05 | 2022 | 2450 | 1239.884 | 1.005 | 1.229 |
| kuairand | P00 | 0.05 | 2024 | 2450 | 579.986 | 1.005 | 1.229 |
| kuairand | P00 | 0.2 | 2022 | 2450 | 603.725 | 1.005 | 1.229 |
| kuairand | P00 | 0.2 | 2024 | 2450 | 970.583 | 1.005 | 1.229 |
| kuairand | P11 | 0.05 | 2022 | 2450 | 1456.694 | 1.143 | 1.52 |
| kuairand | P11 | 0.05 | 2024 | 2450 | 1068.92 | 1.144 | 1.52 |
| kuairand | P11 | 0.2 | 2022 | 2450 | 732.992 | 1.143 | 1.52 |
| kuairand | P11 | 0.2 | 2024 | 2450 | 976.558 | 1.144 | 1.52 |
| movielens | A0 | 0.0 | 2022 | 1000 | 233.106 | 0.91 | 0.979 |
| movielens | A0 | 0.0 | 2024 | 1000 | 398.154 | 0.91 | 0.979 |
| movielens | A1 | 0.0 | 2022 | 1000 | 289.721 | 0.919 | 0.971 |
| movielens | A1 | 0.0 | 2024 | 1000 | 568.483 | 0.919 | 0.971 |
| movielens | P00 | 0.05 | 2022 | 1000 | 243.558 | 0.911 | 0.98 |
| movielens | P00 | 0.05 | 2024 | 1000 | 406.549 | 0.911 | 0.98 |
| movielens | P00 | 0.2 | 2022 | 1000 | 298.511 | 0.911 | 0.98 |
| movielens | P00 | 0.2 | 2024 | 1000 | 411.371 | 0.911 | 0.98 |
| movielens | P11 | 0.05 | 2022 | 1000 | 375.907 | 1.013 | 1.258 |
| movielens | P11 | 0.05 | 2024 | 1000 | 491.43 | 1.013 | 1.258 |
| movielens | P11 | 0.2 | 2022 | 1000 | 389.457 | 1.013 | 1.258 |
| movielens | P11 | 0.2 | 2024 | 1000 | 235.878 | 1.013 | 1.258 |

## 2026-11-03 固定λ=.05补充：新增8条、复用12条

以下为新旧匹配的20条逻辑轨迹，原轨迹不计为新增。完整240行验证/test端点及全部六格、阶段和状态hash见[supplement/endpoints.csv](supplement/endpoints.csv)。

| dataset | arm | seed | epoch | macro_auc | macro_logloss |
| --- | --- | --- | --- | --- | --- |
| kuairand | A0 | 2022 | 33 | 0.6734721 | 0.4362044 |
| kuairand | A0 | 2024 | 50 | 0.6848714 | 0.4345008 |
| kuairand | P00 | 2022 | 34 | 0.6764533 | 0.4350595 |
| kuairand | P00 | 2024 | 37 | 0.6704084 | 0.4378288 |
| kuairand | P01 | 2022 | 42 | 0.6898648 | 0.431583 |
| kuairand | P01 | 2024 | 34 | 0.6811945 | 0.4373841 |
| kuairand | P10 | 2022 | 48 | 0.6871815 | 0.428641 |
| kuairand | P10 | 2024 | 30 | 0.6744648 | 0.4361743 |
| kuairand | P11 | 2022 | 35 | 0.6828313 | 0.4336474 |
| kuairand | P11 | 2024 | 28 | 0.6783196 | 0.4365394 |
| movielens | A0 | 2022 | 26 | 0.7795384 | 0.4949422 |
| movielens | A0 | 2024 | 20 | 0.7790776 | 0.49505 |
| movielens | P00 | 2022 | 23 | 0.7782234 | 0.4923124 |
| movielens | P00 | 2024 | 21 | 0.7780118 | 0.4929313 |
| movielens | P01 | 2022 | 20 | 0.7782691 | 0.4922506 |
| movielens | P01 | 2024 | 23 | 0.778197 | 0.4933864 |
| movielens | P10 | 2022 | 22 | 0.7786838 | 0.4915848 |
| movielens | P10 | 2024 | 22 | 0.7786462 | 0.4949229 |
| movielens | P11 | 2022 | 23 | 0.7789351 | 0.4919056 |
| movielens | P11 | 2024 | 22 | 0.7778577 | 0.4946842 |

### 逐seed组件效应与交互（EMA selected test）

| dataset | seed | contrast | metric | difference |
| --- | --- | --- | --- | --- |
| kuairand | 2022 | ECCM_static | macro_auc | 0.0134115 |
| kuairand | 2022 | ECCM_static | macro_logloss | -0.0034765 |
| kuairand | 2022 | ECCM_dynamic | macro_auc | -0.0043502 |
| kuairand | 2022 | ECCM_dynamic | macro_logloss | 0.0050064 |
| kuairand | 2022 | dynamic_Adam | macro_auc | 0.0107282 |
| kuairand | 2022 | dynamic_Adam | macro_logloss | -0.0064185 |
| kuairand | 2022 | dynamic_ECCM | macro_auc | -0.0070335 |
| kuairand | 2022 | dynamic_ECCM | macro_logloss | 0.0020644 |
| kuairand | 2022 | interaction | macro_auc | -0.0177617 |
| kuairand | 2022 | interaction | macro_logloss | 0.0084829 |
| kuairand | 2022 | P00_vs_A0 | macro_auc | 0.0029812 |
| kuairand | 2022 | P00_vs_A0 | macro_logloss | -0.0011449 |
| kuairand | 2022 | P01_vs_A0 | macro_auc | 0.0163927 |
| kuairand | 2022 | P01_vs_A0 | macro_logloss | -0.0046214 |
| kuairand | 2022 | P10_vs_A0 | macro_auc | 0.0137094 |
| kuairand | 2022 | P10_vs_A0 | macro_logloss | -0.0075635 |
| kuairand | 2022 | P11_vs_A0 | macro_auc | 0.0093592 |
| kuairand | 2022 | P11_vs_A0 | macro_logloss | -0.0025571 |
| kuairand | 2024 | ECCM_static | macro_auc | 0.0107862 |
| kuairand | 2024 | ECCM_static | macro_logloss | -0.0004448 |
| kuairand | 2024 | ECCM_dynamic | macro_auc | 0.0038548 |
| kuairand | 2024 | ECCM_dynamic | macro_logloss | 0.0003651 |
| kuairand | 2024 | dynamic_Adam | macro_auc | 0.0040565 |
| kuairand | 2024 | dynamic_Adam | macro_logloss | -0.0016545 |
| kuairand | 2024 | dynamic_ECCM | macro_auc | -0.0028749 |
| kuairand | 2024 | dynamic_ECCM | macro_logloss | -0.0008446 |
| kuairand | 2024 | interaction | macro_auc | -0.0069314 |
| kuairand | 2024 | interaction | macro_logloss | 0.0008099 |
| kuairand | 2024 | P00_vs_A0 | macro_auc | -0.0144631 |
| kuairand | 2024 | P00_vs_A0 | macro_logloss | 0.003328 |
| kuairand | 2024 | P01_vs_A0 | macro_auc | -0.0036769 |
| kuairand | 2024 | P01_vs_A0 | macro_logloss | 0.0028832 |
| kuairand | 2024 | P10_vs_A0 | macro_auc | -0.0104066 |
| kuairand | 2024 | P10_vs_A0 | macro_logloss | 0.0016735 |
| kuairand | 2024 | P11_vs_A0 | macro_auc | -0.0065518 |
| kuairand | 2024 | P11_vs_A0 | macro_logloss | 0.0020386 |
| movielens | 2022 | ECCM_static | macro_auc | 4.57e-05 |
| movielens | 2022 | ECCM_static | macro_logloss | -6.18e-05 |
| movielens | 2022 | ECCM_dynamic | macro_auc | 0.0002513 |
| movielens | 2022 | ECCM_dynamic | macro_logloss | 0.0003208 |
| movielens | 2022 | dynamic_Adam | macro_auc | 0.0004604 |
| movielens | 2022 | dynamic_Adam | macro_logloss | -0.0007275 |
| movielens | 2022 | dynamic_ECCM | macro_auc | 0.000666 |
| movielens | 2022 | dynamic_ECCM | macro_logloss | -0.000345 |
| movielens | 2022 | interaction | macro_auc | 0.0002056 |
| movielens | 2022 | interaction | macro_logloss | 0.0003826 |
| movielens | 2022 | P00_vs_A0 | macro_auc | -0.001315 |
| movielens | 2022 | P00_vs_A0 | macro_logloss | -0.0026299 |
| movielens | 2022 | P01_vs_A0 | macro_auc | -0.0012693 |
| movielens | 2022 | P01_vs_A0 | macro_logloss | -0.0026917 |
| movielens | 2022 | P10_vs_A0 | macro_auc | -0.0008546 |
| movielens | 2022 | P10_vs_A0 | macro_logloss | -0.0033574 |
| movielens | 2022 | P11_vs_A0 | macro_auc | -0.0006033 |
| movielens | 2022 | P11_vs_A0 | macro_logloss | -0.0030366 |
| movielens | 2024 | ECCM_static | macro_auc | 0.0001852 |
| movielens | 2024 | ECCM_static | macro_logloss | 0.0004551 |
| movielens | 2024 | ECCM_dynamic | macro_auc | -0.0007885 |
| movielens | 2024 | ECCM_dynamic | macro_logloss | -0.0002387 |
| movielens | 2024 | dynamic_Adam | macro_auc | 0.0006344 |
| movielens | 2024 | dynamic_Adam | macro_logloss | 0.0019916 |
| movielens | 2024 | dynamic_ECCM | macro_auc | -0.0003393 |
| movielens | 2024 | dynamic_ECCM | macro_logloss | 0.0012978 |
| movielens | 2024 | interaction | macro_auc | -0.0009737 |
| movielens | 2024 | interaction | macro_logloss | -0.0006938 |
| movielens | 2024 | P00_vs_A0 | macro_auc | -0.0010658 |
| movielens | 2024 | P00_vs_A0 | macro_logloss | -0.0021187 |
| movielens | 2024 | P01_vs_A0 | macro_auc | -0.0008806 |
| movielens | 2024 | P01_vs_A0 | macro_logloss | -0.0016636 |
| movielens | 2024 | P10_vs_A0 | macro_auc | -0.0004314 |
| movielens | 2024 | P10_vs_A0 | macro_logloss | -0.0001271 |
| movielens | 2024 | P11_vs_A0 | macro_auc | -0.0012199 |
| movielens | 2024 | P11_vs_A0 | macro_logloss | -0.0003658 |

全部指标的均值、样本SD、方向计数、95%配对t区间见[supplement/effects_summary.csv](supplement/effects_summary.csv)，全部单seed差见[effects_by_seed.csv](supplement/effects_by_seed.csv)。
