# MDMT baseline comparison

Numerical entries come only from explicit evaluation files. Smoke fixtures are excluded.
HyperGate settings marked UNVERIFIED_PROPOSAL remain disclosed.

## Formal selected E

| Dataset | Method | Seed | Status | Macro AUC | Macro LogLoss | Natural AUC | Natural LogLoss |
|---|---|---:|---|---:|---:|---:|---:|
| movielens | M3OE_FIXED1 | 2022 | EVALUATED | 0.760302 | 0.510894 | 0.761524 | 0.509571 |
| movielens | M3OE_FIXED1 | 2023 | EVALUATED | 0.759770 | 0.515904 | 0.760873 | 0.514310 |
| movielens | M3OE_FIXED1 | 2024 | EVALUATED | 0.759056 | 0.514066 | 0.760352 | 0.511663 |
| movielens | M3OE_FIXED1 | 2025 | EVALUATED | 0.758263 | 0.511271 | 0.759729 | 0.508521 |
| movielens | M3OE_FIXED1 | 2026 | EVALUATED | 0.759837 | 0.508555 | 0.761202 | 0.506109 |
| movielens | HYPERGATE | 2022 | EVALUATED | 0.767500 | 0.496835 | 0.768723 | 0.495478 |
| movielens | HYPERGATE | 2023 | EVALUATED | 0.767961 | 0.500095 | 0.769173 | 0.498119 |
| movielens | HYPERGATE | 2024 | EVALUATED | 0.767252 | 0.499353 | 0.768501 | 0.497502 |
| movielens | HYPERGATE | 2025 | EVALUATED | 0.765339 | 0.494323 | 0.766877 | 0.492733 |
| movielens | HYPERGATE | 2026 | EVALUATED | 0.766072 | 0.498153 | 0.767388 | 0.496430 |
| movielens | AUTOIFS | 2022 | EVALUATED | 0.762493 | 0.498695 | 0.763932 | 0.497008 |
| movielens | AUTOIFS | 2023 | EVALUATED | 0.764644 | 0.498252 | 0.765934 | 0.496710 |
| movielens | AUTOIFS | 2024 | EVALUATED | 0.764080 | 0.496600 | 0.765379 | 0.495016 |
| movielens | AUTOIFS | 2025 | EVALUATED | 0.762027 | 0.503694 | 0.763445 | 0.502310 |
| movielens | AUTOIFS | 2026 | EVALUATED | 0.761737 | 0.499291 | 0.763222 | 0.497559 |
| kuairand | M3OE_FIXED1 | 2022 | EVALUATED | 0.667816 | 0.435195 | 0.701585 | 0.420046 |
| kuairand | M3OE_FIXED1 | 2023 | EVALUATED | 0.669260 | 0.433000 | 0.699936 | 0.409069 |
| kuairand | M3OE_FIXED1 | 2024 | EVALUATED | 0.670445 | 0.436677 | 0.700854 | 0.417073 |
| kuairand | M3OE_FIXED1 | 2025 | EVALUATED | 0.661464 | 0.440490 | 0.700906 | 0.416954 |
| kuairand | M3OE_FIXED1 | 2026 | EVALUATED | 0.674049 | 0.437715 | 0.702733 | 0.426194 |
| kuairand | HYPERGATE | 2022 | EVALUATED | 0.685246 | 0.435291 | 0.708984 | 0.429734 |
| kuairand | HYPERGATE | 2023 | EVALUATED | 0.679912 | 0.435869 | 0.707242 | 0.432305 |
| kuairand | HYPERGATE | 2024 | EVALUATED | 0.689501 | 0.430505 | 0.707344 | 0.422398 |
| kuairand | HYPERGATE | 2025 | EVALUATED | 0.683036 | 0.434638 | 0.707962 | 0.428175 |
| kuairand | HYPERGATE | 2026 | EVALUATED | 0.677398 | 0.430104 | 0.699496 | 0.420495 |
| kuairand | AUTOIFS | 2022 | EVALUATED | 0.674888 | 0.432509 | 0.708639 | 0.425772 |
| kuairand | AUTOIFS | 2023 | EVALUATED | 0.677117 | 0.430239 | 0.710191 | 0.418570 |
| kuairand | AUTOIFS | 2024 | EVALUATED | 0.672965 | 0.431733 | 0.709526 | 0.422498 |
| kuairand | AUTOIFS | 2025 | EVALUATED | 0.675595 | 0.431318 | 0.709359 | 0.423915 |
| kuairand | AUTOIFS | 2026 | EVALUATED | 0.679009 | 0.435278 | 0.709643 | 0.431965 |

## Formal selected E seed summaries

Mean and sample SD use the seed as the unit; incomplete seeds are excluded.

| Dataset | Method | Metric | n seeds | Mean | Sample SD |
|---|---|---|---:|---:|---:|
| movielens | M3OE_FIXED1 | macro_auc | 5 | 0.759446 | 0.000797 |
| movielens | M3OE_FIXED1 | macro_logloss | 5 | 0.512138 | 0.002874 |
| movielens | M3OE_FIXED1 | natural_auc | 5 | 0.760736 | 0.000710 |
| movielens | M3OE_FIXED1 | natural_logloss | 5 | 0.510035 | 0.003116 |
| movielens | HYPERGATE | macro_auc | 5 | 0.766825 | 0.001084 |
| movielens | HYPERGATE | macro_logloss | 5 | 0.497752 | 0.002281 |
| movielens | HYPERGATE | natural_auc | 5 | 0.768132 | 0.000961 |
| movielens | HYPERGATE | natural_logloss | 5 | 0.496053 | 0.002113 |
| movielens | AUTOIFS | macro_auc | 5 | 0.762996 | 0.001291 |
| movielens | AUTOIFS | macro_logloss | 5 | 0.499307 | 0.002649 |
| movielens | AUTOIFS | natural_auc | 5 | 0.764383 | 0.001207 |
| movielens | AUTOIFS | natural_logloss | 5 | 0.497721 | 0.002736 |
| kuairand | M3OE_FIXED1 | macro_auc | 5 | 0.668607 | 0.004612 |
| kuairand | M3OE_FIXED1 | macro_logloss | 5 | 0.436616 | 0.002798 |
| kuairand | M3OE_FIXED1 | natural_auc | 5 | 0.701203 | 0.001037 |
| kuairand | M3OE_FIXED1 | natural_logloss | 5 | 0.417867 | 0.006184 |
| kuairand | HYPERGATE | macro_auc | 5 | 0.683018 | 0.004696 |
| kuairand | HYPERGATE | macro_logloss | 5 | 0.433282 | 0.002756 |
| kuairand | HYPERGATE | natural_auc | 5 | 0.706206 | 0.003814 |
| kuairand | HYPERGATE | natural_logloss | 5 | 0.426621 | 0.004994 |
| kuairand | AUTOIFS | macro_auc | 5 | 0.675915 | 0.002285 |
| kuairand | AUTOIFS | macro_logloss | 5 | 0.432216 | 0.001898 |
| kuairand | AUTOIFS | natural_auc | 5 | 0.709472 | 0.000560 |
| kuairand | AUTOIFS | natural_logloss | 5 | 0.424544 | 0.004920 |

## Formal fixed-final E

| Dataset | Method | Seed | Status | Macro AUC | Macro LogLoss | Natural AUC | Natural LogLoss |
|---|---|---:|---|---:|---:|---:|---:|
| movielens | M3OE_FIXED1 | 2022 | EVALUATED | 0.759183 | 0.542105 | 0.760299 | 0.539472 |
| movielens | M3OE_FIXED1 | 2023 | EVALUATED | 0.758240 | 0.532933 | 0.759408 | 0.530519 |
| movielens | M3OE_FIXED1 | 2024 | EVALUATED | 0.754528 | 0.549238 | 0.755556 | 0.546636 |
| movielens | M3OE_FIXED1 | 2025 | EVALUATED | 0.757681 | 0.535620 | 0.758720 | 0.534295 |
| movielens | M3OE_FIXED1 | 2026 | EVALUATED | 0.756481 | 0.533042 | 0.757864 | 0.529690 |
| movielens | HYPERGATE | 2022 | EVALUATED | 0.763379 | 0.531104 | 0.764424 | 0.528950 |
| movielens | HYPERGATE | 2023 | EVALUATED | 0.761603 | 0.533581 | 0.762746 | 0.530415 |
| movielens | HYPERGATE | 2024 | EVALUATED | 0.759959 | 0.532806 | 0.761053 | 0.529542 |
| movielens | HYPERGATE | 2025 | EVALUATED | 0.766986 | 0.515565 | 0.768039 | 0.513370 |
| movielens | HYPERGATE | 2026 | EVALUATED | 0.762270 | 0.527146 | 0.763566 | 0.524050 |
| movielens | AUTOIFS | 2022 | EVALUATED | 0.762970 | 0.496453 | 0.764423 | 0.494734 |
| movielens | AUTOIFS | 2023 | EVALUATED | 0.765044 | 0.507349 | 0.766321 | 0.506055 |
| movielens | AUTOIFS | 2024 | EVALUATED | 0.764958 | 0.503556 | 0.766197 | 0.502289 |
| movielens | AUTOIFS | 2025 | EVALUATED | 0.762151 | 0.498450 | 0.763612 | 0.496706 |
| movielens | AUTOIFS | 2026 | EVALUATED | 0.762675 | 0.503079 | 0.764151 | 0.501653 |
| kuairand | M3OE_FIXED1 | 2022 | EVALUATED | 0.672570 | 0.437703 | 0.703934 | 0.416838 |
| kuairand | M3OE_FIXED1 | 2023 | EVALUATED | 0.661472 | 0.441086 | 0.701276 | 0.419566 |
| kuairand | M3OE_FIXED1 | 2024 | EVALUATED | 0.667442 | 0.448246 | 0.703855 | 0.438502 |
| kuairand | M3OE_FIXED1 | 2025 | EVALUATED | 0.664618 | 0.445590 | 0.704961 | 0.425986 |
| kuairand | M3OE_FIXED1 | 2026 | EVALUATED | 0.673870 | 0.445114 | 0.703103 | 0.431526 |
| kuairand | HYPERGATE | 2022 | EVALUATED | 0.676303 | 0.443965 | 0.710278 | 0.432498 |
| kuairand | HYPERGATE | 2023 | EVALUATED | 0.679684 | 0.443170 | 0.709674 | 0.444344 |
| kuairand | HYPERGATE | 2024 | EVALUATED | 0.684074 | 0.438047 | 0.712125 | 0.428987 |
| kuairand | HYPERGATE | 2025 | EVALUATED | 0.673873 | 0.446263 | 0.708649 | 0.445110 |
| kuairand | HYPERGATE | 2026 | EVALUATED | 0.677596 | 0.438230 | 0.708151 | 0.432734 |
| kuairand | AUTOIFS | 2022 | EVALUATED | 0.672969 | 0.434412 | 0.707636 | 0.429231 |
| kuairand | AUTOIFS | 2023 | EVALUATED | 0.674364 | 0.435529 | 0.708938 | 0.431899 |
| kuairand | AUTOIFS | 2024 | EVALUATED | 0.671217 | 0.433639 | 0.708491 | 0.426126 |
| kuairand | AUTOIFS | 2025 | EVALUATED | 0.675002 | 0.434132 | 0.708463 | 0.431776 |
| kuairand | AUTOIFS | 2026 | EVALUATED | 0.676076 | 0.435592 | 0.709321 | 0.431127 |

## Development Q selected (not formal evidence)

| Dataset | Method | Seed | Status | Macro AUC | Macro LogLoss | Natural AUC | Natural LogLoss |
|---|---|---:|---|---:|---:|---:|---:|
| movielens | M3OE_FIXED1 | 2022 | NOT_RUN |  |  |  |  |
| movielens | M3OE_FIXED1 | 2023 | NOT_RUN |  |  |  |  |
| movielens | HYPERGATE | 2022 | EVALUATED | 0.812700 | 0.471111 | 0.812706 | 0.470850 |
| movielens | HYPERGATE | 2023 | EVALUATED | 0.807629 | 0.475071 | 0.808052 | 0.474138 |
| movielens | AUTOIFS | 2022 | EVALUATED | 0.804029 | 0.478704 | 0.804856 | 0.477195 |
| movielens | AUTOIFS | 2023 | EVALUATED | 0.807952 | 0.474926 | 0.808334 | 0.473815 |
| kuairand | M3OE_FIXED1 | 2022 | NOT_RUN |  |  |  |  |
| kuairand | M3OE_FIXED1 | 2023 | NOT_RUN |  |  |  |  |
| kuairand | HYPERGATE | 2022 | EVALUATED | 0.715623 | 0.459294 | 0.743505 | 0.507624 |
| kuairand | HYPERGATE | 2023 | EVALUATED | 0.709398 | 0.461117 | 0.740336 | 0.509034 |
| kuairand | AUTOIFS | 2022 | EVALUATED | 0.697753 | 0.462080 | 0.741513 | 0.506700 |
| kuairand | AUTOIFS | 2023 | EVALUATED | 0.697397 | 0.462513 | 0.740927 | 0.507306 |

## Custom method reserved

| Method | Status | Macro AUC | Macro LogLoss |
|---|---|---:|---:|
| CUSTOM_METHOD_TBD | NOT_RUN |  |  |
