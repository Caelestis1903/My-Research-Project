# Cycle03 literature and novelty boundary

Primary literature was browsed for this handoff. The papers below support mathematical ingredients and experimental questions; none supplies evidence of efficacy for this repository's natural MovieLens/KuaiRand M3oE protocol. The local history and exact C2 source take precedence for what was actually tested here. This is a focused search, not a claim of exhaustive priority review.

## Timing as an intervention

[Golatkar, Achille and Soatto, *Time Matters in Regularizing Deep Networks: Weight Decay and Data Augmentation Affect Early Learning Dynamics, Matter Little Near Convergence*, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/file/87784eca6b0dea1dff92478fb786b401-Paper.pdf) examines when regularization is applied, including temporal windows. It supplies precedent for timing experiments, not a prediction that this score loss should be early-only. Its image models, data and regularizers differ from M3oE and NQR. Cycle03's equal-length windows are a local falsifier motivated by C2's stage reversal. No claim is made that the chronological schedule is a new curriculum algorithm.

## Conditional ranking and moment calculations

[Ying, Wen and Lyu, *Stochastic Online AUC Maximization*, NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/c52f1bd66cc19d05628bd8bf27af3ad6-Paper.pdf) formulates conditional positive/negative square AUC risk and a representation suitable for online optimization. Thus pair-square ranking and moment-based computation are established ingredients. C2's natural same-class completion and the proposed opportunity-flattened comparison have different prevalence weighting from the paper's conditional risk. Its optimization results cannot be transferred to this nonlinear network, probability scores, source Adam or selected EMA. The identity `mean(1-p_plus+p_minus)^2=(1-mu_plus+mu_minus)^2+v_plus+v_minus` also follows directly by expanding the square; it is not an invention attributable to a new arm name.

## Parameter neighborhoods

[Foret, Kleiner, Mobahi and Neyshabur, *Sharpness-Aware Minimization for Efficiently Improving Generalization*, ICLR 2021](https://arxiv.org/pdf/2010.01412) gives the parameter-neighborhood minimax loss and first-order SAM update. [The authors' implementation](https://github.com/google-research/sam) identifies the existing method. SAM is a genuine mathematical alternative to the score moments and temporal schedule, but it was already an unselected C2 proposal. A new name or use of source Adam would not establish algorithmic novelty. No radius or SAM run is authorized in this cycle.

## Nearby gradient manipulation is not equivalent

[Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) projects a conflicting task gradient against another task gradient. This illustrates why a geometric projection must be identified by its space and constraint. Historical DT-PGNA in this repository projected a feature perturbation using a pair-specific constraint; it was not PCGrad on the model's task gradients. Neither operation appears in the chosen scalar temporal loss. C1's actual addendum, not the common word “projection,” determines that local distinction.

## Claim register

| Claim | Status |
|---|---|
| New square-ranking objective or moment identity | REJECTED; established ingredients. |
| New globally novel curriculum or SAM variant | REJECTED; not claimed or established. |
| Locally untested exact EARLY10/LATE10 schedule on frozen C2 NQR | SUPPORTED by the reviewed experiment index and current files; this is a local protocol distinction, not a priority claim. |
| Timing explains C2's selected deficit | UNPROVED; target of the fixed-E50 contrast and primary selected-CLEAN falsifier. |
| Equal scheduled exposure isolates intrinsic phase from gradient dose | REJECTED; it matches epochs/rows/updates/lambda budget, not realized gradients, Adam history, pair opportunities or EMA recency. |
| Natural domain size is the causal driver | WEAK evidence; prevalence, features, task, stage and sharing remain competing explanations. |
| A ranking/calibration gain has been established | REJECTED for the candidate. C2's relative LL findings do not establish selected Macro AUC efficacy. |

The selected plan makes no novelty case from failed PAIR/DT-PGNA/TGA/NDP/TV/AXIS/NQR families. Its justification is information gained from a minimal falsifier. If a publication requires a novel effective algorithm, this handoff has not met that requirement; the current decision only advances a scientific experiment to formal design.
