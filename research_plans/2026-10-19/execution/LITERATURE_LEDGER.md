# Literature scope and novelty risks — checked 2026-10-19

This ledger is independent of formal outcomes; it does not claim that a mechanism was observed or that novelty is established.

|Primary source|Prior contribution / boundary for this study|
|---|---|
|[Loshchilov & Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)|Adaptive coupled L2 and decoupled shrinkage are different algorithms. Demonstrating AdamW changes factor norms is not a new optimization method.|
|[Hoff, Lasso, fractional norm and structured sparse estimation using a Hadamard product parametrization (2017)](https://doi.org/10.1016/j.csda.2017.06.007)|Hadamard-factor L2 penalties can represent sparsity penalties, including structured variants. Product-factor shrinkage alone is not a new theoretical discovery.|
|[Smoothing the Edges: Smooth Optimization for Sparse Regularization Using Hadamard Overparametrization (2026)](https://doi.org/10.1007/s10994-026-06997-0)|Recent general treatment further narrows novelty of explaining multiplicative factors as surrogate sparse regularization. This does not itself establish the dynamics of a nonlinear MDMT model.|
|[Pezeshki et al., Gradient Starvation (2020)](https://arxiv.org/abs/2011.09468)|Feature learning competition under cross-entropy is already studied. A weak path may fit this broader phenomenon, but module deletion in our architectures is not the theorem's identical setting.|
|[Kosson et al., Rotational Equilibrium (2023)](https://arxiv.org/abs/2305.17212)|Weight decay, normalization and adaptive updates jointly affect weight magnitude and effective rotation; parameter norm cannot stand in for prediction contribution.|
|[Yeşil & Turgut, Star+ (2024)](https://arxiv.org/abs/2406.16568)|Additive/adaptive/concatenation/gating fusion alternatives to STAR are prior art. The additive G2 control is a mechanism discriminator, not an original algorithm claim.|
|[Bernstein et al., Learning compositional functions via multiplicative weight updates (2020)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/9a32ef65c42085537062753ec435750f-Abstract.html)|Compositional optimization/relative weight updates have extensive prior work. No proposed optimizer here is claimed novel merely because it preserves a multiplicative path.|

## Analytic competing explanation (not an empirical result)

For a single effective coordinate across D domains, write w_d=s*v_d and use a coupled quadratic penalty (lambda/2)*(s²+sum_d v_d²). For fixed effective vector w, minimizing over the scale gauge gives

min_s (lambda/2)*(s²+||w||²/s²) = lambda*||w||_2.

For w=0 the infimum is attained by zero factors. The scalar D=1 case is lambda*|w|. This is a direct algebraic instance of established Hadamard/structured-sparsity theory, not a new theorem. It suggests a group-sparsity interpretation of simultaneously regularizing both factors; it does not prove finite-time Adam collapse, prove the network reaches that infimum, or imply predictive harm. PN, nonlinearities, downstream scale freedom and state history limit direct global conclusions. A factor's norm can also shrink while its partner grows without changing the function.

G1's single-factor/both-factor WD controls, full-model WD0/AdamW and local update counterfactuals discriminate regularizer geometry from global optimizer effects. G2 no-bypass and additive structures discriminate pathway competition and parameterization. Functional contribution and selected/final cell risk, rather than norms, decide whether any change is useful. Literature coverage is a separate axis from empirical reproducibility and causal usefulness; no empirical direction is canceled by this ledger.

For the additive control w_d=s+v_d with the same number of factors, minimizing (lambda/2)*(s²+sum_d v_d²) over s gives (lambda/2)*(sum_d w_d²-(sum_d w_d)²/(D+1)). Thus equal nominal parameter count does not hold induced regularization geometry fixed. This elementary scale-elimination calculation is a competing explanation, not a new method/theorem. The additive and multiplicative controls also have different effective initialization magnitudes when the same Xavier factor recipe is used; a positive additive outcome alone cannot isolate regularizer geometry, gradient conditioning and initialization. G1 retains identical factor initialization for its decay interventions; G2 bypass calibration addresses initial pathway balance without asserting these are identical questions.
