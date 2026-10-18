# N19 initial implementation report

Sources: n19_data.py (full populations, fixed meta/selection/development and declared
controlled mechanisms), n19_models.py (full trainable model variants),
n19_objectives.py (noisy observation likelihoods and joint masks), n19_train.py
(from-scratch full-epoch runner), n19_queue.py (mechanical GPU scheduling only).
P4 reliability alternative will have an addendum before its implementation/run.
P3 dictionary and P5 coalition alternatives already have implementation but are
not used to select/stop their direction before primary development results.

The full M3oE forward is ported into a new wrapper, preserving historical source.
Shared STAR/experts are computed once for two tasks, towers evaluated only on
actual-domain rows; rowwise LayerNorm makes this numerically equivalent to the
historical full-row tower computation. Basic paired forward error was 5.96e−8.
All source experts remain trainable. Routing retains historical emb.detach().
Natural fixed-fusion objective is Sample BCE, one Adam. Explicit original and
original_sample arms enable learned fusion; literal original reference retains
Equal objective, main optimizer overlap and extra epoch Adam architecture update.
These choices are not described as default bug fixes or mixed as one protocol.

Basic checks passed: syntax; full forward/backward for rank, codebook, interaction,
regret/coalition and mask models; all independent/axis/nested/joint channel
likelihoods finite and differentiable; stability feature/schema shape consistent.
Results in basic_checks.json. These are not research results or model training.

Large full model/optimizer/RNG/sampler checkpoints are kept under
/home_new/xzf_n19_artifacts, separated by suite/dataset/method/seed/tag. Raw configs,
epoch selection/development metrics, selected/final validation predictions,
mechanism readings, progress and receipts live in N19/runs. Test reading remains
deferred. Resume restores state and truncates only unfinished epoch rows in its
own new run. No historical artifact is overwritten.

P2 visible clean training labels are zero outside the declared audit before the
likelihood call. P6 unqueried labels are zero-filled before losses; copula update
uses jointly revealed labels only. P6 annotation/execution cost is a per-visit
controlled simulator, not real annotation savings with reusable label caches.

GPU queue supports independent device admission, max5 processes/device and predicted
memory<90%; task count and measurements determine actual concurrency. It reports
only commands, exit codes and raw statuses, never changes research decisions.
