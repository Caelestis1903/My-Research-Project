# P5 same-function action derivation and review

To finish the specified decision-consequence check, use an explicit analytic
construction, not retrospectively fabricated trained D3 gates. Two source
records have labels (1,0), current predictions (.5,.5) and g_sp=(.2,.8).
Swapping both softmax outputs preserves each product g_sp*g_sh=.16 and every
published product-fusion forward value (including similarly generated bias).
Freeze this equivalent function and restrict the action to a common final output
bias perturbation u. Deployment weights are declared (.8,.2), rather than inferred
from gate names. This is a one-dimensional same-function perturbation family.

Specificity weights .1+.5*g_sp are (.2,.5) and (.5,.2). The weighted source BCE
derivatives at bias zero are +.075 and -.075; the declared deployment derivative
is -.3. Hence small positive-learning-rate gradient steps from the two equivalent
proxy orientations have opposite signs of deployment risk change. Exact BCE
readout uses lr=.1, without a step-size search. Ordinary weighting by known
deployment population supplies the correct derivative and solves this example.

This is not an MDMT-specific new failure, a simulation of the full D3 optimizer,
or an explanation of official D3 empirical results. It strengthens the narrow
semantic audit beyond a changed gate number while falsifying the novelty of
replacing it with conventional validation/population utility in this example.

Primary pre-execution review: **PASS**. The source weighting derivative holds
gates fixed under the declared output-bias action; all gates belong to [.0,1.0]
and normalization is explicit. This is a theoretical analysis completion, not
a new trainable algorithm experiment. Save exact numbers in
`theory_data/P5_same_action_decision.json`.
