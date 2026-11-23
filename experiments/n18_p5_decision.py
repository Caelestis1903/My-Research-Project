"""P5 declared same-function action counterexample; not official D3 training."""
import numpy as np
from scipy.special import expit
from n18_common import OUT,dump

def main():
    # Two source records, same current output .5, deterministic labels 1 and 0.
    # D3's product gates both equal .16 in both equivalent orientations.
    gates=np.array([.2,.8]);labels=np.array([1.,0.]);deployment=np.array([.8,.2])
    probability=np.full(2,.5);lr=.1
    risk=lambda z:float(np.dot(deployment,np.logaddexp(0,z)-labels*z))
    rows=[]
    for name,g in [('original',gates),('swapped',1-gates)]:
        weights=.1+.5*g;gradient=float(np.mean(weights*(probability-labels)))
        update=-lr*gradient
        rows.append(dict(orientation=name,gsp=g.tolist(),gate_product=(g*(1-g)).tolist(),
            weighted_loss_shared_output_bias_gradient=gradient,bias_update=update,
            target_risk_before=risk(0),target_risk_after=risk(update),target_risk_delta=risk(update)-risk(0)))
    gradient=float(np.dot(deployment,probability-labels));update=-lr*gradient
    dump(OUT/'theory_data/P5_same_action_decision.json',dict(status='COMPLETE',
        scope='analytic two-source shared-output-bias action; product fusion held fixed; no official D3 training',
        new_algorithm=False,current_predictions=probability.tolist(),labels=labels.tolist(),deployment_weights=deployment.tolist(),
        learning_rate=lr,orientations=rows,conventional_known_population_gradient=dict(gradient=gradient,bias_update=update,
        target_risk_delta=risk(update)-risk(0)),
        conclusion_scope='nonunique proxy can reverse risk-sign of this fixed action family; ordinary known-population weighting resolves the construction'))

if __name__=='__main__':main()
