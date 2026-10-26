"""Focused coordinate-equivalence checks, never research performance data."""
from __future__ import annotations
import numpy as np
from scipy.special import expit
from .numerical_reader import make_free_whitening,original_components,transformed_objective,solve_free_full
from .readers import objective_gradient


def verify_equivalence(b,x,y,lam,config,*,fixture_identity=None):
    b=np.asarray(b,np.float64);x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    ridge=config['reader']['coefficient_ridge'];wht=make_free_whitening(b,x,lam,ridge);rng=np.random.default_rng(9142501)
    points=[np.r_[1.,0.,np.zeros(x.shape[1])],np.r_[.25,-4.,rng.normal(0,.05,x.shape[1])],np.r_[4.,4.,rng.normal(0,.05,x.shape[1])],np.r_[1.13,-.21,rng.normal(0,.05,x.shape[1])]]
    records=[]
    for point in points:
        z=wht.to_transformed(point);mapped=wht.to_original(z);direct=original_components(point,b,x,y,lam,ridge)
        mapped_components=original_components(mapped,b,x,y,lam,ridge)
        correction=(x@wht.matrix)@z[2:]
        logits=z[0]*b+z[1]+correction
        explicit=dict(bce=float(np.mean(np.logaddexp(0,logits)-y*logits)),composite_penalty=float(lam*np.mean(correction**2)/2),original_coefficient_ridge=float(ridge*(z[2:]@(wht.matrix.T@wht.matrix)@z[2:])/2))
        explicit['objective']=sum(explicit.values())
        errors={k:abs(explicit[k]-direct[k]) for k in ('bce','composite_penalty','original_coefficient_ridge','objective')}
        limits={k:1e-12+1e-10*abs(direct[k]) for k in errors}
        if not all(errors[k]<=limits[k] for k in errors):raise AssertionError(('Objective component equivalence',errors,limits))
        old_value,old_gradient=objective_gradient(mapped,b,x,y,lam,ridge)
        value,grad=transformed_objective(z,wht,b,x,y,lam,ridge,objective_gradient)
        chain=np.r_[old_gradient[:2],wht.matrix.T@old_gradient[2:]]
        direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction);step=1e-5
        plus=transformed_objective(z+step*direction,wht,b,x,y,lam,ridge,objective_gradient)[0]
        minus=transformed_objective(z-step*direction,wht,b,x,y,lam,ridge,objective_gradient)[0]
        analytic=float(grad@direction);numeric=float((plus-minus)/(2*step));fd_error=abs(analytic-numeric)
        if fd_error>1e-7*(1+abs(analytic)):raise AssertionError(('Directional gradient equivalence',fd_error,analytic,numeric))
        records.append(dict(alpha=float(point[0]),beta=float(point[1]),original_inverse_map_max_error=float(np.max(np.abs(mapped-point))),logits_max_error=float(np.max(np.abs(logits-direct['logits']))),components_abs_errors=errors,chain_gradient_max_error=float(np.max(np.abs(grad-chain))),directional_gradient_abs_error=fd_error,original_objective_evaluator_error=abs(value-old_value),alpha_beta_constraints_unchanged=True))
    theta,solver=solve_free_full(b,x,y,lam,[(None,None)]*x.shape[1],config,objective_gradient=objective_gradient)
    fitted=original_components(theta,b,x,y,lam,ridge)
    return dict(status='PASS' if solver['accepted'] else 'EQUIVALENCE_PASS_SOLVER_FAILURE',kind='coordinate_numerics_only_not_research_performance',fixture_identity=fixture_identity,rows=len(y),features=x.shape[1],component_checks=records,preconditioner=wht.receipt,solver=solver,fitted_logit_hash=__import__('hashlib').sha256(np.ascontiguousarray(fitted['logits']).tobytes()).hexdigest(),outer_metrics_used=False)


def verify_synthetic(config):
    rng=np.random.default_rng(81029);latent=rng.normal(size=(320,5));mix=rng.normal(size=(5,12));x=latent@mix+1e-3*rng.normal(size=(320,12));b=rng.normal(size=320);y=rng.binomial(1,expit(.9*b+.1*latent[:,0]))
    r=verify_equivalence(b,x,y,.001,config,fixture_identity={'scope':'synthetic_correlated_matrix_basic_check','not_failed_research_fixture':True})
    from .numerical_reader import solve_free_full
    try:solve_free_full(b,x,y,.001,[(-1.,1.)]*12,config)
    except ValueError:r['bounded_unit_readers_rejected']=True
    else:raise AssertionError('Bounded reader coordinates must remain unchanged')
    return r
