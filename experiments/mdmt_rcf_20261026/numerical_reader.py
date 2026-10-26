"""Equivalent coordinates for wholly unbounded FULL reader coefficients.

This module changes only optimizer coordinates. Its objective is always evaluated
in the original alpha/beta/w coordinates, with the original w ridge and boxes.
No training state, model tensor or optimizer snapshot is written by this module.
"""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from .state import array_hash


@dataclass(frozen=True)
class FreeWhitening:
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    receipt: dict

    def to_original(self, transformed):
        z=np.asarray(transformed,dtype=np.float64)
        return np.r_[z[:2],self.matrix@z[2:]]

    def to_transformed(self, original):
        theta=np.asarray(original,dtype=np.float64)
        return np.r_[theta[:2],np.linalg.solve(self.matrix,theta[2:])]

    def pull_gradient(self, original_gradient):
        g=np.asarray(original_gradient,dtype=np.float64)
        return np.r_[g[:2],self.matrix.T@g[2:]]


def original_components(theta,b,x,y,lam,ridge):
    """Audit components of the unchanged original objective."""
    theta=np.asarray(theta,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    correction=x@theta[2:];logits=theta[0]*b+theta[1]+correction
    bce=float(np.mean(np.logaddexp(0,logits)-y*logits))
    composite=float(lam*np.mean(correction*correction)/2)
    coefficient=float(ridge*np.dot(theta[2:],theta[2:])/2)
    return dict(logits=logits,correction=correction,bce=bce,composite_penalty=composite,
                original_coefficient_ridge=coefficient,objective=bce+composite+coefficient)


def original_objective_gradient(theta,b,x,y,lam,ridge):
    # Deliberately the same original-coordinate expression as readers.py.
    alpha,beta=theta[:2];w=theta[2:];correction=x@w;logits=alpha*b+beta+correction
    error=expit(logits)-y;n=len(y)
    value=np.mean(np.logaddexp(0,logits)-y*logits)+lam*np.mean(correction**2)/2+ridge*np.dot(w,w)/2
    gradient=np.r_[np.dot(error,b)/n,np.mean(error),x.T@error/n+lam*x.T@correction/n+ridge*w]
    return float(value),gradient


def projected_kkt(theta,gradient,bounds):
    projected=np.array(gradient,dtype=np.float64,copy=True)
    for j,(lo,hi) in enumerate(bounds):
        if lo is not None and theta[j]<=lo+1e-9:projected[j]=min(projected[j],0.)
        if hi is not None and theta[j]>=hi-1e-9:projected[j]=max(projected[j],0.)
    return float(np.max(np.abs(projected))) if len(projected) else 0.


def make_free_whitening(b,x,lam,ridge=1e-6):
    b=np.asarray(b,dtype=np.float64);x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or b.shape!=(len(x),) or not len(x) or x.shape[1]==0:
        raise ValueError('Whitening requires nonempty [N,M] free FULL features')
    if not np.isfinite(x).all() or not np.isfinite(b).all() or not np.isfinite(lam) or not np.isfinite(ridge) or lam<0 or ridge<=0:
        raise ValueError('Invalid free-reader matrix or original regularization')
    n,m=x.shape;p0=expit(b);v0=p0*(1-p0)
    gram=x.T@x/n
    h0=(x.T*v0)@x/n+lam*gram+ridge*np.eye(m)
    symmetry_before=float(np.max(np.abs(h0-h0.T)))
    h0=(h0+h0.T)/2
    eigenvalues,eigenvectors=np.linalg.eigh(h0)
    if not np.isfinite(eigenvalues).all() or eigenvalues[0]<=0:
        raise FloatingPointError('Free-reader Hessian is not numerically positive definite; original ridge unchanged')
    matrix=(eigenvectors/np.sqrt(eigenvalues))@eigenvectors.T
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix)<m:
        raise FloatingPointError('Free-reader inverse square root is not numerically invertible')
    reconstructed=(eigenvectors*eigenvalues)@eigenvectors.T
    whitened=matrix.T@h0@matrix
    scale=np.maximum(np.std(x,axis=0),np.finfo(np.float64).tiny)
    receipt=dict(revision='NUMERICAL_DELIVERY_V1_FREE_W_ONLY',rows=n,features=m,
        original_start='alpha1_beta0_w0',lambda_value=float(lam),original_ridge=float(ridge),
        original_hessian_hash=array_hash(h0),preconditioner_hash=array_hash(matrix),
        hessian_eigenvalues=eigenvalues.tolist(),hessian_condition=float(eigenvalues[-1]/eigenvalues[0]),
        preconditioner_condition=float(np.sqrt(eigenvalues[-1]/eigenvalues[0])),
        symmetry_before=float(symmetry_before),hessian_reconstruction_max_error=float(np.max(np.abs(reconstructed-h0))),
        whitening_identity_max_error=float(np.max(np.abs(whitened-np.eye(m)))),
        feature_numerical_rank=int(np.linalg.matrix_rank(x)),feature_std_min=float(scale.min()),feature_std_max=float(scale.max()),
        feature_column_norms=np.linalg.norm(x,axis=0).tolist(),eigenvalue_adjustment='NONE',objective_or_ridge_changes=False)
    return FreeWhitening(matrix,eigenvalues,eigenvectors,receipt)


def transformed_objective(z,whitening,b,x,y,lam,ridge,objective_gradient=None):
    objective=original_objective_gradient if objective_gradient is None else objective_gradient
    theta=whitening.to_original(z);value,gradient=objective(theta,b,x,y,lam,ridge)
    return value,whitening.pull_gradient(gradient)


def solve_free_full(b,x,y,lam,bounds,config,*,objective_gradient=None):
    """Return original coefficients and an honest registered-budget receipt.

A numerical failure is returned as accepted=False (with its actual coefficient
and residuals) so a caller can preserve the failure evidence before raising.
"""
    b=np.asarray(b,np.float64);x=np.asarray(x,np.float64);y=np.asarray(y,np.float64)
    if len(bounds)!=x.shape[1] or not all(lo is None and hi is None for lo,hi in bounds):
        raise ValueError('Equivalent whitening applies only to wholly unbounded FULL feature coefficients')
    if y.shape!=(len(b),) or not np.isfinite(y).all():raise ValueError('Invalid training labels')
    ridge=float(config['reader']['coefficient_ridge']);whitening=make_free_whitening(b,x,lam,ridge)
    box=[(.25,4.),(-4.,4.)]+[(None,None)]*x.shape[1]
    start=np.r_[1.,0.,np.zeros(x.shape[1])]
    objective=original_objective_gradient if objective_gradient is None else objective_gradient
    fn=lambda z:transformed_objective(z,whitening,b,x,y,lam,ridge,objective)
    attempts=[];started=time.monotonic();result=None
    for maxiter in (config['reader']['max_iter'],config['reader']['numerical_retry_max_iter']):
        attempt_start=time.monotonic()
        result=minimize(fn,start,jac=True,method='L-BFGS-B',bounds=box,
            options=dict(maxiter=int(maxiter),gtol=float(config['reader']['gtol']),ftol=1e-12,maxls=40))
        theta=whitening.to_original(result.x);value,gradient=objective(theta,b,x,y,lam,ridge)
        transformed_value,transformed_gradient=fn(result.x)
        original_residual=projected_kkt(theta,gradient,box)
        transformed_residual=projected_kkt(result.x,transformed_gradient,box)
        free=[j for j,(lo,hi) in enumerate(box) if (lo is None or theta[j]>lo+1e-9) and (hi is None or theta[j]<hi-1e-9)]
        attempts.append(dict(maxiter=int(maxiter),scipy_success=bool(result.success),message=str(result.message),
            iterations=int(result.nit),evaluations=int(result.nfev),gradient_evaluations=int(result.njev),objective=float(value),
            original_projected_kkt=float(original_residual),transformed_projected_kkt=float(transformed_residual),
            original_free_gradient_max=float(np.max(np.abs(gradient[free]))) if free else 0.,
            original_gtol_satisfied=bool(original_residual<=config['reader']['gtol']),
            transformed_gtol_satisfied=bool(transformed_residual<=config['reader']['gtol']),
            alpha=float(theta[0]),beta=float(theta[1]),alpha_beta_feasible=bool(.25<=theta[0]<=4 and -4<=theta[1]<=4),
            mapped_objective_agreement=float(abs(value-transformed_value)),seconds=time.monotonic()-attempt_start))
        # Preserve the existing same-start numerical retry contract: only an
        # unsuccessful SciPy termination triggers the registered 600 retry.
        if result.success:break
    finite=bool(np.isfinite(theta).all() and np.isfinite(value))
    accepted=bool(finite and (result.success or original_residual<=1e-5))
    receipt=dict(status='CONVERGED' if result.success else 'KKT_CONVERGED' if accepted else 'NUMERICAL_FAILURE',
        accepted=accepted,rows=len(y),objective=float(value),iterations=int(result.nit),
        kkt_residual=float(original_residual),free_gradient_max=attempts[-1]['original_free_gradient_max'],
        original_gtol_satisfied=attempts[-1]['original_gtol_satisfied'],
        transformed_kkt_residual=float(transformed_residual),retry=len(attempts)>1,message=str(result.message),bounds=box,
        numerical_revision='NUMERICAL_DELIVERY_V1_FREE_W_ONLY',preconditioner=whitening.receipt,
        attempts=attempts,elapsed_seconds=time.monotonic()-started,
        original_acceptance_rule='finite AND (SciPy success OR original projected KKT <=1e-5 after registered retry)',
        outer_labels_or_metrics_used=False,coefficient_coordinates='original_alpha_beta_w')
    return theta,receipt
