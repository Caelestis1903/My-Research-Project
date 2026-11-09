"""Paper Eq.(4) CCP; physical alpha retains the raw Gram's absolute scale."""
import time
import warnings
import numpy as np
import cvxpy as cp

class CCPSolver:
    def __init__(self,niter=100,tol=1e-6,ridge=1e-10):
        self.niter,self.tol,self.ridge=niter,tol,ridge
        self.problems={}

    def problem(self,n):
        if n not in self.problems:
            a=cp.Variable(n,nonneg=True); H=cp.Parameter((n,n)); q=cp.Parameter(n)
            constraints=[-cp.log(a)-cp.log(H@a)<=0]
            self.problems[n]=(a,H,q,cp.Problem(cp.Minimize(q@a),constraints),cp.Problem(cp.Minimize(cp.sum(a)),constraints))
        return self.problems[n]

    def solve(self,gram,warm=None):
        start=time.perf_counter(); G=np.asarray(gram,dtype=np.float64); G=(G+G.T)/2
        row=dict(success=False,reason=None,iterations=0,solver_status=None,seconds=0.,gram=G.tolist())
        def finish(alpha,reason):
            row['reason']=reason; row['seconds']=time.perf_counter()-start
            return alpha,row
        if G.ndim!=2 or G.shape[0]!=G.shape[1] or not np.isfinite(G).all() or np.any(np.diag(G)<=1e-24): return finish(None,'near_zero_or_invalid_gram')
        s=float(np.trace(G)/len(G)); H=G/s+self.ridge*np.eye(len(G))
        eig=np.linalg.eigvalsh(G); cond=np.linalg.cond(G)
        row.update(scale=s,eigenvalues=eig.tolist(),condition=float(cond) if np.isfinite(cond) else None,diagonal=np.diag(G).tolist())
        a,Hpar,q,problem,feasible=self.problem(len(G)); Hpar.value=H
        prev=np.ones(len(G)) if warm is None else np.asarray(warm)*np.sqrt(s)
        options=dict(solver='CLARABEL',max_iter=200,tol_gap_abs=min(1e-7,self.tol/10),tol_gap_rel=min(1e-7,self.tol/10),tol_feas=min(1e-7,self.tol/10),warm_start=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                if not np.isfinite(prev).all() or np.any(prev<=0) or np.any(H@prev<=0):
                    feasible.solve(**options)
                    if feasible.status not in ('optimal','optimal_inaccurate') or a.value is None: return finish(None,'no_feasible_initialization')
                    prev=a.value.copy()
                for it in range(self.niter):
                    q.value=1/prev+H.T@(1/(H@prev)); a.value=prev
                    problem.solve(**options); row['solver_status']=problem.status; row['iterations']=it+1
                    if problem.status not in ('optimal','optimal_inaccurate') or a.value is None: return finish(None,'conic_status')
                    current=np.asarray(a.value).copy()
                    if not np.isfinite(current).all() or np.any(current<=0) or np.any(H@current<=0): return finish(None,'invalid_ccp_solution')
                    reg=float(np.linalg.norm(H@current-1/current)/(np.linalg.norm(H@current)+np.linalg.norm(1/current)))
                    alpha=current/np.sqrt(s); utility=G@alpha; residual=float(np.max(np.abs(alpha*utility-1)))
                    row.update(regularized_residual=reg,raw_residual=residual,alpha=alpha.tolist(),utility=utility.tolist(),alpha_sum=float(alpha.sum()),alpha_min=float(alpha.min()))
                    prev=current
                    if reg<=self.tol and residual<=1e-4: break
            if reg>self.tol or residual>1e-4 or np.any(utility<=0): return finish(None,'residual_or_utility')
            row['success']=True
            return finish(alpha,None)
        except (cp.error.SolverError,ValueError,FloatingPointError,ZeroDivisionError) as error:
            row['exception']=repr(error); return finish(None,'solver_exception')
