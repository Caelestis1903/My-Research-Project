"""One post-Adam update with innovation measured from the old shadow."""
import torch


@torch.no_grad()
def update(phi, theta, q, rho0, h):
    innovation = theta-phi if q.shape[1] and h != 1 else None
    phi.mul_(rho0).add_(theta, alpha=1-rho0)
    if innovation is not None:
        phi.add_(q@(q.T@innovation), alpha=rho0-rho0**(1/h))

