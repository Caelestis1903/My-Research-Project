"""Actual-model numerical checks used by the full-batch engineering smoke."""
import copy
import torch
from .core import forward, sensitivity


def check_sensitivity(model, small):
    embedding32, logits32, gradient32 = sensitivity(model, small)
    with torch.no_grad():
        torch.testing.assert_close(forward(model,small)[1],logits32,rtol=0,atol=0)
    _, z, embedding = forward(model,small,probe=True)
    maximum = 0.
    for t in range(2):
        summed = torch.autograd.grad(z[:,t].sum(),embedding,retain_graph=True)[0]
        for row in range(small.shape[1]):
            one = torch.autograd.grad(z[row,t],embedding,retain_graph=True)[0]
            _, _, alone = sensitivity(model,small[:,row:row+1])
            torch.testing.assert_close(summed[row],one[row],rtol=1e-5,atol=1e-6)
            torch.testing.assert_close(summed[row],alone[0,t],rtol=1e-5,atol=1e-6)
            maximum = max(maximum,float((summed[row]-alone[0,t]).abs().max()))
    probe = copy.deepcopy(model).double()
    embedding64, _, gradient64 = sensitivity(probe,small.double())
    torch.testing.assert_close(gradient32.double(),gradient64,rtol=1e-2,atol=1e-4)
    direction = torch.ones_like(embedding64)/embedding64.shape[1]**.5
    with torch.no_grad():
        finite = (forward(probe,small.double(),1e-5*direction)[1]-
                  forward(probe,small.double(),-1e-5*direction)[1])/(2e-5)
    derivative = (gradient64*direction[:,None]).sum(2)
    torch.testing.assert_close(finite,derivative,rtol=1e-3,atol=1e-7)
    assert all(p.grad is None for p in model.parameters())
    return dict(probe_dtype='float64',step=1e-5,fd_rtol=1e-3,fd_atol=1e-7,
        fp32_rtol=1e-2,fp32_atol=1e-4,fd_max_absolute_error=float((finite-derivative).abs().max()),
        fp32_gradient_max_absolute_error=float((gradient32.double()-gradient64).abs().max()),
        unrelated_row_gradient_max_absolute_error=maximum)
