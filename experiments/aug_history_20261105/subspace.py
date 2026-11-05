"""At most two response columns; no dense parameter-space projector."""
import torch


@torch.no_grad()
def orthonormal_basis(columns):
    if columns.ndim != 2 or columns.shape[1] > 2 or not torch.isfinite(columns).all():
        raise ValueError('Invalid response columns')
    double = columns.double()
    norms = torch.linalg.vector_norm(double, dim=0)
    normalized = double[:, norms > 0]/norms[norms > 0]
    if normalized.shape[1] == 0:
        return columns[:, :0].clone(), dict(rank=0, column_norms=norms.tolist(), orthogonality_error=0.)
    gram = normalized.T @ normalized
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    threshold = max(1e-12, 1e-10*float(eigenvalues[-1]))
    if float(eigenvalues[0]) < -threshold:
        raise FloatingPointError('Negative response Gram eigenvalue')
    keep = eigenvalues > threshold
    q64 = normalized @ (eigenvectors[:, keep]/eigenvalues[keep].sqrt())
    error64 = float((q64.T@q64-torch.eye(q64.shape[1], device=q64.device)).abs().max())
    if error64 > 1e-10:
        raise FloatingPointError(f'Response orthogonality error: {error64}')
    q = q64.to(columns.dtype)
    error = float((q.double().T@q.double()-torch.eye(q.shape[1], device=q.device)).abs().max())
    if error > 1e-6:
        raise FloatingPointError(f'FP32 response orthogonality error: {error}')
    return q, dict(rank=q.shape[1], column_norms=norms.tolist(), orthogonality_error=error,
                   orthogonality_error_fp64=error64)


def signed_basis(q, signs):
    return q*signs[:, None]

