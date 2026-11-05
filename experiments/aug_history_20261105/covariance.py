"""FP64 drift statistics and positive trace-sixteen matrix-exponential roots."""
import hashlib
import torch


def psd_eigh(matrix):
    if matrix.shape[-2:]!=(16,16) or not bool(torch.isfinite(matrix).all()):
        raise ValueError('Invalid field statistic')
    matrix=(matrix.double()+matrix.double().transpose(-1,-2))/2
    values,vectors=torch.linalg.eigh(matrix)
    tolerance=torch.maximum(values[...,-1:].abs()*1e-10,values.new_tensor(1e-12))
    if bool((values < -tolerance).any()):
        raise FloatingPointError('Drift statistic is not positive semidefinite')
    return values.clamp_min(0),vectors


@torch.no_grad()
def statistics(u):
    if u.ndim!=4 or u.shape[1]!=2 or u.shape[-1]!=16 or not len(u) or not bool(torch.isfinite(u).all()):
        raise ValueError('Expected finite drift [N,2,fields,16] with N>0')
    value=u.detach().cpu().double()
    norms=torch.linalg.vector_norm(value,dim=-1,keepdim=True)
    unit=torch.where(norms>1e-12,value/norms.clamp_min(1e-12),torch.zeros_like(value))
    matrix=torch.einsum('ntfi,ntfj->tfij',unit,unit)/len(unit)
    values,vectors=psd_eigh(matrix)
    if bool((values.sum(-1)>1+1e-10).any()):
        raise FloatingPointError('Unit-drift statistic trace exceeds one')
    return (vectors*values.unsqueeze(-2))@vectors.transpose(-1,-2)


@torch.no_grad()
def covariance_root(matrix,gamma):
    values,vectors=psd_eigh(matrix)
    identity=torch.eye(16,dtype=torch.float64,device=matrix.device).expand_as(matrix)
    if gamma==0:
        return identity.clone()
    exponent=gamma*values
    weights=(exponent-exponent.max(-1,keepdim=True).values).exp()
    spectrum=16*weights/weights.sum(-1,keepdim=True)
    root=(vectors*spectrum.sqrt().unsqueeze(-2))@vectors.transpose(-1,-2)
    return torch.where(matrix.eq(0).all(-1).all(-1)[...,None,None],identity,root).detach()


def fixed_rotations(dataset,seed,fields):
    matrices=[]
    for field in range(fields):
        key=f'AH1105|{dataset}|{seed}|0|C_ROTATION_F{field}|0|0'
        value=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
        generator=torch.Generator().manual_seed(value)
        q,r=torch.linalg.qr(torch.randn(16,16,dtype=torch.float64,generator=generator))
        matrices.append(q*torch.where(r.diag()>=0,1.,-1.))
    return torch.stack(matrices)


def controlled_statistics(matrix,mode,rotation=None):
    if mode=='merged':
        return matrix.mean(0,keepdim=True).expand_as(matrix).clone()
    if mode=='rotated':
        return rotation[None]@matrix@rotation[None].transpose(-1,-2)
    if mode!='task':
        raise ValueError('Unknown conditional covariance control')
    return matrix
