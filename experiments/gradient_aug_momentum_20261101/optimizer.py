"""Source-separated first moments with a standard mixed-gradient second moment."""
import math
import torch


class SADMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr=.01, betas=(.9, .999), eps=1e-8,
                 weight_decay=0., beta_delta=.99, lambda_aug=.5):
        if not 0 < beta_delta < 1:
            raise ValueError('beta_delta=0 uses ordinary mixed-gradient Adam')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                    beta_delta=beta_delta, lambda_aug=lambda_aug))

    @torch.no_grad()
    def step(self, real_grads, delta_grads, diagnose=False):
        pairs = iter(zip(real_grads, delta_grads))
        sums = dict(q2=0., m2=0., qhat2=0., mhat2=0., delta2=0., qdot=0.,
                    update2=0., reference2=0., update_dot=0., difference2=0.)
        for group in self.param_groups:
            beta1, beta2 = group['betas']; bd = group['beta_delta']; lam = group['lambda_aug']
            for param in group['params']:
                gr, delta = next(pairs)
                if gr is None and (delta is None or lam == 0):
                    continue
                state = self.state[param]
                if not state:
                    state.update(step=torch.tensor(0.), m_real=torch.zeros_like(param),
                                 q_delta=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param))
                r = torch.zeros_like(param) if gr is None else gr.detach()
                if group['weight_decay']:
                    r = r.add(param, alpha=group['weight_decay'])
                d = torch.zeros_like(param) if delta is None else delta.detach()
                mixed = r.add(d, alpha=lam)
                mr, q, v = state['m_real'], state['q_delta'], state['exp_avg_sq']
                reference_m = (mr.add(q, alpha=lam).lerp(mixed, 1-beta1)) if diagnose else None
                state['step'].add_(1)
                count = int(state['step'].item())
                mr.lerp_(r, 1-beta1); q.lerp_(d, 1-bd)
                v.mul_(beta2).addcmul_(mixed, mixed, value=1-beta2)
                bc1, bcd, bc2 = 1-beta1**count, 1-bd**count, 1-beta2**count
                denominator = (v.sqrt()/math.sqrt(bc2)).add_(group['eps'])
                numerator = mr/bc1 + lam*q/bcd
                if diagnose:
                    mh, qh = mr/bc1, q/bcd
                    direction = numerator/denominator
                    reference = (reference_m/bc1)/denominator
                    for name, value in [('q2', q), ('m2', mr), ('qhat2', qh), ('mhat2', mh),
                                        ('delta2', d), ('update2', direction), ('reference2', reference),
                                        ('difference2', direction-reference)]:
                        sums[name] += float(value.double().square().sum())
                    sums['qdot'] += float((qh.double()*d.double()).sum())
                    sums['update_dot'] += float((direction.double()*reference.double()).sum())
                param.addcdiv_(numerator, denominator, value=-group['lr'])
        if not diagnose:
            return None
        lam = self.param_groups[0]['lambda_aug']
        cosine = sums['qdot']/math.sqrt(sums['qhat2']*sums['delta2']) if sums['qhat2']*sums['delta2'] else None
        update_cosine = sums['update_dot']/math.sqrt(sums['update2']*sums['reference2']) if sums['update2']*sums['reference2'] else None
        return dict(measurement='last_actual_batch', separated_state=True,
            q_delta_norm=math.sqrt(sums['q2']), m_real_norm=math.sqrt(sums['m2']),
            injection_ratio=lam*math.sqrt(sums['qhat2'])/(math.sqrt(sums['mhat2'])+1e-24),
            q_delta_cosine=cosine, cosine_reason='' if cosine is not None else 'zero_norm',
            local_adam_update_cosine=update_cosine,
            local_adam_relative_difference=math.sqrt(sums['difference2'])/(math.sqrt(sums['reference2'])+1e-24),
            local_reference='one_step_from_mapped_same_state_not_A1_trajectory')
