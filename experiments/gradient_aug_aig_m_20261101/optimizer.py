"""Training-only group innovation gates with the frozen SAD second moment."""
import math
import torch


def parameter_groups(model, target):
    result = []
    for name, _ in model.named_parameters():
        embedding = name.startswith('embedding.embed_dict.')
        path = (name in ('shared_weight', 'shared_bias', f'slot_weight.{target}', f'slot_bias.{target}') or
                name.startswith(('skip_conn.', 'star_mlp.', 'expert.', 'task_expert.',
                    f'domain_expert.{target}.', f'gate.{target}.', f'gate.{3+target}.',
                    f'tower.{target}.', f'tower.{3+target}.')))
        result.append('E' if embedding else 'P' if path else 'U')
    return result


@torch.no_grad()
def innovation(optimizer, parameters, groups, real, delta):
    values = {group: [[], [], []] for group in ('E', 'P')}
    for param, group, g, d in zip(parameters, groups, real, delta):
        if group == 'U':
            continue
        q = optimizer.state.get(param, {}).get('q_delta')
        if g is None and d is None and q is None:
            continue
        zero = torch.zeros_like(param)
        q = zero if q is None else q
        g = zero if g is None else g
        d = zero if d is None else d
        for bucket, tensor in zip(values[group], (d-q, q, g)):
            bucket.append(tensor.double().square().sum())
    result = {}
    for group, buckets in values.items():
        norms = [math.sqrt(float(torch.stack(items).sum())) if items else 0. for items in buckets]
        result[group] = norms[0] / (norms[1] + norms[2] + 1e-12)
    return result


class AIGMomentum(torch.optim.Optimizer):
    def __init__(self, parameters, groups, lr=.01, weight_decay=0., beta_delta=.95,
                 lambda_min=0., lambda_max=.5, thresholds=None, slope=4.):
        if not 0 < beta_delta < 1 or not 0 <= lambda_min <= lambda_max <= 1:
            raise ValueError('Invalid AIG coefficients')
        parameters = list(parameters)
        if len(parameters) != len(groups) or not set(groups) <= {'E', 'P', 'U'}:
            raise ValueError('Parameter group mapping mismatch')
        self.groups = list(groups)
        super().__init__(parameters, dict(lr=lr, weight_decay=weight_decay, beta_delta=beta_delta,
            lambda_min=lambda_min, lambda_max=lambda_max, thresholds=dict(thresholds or {'E':0., 'P':0.}),
            slope=slope, betas=(.9,.999), eps=1e-8))

    @torch.no_grad()
    def step(self, real_grads, delta_grads, diagnose=False):
        options = self.param_groups[0]
        params = options['params']
        real, delta = list(real_grads), list(delta_grads)
        u = innovation(self, params, self.groups, real, delta)
        gates = {g: 1/(1+math.exp(-max(-700., min(700., options['slope']*(u[g]-options['thresholds'][g])))))
                 for g in ('E', 'P')}
        lambdas = {g:options['lambda_min']+(options['lambda_max']-options['lambda_min'])*gates[g] for g in gates}
        sums = {g:dict(q2=0., m2=0., qhat2=0., mhat2=0.) for g in ('E', 'P')}
        b1, b2 = options['betas']; bd = options['beta_delta']
        for param, group, gr, d in zip(params, self.groups, real, delta):
            lam = 0. if group == 'U' else lambdas[group]
            if gr is None and (d is None or lam == 0):
                continue
            state = self.state[param]
            if not state:
                state.update(step=torch.tensor(0.), m_real=torch.zeros_like(param), exp_avg_sq=torch.zeros_like(param))
                if group != 'U':
                    state['q_delta'] = torch.zeros_like(param)
            r = torch.zeros_like(param) if gr is None else gr.detach()
            if options['weight_decay']:
                r = r.add(param, alpha=options['weight_decay'])
            d = torch.zeros_like(param) if d is None else d.detach()
            state['step'].add_(1); step = int(state['step'])
            m, v = state['m_real'], state['exp_avg_sq']
            m.lerp_(r, 1-b1)
            mixed = r.add(d, alpha=lam)
            v.mul_(b2).addcmul_(mixed, mixed, value=1-b2)
            numerator = m/(1-b1**step)
            if group != 'U':
                q = state['q_delta']; q.lerp_(d, 1-bd)
                qhat = q/(1-bd**step)
                numerator = numerator + lam*qhat
                if diagnose:
                    for key, tensor in [('q2',q),('m2',m),('qhat2',qhat),('mhat2',m/(1-b1**step))]:
                        sums[group][key] += float(tensor.double().square().sum())
            denominator = (v.sqrt()/math.sqrt(1-b2**step)).add_(options['eps'])
            if group == 'U':
                param.addcdiv_(m, denominator, value=-options['lr']/(1-b1**step))
            else:
                param.addcdiv_(numerator, denominator, value=-options['lr'])
        records = []
        for group in ('E', 'P'):
            record = dict(group=group, u=u[group], gate=gates[group], lambda_t=lambdas[group])
            if diagnose:
                values = sums[group]
                record.update(q_norm=math.sqrt(values['q2']), m_norm=math.sqrt(values['m2']),
                    injection_ratio=lambdas[group]*math.sqrt(values['qhat2'])/(math.sqrt(values['mhat2'])+1e-24))
            records.append(record)
        return records
