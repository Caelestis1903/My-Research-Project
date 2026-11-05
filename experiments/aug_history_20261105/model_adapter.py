"""Source probability forward plus a read-only raw-tower logit accessor."""
import torch
from experiments.mdmt_history_information_20261103.vendor.v3.vendor.ema02.vendor.gradient_aug_20261031_v2.views import MDMTRec, make_model


class ModelAdapter(MDMTRec):
    def forward_logits(self, x, params=None, delta=None, probe_input_full_graph=False):
        if delta is not None or probe_input_full_graph:
            raise NotImplementedError('Full-input probes belong to the triggered C implementation')
        outputs, handles = {}, []
        try:
            for index, tower in enumerate(self.tower):
                handles.append(tower.register_forward_hook(
                    lambda module, inputs, output, index=index: outputs.__setitem__(index, output.squeeze(1))))
            if params is None:
                self(x)
            else:
                torch.func.functional_call(self, params, (x,))
            result = []
            for task in range(2):
                value = torch.zeros(x.shape[1], device=x.device, dtype=outputs[0].dtype)
                for domain in range(3):
                    value = torch.where(x[-1] == domain, outputs[3*task+domain], value)
                result.append(value)
            return torch.stack(result, dim=1)
        finally:
            for handle in handles:
                handle.remove()


def build_model(bundle, config):
    model = make_model(bundle, config['source'])
    model.__class__ = ModelAdapter
    if sum(p.numel() for p in model.parameters()) != config['parameter_count']:
        raise ValueError('Source parameter count changed')
    return model

