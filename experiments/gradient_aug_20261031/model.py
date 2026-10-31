"""Fixed1 backbone with a non-parametric, field-aligned embedding view."""
import torch
from .reference import MDMTRec as FixedReference
from .basic.features import DenseFeature, SparseFeature


def perturb_embedding(embedding, slices, relative_std, generator):
    result = embedding.clone()
    for start, end in slices:
        field = embedding[:, start:end]
        rms = field.square().mean(dim=1, keepdim=True).sqrt().detach()
        noise = torch.randn(field.shape, dtype=field.dtype, device=field.device, generator=generator)
        result[:, start:end] = field + relative_std * rms * noise
    return result


class MDMTRec(FixedReference):
    def forward(self, x, test_flag=False, *, relative_std=None, generator=None):
        if relative_std is None:
            return super().forward(x, test_flag=test_flag)
        # The reference consumes this output before STAR and skip; no stateful batch layers exist.
        handle = self.embedding.register_forward_hook(
            lambda module, inputs, output: perturb_embedding(output, self.perturb_slices, relative_std, generator))
        try:
            return super().forward(x, test_flag=test_flag)
        finally:
            handle.remove()


def make_model(bundle, dataset_config, reference=False):
    features = [DenseFeature(i) if name in bundle.features.dense_columns else
                SparseFeature(i, bundle.features.vocab_sizes[name], embed_dim=16)
                for i, name in enumerate(bundle.features.columns)]
    cls = FixedReference if reference else MDMTRec
    model = cls(features, domain_num=3, task_num=2, fcn_dims=[512,256,256,64], expert_num=4,
                exp_d=1, exp_t=1, bal_d=1, bal_t=1)
    sparse = [(bundle.features.columns[f.name], f) for f in features if isinstance(f, SparseFeature)]
    dense = [(bundle.features.columns[f.name], f) for f in features if isinstance(f, DenseFeature)]
    offset = 0
    slices = {}
    for name, feature in sparse + dense:
        slices[name] = (offset, offset + feature.embed_dim)
        offset += feature.embed_dim
    assert offset == model.input_dim
    wanted = dataset_config['perturbable_sparse_fields']
    assert set(wanted) == {name for name, _ in sparse if name != 'domain_indicator'}
    if not reference:
        model.field_slices = slices
        model.perturb_slices = [slices[name] for name in wanted]
    assert not any(isinstance(m, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) for m in model.modules())
    assert not any(name.startswith(('_weight_', 'dbo_')) for name, _ in model.named_parameters())
    return model
