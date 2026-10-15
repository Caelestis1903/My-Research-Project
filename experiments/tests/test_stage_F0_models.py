import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
for path in (os.path.join(ROOT, "m3oe"), os.path.join(HERE, "history_exp", "first")):
    if path not in sys.path:
        sys.path.insert(0, path)

from p7_models import CGC_Layer
from stage_F0_data import MOVIELENS_COLUMNS, MOVIELENS_SPARSE_COLUMNS
from stage_F0_models import (
    F0CGC1Layer,
    build_f0_model,
    build_semantically_paired_model,
    cgc1_parameter_formula,
    legacy_m3oe_tensor_compatibility_fixture,
    m3oe_parameter_formula,
    mmoe_parameter_formula,
    non_embedding_parameter_count,
)
from stage_F0_protocol import FROZEN_WIDTHS
from stage_S1_data import FeatureSchema


def _schema():
    return FeatureSchema(
        dataset="movielens", columns=MOVIELENS_COLUMNS, dense_columns=("age",),
        sparse_columns=MOVIELENS_SPARSE_COLUMNS,
        vocab_sizes={name: (3 if name == "domain_indicator" else 9)
                     for name in MOVIELENS_SPARSE_COLUMNS},
    )


def test_all_six_frozen_widths_match_actual_model_counts():
    schema = _schema()
    formulas = {"m3oe": m3oe_parameter_formula, "mmoe": mmoe_parameter_formula,
                "cgc1": cgc1_parameter_formula}
    for backbone in ("m3oe", "mmoe", "cgc1"):
        for scope in ("mdmt", "sdmt"):
            d = 3 if scope == "mdmt" else 1
            width = FROZEN_WIDTHS[(backbone, scope)]
            model = build_f0_model(schema, d, 2, backbone, width["c"], seed=17)
            assert non_embedding_parameter_count(model) == width["non_embedding"]
            assert formulas[backbone](d, 2, width["c"]) == width["non_embedding"]
            del model


def test_m3oe_matches_legacy_s1_forward_at_frozen_joint_width():
    result = legacy_m3oe_tensor_compatibility_fixture(_schema(), batch_size=7)
    assert result["equivalent"], result


def test_cgc1_retains_the_old_consumed_cell_outputs():
    torch.manual_seed(991)
    old = CGC_Layer(4, 1, 6, 113, [32, 8], [8]).eval()
    new = F0CGC1Layer(6, 113, 32, 8).eval()
    new_state = new.state_dict()
    compatible = {key: value for key, value in old.state_dict().items()
                  if key in new_state and value.shape == new_state[key].shape}
    new.load_state_dict(compatible, strict=True)
    x = torch.randn(5, 113)
    with torch.no_grad():
        old_cells = old([x] * 7)[:6]
        new_cells = new(x)
    for expected, observed in zip(old_cells, new_cells):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_m3oe_single_domain_single_task_degeneracy_is_finite():
    model = build_f0_model(_schema(), 1, 1, "m3oe", 8, seed=4).eval()
    x = torch.zeros((8, 3), dtype=torch.float32)
    with torch.no_grad():
        value = model(x)
    assert value.shape == (3, 1)
    assert torch.isfinite(value).all()
    assert not any(name.startswith("_weight_bal_") for name, _ in model.named_parameters())


def test_production_builder_reuses_one_mdmt_source_and_hashes_physical_embedding_rows():
    schema = _schema()
    for backbone in ("m3oe", "mmoe", "cgc1"):
        mdmt, mdmt_audit = build_semantically_paired_model(
            schema, backbone=backbone, domain_num=3, task_num=2,
            c=FROZEN_WIDTHS[(backbone, "mdmt")]["c"], seed=2022,
            model_seed=77891)
        source_hash = mdmt_audit["source_state_sha256"]
        assert mdmt_audit["target_state_sha256"] == source_hash
        for domain in (0, 1, 2):
            target, audit = build_semantically_paired_model(
                schema, backbone=backbone, domain_num=1, task_num=2,
                c=FROZEN_WIDTHS[(backbone, "sdmt")]["c"], seed=2022,
                physical_domain=domain, model_seed=77891)
            assert audit["source_state_sha256"] == source_hash
            remap = audit["local_embedding_remap"]
            assert remap["physical_domain"] == domain
            assert remap["local_row_matches_physical_source"] is True
            assert remap["global_table_shape_preserved"] is True
            parameter = dict(target.named_parameters())
            if backbone == "m3oe":
                slot_bias = parameter["slot_bias.0"]
                assert torch.count_nonzero(slot_bias).item() == 0
                assert any(row["target"] == "slot_bias.0" for row in audit["unmatched"])
