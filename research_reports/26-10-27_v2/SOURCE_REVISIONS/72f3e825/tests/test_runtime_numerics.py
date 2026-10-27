import json
import numpy as np
import pytest
import torch
from experiments.mdmt_relation_basis_v2.diagnostics import compare_raw_logits


def test_cross_batch_relative_fp32_and_failure_evidence():
    cached = torch.tensor([[-4.291538238525391]], dtype=torch.float32)
    raw = torch.tensor([[-4.291540622711182]], dtype=torch.float32)
    record = compare_raw_logits(raw, cached, np.array([205]))
    assert record['passed'] and record['finite']
    assert record['max_abs'] == 2.384185791015625e-6
    assert record['max_index'] == [0, 0]
    assert record['row_id_first'] == 205
    assert record['allowed_at_max'] > record['max_abs']
    for actual, expected, reason in [
        (torch.tensor([[2e-6]]), torch.zeros(1, 1), 'logit_tolerance_exceeded'),
        (torch.tensor([[1.01]]), torch.ones(1, 1), 'logit_tolerance_exceeded'),
        (torch.tensor([[float('nan')]]), torch.zeros(1, 1), 'nonfinite_logits'),
        (torch.tensor([[float('inf')]]), torch.zeros(1, 1), 'nonfinite_logits'),
        (torch.zeros(1, 2), torch.zeros(1, 1), 'shape_mismatch'),
    ]:
        with pytest.raises(ValueError) as error:
            compare_raw_logits(actual, expected)
        evidence = json.loads(str(error.value).split(': ', 1)[1])
        assert evidence['passed'] is False
        assert evidence['reason'] == reason
        assert evidence['atol'] == 1e-6 and evidence['rtol'] == 1.3e-6

