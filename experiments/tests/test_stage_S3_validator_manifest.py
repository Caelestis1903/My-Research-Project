from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest
from argparse import Namespace

try:
    import torch
except ImportError:  # pragma: no cover - dependency-light test environments
    torch = None

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_S3_orchestrator import (  # noqa: E402
    _checkpoint_state_numerically_equal,
    _dataset_identity_reasons,
)
import stage_S3_orchestrator as orchestrator  # noqa: E402
from stage_S3_protocol import build_fresh_matrix, build_launch_plan  # noqa: E402


class StageS3ValidatorManifestTests(unittest.TestCase):
    def test_dataset_label_is_canonicalized_but_dataset_key_is_exact(self):
        task = {"dataset": "kuairand"}
        manifest = {"dataset": "KuaiRand-Pure", "dataset_key": "kuairand"}
        self.assertEqual(_dataset_identity_reasons(task, manifest), [])

        wrong_key = dict(manifest, dataset_key="KuaiRand-Pure")
        self.assertEqual(_dataset_identity_reasons(task, wrong_key),
                         ["manifest_mismatch:dataset_key"])

        wrong_label = dict(manifest, dataset="MovieLens-1M")
        self.assertEqual(_dataset_identity_reasons(task, wrong_label),
                         ["manifest_mismatch:dataset"])
        missing_key = dict(manifest)
        missing_key.pop("dataset_key")
        self.assertEqual(_dataset_identity_reasons(task, missing_key),
                         ["manifest_mismatch:dataset_key"])
        noncanonical_key = dict(manifest, dataset_key="KuaiRand")
        self.assertEqual(_dataset_identity_reasons(task, noncanonical_key),
                         ["manifest_mismatch:dataset_key"])

    @unittest.skipIf(torch is None, "PyTorch is required for checkpoint semantics")
    def test_final_common49_uses_checkpoint_state_not_serialized_hash(self):
        with tempfile.TemporaryDirectory(prefix="stage_s3_validator_") as root:
            left_path = os.path.join(root, "final_checkpoint.pth")
            right_path = os.path.join(root, "common_epoch_49.pth")
            state = {"weight": torch.tensor([1.0, 2.0]),
                     "nested": {"bias": torch.tensor([3.0])}}
            torch.save(state, left_path)
            torch.save(state, right_path)
            with open(left_path, "rb") as stream:
                left_hash = hashlib.sha256(stream.read()).hexdigest()
            with open(right_path, "rb") as stream:
                right_hash = hashlib.sha256(stream.read()).hexdigest()
            self.assertNotEqual(left_hash, right_hash)
            self.assertTrue(_checkpoint_state_numerically_equal(left_path, right_path))

            changed = {"weight": torch.tensor([1.0, 2.1]),
                       "nested": {"bias": torch.tensor([3.0])}}
            torch.save(changed, right_path)
            self.assertFalse(_checkpoint_state_numerically_equal(left_path, right_path))
            torch.save({"weight": torch.tensor([1.0, 2.0], dtype=torch.float64),
                        "nested": {"bias": torch.tensor([3.0])}}, right_path)
            self.assertFalse(_checkpoint_state_numerically_equal(left_path, right_path))
            torch.save({"weight": torch.tensor([1.0, 2.0])}, right_path)
            self.assertFalse(_checkpoint_state_numerically_equal(left_path, right_path))
            with open(right_path, "wb") as stream:
                stream.write(b"not a checkpoint")
            self.assertIsNone(_checkpoint_state_numerically_equal(left_path, right_path))

    def test_frozen_validator_repair_rebinds_only_a_temporary_exact_queue_copy(self):
        raw_root = os.path.join(HERE, "stage_S3_raw_results")
        frozen_queue = os.path.join(HERE, "stage_S3_queue_manifest_A1.json")
        frozen_hash = orchestrator._file_hash(frozen_queue)
        if frozen_hash != orchestrator._VALIDATOR_REPAIR_QUEUE_SHA256:
            self.skipTest("live queue has advanced beyond the seed-2023 validator incident")
        tasks = build_fresh_matrix(raw_root)
        tasks = orchestrator.annotate_pairs(tasks, build_launch_plan(tasks))
        current_source = orchestrator.source_inventory(orchestrator.ROOT)
        current_digest = orchestrator.source_inventory_digest(current_source)
        authority = {
            "review_report_sha256": "fixture-review",
            "reviewed_source_inventory_sha256": current_digest,
        }
        with tempfile.TemporaryDirectory(prefix="stage_s3_rebind_") as root:
            queue_path = os.path.join(root, "queue.json")
            shutil.copy2(frozen_queue, queue_path)
            queue = orchestrator._read_json(queue_path)
            with self.assertRaisesRegex(RuntimeError, "source inventory drift"):
                orchestrator._apply_reviewed_validator_rebind(
                    queue_path, queue,
                    Namespace(raw_root=raw_root, allow_reviewed_validator_rebind=False),
                    current_source, authority, tasks, process_inventory=lambda: [])
            self.assertEqual(orchestrator._file_hash(queue_path), frozen_hash)
            args = Namespace(raw_root=raw_root, allow_reviewed_validator_rebind=True)
            orchestrator._apply_reviewed_validator_rebind(
                queue_path, queue, args, current_source, authority, tasks,
                process_inventory=lambda: [])
            rebound = orchestrator._read_json(queue_path)
            self.assertEqual(rebound["source_inventory_digest"], current_digest)
            self.assertEqual(rebound["validator_repair_rebind"]["changed_source_files"],
                             ["experiments/stage_S3_orchestrator.py"])
            self.assertEqual(rebound["launch_group_states"][
                "s3_kuairand_natural_seed2023"], "ARCHIVED_WAVE_1")
            self.assertEqual(orchestrator._validate_amendment_queue_history(rebound), [])
        self.assertEqual(orchestrator._file_hash(frozen_queue), frozen_hash)


if __name__ == "__main__":
    unittest.main()
