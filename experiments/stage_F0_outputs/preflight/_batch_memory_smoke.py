"""Synthetic FP32 forward/backward resource probe; never performs optimizer steps."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from stage_F0_data import load_movielens_natural
from stage_F0_models import build_f0_model
from stage_F0_protocol import width_for
from stage_F0_queue import QueueFileLock, acquire_process_slot
from stage_F0_run import setup_deterministic_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    slot = acquire_process_slot(role="F0-target-batch-memory-smoke")
    gpu_lock = QueueFileLock(os.path.join(args.output, "gpu_locks", "gpu{}.lock".format(args.gpu)))
    try:
        device = torch.device("cuda:{}".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        torch.cuda.init()
        setup_deterministic_runtime(901733 + args.gpu, device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        bundle = load_movielens_natural()
        results = []
        for model_name in args.models:
            c = int(width_for(model_name, "mdmt")["c"])
            for batch_size in (4096, 40960):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                model = build_f0_model(bundle.features, 3, 2, model_name, c, seed=77119).to(device)
                model.train()
                rng = np.random.default_rng(4811 + batch_size)
                x = np.empty((batch_size, len(bundle.features.columns)), dtype=np.float32)
                for j, col in enumerate(bundle.features.columns):
                    if col == "age":
                        x[:, j] = rng.random(batch_size, dtype=np.float32)
                    elif col == "domain_indicator":
                        x[:, j] = rng.integers(0, 3, size=batch_size, dtype=np.int32)
                    else:
                        x[:, j] = rng.integers(0, bundle.features.vocab_sizes[col], size=batch_size, dtype=np.int32)
                y = rng.integers(0, 2, size=(batch_size, 2), dtype=np.int32).astype(np.float32)
                xt = torch.from_numpy(x.T.copy()).to(device=device, dtype=torch.float32)
                yt = torch.from_numpy(y).to(device=device, dtype=torch.float32)
                state0 = hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                                                   for t in model.state_dict().values())).hexdigest()
                started = time.perf_counter()
                output1 = model(xt)
                loss1 = torch.nn.functional.binary_cross_entropy(output1, yt)
                loss1.backward()
                torch.cuda.synchronize(device)
                elapsed1 = time.perf_counter() - started
                grads1 = [None if p.grad is None else p.grad.detach().cpu().clone()
                          for p in model.parameters()]
                model.zero_grad(set_to_none=True)
                started = time.perf_counter()
                output2 = model(xt)
                loss2 = torch.nn.functional.binary_cross_entropy(output2, yt)
                loss2.backward()
                torch.cuda.synchronize(device)
                elapsed2 = time.perf_counter() - started
                same_output = torch.equal(output1.detach(), output2.detach())
                same_loss = torch.equal(loss1.detach(), loss2.detach())
                same_grads = all((a is None and p.grad is None) or
                                 (a is not None and p.grad is not None and torch.equal(a, p.grad.detach().cpu()))
                                 for a, p in zip(grads1, model.parameters()))
                finite = bool(torch.isfinite(output1).all() and torch.isfinite(loss1) and
                              all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                state1 = hashlib.sha256(b"".join(t.detach().cpu().contiguous().numpy().tobytes()
                                                   for t in model.state_dict().values())).hexdigest()
                free, total = torch.cuda.mem_get_info(device)
                results.append({"backbone": model_name, "scope": "mdmt", "batch_size": batch_size,
                                "c": c, "status": "PASS" if finite and same_output and same_loss and same_grads and state0 == state1 else "FAIL",
                                "synthetic_only": True, "optimizer_step_performed": False,
                                "test_data_read": False, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                "tf32": False, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                "output_equal_repeat": same_output, "loss_equal_repeat": same_loss,
                                "gradients_equal_repeat": same_grads, "parameters_unchanged": state0 == state1,
                                "finite": finite, "elapsed_two_forward_backward_seconds": elapsed1 + elapsed2,
                                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                                "gpu_free_bytes_after": int(free), "gpu_total_bytes": int(total),
                                "loss_value_raw_synthetic": float(loss1.detach().cpu())})
                del model, x, y, xt, yt, output1, output2, loss1, loss2, grads1
                torch.cuda.empty_cache()
        payload = {"gpu_index": args.gpu, "gpu_name": torch.cuda.get_device_name(args.gpu),
                   "results": results, "status": "PASS" if all(x["status"] == "PASS" for x in results) else "FAIL",
                   "scope_note": "Synthetic deterministic forward/backward memory smoke only; no optimizer update, no real rows, no validation/test metrics."}
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "batch_memory_smoke_gpu{}.json".format(args.gpu)), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(json.dumps({"gpu": args.gpu, "status": payload["status"], "runs": len(results)}))
    except BaseException as exc:
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "batch_memory_smoke_gpu{}_error.json".format(args.gpu)), "w") as f:
            json.dump({"gpu_index": args.gpu, "status": "FAIL", "error": repr(exc)}, f, indent=2)
        raise
    finally:
        gpu_lock.close()
        slot.close()


if __name__ == "__main__":
    main()
