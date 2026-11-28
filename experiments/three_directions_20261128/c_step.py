"""Direction C's warmup verification, immutable transition and pair objective."""
import numpy as np
import torch
from torch.nn import functional as F

from . import a_step
from .c0_audit import require
from .memory_bank import MODES, TEACHER_EPOCH, TEMPERATURE, bank_metadata, draw_pair, prepare_banks, validate_bank
from .protocol import v3


def make_training_parts(model, job, bundle, seed_inputs, verify_inputs):
    optimizer, stock = a_step.make_training_parts(model, job, bundle)
    return optimizer, DirectionCAdapter(stock, job.seed, job.arm, seed_inputs, verify_inputs)


def pair_loss(margin, pair, config, arm):
    """Return the event mean. Direction A applies the stock importance weight."""
    if pair["memory_source"] != "bank" or arm == "C4":
        return F.softplus(-margin / config["tau"]).mean()
    target_margin = torch.as_tensor(pair["target_margins"], dtype=torch.float32, device=margin.device)
    with torch.no_grad():
        targets = (target_margin / TEMPERATURE).sigmoid()
    return F.binary_cross_entropy_with_logits(margin.float() / TEMPERATURE, targets)


def prefix_step(row_hash, diagnostics):
    condition = diagnostics["condition"]
    return [row_hash, None if condition is None else list(condition),
            diagnostics["w"], diagnostics["q"], diagnostics["pair_index_hash"]]


def _run_digests(optimizer, loader, rng):
    return {f"{name}_hash": v3.state_digest(state)
            for name, state in (("optimizer", optimizer), ("loader", loader), ("rng", rng))}


class DirectionCAdapter:
    def __init__(self, stock, seed, arm, inputs, verify_inputs):
        self.stock = stock
        self.sampler = stock.sampler
        self.seed = seed
        self.arm = arm
        self.inputs = inputs
        self.verify_inputs = verify_inputs
        self.bank = self.boundary = None

    def verify_init(self, model_hash):
        expected = self.inputs["prefix"]["init_model_hash"]
        require(model_hash == expected, f"C1 P0 initialization hash mismatch: expected {expected}, got {model_hash}")

    def _prepare(self):
        self.verify_inputs(self.inputs)
        banks, summaries = prepare_banks(self.sampler, self.seed, self.inputs)
        bank = banks[MODES[self.arm]]
        return bank, bank_metadata(bank, summaries)

    def finish_epoch(self, epoch, global_step, order_hash, model, optimizer, ema, loader, device):
        """Install and describe the bank immediately after the E10 update."""
        if epoch > TEACHER_EPOCH:
            return None
        prefix = self.inputs["prefix"]
        expected = (epoch * prefix["steps_per_epoch"], prefix["order_hashes"][epoch - 1])
        observed = (global_step, order_hash)
        require(observed == expected, f"C1 P0 row order or step mismatch E{epoch:02d}: expected {expected}, got {observed}")
        if epoch != TEACHER_EPOCH:
            return None
        teacher = self.inputs["teacher"]
        online_hash = v3.state_digest(model.state_dict())
        ema_hash = v3.state_digest(ema.state_dict())
        expected = (teacher["global_step"], teacher["online_hash"], teacher["model_hash"])
        observed = (global_step, online_hash, ema_hash)
        require(self.bank is None and observed == expected,
                f"C1 P0 E10 online model, fixed-EMA teacher or step identity mismatch: expected {expected}, got {observed}")
        bank, metadata = self._prepare()
        self.boundary = dict(epoch=epoch, global_step=global_step, online_hash=online_hash, ema_hash=ema_hash,
                             prefix_hash=prefix["hash"],
                             **_run_digests(optimizer.state_dict(), loader.get_state(), v3.rng_state(device)))
        self.bank = bank
        return dict(prefix=self.boundary, bank=metadata)

    def state_dict(self):
        return dict(sampler=self.sampler.state_dict(), arm=self.arm, bank=self.bank, boundary=self.boundary)

    def restore_epoch(self, state):
        """Restore the adapter with the enclosing verified complete epoch."""
        adapter = state["adapter"]
        epoch, global_step = state["epoch"], state["global_step"]
        prefix, teacher = self.inputs["prefix"], self.inputs["teacher"]
        require(adapter.keys() == {"sampler", "arm", "bank", "boundary"}
                and adapter["arm"] == self.arm and adapter["sampler"] == self.sampler.state_dict()
                and 1 <= epoch <= 50 and global_step == epoch * prefix["steps_per_epoch"],
                "Restored C1 sampler identity or epoch boundary mismatch")
        bank, boundary = adapter["bank"], adapter["boundary"]
        if epoch < TEACHER_EPOCH:
            require(bank is None and boundary is None and
                    state["epoch_metrics"]["order_hash"] == prefix["order_hashes"][epoch - 1],
                    "C1 warmup restore requires the verified P0 row order and a pending bank")
            self.bank = self.boundary = None
            return
        expected = dict(epoch=TEACHER_EPOCH, global_step=teacher["global_step"], online_hash=teacher["online_hash"],
                        ema_hash=teacher["model_hash"], prefix_hash=prefix["hash"])
        require(bank is not None and boundary is not None
                and all(boundary.get(k) == v for k, v in expected.items())
                and set(boundary) == set(expected) | {"optimizer_hash", "loader_hash", "rng_hash"},
                "Restored C1 E10 teacher or prefix identity mismatch")
        if epoch == TEACHER_EPOCH:
            require(v3.state_digest(state["online"]) == teacher["online_hash"]
                    and v3.state_digest(state["ema"]) == teacher["model_hash"]
                    and state["epoch_metrics"]["order_hash"] == prefix["order_hashes"][epoch - 1]
                    and all(boundary[k] == v for k, v in
                            _run_digests(state["optimizer"], state["loader"], state["rng"]).items()),
                    "Restored C1 E10 model, optimizer, loader or RNG differs from the committed prefix")
        rebuilt, _ = self._prepare()
        validate_bank(bank, rebuilt)
        self.bank, self.boundary = rebuilt, dict(boundary)

    def step(self, model, optimizer, ema, x, y, row_ids, config, context):
        args = (model, optimizer, ema, x, y, row_ids, config, context)
        if context["epoch"] <= TEACHER_EPOCH:
            result = self.stock.step(*args)
            diagnostics = result["diagnostics"]
            step = context["global_step"]
            expected = self.inputs["prefix"]["batches"]
            require(1 <= step <= len(expected), f"C1 logical step {step} falls outside the committed P0 prefix")
            observed = prefix_step(v3.state_digest(row_ids), diagnostics)
            require(observed == expected[step - 1],
                    f"C1 P0 prefix batch or pair mismatch at step {step}: expected {expected[step - 1]}, got {observed}")
            has_event = diagnostics["condition"] is not None
            diagnostics.update(memory_phase="warmup", memory_source="current" if has_event else "none",
                               memory_miss=None if has_event else "no_event",
                               current_pair_index_hash=diagnostics["pair_index_hash"])
            return result

        require(self.bank is not None, "C1 memory exposure requires a verified E10 bank")
        current = applied = detached_margin = None

        def select_pair(event_context):
            nonlocal current, applied
            current = self.sampler.sample_macro6(event_context)
            applied = draw_pair(self.bank, current, event_context)
            return applied

        def reduce_pair(margin, pair, pair_config):
            nonlocal detached_margin
            detached_margin = margin.detach()
            return pair_loss(margin, pair, pair_config, self.arm)

        result = self.stock.step(*args, select_pair=select_pair, reduce_pair=reduce_pair)
        diagnostics = result["diagnostics"]
        diagnostics.update(memory_phase="memory", bank_hash=self.bank["bank_hash"],
                           current_pair_index_hash=None if current is None else current["pair_index_hash"])
        if applied is None:
            diagnostics.update(memory_source="none", memory_miss="no_event")
        else:
            diagnostics.update(memory_source=applied["memory_source"], memory_miss=applied["memory_miss"])
        if detached_margin is not None:
            positive, zero = [int(v) for v in torch.stack(((detached_margin > 0).sum(), (detached_margin == 0).sum())).tolist()]
            diagnostics.update(pair_margin_positive=positive, pair_margin_tied=zero,
                               pair_margin_negative=len(detached_margin) - positive - zero)
        if applied is not None and applied["memory_source"] == "bank":
            target = applied["target_margins"]
            diagnostics.update(memory_bank_size=applied["bank_size"], draw_positions_hash=applied["draw_positions_hash"],
                               teacher_margin=dict(min=float(target.min()), max=float(target.max()),
                                                   mean=float(target.mean())),
                               target_probability_mean=float(np.mean(1. / (1. + np.exp(-target))))
                               if self.arm != "C4" else 1.)
        return result
