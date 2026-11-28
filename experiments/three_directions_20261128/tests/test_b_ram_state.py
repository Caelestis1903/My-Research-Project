import copy
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import torch

from experiments.conditional_history_shared_update_20261122.protocol import read, v3
from experiments.conditional_history_shared_update_20261122.runner import select
from experiments.three_directions_20261128.ram_state import Journal, _opened, _unlink

BINDING = dict(study="ram-check", stage="B1", dataset="fixture", seed=2022, arm="B2", source_hash="source",
               config_hash="config", data_hash="data", scope_hash="scope", job_key="fixture")
CHILD = """
import os
import sys
from multiprocessing.shared_memory import SharedMemory
from experiments.three_directions_20261128.ram_state import Journal
from experiments.three_directions_20261128.tests.test_b_ram_state import BINDING, train_reduced
journal = Journal(sys.argv[1], BINDING)
with journal.writer():
    attempt = journal.new_attempt("gpu-fixture")
    train_reduced(journal, attempt, 2)
    with journal._mutex():
        value = journal._load()
        value["pending"] = sys.argv[2]
        journal._save(value)
    shm = SharedMemory(sys.argv[2], create=True, size=5, track=False)
    shm.buf[:] = b"short"
    os._exit(0)
"""


def train_reduced(journal, attempt, final_epoch, state=None):
    v3.setup(2022)
    model = torch.nn.Linear(1, 1)
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters())
    loader = torch.Generator().manual_seed(12022)
    start = 0
    if state is not None:
        start = state["epoch"]
        model.load_state_dict(state["online"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        loader.set_state(state["loader"])
        v3.restore_rng(state["rng"], torch.device("cpu"))
    for epoch in range(start + 1, final_epoch + 1):
        indices = torch.randperm(2, generator=loader)
        for i in indices:
            optimizer.zero_grad()
            model(torch.randn(1) + i).square().sum().backward()
            optimizer.step()
            v3.update_ema(ema, model, .98)
        state = v3.cpu_copy(dict(epoch=epoch, global_step=2*epoch, cursor=2,
                                 permutation=indices.numpy(), online=model.state_dict(), ema=ema.state_dict(),
                                 optimizer=optimizer.state_dict(), rng=v3.rng_state(torch.device("cpu")),
                                 loader=loader.get_state()))
        if journal is not None:
            journal.commit_epoch(attempt, state)
    return state


def test_cross_process_ram_interruption_fallback_resume_and_selected(tmp_path):
    name = "phase03_3d_" + uuid4().hex
    try:
        subprocess.run([sys.executable, "-c", CHILD, str(tmp_path), name], check=True)
        journal = Journal(tmp_path, BINDING)
        initial = read(journal.path)
        assert initial["pending"] == name and len(initial["epochs"]) == 2
        with journal.writer():
            attempt = journal.new_attempt("gpu-second")
            assert read(journal.path)["pending"] is None
            assert not Path("/dev/shm", name).exists()
            newest, receipt = journal.recover()
            assert newest["epoch"] == 2
            assert receipt["identity"]["storage_generation"] == attempt["generation"]
            assert receipt["identity"]["gpu"] == "gpu-fixture"
            assert receipt["identity"]["attempt"] != attempt["id"]
            with _opened(receipt["name"]) as segment:
                segment.buf[0] ^= 1
            prior, receipt = journal.recover()
            assert prior["epoch"] == 1 and receipt == initial["epochs"][1]
            resumed = train_reduced(journal, attempt, 3, state=prior)
            uninterrupted = train_reduced(None, None, 3)
            assert v3.equal(resumed, uninterrupted)
            model = torch.nn.Linear(1, 1)
            model.load_state_dict(resumed["online"])
            selected, epochs, scores = {}, {}, {}
            select(2, "online", model, dict(macro_auc=.7, macro_logloss=.7), selected, epochs, scores)
            with torch.no_grad():
                model.weight.add_(1)
            select(3, "online", model, dict(macro_auc=.7, macro_logloss=.1), selected, epochs, scores)
            assert epochs["online"] == 2 and scores["online"]["macro_logloss"] == .7
            readouts = {"online-selected": ("online", 2, selected["online"]),
                        "fixed_ema-selected": ("fixed_ema", 3, resumed["ema"])}
            required = {}
            for key, (kind, epoch, model_state) in readouts.items():
                model_receipt = journal.commit_model(attempt, key, kind, epoch, model_state, 2*epoch)
                required[key] = {k: model_receipt["identity"][k] for k in ("readout", "epoch", "model_hash")}
                packet, actual = journal.read_model(key)
                assert actual == model_receipt and v3.equal(packet, model_state)
            journal.complete(attempt, dict(epoch=3), required)
            assert journal.completed() == dict(epoch=3)
            assert read(journal.path)["epochs"] == []
        assert not list(tmp_path.glob("*.pt"))
    finally:
        _unlink(name)
        journal_path = tmp_path / "RAM_JOURNAL.json"
        if journal_path.exists():
            final = read(journal_path)
            active = [*final["epochs"], *final["compact"].values()]
            for item in active:
                _unlink(item["name"])
            for stale in [*final["garbage"], final["pending"]]:
                if stale:
                    _unlink(stale)
