"""Named RAM packets with a small atomic metadata journal per trajectory."""
from contextlib import contextmanager
import fcntl
import hashlib
import io
import os
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import shutil
from uuid import uuid4

import torch

from experiments.conditional_history_shared_update_20261122.protocol import read, v3, write

SHM_ROOT = Path("/dev/shm")


@contextmanager
def lock(path, *, nonblocking=False):
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield


def _unlink(name):
    try:
        with _opened(name) as ref:
            ref.unlink()
    except FileNotFoundError:
        pass


@contextmanager
def _opened(name, **kwargs):
    segment = SharedMemory(name=name, track=False, **kwargs)
    try:
        yield segment
    finally:
        segment.close()


def _read_packet(receipt, expected):
    identity = receipt["identity"]
    if any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("RAM receipt identity differs from its frozen trajectory")
    name, size = receipt["name"], receipt["size"]
    if not name.startswith("phase03_3d_") or size <= 0:
        raise ValueError("Invalid study RAM reference")
    with _opened(name) as segment:
        if size != segment.size:
            raise ValueError("RAM segment size mismatch")
        payload = bytes(segment.buf)
    if hashlib.sha256(payload).hexdigest() != receipt["sha256"]:
        raise ValueError("RAM segment byte hash mismatch")
    packet = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
    if packet.get("identity") != identity or v3.state_digest(packet) != receipt["packet_digest"]:
        raise ValueError("RAM packet identity or decoded digest mismatch")
    state = packet["state"]
    if identity["kind"] == "epoch":
        if (state["epoch"] != identity["epoch"] or state["global_step"] != identity["global_step"]
                or state["cursor"] != len(state["permutation"])):
            raise ValueError("RAM epoch requires a verified full-boundary cursor")
    elif v3.state_digest(state["model"]) != identity["model_hash"]:
        raise ValueError("Compact RAM model hash mismatch")
    return state


class Journal:
    """Access one job journal under a job writer lock held by the worker."""

    def __init__(self, directory, binding):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "RAM_JOURNAL.json"
        self.binding = dict(binding)

    def writer(self):
        return lock(self.directory / "WRITER.lock", nonblocking=True)

    def _mutex(self):
        return lock(self.directory / "JOURNAL.lock")

    def _load(self):
        if not self.path.exists():
            return dict(binding=self.binding, generation=uuid4().hex, attempts=[], acknowledged=0,
                        epochs=[], compact={}, pending=None, garbage=[], complete=None)
        journal = read(self.path)
        if journal["binding"] != self.binding:
            raise ValueError("RAM journal is bound to a different trajectory identity")
        return journal

    def _save(self, journal):
        write(self.path, journal)

    def _clear_garbage(self, journal):
        if journal["pending"] is not None:
            journal["garbage"].append(journal["pending"])
            journal["pending"] = None
            self._save(journal)
        if journal["garbage"]:
            for name in journal["garbage"]:
                _unlink(name)
            journal["garbage"].clear()
            self._save(journal)

    def new_attempt(self, device):
        with self._mutex():
            journal = self._load()
            self._clear_garbage(journal)
            ident = dict(id=uuid4().hex, gpu=device, pid=os.getpid(),
                         number=len(journal["attempts"]) + 1, generation=journal["generation"])
            journal["attempts"].append(ident)
            self._save(journal)
            return ident

    def _expected(self, journal, receipt):
        identity = receipt["identity"]
        writers = [a for a in journal["attempts"] if a["id"] == identity["attempt"]]
        if (len(writers) != 1 or writers[0]["gpu"] != identity["gpu"]
                or writers[0]["generation"] != journal["generation"]):
            raise ValueError("RAM writer attempt is unavailable or mismatched")
        return dict(self.binding, storage_generation=journal["generation"],
                    attempt=writers[0]["id"], gpu=writers[0]["gpu"])

    def recover(self):
        with self._mutex():
            journal = self._load()
            self._clear_garbage(journal)
            for idx, receipt in enumerate(journal["epochs"]):
                try:
                    state = _read_packet(receipt, self._expected(journal, receipt))
                except (OSError, ValueError, RuntimeError, EOFError):
                    continue
                if idx:
                    journal["garbage"].extend(r["name"] for r in journal["epochs"][:idx])
                    journal["epochs"] = journal["epochs"][idx:]
                    self._save(journal)
                    self._clear_garbage(journal)
                return state, receipt
            if journal["acknowledged"] and not journal["complete"]:
                raise RuntimeError("Previously verified trajectory has no readable epoch boundary")
            return None, None

    def commit_epoch(self, attempt, state):
        return self._publish(attempt, state, dict(kind="epoch", epoch=state["epoch"],
                                                 global_step=state["global_step"]))

    def commit_model(self, attempt, key, readout, epoch, state, global_step):
        packet = dict(model=state)
        return self._publish(attempt, packet, dict(kind="compact_model", key=key, readout=readout,
                             epoch=epoch, global_step=global_step, model_hash=v3.state_digest(state)))

    def _publish(self, attempt, state, fields):
        with self._mutex():
            journal = self._load()
            if not journal["attempts"] or attempt != journal["attempts"][-1] or journal["complete"] is not None:
                raise ValueError("RAM update requires the current unfinished writer attempt")
            self._clear_garbage(journal)
            if fields["kind"] == "epoch":
                last = journal["epochs"][0]["identity"]["epoch"] if journal["epochs"] else 0
                if fields["epoch"] != last + 1:
                    raise ValueError("RAM epoch must follow the last verified boundary")
            else:
                current = journal["compact"].get(fields["key"])
                if current and all(current["identity"].get(k) == v for k, v in fields.items()):
                    try:
                        _read_packet(current, self._expected(journal, current))
                    except (OSError, ValueError, RuntimeError, EOFError):
                        pass
                    else:
                        return current
            name = "phase03_3d_" + uuid4().hex
            journal["pending"] = name
            self._save(journal)
            identity = dict(self.binding, **fields, storage_generation=journal["generation"],
                            attempt=attempt["id"], gpu=attempt["gpu"])
            packet = dict(identity=identity, state=state)
            buffer = io.BytesIO()
            torch.save(packet, buffer)
            payload = buffer.getbuffer()
            size = len(payload)
            if shutil.disk_usage(SHM_ROOT).free < size:
                raise OSError("Insufficient /dev/shm capacity for the named RAM transaction")
            # Distinct names and verified journal ownership make recovery independent of writer lifetime.
            with _opened(name, create=True, size=size) as segment:
                # Reserve physical tmpfs blocks up front so concurrent writers receive ENOSPC instead of SIGBUS.
                fd = os.open(SHM_ROOT / name, os.O_RDWR)
                try:
                    os.posix_fallocate(fd, 0, size)
                finally:
                    os.close(fd)
                segment.buf[:] = payload
            receipt = dict(name=name, size=size, sha256=hashlib.sha256(payload).hexdigest(),
                           packet_digest=v3.state_digest(packet), identity=identity)
            _read_packet(receipt, self._expected(journal, receipt))
            if fields["kind"] == "epoch":
                journal["garbage"].extend(r["name"] for r in journal["epochs"][1:])
                journal["epochs"] = [receipt, *journal["epochs"][:1]]
                journal["acknowledged"] = max(journal["acknowledged"], fields["epoch"])
            else:
                if old := journal["compact"].get(fields["key"]):
                    journal["garbage"].append(old["name"])
                journal["compact"][fields["key"]] = receipt
            journal["pending"] = None
            self._save(journal)
            self._clear_garbage(journal)
            return receipt

    def read_model(self, key, *, expected=None):
        with self._mutex():
            journal = self._load()
            receipt = journal["compact"][key]
            if expected and any(receipt["identity"].get(k) != v for k, v in expected.items()):
                raise ValueError("Compact RAM read differs from the expected model identity")
            verified = dict(self._expected(journal, receipt), key=key)
            return _read_packet(receipt, verified)["model"], receipt

    def complete(self, attempt, result, required):
        with self._mutex():
            journal = self._load()
            if attempt != journal["attempts"][-1] or not journal["epochs"]:
                raise ValueError("Completion requires a verified final epoch and the active writer")
            if journal["epochs"][0]["identity"]["epoch"] != result["epoch"]:
                raise ValueError("Completion epoch differs from epoch journal")
            if not {"online-selected", "fixed_ema-selected"}.issubset(required):
                raise ValueError("Completion requires both verified readout selections")
            self._verify_models(journal, required)
            journal["complete"] = dict(result=result, required=required)
            journal["garbage"].extend(r["name"] for r in journal["epochs"])
            journal["epochs"].clear()
            self._save(journal)
            self._clear_garbage(journal)

    def _verify_models(self, journal, required):
        for key, fields in required.items():
            receipt = journal["compact"][key]
            _read_packet(receipt, dict(self._expected(journal, receipt), key=key, **fields))

    def completed(self):
        with self._mutex():
            journal = self._load()
            self._clear_garbage(journal)
            if completion := journal["complete"]:
                self._verify_models(journal, completion["required"])
                return completion["result"]
        return None
