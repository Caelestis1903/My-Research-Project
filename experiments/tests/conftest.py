"""Adopt a process slot reserved by run_f0_python_task.sh before Python."""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from stage_F0_queue import adopt_reserved_process_slot


def pytest_sessionstart(session):
    required = ("F0_RESERVED_PROCESS_SLOT_FD", "F0_RESERVED_PROCESS_SLOT_INDEX",
                "F0_RESERVED_PROCESS_SLOT_OWNER_PID", "F0_RESERVED_PROCESS_SLOT_ROLE")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        import pytest
        pytest.exit("tests must start through run_f0_python_task.sh; missing prelaunch slot "
                    "identity: {}".format(", ".join(missing)), returncode=75)
    session.config._f0_process_slot = adopt_reserved_process_slot(
        fd=int(os.environ["F0_RESERVED_PROCESS_SLOT_FD"]),
        index=int(os.environ["F0_RESERVED_PROCESS_SLOT_INDEX"]),
        owner_pid=int(os.environ["F0_RESERVED_PROCESS_SLOT_OWNER_PID"]),
        role=os.environ["F0_RESERVED_PROCESS_SLOT_ROLE"],
        scheduler_dir=os.environ.get("F0_PROCESS_SCHEDULER_ROOT"))


def pytest_sessionfinish(session, exitstatus):
    lease = getattr(session.config, "_f0_process_slot", None)
    if lease is not None:
        lease.close()
