"""One-shot P7 orchestrator launcher.

Spawns the orchestrator as a fully detached child using
subprocess.Popen with DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
flags on Windows, plus close_fds so it survives this script's exit.
"""
import os
import subprocess
import sys

ROOT = r"D:\CODE\phase0-3"
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
ORCH = os.path.join(ROOT, "experiments", "p7_orchestrator.py")
OUT = os.path.join(ROOT, "experiments", "logs", "p7_orchestrator.out")

# DETACHED_PROCESS (0x00000008) | CREATE_NEW_PROCESS_GROUP (0x00000200)
flags = 0x00000008 | 0x00000200

with open(OUT, "ab", buffering=0) as logf:
    p = subprocess.Popen(
        [PY, "-u", ORCH, "--concurrency", "2"],
        stdout=logf, stderr=subprocess.STDOUT,
        cwd=ROOT, close_fds=True,
        creationflags=flags,
    )
print(f"launched orchestrator pid={p.pid}")
