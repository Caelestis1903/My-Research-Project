"""Current-scope controls and measured, independent GPU admission."""
import json
import math
import re
import subprocess
import time
from pathlib import Path

from experiments.conditional_history_shared_update_20261122.scheduler import MIB, smi

STUDY = 'history_information_20261124'
ROOT = Path(__file__).resolve().parents[2]
WORKSPACES = (ROOT, ROOT.with_name('phase0-3_tvg_rm_20261116'))
SENTINELS = ('USER_STOPPED', 'PAUSE_TRAINING')


def sample():
    memory = {k.split(':')[0]:int(k.split()[1])*1024 for k in Path('/proc/meminfo').read_text().splitlines()
              if k.startswith(('MemAvailable:','MemTotal:'))}
    total, used = memory['MemTotal'],memory['MemTotal']-memory['MemAvailable']
    devices, processes, errors = [], [], []
    for fields in smi('--query-gpu=index,uuid,memory.used,memory.total,power.draw,power.limit'):
        try:
            i, uuid, occupied, capacity, watts, limit = fields
            row = dict(index=int(i),uuid=uuid,used=int(occupied)*MIB,total=int(capacity)*MIB,
                       watts=float(watts),limit=float(limit))
            eligible(row,used/total)
            devices.append(row)
        except ValueError:
            errors.append(dict(kind='GPU_CURRENT_TELEMETRY',raw=fields))
    try:
        processes = [dict(pid=int(pid),uuid=uuid,bytes=int(occupied)*MIB)
                     for pid,uuid,occupied in smi('--query-compute-apps=pid,gpu_uuid,used_memory')]
        process_available = True
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        process_available = False
        errors.append(dict(kind='GPU_PROCESS_TELEMETRY',error=str(exc)))
    return dict(time=time.time(),host_ram_used=used/total,host_ram_used_bytes=used,
                host_ram_total_bytes=total,devices=devices,processes=processes,
                process_available=process_available,errors=errors)


def register_stop_controls(artifact_root):
    controls = [dict(workspace=str(root), path=str(root/'artifacts'/STUDY/name),
                     scope=STUDY, status=name) for root in WORKSPACES for name in SENTINELS]
    receipt = dict(study_id=STUDY, controls=controls,
                   archived=[dict(workspace=str(WORKSPACES[1]), study='26-11-16_augmentation_gradient_momentum',
                                  dispositions=['STOP_AFTER_CURRENT_REPORT', 'PAUSE_TRAINING'], scope='cycle04/05')])
    path = Path(artifact_root)/'STOP_CONTROLS.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    return receipt


def stop_status(artifact_root):
    receipt = json.loads((Path(artifact_root)/'STOP_CONTROLS.json').read_text(encoding='utf-8'))
    expected = {(str(root), str(root/'artifacts'/STUDY/status), STUDY, status)
                for root in WORKSPACES for status in SENTINELS}
    controls = receipt['controls']
    actual = {(c['workspace'], c['path'], c['scope'], c['status']) for c in controls}
    if receipt['study_id'] != STUDY or actual != expected or len(actual) != len(controls):
        raise ValueError('S0 stop receipt must cover both registered current workspaces')
    for status in SENTINELS:
        for row in controls:
            if row['status'] == status and Path(row['path']).exists():
                return status, row['path']
    return None


def stop_requested(artifact_root):
    return stop_status(artifact_root) is not None


def eligible(row, host_fraction, *, physical_fit=False, cuda_usable=None, full_process=None):
    values = (host_fraction, row['used'], row['total'], row['watts'], row['limit'])
    if not all(math.isfinite(value) and value >= 0 for value in values) or not row['total'] or not row['limit']:
        raise ValueError('GPU or host resource telemetry unavailable')
    if host_fraction >= .90 or row['used'] >= .85 * row['total'] or row['watts'] >= .85 * row['limit']:
        return False
    if physical_fit:
        return bool(cuda_usable and full_process and row['used'] + full_process <= cuda_usable)
    return True


def residency(measurements, pid, gpu_uuid):
    return max((p['bytes'] for p in measurements['processes']
                if p['pid'] == pid and p['uuid'] == gpu_uuid), default=None)


def quiescent_capacity(row, processes, cuda_usable):
    return bool(cuda_usable and row['used'] <= row['total']-cuda_usable
                and not any(p['uuid']==row['uuid'] for p in processes))


def allocation_bytes(error):
    match = re.search(r'Tried to allocate ([\d.]+) (GiB|MiB|KiB|bytes)', str(error))
    if match is None:
        return None
    number, unit = match.groups()
    rounding = .5*10**(-len(number.partition('.')[2])) if unit != 'bytes' else 0
    return math.ceil((float(number)+rounding)*{'GiB':2**30,'MiB':2**20,'KiB':2**10,'bytes':1}[unit])
