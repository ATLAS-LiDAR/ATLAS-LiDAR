"""Shared pieces of the ATLAS evaluation harness: GPU scheduling, manifests, receipts.

Nothing here imports torch or a detector, so it can be unit-tested anywhere.
"""
import csv
import fcntl
import hashlib
import json
import os
import re
import threading
from pathlib import Path

# Markers a CUDA driver prints when a GPU has an uncorrectable memory error. A job that
# dies with one of these takes its GPU out of the pool instead of failing every job that
# would be scheduled on it next.
ECC_MARKERS = ('uncorrectable ecc', 'cuda_error_ecc_uncorrectable')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value, **kwargs):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, **kwargs))
    tmp.replace(path)


def gpu_slots(gpus=None, exclude='', max_parallel=None, slots_per_gpu=1):
    """GPU ids to schedule on, each repeated `slots_per_gpu` times.

    Defaults to CUDA_VISIBLE_DEVICES. Several slots per GPU let one job's CPU-bound
    metric phase overlap another job's inference on the same device.
    """
    visible = gpus if gpus is not None else os.environ.get('CUDA_VISIBLE_DEVICES')
    if not visible:
        raise ValueError('No GPUs: pass --gpus or set CUDA_VISIBLE_DEVICES')
    ids = [g.strip() for g in visible.split(',') if g.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('GPU ids must be unique')
    excluded = {g.strip() for g in (exclude or '').split(',') if g.strip()}
    ids = [g for g in ids if g not in excluded]
    if max_parallel:
        ids = ids[:max_parallel]
    if not ids:
        raise ValueError('No GPU slots remain after exclusions')
    if slots_per_gpu < 1:
        raise ValueError('slots_per_gpu must be >= 1')
    return [g for _ in range(slots_per_gpu) for g in ids]


class HealthyGpuPool:
    """Hands out GPU slots; a slot released as unhealthy is never handed out again."""

    def __init__(self, slots):
        self.available = list(slots)
        self.healthy = set(slots)
        self.condition = threading.Condition()

    def acquire(self):
        with self.condition:
            while not self.available:
                if not self.healthy:
                    raise RuntimeError('No healthy GPU slots remain')
                self.condition.wait()
            return self.available.pop(0)

    def release(self, gpu, healthy=True):
        with self.condition:
            if healthy:
                self.available.append(gpu)
            else:
                self.healthy.discard(gpu)
                self.available = [g for g in self.available if g != gpu]
            self.condition.notify_all()


def log_reports_ecc(log_path):
    text = Path(log_path).read_text(errors='replace').lower()
    return any(marker in text for marker in ECC_MARKERS)


def worker_env(base_env, gpu, threads, pythonpath=()):
    """Environment for a single-GPU, non-distributed worker process."""
    env = dict(base_env)
    paths = [str(p) for p in pythonpath if Path(p).exists()]
    if env.get('PYTHONPATH'):
        paths.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(paths)
    threads = str(threads)
    env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
               MKL_NUM_THREADS=threads, TF_NUM_INTRAOP_THREADS=threads, TF_NUM_INTEROP_THREADS='1',
               TF_FORCE_GPU_ALLOW_GROWTH='true', PYTHONUNBUFFERED='1', MPLBACKEND='Agg')
    # Launched inside a distributed allocation, a worker must still be a plain process.
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE'):
        env.pop(key, None)
    return env


def merge_summary(out, rows, key_fields, flatten=None):
    """Merge rows into out/summary.{json,csv}; a failed retry never replaces a success."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.summary.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out / 'summary.json'
        old = json.loads(path.read_text()) if path.exists() else []
        merged = {tuple(r.get(k) for k in key_fields): r for r in old}
        for row in rows:
            key = tuple(row.get(k) for k in key_fields)
            if row.get('status') != 'ok' and merged.get(key, {}).get('status') == 'ok':
                continue
            merged[key] = row
        rows = list(merged.values())
        atomic_json(path, rows)
        flat = [flatten(r) for r in rows] if flatten else rows
        fields = list(dict.fromkeys(k for r in flat for k in r))
        tmp = out / 'summary.csv.tmp'
        with tmp.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
        tmp.replace(out / 'summary.csv')
    return rows


def _resolve(value, base):
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else base / path


def load_manifest(path, required=('cfg', 'ckpt'), default_repo=None):
    """Read a model manifest.

    {"<name>": {"repo": ..., "cfg": ..., "ckpt": ..., "epochs": [...], ...}, ...}

    Environment variables and ~ are expanded. A relative `repo` or `ckpt` is relative to
    the manifest's directory; a relative `cfg` is relative to `repo`. A model without a
    `repo` gets `default_repo` when one is given. A `ckpt` containing `{epoch}` together with an `epochs` list
    expands to one model per epoch, named `<name>_ep<epoch>`. Keys starting with `_` are
    comments.
    """
    path = Path(path).resolve()
    base = path.parent
    specs = {}
    for name, raw in json.loads(path.read_text()).items():
        if name.startswith('_'):
            continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError(f'Invalid model name: {name!r}')
        missing = [k for k in required if k not in raw]
        if missing:
            raise ValueError(f'{name}: missing {", ".join(missing)}')
        spec = {k: v for k, v in raw.items() if not k.startswith('_')}
        if 'repo' not in spec and default_repo is not None:
            spec['repo'] = str(default_repo)
        if 'repo' in spec:
            spec['repo'] = str(_resolve(spec['repo'], base).resolve())
        cfg_base = Path(spec['repo']) if 'repo' in spec else base
        spec['cfg'] = str(_resolve(spec['cfg'], cfg_base))
        spec['pythonpath'] = [str(_resolve(p, base)) for p in spec.get('pythonpath', [])]
        if 'python' in spec:
            spec['python'] = os.path.expanduser(os.path.expandvars(spec['python']))
        epochs = spec.pop('epochs', None)
        ckpt = str(_resolve(spec['ckpt'], base))
        if epochs is None:
            if '{epoch}' in ckpt:
                raise ValueError(f'{name}: checkpoint has {{epoch}} but no "epochs" list')
            specs[name] = dict(spec, ckpt=ckpt)
            continue
        if '{epoch}' not in ckpt:
            raise ValueError(f'{name}: "epochs" given but checkpoint has no {{epoch}}')
        for epoch in epochs:
            specs[f'{name}_ep{epoch}'] = dict(spec, ckpt=ckpt.format(epoch=epoch),
                                              base_model=name, epoch=int(epoch))
    if not specs:
        raise ValueError('Empty manifest: ' + str(path))
    return specs
