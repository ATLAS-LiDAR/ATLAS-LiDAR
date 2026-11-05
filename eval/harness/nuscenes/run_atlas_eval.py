#!/usr/bin/env python3
"""Evaluate nuScenes detectors on ATLAS-nuScenes: ASR plus AP/mAP/NDS on the attacked frames.

One job per (model, configuration), each in its own single-GPU process. Models are listed
in a JSON manifest (see manifest.example.json); two backends are supported:

  det3d  CenterPoint / CenterFormer style repositories, run unmodified: the harness swaps
         the point-cloud loader for one that reads the attacked clouds.
  pcdet  OpenPCDet-style repositories (LION, DSVT, OpenPCDet, ...) with the ATLAS hook
         from eval/detector-patches/ applied.

Configurations are the directory names under <atlas_data>/data/ (car: inject_global_easy,
removal_az20, ...; other targets: inject_global_pedestrian_easy, ...). The special
configuration `clean` runs the full validation split with the standard devkit metrics, so
clean and attacked rows land in one summary.

  python eval/harness/nuscenes/run_atlas_eval.py --manifest models.json \\
      --atlas_data /data/ATLAS-nuScenes --dataroot /data/nuscenes/v1.0-trainval \\
      --output_dir eval/output/nuscenes --gpus 0,1,2,3 --resume

Only the attacked frames shipped in ATLAS-nuScenes are evaluated, each with its own
pre-accumulated 10-sweep cloud; no clean sweeps are added. --dry_run checks paths only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import importlib.util
import json
import logging
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common  # noqa: E402

CLEAN = 'clean'
CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
HARNESS_CODE = [HERE / 'run_atlas_eval.py', HERE / 'atlas_eval.py', HERE / 'attacked_metrics.py',
                HERE / 'numpy_compat.py', HERE.parent / 'common.py']
DEVKIT_META = dict(use_camera=False, use_lidar=True, use_radar=False, use_map=False, use_external=False)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def atlas_module():
    return module(HERE / 'atlas_eval.py', 'atlas_eval_harness')


def target_class_for(config):
    # Mirrors atlas_eval.target_class_for without importing numpy.
    parts = config.split('_')
    for token, name in (('car', 'car'), ('pedestrian', 'pedestrian'), ('cyclist', 'bicycle')):
        if token in parts[1:]:
            return name
    return 'car'


def default_iou(target_class):
    return {'car': 0.3, 'pedestrian': 0.5, 'bicycle': 0.5}.get(target_class, 0.5)


def is_centerformer(spec):
    return spec.get('variant', Path(spec['repo']).name.lower()) == 'centerformer'


# ---------------------------------------------------------------- inference

class AtlasPoints:
    """det3d loader replacement: read the attacked cloud; voxelization and model are untouched."""

    def __init__(self, cache, records):
        self.cache, self.records = Path(cache), records

    def __call__(self, res, info):
        import numpy as np
        points = np.load(self.cache / self.records[info['token']]['points_file'], allow_pickle=False)[:, :5].copy()
        res['type'] = 'NuScenesDataset'
        res['lidar'].update(points=points[:, :4], times=points[:, 4:5], combined=points, points_num=len(points))
        return res, info


def normalize_det3d(output, classes):
    """det3d [x, y, z, w, l, h, vx, vy, r] -> OpenPCDet [x, y, z, l, w, h, yaw, vx, vy]."""
    import numpy as np
    raw = output['box3d_lidar'].detach().cpu().numpy()
    if raw.ndim != 2 or raw.shape[1] != 9:
        raise ValueError(f'Expected native nuScenes 9D boxes, got {raw.shape}')
    labels = output['label_preds'].detach().cpu().numpy().astype(int)
    if ((labels < 0) | (labels >= len(classes))).any():
        raise ValueError('Invalid class labels')
    boxes = raw[:, [0, 1, 2, 4, 3, 5, 8, 6, 7]].copy()
    boxes[:, 6] = -raw[:, 8] - np.pi / 2
    return dict(metadata=output['metadata'], boxes_lidar=boxes, score=output['scores'].detach().cpu().numpy(),
                name=np.asarray(classes)[labels], pred_labels=labels + 1)


def oriented_iou(boxes, target):
    """Rotated 3D IoU of each box with `target` (centered boxes, [x, y, z, l, w, h, yaw])."""
    import numpy as np
    from shapely.geometry import Polygon

    def polygon(box):
        x, y, _, l, w, _, a = box[:7]
        c, s = np.cos(a), np.sin(a)
        corners = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]) * .5
        return Polygon(corners @ np.array([[c, s], [-s, c]]) + [x, y])
    gt = polygon(target)
    result = []
    for box in boxes:
        height = max(0., min(box[2] + box[5] / 2, target[2] + target[5] / 2)
                     - max(box[2] - box[5] / 2, target[2] - target[5] / 2))
        inter = gt.intersection(polygon(box)).area * height if height else 0.
        result.append(inter / max(float(np.prod(box[3:6]) + np.prod(target[3:6]) - inter), 1e-8))
    return np.asarray(result)


def infer_det3d(args, spec, frames, cache):
    import torch
    from det3d.datasets import build_dataset, build_dataloader
    from det3d.models import build_detector
    from det3d.torchie import Config
    from det3d.torchie.apis import batch_processor
    from det3d.torchie.trainer import load_checkpoint
    from tqdm import tqdm
    cfg = Config.fromfile(spec['cfg'])
    cfg.data.val.root_path = str(args.dataroot)
    if is_centerformer(spec):
        # CenterFormer's neck reads these from its own config; keep them equal to test_cfg.
        for key in ('obj_num', 'score_threshold'):
            if key in cfg.test_cfg:
                cfg.model['neck'][key] = cfg.test_cfg[key]
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # Fail on incompatible weights rather than silently reporting a partially loaded model.
    load_checkpoint(model, spec['ckpt'], map_location='cpu', strict=True)
    print('Checkpoint loaded strictly:', spec['ckpt'], flush=True)
    dataset = build_dataset(cfg.data.val)
    if frames is not None:
        wanted = {r['sample_token']: r for r in frames}
        dataset._nusc_infos = [r for r in dataset._nusc_infos if r['token'] in wanted]
        if len(dataset._nusc_infos) != len(wanted) or {r['token'] for r in dataset._nusc_infos} != set(wanted):
            raise ValueError('ATLAS tokens missing or duplicated in native validation infos')
        transforms = dataset.pipeline.transforms
        if type(transforms[0]).__name__ != 'LoadPointCloudFromFile':
            raise ValueError('Unexpected native input pipeline; refusing to append clean sweeps')
        transforms[0] = AtlasPoints(cache, wanted)
    loader = build_dataloader(dataset, batch_size=args.batch_size, workers_per_gpu=args.workers,
                              dist=False, shuffle=False)
    model.cuda().eval()
    annos = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.configs[0]):
            predictions = batch_processor(model, batch, train_mode=False, local_rank=0)
            annos.extend(normalize_det3d(pred, dataset._class_names) for pred in predictions)
    return annos, None


def infer_pcdet(args, spec, out, clean):
    import torch
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    from tqdm import tqdm
    try:
        from pcdet.datasets.nuscenes import atlas_eval as repo_hook
    except ImportError as exc:
        raise RuntimeError(spec['repo'] + ' lacks the ATLAS hook; apply eval/detector-patches/') from exc
    cfg_from_yaml_file(spec['cfg'], cfg)
    cfg.LOCAL_RANK = 0
    # NuScenesDataset appends VERSION to DATA_PATH; absolute info paths support any metadata root.
    cfg.DATA_CONFIG.DATA_PATH = str(args.dataroot.parent)
    cfg.DATA_CONFIG.INFO_PATH['test'] = [str(args.dataroot / 'nuscenes_infos_10sweeps_val.pkl')]
    if clean:
        cfg.DATA_CONFIG.pop('ATLAS_MANIFEST', None)
    else:
        repo_hook.configure(args, cfg)
    logger = common_utils.create_logger(out / 'eval.log', rank=0)
    dataset, loader, _ = build_dataloader(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
                                          batch_size=args.batch_size, dist=False, workers=args.workers,
                                          logger=logger, training=False)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    state = torch.load(spec['ckpt'], map_location='cpu')['model_state']
    if any(torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):
        raise ValueError('Non-finite checkpoint weights')
    # The repository's own loader adapts spconv layouts; strict=True still requires every tensor.
    model._load_state_dict(state, strict=True)
    del state
    print('Checkpoint loaded strictly:', spec['ckpt'], flush=True)
    model.cuda().eval()
    annos, recalled, gt = [], 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=args.configs[0]):
            load_data_to_gpu(batch)
            predictions, counts = model(batch)
            recalled += counts.get('rcnn_0.3', 0)
            gt += counts.get('gt', 0)
            annos.extend(dataset.generate_prediction_dicts(batch, predictions, cfg.CLASS_NAMES))
    return annos, float(recalled / max(gt, 1))


# ---------------------------------------------------------------- worker

def worker(args):
    spec = json.loads(args._spec)
    repo = Path(spec['repo'])
    os.chdir(repo / 'tools' if spec['backend'] == 'pcdet' else repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    torch.from_numpy(np.zeros(1, dtype=np.float32))  # fails early on a NumPy/PyTorch ABI mismatch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    np.random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.set_device(0)
    config = args.configs[0]
    clean = config == CLEAN
    out = args.output_dir
    atlas = atlas_module()
    frames = None
    if not clean:
        args.atlas_config = config
        args.atlas_attack = config.split('_')[0]
        args.atlas_cache = args.cache_root / config
        args.atlas_target_class = args.target_class
        args.atlas_iou_thresh, args.atlas_score_thresh = args.iou_thresh, args.score_thresh
        args.atlas_cache.mkdir(parents=True, exist_ok=True)
        # Several models may prepare the same configuration at once.
        with (args.atlas_cache / '.prepare.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            atlas.prepare_cache(args)
        frames = json.loads((args.atlas_cache / 'manifest.json').read_text())['frames']

    annos_path, receipt = out / 'result.pkl', out / 'inference.json'
    signature = json.loads(args._signature)
    if args.resume and receipt.exists() and annos_path.exists() and json.loads(receipt.read_text()) == signature:
        with annos_path.open('rb') as f:
            annos = pickle.load(f)
        recall = json.loads((out / 'recall.json').read_text())
    else:
        if spec['backend'] == 'pcdet':
            annos, recall = infer_pcdet(args, spec, out, clean)
        else:
            annos, recall = infer_det3d(args, spec, frames, args.atlas_cache if frames else None)
        with annos_path.open('wb') as f:
            pickle.dump(annos, f)
        common.atomic_json(out / 'recall.json', recall)
        common.atomic_json(receipt, signature)

    tokens = [a['metadata']['token'] for a in annos]
    if len(tokens) != len(set(tokens)):
        raise ValueError('Duplicate prediction tokens')
    if clean:
        summary = dict(attack_type=CLEAN, metrics_scope='clean_full_val')
    else:
        dataset = SimpleNamespace(atlas_by_token={r['sample_token']: r for r in frames})
        atlas.evaluate(dataset, annos, args, out, logging.getLogger('atlas'), iou_fn=oriented_iou)
        summary = json.loads((out / 'atlas_asr.json').read_text())['summary']
        summary.update(metrics_scope='attacked_frames_only')
    summary.update(metrics_num_samples=len(tokens), **{'recall_rcnn_0.3': recall})

    if not args.asr_only:
        module(HERE / 'numpy_compat.py', 'numpy_compat').patch_nuscenes_numpy()
        utils_repo = repo if spec['backend'] == 'pcdet' else args.pcdet_repo
        utils = module(utils_repo / 'pcdet/datasets/nuscenes/nuscenes_utils.py', 'nuscenes_utils')
        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.config import config_factory
        nusc = NuScenes(version='v1.0-trainval', dataroot=str(args.dataroot), verbose=True)
        payload = utils.transform_det_annos_to_nusc_annos(annos, nusc)
        payload['meta'] = DEVKIT_META
        prediction_path = out / 'results_nusc.json'
        common.atomic_json(prediction_path, payload)
        detection_cfg = config_factory('detection_cvpr_2019')
        if clean:
            from nuscenes.eval.detection.evaluate import NuScenesEval
            evaluator = NuScenesEval(nusc, config=detection_cfg, result_path=str(prediction_path),
                                     eval_set='val', output_dir=str(out), verbose=True)
        else:
            from nuscenes.eval.common.loaders import load_gt
            from nuscenes.eval.detection.data_classes import DetectionBox
            metrics_mod = module(HERE / 'attacked_metrics.py', 'attacked_metrics')
            gt = load_gt(nusc, 'val', DetectionBox, verbose=True)
            evaluator = metrics_mod.make_evaluator(nusc, detection_cfg, prediction_path, out, set(tokens), gt)
        evaluator.main(plot_examples=0, render_curves=False)
        metrics = json.loads((out / 'metrics_summary.json').read_text())
        summary.update(mAP=metrics['mean_ap'], NDS=metrics['nd_score'],
                       **{'AP_' + c: metrics['mean_dist_aps'][c] for c in CLASSES})
    common.atomic_json(out / 'values.json', summary)


def environment(spec, gpu, args):
    extra = list(spec.get('pythonpath', [])) + [str(p) for p in args.pythonpath] + [spec['repo']]
    repo = Path(spec['repo'])
    if spec['backend'] == 'pcdet':
        extra += sorted(str(p) for p in (repo / 'pcdet/ops/mamba/build').glob('lib.*'))
    else:
        ops = repo / 'det3d/models/ops'
        extra += [str(ops)] + sorted(str(p) for p in (ops / 'build').glob('lib.*'))
    env = common.worker_env(os.environ, gpu, args.threads_per_task, extra)
    env.setdefault('MPLCONFIGDIR', str(args.output_dir / '.mplconfig'))
    return env


# ---------------------------------------------------------------- scheduler

def discover_configs(atlas_data):
    root = Path(atlas_data) / 'data'
    return sorted(p.name for p in root.iterdir() if p.is_dir() and any(p.glob('train-*.parquet'))) if root.is_dir() else []


def code_files(spec):
    files = list(HARNESS_CODE)
    backend_root = Path(spec['repo']) / ('pcdet' if spec['backend'] == 'pcdet' else 'det3d')
    for subdir in ('models', 'datasets'):
        files += sorted((backend_root / subdir).rglob('*.py'))
    return files


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--manifest', type=Path, required=True, help='JSON model manifest (see manifest.example.json)')
    p.add_argument('--atlas_data', type=Path, help='ATLAS-nuScenes root (contains data/<config>/train-*.parquet)')
    p.add_argument('--dataroot', type=Path, required=True,
                   help='nuScenes root holding v1.0-trainval/ metadata and nuscenes_infos_10sweeps_val.pkl')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--models', nargs='+', help='Subset of manifest models (default: all)')
    p.add_argument('--configs', nargs='+', help=f'Configurations, or "{CLEAN}" (default: every one in --atlas_data)')
    p.add_argument('--cache_root', type=Path, help='Unpacked point clouds (default: <atlas_data>/.eval_cache)')
    p.add_argument('--pcdet_repo', type=Path, default=HERE.parents[1] / 'phim',
                   help='OpenPCDet-style repository providing nuscenes_utils for det3d models (default: eval/phim)')
    p.add_argument('--pythonpath', nargs='*', type=Path, default=[], help='Extra PYTHONPATH entries for every worker')
    p.add_argument('--target_class', help='Override the target class inferred from each configuration name')
    p.add_argument('--iou_thresh', type=float, help='Override the IoU threshold (default: 0.3 car, 0.5 pedestrian/bicycle)')
    p.add_argument('--score_thresh', type=float, default=0.)
    p.add_argument('--gpus', help='Comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES)')
    p.add_argument('--exclude_gpus', default='')
    p.add_argument('--max_parallel', type=int, help='Use at most this many GPUs')
    p.add_argument('--slots_per_gpu', type=int, default=1, help='Concurrent jobs per GPU')
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads_per_task', type=int, default=2)
    p.add_argument('--resume', action='store_true', help='Skip finished jobs whose inputs are unchanged')
    p.add_argument('--dry_run', action='store_true', help='Validate inputs and list the jobs, then exit')
    p.add_argument('--asr_only', action='store_true', help='Skip AP/mAP/NDS')
    p.add_argument('--_spec', help=argparse.SUPPRESS)
    p.add_argument('--_signature', help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.pythonpath = [Path(v).resolve() for v in args.pythonpath]
    if args.batch_size < 1 or args.workers < 0 or args.threads_per_task < 1 or args.slots_per_gpu < 1:
        p.error('Invalid batch_size / workers / threads_per_task / slots_per_gpu')
    if args.iou_thresh is not None and not 0 < args.iou_thresh <= 1 or not 0 <= args.score_thresh <= 1:
        p.error('IoU threshold must be in (0,1]; score threshold in [0,1]')
    if args.cache_root is None and args.atlas_data is not None:
        args.cache_root = args.atlas_data / '.eval_cache'
    return p, args


def main(argv=None):
    p, args = parse_args(argv)
    if args._spec:
        worker(args)
        return
    specs = common.load_manifest(args.manifest)
    models = args.models or list(specs)
    for name in models:
        if name not in specs:
            p.error('Unknown model: ' + name)
        spec = specs[name]
        if spec.get('backend') not in ('det3d', 'pcdet'):
            p.error(f'{name}: backend must be det3d or pcdet')
        if 'repo' not in spec:
            p.error(f'{name}: missing repo')
        for key in ('repo', 'cfg', 'ckpt'):
            if not Path(spec[key]).exists():
                p.error(f'{name}: missing {key}: {spec[key]}')
    configs = args.configs or ([CLEAN] if args.atlas_data is None else discover_configs(args.atlas_data))
    for config in configs:
        if config == CLEAN:
            continue
        if not re.fullmatch(r'(inject|removal)_[A-Za-z0-9_]+', config):
            p.error('Invalid configuration name: ' + config)
        if args.atlas_data is None:
            p.error('--atlas_data is required for attacked configurations')
        if not list((args.atlas_data / 'data' / config).glob('train-*.parquet')):
            p.error('Missing ATLAS shards: ' + config)
    if not configs:
        p.error('No configurations found')
    if not (args.dataroot / 'v1.0-trainval/sample.json').is_file():
        p.error('Missing nuScenes metadata: ' + str(args.dataroot / 'v1.0-trainval'))
    if not args.asr_only and any(specs[m]['backend'] == 'det3d' for m in models) \
            and not (args.pcdet_repo / 'pcdet/datasets/nuscenes/nuscenes_utils.py').is_file():
        p.error('det3d models need --pcdet_repo (an OpenPCDet-style checkout) for AP/mAP/NDS')
    try:
        slots = common.gpu_slots(args.gpus, args.exclude_gpus, args.max_parallel, args.slots_per_gpu)
    except ValueError as exc:
        if not args.dry_run:
            p.error(str(exc))
        slots = []

    def thresholds(config):
        target = args.target_class or target_class_for(config)
        return target, args.iou_thresh if args.iou_thresh is not None else default_iou(target)

    for name in models:
        print(f'{name:32s} {specs[name]["backend"]:6s} {specs[name]["ckpt"]}', flush=True)
    for config in configs:
        print(f'{config:40s} ' + ('full validation split' if config == CLEAN else
              'target {} @ IoU {}'.format(*thresholds(config))), flush=True)
    print(f'GPU slots: {slots}; jobs: {len(models) * len(configs)}; summary: {args.output_dir / "summary.csv"}',
          flush=True)
    if args.dry_run:
        return
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume:
        p.error('Output exists; use --resume or a new --output_dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pool = common.HealthyGpuPool(slots)
    signatures = {}
    for name in models:
        spec = specs[name]
        signatures[name] = dict(model=spec, checkpoint_sha=common.sha(spec['ckpt']), cfg_sha=common.sha(spec['cfg']),
                                code={str(f): common.sha(f) for f in code_files(spec)},
                                score=args.score_thresh, batch_size=args.batch_size, asr_only=args.asr_only,
                                dataroot=str(args.dataroot))

    def run(name, config):
        out = args.output_dir / name / config
        out.mkdir(parents=True, exist_ok=True)
        log, receipt = out / 'worker.log', out / 'done.json'
        target, iou = thresholds(config)
        shards = [] if config == CLEAN else [
            (f.name, f.stat().st_size, f.stat().st_mtime_ns)
            for f in sorted((args.atlas_data / 'data' / config).glob('train-*.parquet'))]
        signature = json.loads(json.dumps(dict(signatures[name], config=config, shards=shards,
                                               target_class=None if config == CLEAN else target, iou=iou)))
        spec = specs[name]
        row = dict(model=name, config=config, status='failed', checkpoint=spec['ckpt'], log_path=str(log))
        row.update({k: spec[k] for k in ('base_model', 'epoch') if k in spec})
        gpu, healthy = None, True
        try:
            if args.resume and receipt.exists():
                old = json.loads(receipt.read_text())
                if old['signature'] != signature:
                    raise ValueError('Inputs changed; use a new --output_dir')
                if old['row']['status'] == 'ok':
                    print('SKIP', name, config, flush=True)
                    return old['row']
            gpu = pool.acquire()
            row['gpu'] = gpu
            cmd = [spec.get('python', sys.executable), '-u', str(Path(__file__).resolve()),
                   '--_spec', json.dumps(spec), '--_signature', json.dumps(signature),
                   '--manifest', str(args.manifest), '--configs', config, '--output_dir', str(out),
                   '--dataroot', str(args.dataroot), '--pcdet_repo', str(args.pcdet_repo),
                   '--batch_size', str(args.batch_size), '--workers', str(args.workers),
                   '--iou_thresh', str(iou), '--target_class', target, '--score_thresh', str(args.score_thresh)]
            if config != CLEAN:
                cmd += ['--atlas_data', str(args.atlas_data), '--cache_root', str(args.cache_root)]
            if args.asr_only:
                cmd.append('--asr_only')
            if args.resume:
                cmd.append('--resume')
            print('RUN', name, config, 'GPU', gpu, 'log=', log, flush=True)
            with log.open('w') as f:
                result = subprocess.run(cmd, env=environment(spec, gpu, args), stdout=f, stderr=subprocess.STDOUT)
            row['exit_code'] = result.returncode
            if result.returncode == 0:
                row.update(json.loads((out / 'values.json').read_text()), status='ok')
            elif common.log_reports_ecc(log):
                healthy = False
                print('QUARANTINE GPU', gpu, flush=True)
            common.atomic_json(receipt, dict(signature=signature, row=row))
        except Exception as exc:
            row['error'] = str(exc)
        finally:
            if gpu is not None:
                pool.release(gpu, healthy=healthy)
        print(row['status'].upper(), name, config, flush=True)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [executor.submit(run, name, config) for config in configs for name in models]
        for future in as_completed(futures):
            rows.append(future.result())
            common.merge_summary(args.output_dir, rows, ('model', 'config'))
    failed = sum(r['status'] != 'ok' for r in rows)
    print(f'Finished {len(rows) - failed}/{len(rows)}; {args.output_dir / "summary.csv"}', flush=True)
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
