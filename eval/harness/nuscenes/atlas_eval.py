"""ATLAS-nuScenes input and target-specific attack success rate (ASR).

This file is used two ways, and must stay identical in both places:

  * by the harness (eval/harness/nuscenes/run_atlas_eval.py), loaded by path;
  * inside an OpenPCDet-style detector repository as pcdet/datasets/nuscenes/atlas_eval.py,
    installed by the patches in eval/detector-patches/, where tools/test.py gains the
    --atlas_* flags below.

It reads the attacked frames of one ATLAS-nuScenes configuration from the Parquet
release, caches each point cloud as .npy, and scores ASR: for injection, the fraction of
attacked frames where the detector reports the phantom target; for removal, the fraction
where the erased real object is no longer detected. The denominator is every shipped
attacked frame, not conditioned on the clean model detecting anything.
"""
import json
from pathlib import Path
import numpy as np

# Class token in an ATLAS configuration name -> nuScenes detection class.
CONFIG_CLASSES = {'car': 'car', 'pedestrian': 'pedestrian', 'cyclist': 'bicycle'}
# Default IoU per target class: 0.3 for cars (the original ATLAS-nuScenes definition);
# 0.5 for small targets, as for Pedestrian/Cyclist in the Waymo ASR patch.
DEFAULT_IOU = {'car': 0.3, 'pedestrian': 0.5, 'bicycle': 0.5}
# spoof_gt may carry a 10th element with the 1-based nuScenes class index.
CLASS_INDEX = {'car': 1}


def target_class_for(config):
    """'inject_global_pedestrian_easy' -> 'pedestrian'; car configurations carry no token."""
    parts = config.split('_')
    for token, name in CONFIG_CLASSES.items():
        if token in parts[1:]:
            return name
    return 'car'


def add_arguments(parser):
    parser.add_argument('--atlas_attack', choices=['inject', 'removal'], default=None,
                        help='Enable ATLAS attacked-frame-only ASR evaluation')
    parser.add_argument('--atlas_data', type=Path, help='Local ATLAS-nuScenes dataset root')
    parser.add_argument('--atlas_config', help='Configuration name, e.g. inject_global_easy or removal_az20')
    parser.add_argument('--atlas_cache', type=Path, help='Optional cache directory for this configuration')
    parser.add_argument('--atlas_target_class', default=None,
                        help='nuScenes class of the attack target (default: inferred from --atlas_config)')
    parser.add_argument('--atlas_iou_thresh', type=float, default=None,
                        help='Rotated 3D IoU threshold (default: 0.3 for car, 0.5 for pedestrian/bicycle)')
    parser.add_argument('--atlas_score_thresh', type=float, default=0.0)


def validate_arguments(args, parser):
    if not args.atlas_attack:
        return
    if args.atlas_data is None or args.atlas_config is None:
        parser.error('--atlas_attack requires --atlas_data and --atlas_config')
    if not args.atlas_config.startswith(args.atlas_attack + '_'):
        parser.error('--atlas_config does not match --atlas_attack')
    if Path(args.atlas_config).name != args.atlas_config:
        parser.error('--atlas_config must be a configuration name')
    if args.atlas_target_class is None:
        args.atlas_target_class = target_class_for(args.atlas_config)
    if args.atlas_iou_thresh is None:
        args.atlas_iou_thresh = DEFAULT_IOU.get(args.atlas_target_class, 0.5)
    if not 0 < args.atlas_iou_thresh <= 1 or not 0 <= args.atlas_score_thresh <= 1:
        parser.error('IoU threshold must be in (0,1]; score threshold in [0,1]')


def cache_dir(args):
    return (args.atlas_cache or Path(args.atlas_data).resolve() / '.eval_cache' / args.atlas_config).resolve()


def prepare_cache(args):
    """Unpack one configuration's Parquet shards to per-frame .npy files, once.

    Rows are streamed one at a time so memory stays bounded by a single point cloud.
    """
    import os
    import uuid
    import pyarrow.parquet as pq
    root = Path(args.atlas_data).resolve()
    cache = cache_dir(args)
    cache.mkdir(parents=True, exist_ok=True)
    shards = sorted((root / 'data' / args.atlas_config).glob('train-*.parquet'))
    if not shards:
        raise ValueError('No local Parquet shards for ' + args.atlas_config)
    signature = json.loads(json.dumps([(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in shards]))
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old['signature'] == signature and all((cache / r['points_file']).is_file() for r in old['frames']):
            return
        raise ValueError('Stale or incomplete cache: choose a new --atlas_cache directory')
    records, seen = [], set()
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            token = row['sample_token']
            if len(token) != 32 or any(c not in '0123456789abcdef' for c in token) or token in seen:
                raise ValueError('Invalid or duplicate sample_token: ' + token)
            seen.add(token)
            points = np.asarray(row.pop('points'), dtype=np.float32).reshape(row['num_points'], row['point_dim'])
            target = np.asarray(row['spoof_gt'], dtype=np.float32)
            if points.shape[1] != 6 or not np.isfinite(points).all():
                raise ValueError('Invalid points for ' + token)
            if target.size not in (7, 10) or not np.isfinite(target[:7]).all() or np.any(target[3:6] <= 0):
                raise ValueError('Invalid spoof_gt for ' + token)
            row['points_file'] = token + '.npy'
            temp = cache / (token + '.' + uuid.uuid4().hex + '.tmp')
            with temp.open('wb') as f:
                np.save(f, points, allow_pickle=False)
            os.replace(temp, cache / row['points_file'])
            records.append(row)
    if not records:
        raise ValueError('Empty ATLAS dataset')
    temp = cache / ('manifest.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps({'signature': signature, 'config': args.atlas_config, 'frames': records}))
    os.replace(temp, manifest_path)


def configure(args, cfg):
    """Point an OpenPCDet NuScenesDataset config at the prepared ATLAS cache."""
    if not getattr(args, 'atlas_attack', None):
        return
    if cfg.DATA_CONFIG.DATASET != 'NuScenesDataset':
        raise ValueError('ATLAS input currently supports NuScenesDataset only')
    import torch.distributed as dist
    # Every rank participates; propagate cache failures instead of hanging at a barrier.
    distributed = dist.is_available() and dist.is_initialized()
    error = [None]
    if not distributed or dist.get_rank() == 0:
        try:
            prepare_cache(args)
        except Exception as exc:
            error[0] = str(exc)
    if distributed:
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError('ATLAS cache preparation failed: ' + error[0])
    cfg.DATA_CONFIG.ATLAS_MANIFEST = str(cache_dir(args) / 'manifest.json')


def attach_dataset(dataset):
    """Restrict a NuScenesDataset to the attacked frames named in ATLAS_MANIFEST."""
    path = dataset.dataset_cfg.get('ATLAS_MANIFEST')
    if not path:
        return
    if dataset.training:
        raise ValueError('ATLAS evaluation must use training=False')
    dataset.atlas_root = Path(path).parent
    frames = json.loads(Path(path).read_text())['frames']
    dataset.atlas_by_token = {r['sample_token']: r for r in frames}
    available = {r['token'] for r in dataset.infos}
    missing = set(dataset.atlas_by_token) - available
    if missing:
        raise ValueError(f'{len(missing)} ATLAS tokens absent from configured evaluation split')
    dataset.infos = [r for r in dataset.infos if r['token'] in dataset.atlas_by_token]
    if len(dataset.infos) != len(frames):
        raise ValueError('ATLAS frame count does not match unique dataset infos')
    dataset.logger.info('ATLAS evaluation: %d attacked frames only', len(dataset.infos))


def summarize(records, attack, iou_thresh, score_thresh, target_class='car'):
    n = len(records)
    successes = sum(r['attack_success'] for r in records)
    summary = dict(attack_type=attack, target_class=target_class, num_target_frames=n,
                   num_success=successes, asr=successes / n if n else None,
                   num_detected=sum(r['target_detected'] for r in records),
                   num_empty_predictions=sum(r['num_predictions'] == 0 for r in records),
                   iou_threshold=iou_thresh, score_threshold=score_thresh,
                   matching=f'same-class oriented 3D IoU; {target_class} target',
                   denominator='all shipped attacked frames (not conditioned on clean detection)')
    kept = [r['n_spoof_kept'] for r in records if r.get('n_spoof_kept') is not None]
    if kept:
        summary['num_target_fully_occluded'] = sum(k == 0 for k in kept)
    return summary


def evaluate(dataset, annos, args, result_dir, logger, iou_fn=None):
    """Score ASR for `annos` (OpenPCDet prediction dicts) against dataset.atlas_by_token."""
    target_class = getattr(args, 'atlas_target_class', None) or target_class_for(args.atlas_config)
    if iou_fn is None:
        import torch
        from pcdet.ops.iou3d_nms import iou3d_nms_utils

        def iou_fn(boxes, target):
            return iou3d_nms_utils.boxes_iou3d_gpu(
                torch.as_tensor(boxes, dtype=torch.float32, device='cuda').contiguous(),
                torch.as_tensor(target[None], dtype=torch.float32, device='cuda').contiguous()
            ).cpu().numpy().reshape(-1)
    result_dir = Path(result_dir)
    records, seen = [], set()
    for anno in annos:
        token = anno['metadata']['token']
        if token in seen:
            raise ValueError('Duplicate prediction token: ' + token)
        seen.add(token)
        row = dataset.atlas_by_token[token]
        target = np.asarray(row['spoof_gt'], dtype=np.float32)
        if target.size == 10 and target_class in CLASS_INDEX and int(target[9]) != CLASS_INDEX[target_class]:
            raise ValueError(f'spoof_gt class does not match the {target_class} target')
        boxes = np.asarray(anno['boxes_lidar'])[:, :7]
        scores = np.asarray(anno['score'])
        mask = (np.asarray(anno['name']) == target_class) & (scores >= args.atlas_score_thresh)
        overlaps = iou_fn(boxes[mask], target[:7]) if mask.any() else np.empty(0)
        best = int(np.argmax(overlaps)) if len(overlaps) else None
        max_iou = float(overlaps[best]) if best is not None else 0.0
        detected = max_iou >= args.atlas_iou_thresh
        kept = row.get('atk_n_spoof_k')
        records.append(dict(sample_token=token, segment=row['segment'], frame_index=row['frame_index'],
                            num_predictions=len(boxes), num_candidate_predictions=int(mask.sum()),
                            max_iou=max_iou, matched_score=float(scores[mask][best]) if detected else None,
                            target_detected=detected, n_spoof_kept=None if kept is None else int(kept),
                            attack_success=detected if args.atlas_attack == 'inject' else not detected))
    if seen != set(dataset.atlas_by_token):
        raise ValueError('Missing ATLAS predictions; refusing to report partial ASR')
    summary = summarize(records, args.atlas_attack, args.atlas_iou_thresh, args.atlas_score_thresh, target_class)
    import pickle
    with (result_dir / 'result_asr.pkl').open('wb') as f:
        pickle.dump({'summary': summary, 'records': records}, f)
    (result_dir / 'atlas_asr.json').write_text(json.dumps({'summary': summary, 'records': records}, indent=2))
    logger.info('ATLAS %s ASR (%s): %.6f (%d/%d), IoU >= %.3f, score >= %.3f',
                args.atlas_attack, target_class, summary['asr'], summary['num_success'],
                summary['num_target_frames'], args.atlas_iou_thresh, args.atlas_score_thresh)
    return {'atlas/asr': summary['asr'], 'atlas/num_success': summary['num_success'],
            'atlas/num_target_frames': summary['num_target_frames']}
