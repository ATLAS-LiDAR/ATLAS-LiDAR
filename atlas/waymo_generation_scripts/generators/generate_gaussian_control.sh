#!/bin/bash
# ==========================================================================
# ATLAS point-injection rebuttal: SCENE-WIDE GAUSSIAN NOISE CONTROL.
#
# Perturbs EVERY point's Cartesian coordinates with independent zero-mean
# Gaussian noise, (x',y',z') = (x,y,z) + eps, eps ~ N(0, sigma^2 I3), on
# exactly the frames the paired ATLAS injection run attacked (32-off/32-on
# schedule), in exactly the same segments. NO points are added or removed.
# Clean (non-attacked) frames are written through untouched.
#
# PAIRING: the ATLAS run's dataset pickles are the source of truth. A frame
# is a target iff its ATLAS frame dict has spoof_gt is not None. The spoof
# box is COPIED into the Gaussian dataset so the injection eval can test for
# false positives at the same phantom location (expected: zero FPs here,
# since no points were injected and sigma-level jitter cannot assemble a
# vehicle-shaped cluster).
#
# CLEAN POINTS come from the same SF dataloader path the ATLAS generator
# used (dataset_sf[idx]['points'] + lag=0 column), so the only difference
# between the two datasets on a target frame is noise-vs-injection.
# CP_SF_CFG MUST MATCH the --cp_sf_cfg of the paired ATLAS injection run
# (point range / feature encoding must be identical).
#
# Severity: sigma follows the CVPR 2023 3D common-corruptions schedule,
# sigma in {0.02, 0.04, 0.06, 0.08, 0.10} m. Set --array=0-4 with the full
# SIGMAS list to generate all five severities in one submission.
#
# CPU-BOUND: no model is run; the cost is dataloading + voxelization in
# workers. No GPU requested.
# ==========================================================================

#SBATCH --account=gts-gchou3
#SBATCH --job-name=gen_gauss_ctrl
#SBATCH --partition=cpu-small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-0

set -euo pipefail

# Launch with:  bash "$0"   (self-submits)

# ----------------------------- EDIT ---------------------------------------
SIGMAS=(0.08)                 # e.g. (0.02 0.04 0.06 0.08 0.10) with --array=0-4
ATLAS_RUN_NAME="global_easy_seed42"   # paired ATLAS injection run
SEED=42                       # noise seed (schedule/segments come from ATLAS)
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

PROJECT_ROOT="/storage/cedar/cedar0/cedarp-gchou3-0/adversarial_lidar/PTT"
NEURIPS_ROOT="$PROJECT_ROOT/neurips"
GENERATION_ROOT="$NEURIPS_ROOT/generation_runs"

OPENPCDET_ROOT="/storage/project/r-gchou3-0/spanse30/OpenPCDet"
# MUST match the --cp_sf_cfg used by the paired ATLAS injection generation.
CP_SF_CFG="$OPENPCDET_ROOT/tools/cfgs/waymo_models/centerpoint.yaml"
CONDA_ENV="/storage/project/r-gchou3-0/spanse30/ptt_env"

SLURM_LOG_DIR="$NEURIPS_ROOT/data_gen_logs"
# --------------------------------------------------------------------------

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    mkdir -p "$SLURM_LOG_DIR"
    exec sbatch \
        --output="$SLURM_LOG_DIR/%x_%A_%a.out" \
        --error="$SLURM_LOG_DIR/%x_%A_%a.err" \
        "$0"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#SIGMAS[@]} )); then
    echo "ERROR: invalid array task ID: $TASK_ID" >&2
    exit 1
fi
SIGMA="${SIGMAS[$TASK_ID]}"
SIGMA_TAG="$(python3 -c "print(f'{int(round(float(\"$SIGMA\")*100)):03d}')")"

RUN_NAME="gaussian_sigma_${SIGMA_TAG}_${ATLAS_RUN_NAME}"
RUN_ROOT="$GENERATION_ROOT/$RUN_NAME"
DATASET_DIR="$RUN_ROOT/dataset"
ANNOS_DIR="$RUN_ROOT/annos"
METRICS_DIR="$RUN_ROOT/metrics"
RUNTIME_LOG="$SLURM_LOG_DIR/${RUN_NAME}.log"

ATLAS_ROOT="$GENERATION_ROOT/$ATLAS_RUN_NAME"
ATLAS_DATASET_DIR="$ATLAS_ROOT/dataset"
ATLAS_ANNOS_DIR="$ATLAS_ROOT/annos"
ATLAS_METRICS_DIR="$ATLAS_ROOT/metrics"

for path in "$OPENPCDET_ROOT" "$CP_SF_CFG" "$ATLAS_DATASET_DIR" "$ATLAS_ANNOS_DIR"; do
    if [[ ! -e "$path" ]]; then
        echo "ERROR: required path does not exist: $path" >&2
        exit 1
    fi
done
mkdir -p "$DATASET_DIR" "$ANNOS_DIR" "$METRICS_DIR" "$SLURM_LOG_DIR"

module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$OPENPCDET_ROOT"
export PYTHONPATH="$OPENPCDET_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$WORKERS"

echo "Generating $RUN_NAME (sigma=${SIGMA} m) paired against $ATLAS_RUN_NAME"

TMP_PY="${SLURM_TMPDIR:-/tmp}/gauss_ctrl_${SLURM_JOB_ID}_${TASK_ID}.py"
trap 'rm -f "$TMP_PY"' EXIT
cat > "$TMP_PY" <<'PY'
"""
Scene-wide Gaussian noise control for the ATLAS point-injection rebuttal.

Reads the paired ATLAS injection run's per-segment dataset pickles to learn
(a) which segments are in the dataset and (b) which frames were attacked
(spoof_gt is not None). For every frame it takes the CLEAN single-frame
cloud from the SF dataloader (identical source to the ATLAS generator:
dataset_sf[idx]['points'] plus a lag=0 column), and on target frames adds
i.i.d. N(0, sigma^2) noise to x, y, z of EVERY point. No points are added
or removed anywhere. The ATLAS spoof box is copied into each target frame
(spoof_gt) so the injection FP evaluation runs unchanged on this dataset.

Output format matches what the injection eval scripts consume:
  dataset/{segment}.pkl  -> list of frame dicts with keys
      frame_id, points (float32 [N, C+1], last col = lag 0.0), pose,
      spoof_gt (copied ATLAS box or None), modified_this_frame,
      noise_sigma_m, attack_run_id, spoof_candidate_count=0,
      spoof_kept_count=0, n_spoof_r=0, n_spoof_k=0
  annos/{segment}_a.pkl  -> provenance + per-frame activity list
  metrics/frames/{segment}_frames.csv, metrics/run_manifest.json
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle as pkl
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcdet_root', required=True)
    parser.add_argument('--cp_sf_cfg', required=True,
                        help='MUST match the paired ATLAS run --cp_sf_cfg')
    parser.add_argument('--atlas_dataset_dir', required=True)
    parser.add_argument('--atlas_annos_dir', required=True)
    parser.add_argument('--atlas_metrics_dir', default='')
    parser.add_argument('--sigma', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir_dataset', required=True)
    parser.add_argument('--save_dir_annos', required=True)
    parser.add_argument('--metrics_dir', required=True)
    parser.add_argument('--log_file', required=True)
    parser.add_argument('--variant_name', required=True)
    parser.add_argument('--workers', type=int, default=8)
    return parser.parse_args()


def setup_logger(log_path):
    logger = logging.getLogger('gen_gauss_ctrl')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def atomic_pickle_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as handle:
        pkl.dump(value, handle)
    os.replace(tmp, path)


def atomic_json_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')

    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(type(obj).__name__)

    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, default=default)
    os.replace(tmp, path)


def atomic_write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.csv')
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(tmp, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def optional_sha256(path, chunk_size=1024 * 1024):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def segment_name_from_info(info):
    pc = info.get('point_cloud')
    if pc and 'lidar_sequence' in pc:
        return str(pc['lidar_sequence'])
    return str(info['frame_id']).rsplit('_', 1)[0]


def discover_atlas_segments(atlas_dataset_dir, atlas_annos_dir):
    """Segments of the paired ATLAS run: dataset pickle + annos must exist."""
    segments = []
    for path in sorted(Path(atlas_dataset_dir).glob('*.pkl')):
        name = path.stem
        if name.endswith('_d'):
            name = name[:-2]
        if (Path(atlas_annos_dir) / f'{name}_a.pkl').is_file():
            segments.append(name)
        else:
            raise RuntimeError(f'ATLAS annos missing for segment {name}')
    if not segments:
        raise RuntimeError(f'No ATLAS dataset pickles in {atlas_dataset_dir}')
    return segments


def atlas_dataset_file(atlas_dataset_dir, segment):
    for candidate in (
        Path(atlas_dataset_dir) / f'{segment}.pkl',
        Path(atlas_dataset_dir) / f'{segment}_d.pkl',
    ):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'No ATLAS dataset pickle for {segment}')


def frame_rng(seed, segment, local_idx):
    """Deterministic per-frame RNG, independent of iteration order."""
    entropy = [int(seed), zlib.crc32(segment.encode('utf-8')), int(local_idx)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def strip_collate(samples):
    """Identity collate that drops voxel tensors before worker IPC."""
    item = samples[0]
    return {
        key: value for key, value in item.items()
        if key not in ('voxels', 'voxel_coords', 'voxel_num_points')
    }


def main():
    args = parse_args()
    script_start = time.perf_counter()
    logger = setup_logger(args.log_file)

    sys.path.insert(0, args.pcdet_root)
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader

    metrics_dir = Path(args.metrics_dir)
    (metrics_dir / 'frames').mkdir(parents=True, exist_ok=True)
    Path(args.save_dir_dataset).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir_annos).mkdir(parents=True, exist_ok=True)

    manifest_path = metrics_dir / 'run_manifest.json'
    manifest = {
        'generator_name': 'generate_gaussian_control',
        'variant_name': args.variant_name,
        'noise_model': 'iid zero-mean Gaussian on x,y,z of every point '
                       '(CVPR 2023 3D common-corruptions global noise)',
        'sigma_m': float(args.sigma),
        'seed': int(args.seed),
        'points_added': 0,
        'points_removed': 0,
        'cp_sf_cfg': str(Path(args.cp_sf_cfg).resolve()),
        'atlas_dataset_dir': str(Path(args.atlas_dataset_dir).resolve()),
        'atlas_annos_dir': str(Path(args.atlas_annos_dir).resolve()),
        'atlas_metrics_dir': (
            str(Path(args.atlas_metrics_dir).resolve())
            if args.atlas_metrics_dir else None
        ),
        'atlas_run_manifest_sha256': (
            optional_sha256(Path(args.atlas_metrics_dir) / 'run_manifest.json')
            if args.atlas_metrics_dir else None
        ),
        'status': 'running',
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    atomic_json_dump(manifest, manifest_path)

    if args.sigma <= 0:
        raise ValueError(f'sigma must be positive, got {args.sigma}')

    selected_segments = discover_atlas_segments(
        args.atlas_dataset_dir, args.atlas_annos_dir
    )
    logger.info('Paired ATLAS run has %d segments', len(selected_segments))
    manifest['selected_segment_count'] = len(selected_segments)

    # Resume: a segment is complete when all three outputs exist.
    completed = set()
    for segment in selected_segments:
        if (
            (Path(args.save_dir_dataset) / f'{segment}.pkl').is_file()
            and (Path(args.save_dir_annos) / f'{segment}_a.pkl').is_file()
            and (metrics_dir / 'frames' / f'{segment}_frames.csv').is_file()
        ):
            completed.add(segment)
    remaining = [s for s in selected_segments if s not in completed]
    logger.info('%d segments already complete, %d remaining',
                len(completed), len(remaining))
    manifest['completed_segment_count_at_start'] = len(completed)
    atomic_json_dump(manifest, manifest_path)

    if not remaining:
        manifest['status'] = 'completed'
        manifest['ended_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json_dump(manifest, manifest_path)
        logger.info('Nothing to do.')
        return

    # ---- clean single-frame points, exactly as the ATLAS generator sourced
    # them: dataset_sf[idx]['points'] (post data-processing), no model. ----
    cfg_from_yaml_file(args.cp_sf_cfg, cfg)
    if os.environ.get('WAYMO_DATA_PATH'):
        cfg.DATA_CONFIG.DATA_PATH = os.environ['WAYMO_DATA_PATH']
    cfg.TAG = Path(args.cp_sf_cfg).stem
    cfg.EXP_GROUP_PATH = 'gaussian_control_generation'
    dataset_sf, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    remaining_set = set(remaining)
    dataset_sf.infos = [
        info for info in dataset_sf.infos
        if segment_name_from_info(info) in remaining_set
    ]
    present = {segment_name_from_info(info) for info in dataset_sf.infos}
    missing = remaining_set - present
    if missing:
        raise RuntimeError(
            'ATLAS segments absent from SF dataset infos: '
            + ', '.join(sorted(missing))
        )

    global_to_segment = {}
    counts = defaultdict(int)
    pose_map = {}
    for global_idx, info in enumerate(dataset_sf.infos):
        segment = segment_name_from_info(info)
        local_idx = counts[segment]
        counts[segment] += 1
        global_to_segment[global_idx] = (segment, local_idx)
        if 'pose' in info:
            pose_map[(segment, local_idx)] = np.asarray(
                info['pose'], dtype=np.float64
            ).reshape(4, 4)

    loader = DataLoader(
        dataset_sf,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=strip_collate,
        drop_last=False,
    )

    sigma = float(args.sigma)
    current_segment = None
    atlas_data = None
    out_frames = None
    out_annos_frames = None
    frame_rows = None
    total_target_frames = 0
    total_frames = 0
    saved_segments = 0

    def flush_current():
        nonlocal current_segment, out_frames, out_annos_frames, frame_rows
        nonlocal saved_segments
        if current_segment is None:
            return
        if len(out_frames) != counts[current_segment]:
            raise RuntimeError(
                f'Incomplete segment {current_segment}: '
                f'{len(out_frames)} != {counts[current_segment]}'
            )
        atomic_pickle_dump(
            out_frames,
            Path(args.save_dir_dataset) / f'{current_segment}.pkl',
        )
        atomic_pickle_dump(
            {
                'segment': current_segment,
                'variant_name': args.variant_name,
                'noise_sigma_m': sigma,
                'seed': int(args.seed),
                'atlas_dataset_dir': str(Path(args.atlas_dataset_dir).resolve()),
                'spoof_annos': out_annos_frames,
            },
            Path(args.save_dir_annos) / f'{current_segment}_a.pkl',
        )
        atomic_write_csv(
            metrics_dir / 'frames' / f'{current_segment}_frames.csv',
            frame_rows,
        )
        saved_segments += 1
        logger.info('Saved: %s', current_segment)

    try:
        for global_idx, item in enumerate(loader):
            segment, local_idx = global_to_segment[global_idx]
            if segment != current_segment:
                flush_current()
                current_segment = segment
                with open(
                    atlas_dataset_file(args.atlas_dataset_dir, segment), 'rb'
                ) as handle:
                    atlas_data = pkl.load(handle)
                if len(atlas_data) != counts[segment]:
                    raise RuntimeError(
                        f'ATLAS/SF frame count mismatch for {segment}: '
                        f'{len(atlas_data)} != {counts[segment]}'
                    )
                out_frames = []
                out_annos_frames = []
                frame_rows = []

            atlas_frame = atlas_data[local_idx]
            frame_id = str(item['frame_id'])
            if frame_id != str(atlas_frame['frame_id']):
                raise AssertionError(
                    f'{frame_id} != {atlas_frame["frame_id"]} '
                    f'({segment}[{local_idx}])'
                )

            pts = np.asarray(item['points'], dtype=np.float32)
            # Same layout the ATLAS generator wrote: SF features + lag=0 col.
            points = np.hstack([
                pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)
            ])
            n_before = int(points.shape[0])

            spoof_gt = atlas_frame.get('spoof_gt')
            active = spoof_gt is not None

            displacement_mean = None
            displacement_max = None
            if active:
                rng = frame_rng(args.seed, segment, local_idx)
                eps = rng.normal(0.0, sigma, size=(points.shape[0], 3))
                points[:, :3] += eps.astype(np.float32)
                norms = np.linalg.norm(eps, axis=1)
                displacement_mean = float(np.mean(norms)) if norms.size else 0.0
                displacement_max = float(np.max(norms)) if norms.size else 0.0
                total_target_frames += 1

            if int(points.shape[0]) != n_before:
                raise AssertionError('Point count changed; control must not '
                                     'add or remove points')

            pose = pose_map.get((segment, local_idx))
            if pose is None:
                pose = np.asarray(atlas_frame.get('pose'), dtype=np.float64)

            out_frames.append({
                'frame_id': frame_id,
                'points': points.astype(np.float32, copy=False),
                'pose': pose,
                'spoof_gt': (
                    np.array(spoof_gt, copy=True) if active else None
                ),
                'attack_run_id': atlas_frame.get('attack_run_id'),
                'modified_this_frame': bool(active),
                'noise_sigma_m': sigma if active else 0.0,
                'spoof_candidate_count': 0,
                'spoof_kept_count': 0,
                'n_spoof_r': 0,
                'n_spoof_k': 0,
            })
            out_annos_frames.append({
                'local_idx': int(local_idx),
                'frame_id': frame_id,
                'active_this_frame': bool(active),
                'modified_this_frame': bool(active),
                'n_points': n_before,
                'noise_sigma_m': sigma if active else 0.0,
            })
            frame_rows.append({
                'variant_name': args.variant_name,
                'segment_id': segment,
                'frame_id': frame_id,
                'local_frame_index': int(local_idx),
                'attack_active': bool(active),
                'noise_sigma_m': sigma if active else 0.0,
                'n_points': n_before,
                'n_points_perturbed': n_before if active else 0,
                'displacement_norm_mean_m': displacement_mean,
                'displacement_norm_max_m': displacement_max,
                'spoof_box_x': None if not active else float(spoof_gt[0]),
                'spoof_box_y': None if not active else float(spoof_gt[1]),
                'spoof_box_z': None if not active else float(spoof_gt[2]),
                'atlas_run_id': atlas_frame.get('attack_run_id'),
            })
            total_frames += 1

        flush_current()

    except Exception:
        import traceback
        logger.error('===== EXCEPTION =====')
        logger.error(traceback.format_exc())
        manifest.update({
            'status': 'failed',
            'ended_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'wall_time_seconds': float(time.perf_counter() - script_start),
            'exception': traceback.format_exc(),
        })
        atomic_json_dump(manifest, manifest_path)
        raise

    manifest.update({
        'status': 'completed',
        'ended_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'wall_time_seconds': float(time.perf_counter() - script_start),
        'frames_processed_this_run': int(total_frames),
        'target_frames_perturbed_this_run': int(total_target_frames),
        'segments_saved_this_run': int(saved_segments),
    })
    atomic_json_dump(manifest, manifest_path)
    logger.info(
        'Done: %d frames (%d perturbed) across %d segments',
        total_frames, total_target_frames, saved_segments,
    )


if __name__ == '__main__':
    main()
PY

srun -u python "$TMP_PY" \
    --pcdet_root "$OPENPCDET_ROOT" \
    --cp_sf_cfg "$CP_SF_CFG" \
    --atlas_dataset_dir "$ATLAS_DATASET_DIR" \
    --atlas_annos_dir "$ATLAS_ANNOS_DIR" \
    --atlas_metrics_dir "$ATLAS_METRICS_DIR" \
    --sigma "$SIGMA" \
    --seed "$SEED" \
    --save_dir_dataset "$DATASET_DIR" \
    --save_dir_annos "$ANNOS_DIR" \
    --metrics_dir "$METRICS_DIR" \
    --log_file "$RUNTIME_LOG" \
    --variant_name "$RUN_NAME" \
    --workers "$WORKERS"

echo "Completed $RUN_NAME"
