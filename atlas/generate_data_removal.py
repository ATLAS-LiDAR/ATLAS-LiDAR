#!/usr/bin/env python
import os
import sys
import argparse
import importlib
import pickle as pkl
import logging
import traceback
import random
from collections import defaultdict

import numpy as np

import removal
importlib.reload(removal)


def unload_pcdet():
    for k in list(sys.modules.keys()):
        if k.startswith('pcdet'):
            del sys.modules[k]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp_mf_root', required=True)
    parser.add_argument('--cp_mf_cfg', required=True)
    parser.add_argument('--cp_sf_cfg', required=True)
    parser.add_argument('--log_file', type=str, required=True)
    parser.add_argument('--save_dir_dataset', required=True)
    parser.add_argument('--save_dir_annos', required=True)
    parser.add_argument('--az_width', type=int, required=True, help='A-HFR wedge azimuth width in degrees')
    parser.add_argument('--seed', required=True, default=42, type=int)
    return parser.parse_args()


def extract_one_runs(d):
    runs = []
    current_run = []

    for idx in sorted(d.keys()):
        if d[idx] == 1:
            current_run.append(idx)
        else:
            if current_run:
                runs.append(current_run)
                current_run = []

    if current_run:
        runs.append(current_run)

    return runs


def track_sequence(dataset, start=32, end=63, desired_range=60.0,
                   target_class=1, max_step=6.0, negative=True):
    """
    Track one GT object over a contiguous global-index window.
    Returns per-frame target boxes used to aim the A-HFR sector.
    """
    T = end - start + 1
    tracked_boxes = [None] * T
    tracked_idxs = [None] * T
    centers = np.full((T, 3), np.nan, dtype=float)
    match_dists = np.full(T, np.nan)
    ranges = np.full(T, np.nan)

    init_data = dataset[start]
    init_boxes = init_data['gt_boxes']

    init_idx, cur_box = removal.select_initial_target(
        init_boxes,
        desired_range=desired_range,
        target_class=target_class,
        negative=negative,
    )
    if cur_box is None:
        return tracked_boxes, centers, {'ok': False, 'reason': 'no_init_target'}

    tracked_boxes[0] = cur_box
    tracked_idxs[0] = init_idx
    centers[0] = cur_box[:3]
    ranges[0] = removal._box_range(cur_box)

    prev_box = cur_box
    prev_prev_box = None

    for t in range(1, T):
        frame_data = dataset[start + t]
        curr_boxes = frame_data['gt_boxes']

        new_box, new_idx, d = removal.match_with_velocity(
            prev_box,
            prev_prev_box,
            curr_boxes,
            target_class=target_class,
            max_step=max_step,
        )

        if new_box is None:
            break

        tracked_boxes[t] = new_box
        tracked_idxs[t] = new_idx
        centers[t] = new_box[:3]
        match_dists[t] = d
        ranges[t] = removal._box_range(new_box)

        prev_prev_box = prev_box
        prev_box = new_box

    valid = ~np.isnan(centers[:, 0])
    valid_centers = centers[valid]
    num_tracked = int(np.sum(valid))
    if len(valid_centers) > 1:
        deltas = np.linalg.norm(np.diff(valid_centers, axis=0), axis=1)
    else:
        deltas = np.array([])

    valid_boxes = [b for b in tracked_boxes if b is not None]
    speeds_mps = np.array([np.linalg.norm([b[7], b[8]]) for b in valid_boxes])

    diagnostics = {
        'ok': True,
        'num_tracked': num_tracked,
        'total_frames': T,
        'init_range_m': float(ranges[0]),
        'range_trend': ranges[valid],
        'match_dists': match_dists,
        'step_deltas_m': deltas,
        'approx_speeds_mps': speeds_mps,
        'max_step_m': float(np.nanmax(deltas)) if len(deltas) > 0 else 0.0,
        'max_speed_mps': float(np.nanmax(speeds_mps)) if num_tracked > 1 else 0.0,
        'mean_match_dist_m': float(np.nanmean(match_dists)) if num_tracked > 1 else 0.0,
    }

    return tracked_boxes, centers, diagnostics


def build_target_schedule(segment_to_global):
    target_frames_global = {}
    target_frames_seg = {}

    for seg, global_indices in segment_to_global.items():
        toggle = False
        targets_global = {}
        targets_seg = {}

        for local_idx, global_idx in enumerate(global_indices):
            if local_idx == 0:
                targets_seg[local_idx] = None
                targets_global[global_idx] = None
            else:
                if local_idx % 32 == 0:
                    toggle = not toggle
                if toggle:
                    targets_seg[local_idx] = 1
                    targets_global[global_idx] = 1
                else:
                    targets_seg[local_idx] = None
                    targets_global[global_idx] = None

        target_frames_global[seg] = targets_global
        target_frames_seg[seg] = targets_seg

    return target_frames_global, target_frames_seg


def setup_segment_removal(segment, dataset_mf, dataset_sf, target_frames_global,
                          global_to_segment, mf_to_sf_global, az_width,
                          p_remove, seed):
    """
    For each contiguous removal run:
      1. Track one real GT object over the run.
      2. Apply A-HFR point removal to each successfully tracked frame.
      3. Cache the attacked single-frame point cloud and per-frame metadata.
    """
    target_frames = target_frames_global[segment].copy()
    removal_cache = {}
    removal_annos = {}
    run_annos = []

    runs_global = extract_one_runs(target_frames)
    for run in runs_global:
        window_min = min(run)
        window_max = max(run)
        run_seed = int(seed + window_min)

        tracked_boxes, centers, diag = track_sequence(
            dataset_mf,
            start=window_min,
            end=window_max,
            negative=True
        )

        run_annos.append({
            'window': [global_to_segment[window_min][1], global_to_segment[window_max][1]],
            'global_window': [int(window_min), int(window_max)],
            'perturbation': 'ahfr_point_removal',
            'seed': run_seed,
            'az_width_deg': int(az_width),
            'p_remove': float(p_remove),
            'tracking_diagnostics': diag
        })

        state = None
        for offset, global_idx in enumerate(sorted(run)):
            local_idx = global_to_segment[global_idx][1]
            box = tracked_boxes[offset] if offset < len(tracked_boxes) else None

            if box is None:
                target_frames[global_idx] = None
                removal_annos[local_idx] = None
                continue

            pts = dataset_sf[mf_to_sf_global[global_idx]]['points'].copy()
            points = np.hstack([
                pts,
                np.zeros((pts.shape[0], 1), dtype=pts.dtype),
            ])

            frame_rng = np.random.default_rng(int(run_seed + global_idx))
            attacked_points, state, meta = removal.spoof_frame_ahfr_roadside(
                points=points,
                target_box=box,
                state=state,
                p_remove=p_remove,
                rng=frame_rng,
                az_width_deg=az_width,
            )

            attacked_points = attacked_points.astype(np.float32, copy=False)
            removal_cache[global_idx] = attacked_points
            target_frames[global_idx] = box.copy()
            removal_annos[local_idx] = meta

    return target_frames, removal_cache, removal_annos, run_annos


def save_segment(save_dir_dataset, save_dir_annos, segment, segment_dataset,
                 removal_annos, run_annos):
    segment_annos = {
        'segment': segment,
        'spoof_annos': removal_annos,
        'run_annos': run_annos,
    }

    dataset_path = f'{save_dir_dataset}/{segment}.pkl'
    anno_path = f'{save_dir_annos}/{segment}_a.pkl'

    with open(dataset_path, 'wb') as f:
        pkl.dump(segment_dataset, f)
    with open(anno_path, 'wb') as f:
        pkl.dump(segment_annos, f)


def main():
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    prr = {
        10: 0.97,
        20: 1.00,
        30: 0.99,
        40: 0.98,
        50: 0.90,
        60: 0.90,
    }
    if args.az_width not in prr:
        raise ValueError(f'Unsupported az_width={args.az_width}. Expected one of {sorted(prr)}')
    p_remove = prr[args.az_width]

    log = logging.getLogger('spoof')
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)

    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)

    unload_pcdet()
    sys.path.insert(0, args.cp_mf_root)

    from pcdet.config import cfg as cfg_cp
    from pcdet.config import cfg_from_yaml_file as cfg_from_yaml_file_cp
    from pcdet.datasets import build_dataloader as build_cp_loader
    from pcdet.utils import common_utils

    sys.path.pop(0)

    cfg_cp.clear()
    cfg_from_yaml_file_cp(args.cp_mf_cfg, cfg_cp)
    logger = common_utils.create_logger()
    logger.info(f'Loaded cfg from {args.cp_mf_cfg}')
    dataset_mf, test_loader_mf, _ = build_cp_loader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=8,
        logger=logger,
        training=False,
    )

    cfg_cp.clear()
    cfg_from_yaml_file_cp(args.cp_sf_cfg, cfg_cp)
    logger = common_utils.create_logger()
    logger.info(f'Loaded cfg from {args.cp_sf_cfg}')
    dataset_sf, _, _ = build_cp_loader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=8,
        logger=logger,
        training=False,
    )

    frame_ids = [info['frame_id'] for info in dataset_sf.infos]
    unique_segments_sf = sorted(list(set(fid.rsplit('_', 1)[0] for fid in frame_ids)))
    mf_segments = set(info['point_cloud']['lidar_sequence'] for info in dataset_mf.infos)
    common_segments = mf_segments.intersection(unique_segments_sf)

    segment_to_global = defaultdict(list)
    global_to_segment = {}

    for global_idx, info in enumerate(dataset_mf.infos):
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments:
            local_idx = len(segment_to_global[segment])
            segment_to_global[segment].append(global_idx)
            global_to_segment[global_idx] = (segment, local_idx)

    target_frames_global, target_frames_seg = build_target_schedule(segment_to_global)

    sf_segment_to_global = defaultdict(list)
    for global_idx, info in enumerate(dataset_sf.infos):
        seg = info['frame_id'].rsplit('_', 1)[0]
        if seg in common_segments:
            sf_segment_to_global[seg].append(global_idx)

    mf_to_sf_global = {}
    for mf_global, (seg, local_idx) in global_to_segment.items():
        if seg in sf_segment_to_global and local_idx < len(sf_segment_to_global[seg]):
            mf_to_sf_global[mf_global] = sf_segment_to_global[seg][local_idx]

    for seg in common_segments:
        if len(segment_to_global[seg]) != len(sf_segment_to_global[seg]):
            raise RuntimeError(
                f'MF/SF length mismatch for {seg}: '
                f'MF={len(segment_to_global[seg])}, SF={len(sf_segment_to_global[seg])}'
            )

    if len(mf_to_sf_global) != len(global_to_segment):
        raise RuntimeError(
            f'Incomplete MF→SF mapping: mapped {len(mf_to_sf_global)} '
            f'of {len(global_to_segment)} MF frames'
        )

    save_dir_dataset = args.save_dir_dataset
    save_dir_annos = args.save_dir_annos
    os.makedirs(save_dir_dataset, exist_ok=True)
    os.makedirs(save_dir_annos, exist_ok=True)

    completed = {
        f[:-4] for f in os.listdir(save_dir_dataset)
        if f.endswith('.pkl')
    }

    segment_dataset = []
    target_frames = None
    removal_cache = None
    removal_annos = None
    run_annos = None
    current_segment = None

    try:
        for frame, batch in enumerate(test_loader_mf):
            segment = batch['frame_id'][0].rsplit('_', 1)[0]
            if segment not in common_segments:
                continue
            if segment in completed:
                continue

            if current_segment is None:
                current_segment = segment
                target_frames, removal_cache, removal_annos, run_annos = setup_segment_removal(
                    current_segment,
                    dataset_mf,
                    dataset_sf,
                    target_frames_global,
                    global_to_segment,
                    mf_to_sf_global,
                    args.az_width,
                    p_remove,
                    args.seed
                )

            if segment != current_segment:
                save_segment(
                    save_dir_dataset,
                    save_dir_annos,
                    current_segment,
                    segment_dataset,
                    removal_annos,
                    run_annos,
                )
                log.info('Saved: %s', current_segment)

                segment_dataset = []
                current_segment = segment
                target_frames, removal_cache, removal_annos, run_annos = setup_segment_removal(
                    current_segment,
                    dataset_mf,
                    dataset_sf,
                    target_frames_global,
                    global_to_segment,
                    mf_to_sf_global,
                    args.az_width,
                    p_remove,
                    args.seed
                )

            if frame in removal_cache:
                single_frame_points = removal_cache[frame]
            else:
                pts = dataset_sf[mf_to_sf_global[frame]]['points']
                single_frame_points = np.hstack([
                    pts,
                    np.zeros((pts.shape[0], 1), dtype=pts.dtype),
                ])

            single_frame_points = single_frame_points.astype(np.float32, copy=False)

            dict_mf_mod = dataset_mf[frame].copy()
            for k in ['voxels', 'voxel_coords', 'voxel_num_points']:
                dict_mf_mod.pop(k, None)

            dict_mf_mod['points'] = single_frame_points
            dict_mf_mod['pose'] = dict_mf_mod['poses'][:4]
            
            local_idx = global_to_segment[frame][1]
            removal_meta = removal_annos.get(local_idx)

            attack_active = (
                target_frames[frame] is not None
                and removal_meta is not None
                and bool(removal_meta.get('active_this_frame', False))
            )
            dict_mf_mod['spoof_gt'] = target_frames[frame].copy() if attack_active else None
            dict_mf_mod['removal_meta'] = removal_annos.get(global_to_segment[frame][1])

            segment_dataset.append(dict_mf_mod)

    except Exception:
        log.error('===== EXCEPTION =====')
        log.error(traceback.format_exc())
        raise

    if segment_dataset and current_segment is not None:
        save_segment(
            save_dir_dataset,
            save_dir_annos,
            current_segment,
            segment_dataset,
            removal_annos,
            run_annos,
        )
        log.info('Saved: %s', current_segment)


if __name__ == '__main__':
    main()
