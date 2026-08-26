#!/usr/bin/env python
import os
import sys
import argparse
import importlib
import pickle as pkl
import logging
from pathlib import Path
import traceback
from tqdm import tqdm
import numpy as np
import random

from dataclasses import dataclass

import helpers_ptt
importlib.reload(helpers_ptt)
import torch.nn.functional as F

from collections import defaultdict

elevations = np.load('waymo_top_lidar_inclinations.npy')
stats = np.load("range_conditioned_stats.npy", allow_pickle=True).item()



def unload_pcdet():
    for k in list(sys.modules.keys()):
        if k.startswith("pcdet"):
            del sys.modules[k]


def import_pcdet(repo_path, alias):
    """
    Import a specific OpenPCDet repo under a unique namespace.
    """
    sys.path.insert(0, repo_path)
    module = importlib.import_module("pcdet")
    sys.modules[alias] = module
    sys.path.pop(0)
    return module

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cp_mf_root", required=True)  #centerpoint multiframe root directory

    parser.add_argument("--cp_mf_cfg", required=True)   #centerpoint multiframe config file

    parser.add_argument("--cp_sf_cfg", required=True)   #centerpoint singleframe config file
    parser.add_argument(
        "--log_file",
        type=str,
        required=True,
        help="Path to runtime log file"
    )
    parser.add_argument("--save_dir_dataset", required=True)
    parser.add_argument("--save_dir_annos", required=True)
    parser.add_argument("--trace_file", required=True)
    parser.add_argument("--seed", required=True, default=42, type=int)


    return parser.parse_args()


def center_trace(selected_trace):
    pts = selected_trace['points'][:, :2]
    box = selected_trace['box']
    
    centered_points_trace = pts - box[:2]

    # original yaw
    yaw = box[6]

    # rotate points so object becomes ego-aligned
    angle = -yaw
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([[c, -s],
                  [s,  c]])

    rotated_trace = centered_points_trace @ R.T

    box_new = box.copy()
    box_new[6] = 0.0
    box_new[0] = 0.0
    box_new[1] = 0.0

    return rotated_trace, box_new

def place_trace(traces, perturbation, window_seg_min, window_seg_max, seed, anchor_pose=None):
    """
    For global_position_fixed:
      anchor_pose is the (4,4) ego pose of the last frame in the spoof run.
      The trace is placed in ego-relative coords of that anchor frame,
      then converted to world/global coordinates. Every historical frame
      will then transform from global back into its own ego frame inside
      process_4frames.
    """
    rng = np.random.default_rng(seed)
    trace = rng.choice(traces)
    relative_position_offset = np.array([rng.uniform(15, 20), rng.uniform(-3, 3)])

    centered_trace, centered_box = center_trace(trace)

    # random small yaw
    yaw_new = rng.uniform(-0.2, 0.2)
    c = np.cos(yaw_new)
    s = np.sin(yaw_new)
    R = np.array([[c, -s],
                  [s,  c]])
    centered_trace = centered_trace @ R.T
    centered_box[6] = float(yaw_new)
  

    # place in ego-relative coords of anchor frame
    front_near_trace_planar = centered_trace + relative_position_offset
    front_near_trace = np.concatenate(
        [front_near_trace_planar, trace['points'][:, 2].reshape(-1, 1)], axis=1
    )

    front_near_box = centered_box.copy()
    front_near_box[0] += relative_position_offset[0]
    front_near_box[1] += relative_position_offset[1]

    # Convert to global coordinates using anchor pose
    assert anchor_pose is not None, "anchor_pose required for global_position_fixed"

    # rotate points by anchor rotation, then translate
    trace_global_xyz = (front_near_trace @ anchor_pose[:3, :3].T) + anchor_pose[:3, 3]
    spoof_points = np.hstack((trace_global_xyz, trace['points'][:, 3:5]))

    # box centre: same rotation + translation
    box_xyz_local = np.array([front_near_box[0], front_near_box[1], front_near_box[2]])
    box_xyz_global = (box_xyz_local @ anchor_pose[:3, :3].T) + anchor_pose[:3, 3]
    front_near_box[0] = box_xyz_global[0]
    front_near_box[1] = box_xyz_global[1]
    front_near_box[2] = box_xyz_global[2]
    front_near_box[6] = front_near_box[6] + np.arctan2(anchor_pose[1, 0], anchor_pose[0, 0])

    annotation = {
        'window': [window_seg_min, window_seg_max],
        'perturbation': perturbation,
        'seed': seed
    }

    return spoof_points, annotation, front_near_box

def transform_spoof_to_ego_frame(spoof_points_global, gt_box_global, target_pose):
    """
    Transform spoof points and GT box from global coords to a specific frame's ego coords.
    
    Args:
        spoof_points_global: (N, 5) points in global coords [x, y, z, intensity, elongation]
        gt_box_global: (10,) GT box center in global coords
        target_pose: (4, 4) ego pose of the target frame (transforms ego→global)
    
    Returns:
        spoof_points_ego: (N, 5) points in ego coords
        gt_box_ego: (10,) GT box in ego coords
    """
    # Transform spoof points from global--> ego
    spoof_homo = np.hstack([spoof_points_global[:, :3], np.ones((spoof_points_global.shape[0], 1))])
    spoof_ego_homo = spoof_homo @ np.linalg.inv(target_pose).T
    spoof_points_ego = np.hstack([spoof_ego_homo[:, :3], spoof_points_global[:, 3:5]])
    
    # Transform GT box center from global --> ego
    gt_box_ego = gt_box_global.copy()
    box_xyz_global = np.array([gt_box_global[0], gt_box_global[1], gt_box_global[2], 1.0])
    box_xyz_ego = box_xyz_global @ np.linalg.inv(target_pose).T
    gt_box_ego[0] = box_xyz_ego[0]
    gt_box_ego[1] = box_xyz_ego[1]
    gt_box_ego[2] = box_xyz_ego[2]
    
    # Transform heading: subtract target frame's yaw
    gt_box_ego[6] = (gt_box_global[6] 
                     - np.arctan2(target_pose[1, 0], target_pose[0, 0]))
    
    return spoof_points_ego, gt_box_ego


def ground_anchor_spoof(points, spoof, gt, radius=2.0):
    """
    points : scene lidar points (NxD)
    spf_cur : spoof points (Mx5 or Mx3 depending on pipeline)

    returns: ground anchored spoof points
    """
    gt_box = gt.copy()        
    spf_cur = spoof.copy()        
    
    scene_xyz = points[:, :3]

    spoof_xy = spf_cur[:, :2]
    center_xy = spoof_xy.mean(axis=0)

    dists = np.linalg.norm(scene_xyz[:, :2] - center_xy, axis=1)
    local_mask = dists < radius

    if np.sum(local_mask) > 20:
        local_points = scene_xyz[local_mask]

        ground_z = np.percentile(local_points[:, 2], 5)
        spoof_bottom_z = np.min(spf_cur[:, 2])

        dz = ground_z - spoof_bottom_z
        
        # print(f"dz: {dz}")
        spf_cur[:, 2] += dz
        gt_box[2] += dz
        # print(f"gt_box z: {gt_box[2]}")

    return spf_cur, gt_box


def spoof_frame(spoof_points, points, gt_box, rng):
    global elevations, stats
    if spoof_points is None:
        return points, gt_box, 0, 0
    
    #Ground anchoring correction
    n_spoof_r = spoof_points.shape[0]
    spf_cur, gt_box = ground_anchor_spoof(points, spoof_points, gt_box) 
        
    #empirically sample intensities and elongations 
    r_avg = helpers_ptt.avg_spoof_range(spf_cur)
    band_stats = helpers_ptt.get_stats_for_range(r_avg, stats)
    P_two = 0.01
    if (band_stats is not None):
        # print("updating intensity and elongation with empirical values")
        I = rng.lognormal(
                band_stats["log_mu"],
                band_stats["log_sigma"],
                size=spf_cur.shape[0]
            )
        I = np.clip(I, band_stats["intensity_p1"], band_stats["intensity_p99"])
        E = rng.choice(band_stats["elong_pool"], size=spf_cur.shape[0])
        spf_cur[:, 3] = I
        spf_cur[:, 4] = E
        
    dtheta = np.deg2rad(0.1358)

    #convert to spherical
    r_trace, theta_trace, phi_trace = helpers_ptt.cart_2_spherical(spf_cur)
    scene_xyz = points[:, :3]
    r_scene, theta_scene, phi_scene = helpers_ptt.cart_2_spherical(scene_xyz)

    #Discretize scene points by (azimuth, beam)
    az_scene = np.mod(theta_scene, 2*np.pi)
    az_idx_scene = np.floor(az_scene / dtheta).astype(np.int32)
    beam_idx_scene = np.abs(elevations[:, None] - phi_scene).argmin(axis=0)

    # Sort scene by (az_idx, beam_idx, range) 
    order = np.lexsort((r_scene, beam_idx_scene, az_idx_scene))
    az_idx_scene = az_idx_scene[order]
    beam_idx_scene = beam_idx_scene[order]
    r_scene = r_scene[order]
    scene_indices_sorted = order

    # Identify group boundaries
    keys = az_idx_scene * 1000 + beam_idx_scene  # unique key
    unique_keys, group_start = np.unique(keys, return_index=True)
    group_end = np.r_[group_start[1:], len(keys)]

    # Build fast lookup map
    key_to_slice = {
        k: (start, end)
        for k, start, end in zip(unique_keys, group_start, group_end)
    }

    remove_spoof = []
    remove_scene = []

    # process spoof points
    for idx, (r, theta, phi) in enumerate(zip(r_trace, theta_trace, phi_trace)):
        az = theta % (2*np.pi)
        az_idx = int(np.floor(az / dtheta))
        beam_idx = np.argmin(np.abs(elevations - phi))
        key = az_idx * 1000 + beam_idx

        if key not in key_to_slice:
            continue

        start, end = key_to_slice[key]
        r_group = r_scene[start:end]
        idx_group = scene_indices_sorted[start:end]

        if len(r_group) == 0:
            continue

        # Case 1: spoof behind first real return
        if r_group[0] < r:
            remove_spoof.append(idx)
        
        # Case 2: spoof in front of all
        elif r < r_group[0]:

            # Spoof becomes first return.
            # Decide if second return survives.
            if rng.random() < P_two:
                # Allow second return: keep nearest scene return as second
                if len(idx_group) > 0:
                    remove_scene.extend(idx_group[1:].tolist())
            else:
                # Only spoof survives
                remove_scene.extend(idx_group.tolist())

        # Case 3: spoof between returns
        else:
            k = np.searchsorted(r_group, r, side='right')

            if k < len(r_group):
                if rng.random() < P_two:
                    # Keep only one scene return behind spoof
                    remove_scene.extend(idx_group[k+1:].tolist())
                else:
                    # Spoof fully occludes behind
                    remove_scene.extend(idx_group[k:].tolist())
    # print("unique remove_spoof:", len(set(remove_spoof)))
    # print("unique remove_scene:", len(set(remove_scene)))

    # apply removals
    mask_spoof = np.ones(spf_cur.shape[0], dtype=bool)
    mask_spoof[remove_spoof] = False
    spoof_points_rc = spf_cur[mask_spoof]
    
    removed_unique = len(set(remove_scene))
    kept_spoof = spoof_points_rc.shape[0]
    # print("expected points diff:", kept_spoof - removed_unique)

    mask_scene = np.ones(points.shape[0], dtype=bool)
    mask_scene[remove_scene] = False
    scene_points_rc = points[mask_scene]

    # preserve lag column
    lag_value = points[0, -1]
    lag_col = np.full((spoof_points_rc.shape[0], 1), lag_value, dtype=np.float32)    
    spf_pts_rc_lag = np.hstack([spoof_points_rc, lag_col])
    # helpers_ptt.plot_trace(spf_pts_rc_lag)

    frame_points_spoof_rc = np.concatenate([scene_points_rc, spf_pts_rc_lag], axis=0)

    return frame_points_spoof_rc, gt_box, n_spoof_r, kept_spoof


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

    # handle tail case
    if current_run:
        runs.append(current_run)

    return runs



def setup_segment_spoofs(segment, target_frames_global, global_to_segment, 
                         seg_localidx_to_pose, traces, seed):
    
    target_frames = target_frames_global[segment].copy()
    target_frames_anno = target_frames_global[segment].copy()
    segment_spoof_annos = []
    
    runs_global = extract_one_runs(target_frames)
    for run in runs_global:
        subset_g = {k: target_frames[k] for k in run}
        subset_g_anno = {k: target_frames_anno[k] for k in run}
        window_min = global_to_segment[min(subset_g)][1]
        window_max = global_to_segment[max(subset_g)][1]
        
        anchor_global_idx = max(subset_g.keys())
        anchor_local_idx = global_to_segment[anchor_global_idx][1]
        anchor_pose = seg_localidx_to_pose[(segment, anchor_local_idx)]
        
        run_seed = int(seed + min(subset_g))
        spoof_points_global, spoof_annotation, box_global = place_trace(
            traces, 'global_position_fixed',
            window_min, window_max, run_seed,anchor_pose=anchor_pose
        )
        
        for key in subset_g.keys():
            target_segment, target_local_idx = global_to_segment[key]
            target_pose = seg_localidx_to_pose[(target_segment, target_local_idx)]
            
            spoof_points_ego, box_ego = transform_spoof_to_ego_frame(
                spoof_points_global, box_global, target_pose
            )
            
            subset_g[key] = spoof_points_ego      # New array ref (safe)
            subset_g_anno[key] = box_ego.copy()   # CRITICAL: .copy() here
        
        segment_spoof_annos.append(spoof_annotation)
        target_frames.update(subset_g)
        target_frames_anno.update(subset_g_anno)
    
    return target_frames, target_frames_anno, segment_spoof_annos

def save_segment(save_dir_dataset, save_dir_annos, segment, segment_dataset,
                 segment_spoof_annos):
    segment_annos = {
        'segment': segment,
        'spoof_annos': segment_spoof_annos
    }

    dataset_path = f'{save_dir_dataset}/{segment}.pkl'
    anno_path = f'{save_dir_annos}/{segment}_a.pkl'

    with open(dataset_path, 'wb') as f:
        pkl.dump(segment_dataset, f)
    with open(anno_path, 'wb') as f:
        pkl.dump(segment_annos, f)
    return



def main():

    args = parse_args()
    log = logging.getLogger("spoof")
    log.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Console handler 
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)

    # File handler 
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    
    
    #load in centerpoint multiframe dataset. 
    unload_pcdet()
    sys.path.insert(0, args.cp_mf_root)

    from pcdet.config import cfg as cfg_cp
    from pcdet.config import cfg_from_yaml_file as cfg_from_yaml_file_cp
    from pcdet.datasets import build_dataloader as build_cp_loader
    from pcdet.utils import common_utils

    sys.path.pop(0)

    # cfg_cp.TAG = 'centerpoint_multiframe'
    # cfg_cp.EXP_GROUP_PATH = 'waymo_multiframe'
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
        logger = logger,
        training=False
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
        logger = logger,
        training=False
    )

    
    frame_ids = [info['frame_id'] for info in dataset_sf.infos]

    # Split the frame_id to get only the segment name (everything before the last underscore)
    unique_segments_sf = sorted(list(set(fid.rsplit('_', 1)[0] for fid in frame_ids)))
    
    mf_segments = set(info['point_cloud']['lidar_sequence']
                  for info in dataset_mf.infos)
    common_segments = mf_segments.intersection(unique_segments_sf)

    # 
    segment_to_global = defaultdict(list)

    global_to_segment = {}

    seg_localidx_to_pose = {}

    for global_idx, info in enumerate(dataset_mf.infos):
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments:
            local_idx = len(segment_to_global[segment])  # current position in segment
            segment_to_global[segment].append(global_idx)
            global_to_segment[global_idx] = (segment, local_idx)
            seg_localidx_to_pose[(segment, local_idx)] = info['pose'].reshape((4, 4))


    segment_counts = defaultdict(int)

    for info in dataset_mf.infos:
        segment = info['point_cloud']['lidar_sequence']
        if segment in common_segments: segment_counts[segment] += 1

    segment_last_local_idx = {seg: count - 1 for seg, count in segment_counts.items()}


    target_frames_global = {}
    target_frames_seg = {}

    for seg, last_ind in segment_last_local_idx.items():
        toggle = False
        targets_global = {}
        targets_seg = {}
        for i in range(last_ind+1):
            if i==0:
                targets_seg[i] = None
                targets_global[segment_to_global[seg][i]] = None
            else:
                rem = i%32
                # print(rem)
                if(not rem):
                    toggle = not toggle
                if toggle:
                    targets_seg[i] = 1
                    targets_global[segment_to_global[seg][i]] = 1
                else:
                    targets_seg[i] = None
                    targets_global[segment_to_global[seg][i]] = None
        target_frames_global[seg] = targets_global
        target_frames_seg[seg] = targets_seg


    sf_segment_to_global = defaultdict(list)

    for global_idx, info in enumerate(dataset_sf.infos):
        seg = info['frame_id'].rsplit('_', 1)[0]
        if seg in common_segments:
            sf_segment_to_global[seg].append(global_idx)
    mf_to_sf_global = {}

    for mf_global, (seg, local_idx) in global_to_segment.items():
        if seg in sf_segment_to_global and local_idx < len(sf_segment_to_global[seg]):
            sf_global = sf_segment_to_global[seg][local_idx]
            mf_to_sf_global[mf_global] = sf_global
    for seg in common_segments:
        if len(segment_to_global[seg]) != len(sf_segment_to_global[seg]):
            raise RuntimeError(
                f"MF/SF length mismatch for {seg}: "
                f"MF={len(segment_to_global[seg])}, SF={len(sf_segment_to_global[seg])}"
            )

        if len(mf_to_sf_global) != len(global_to_segment):
            raise RuntimeError(
                f"Incomplete MF→SF mapping: mapped {len(mf_to_sf_global)} "
                f"of {len(global_to_segment)} MF frames"
            )
            
            
    segment_dataset = []
    segment_spoof_annos = []
    save_dir_dataset = args.save_dir_dataset
    os.makedirs(save_dir_dataset, exist_ok=True)
    save_dir_annos = args.save_dir_annos
    os.makedirs(save_dir_annos, exist_ok=True)
    
    completed = set(
        f[:-4] for f in os.listdir(save_dir_dataset)
    )
    
    
    traces_file = args.trace_file
    with open(traces_file, 'rb') as file:
        traces = pkl.load(file)

    target_frames = None
    current_segment = None
    target_frames_anno = None
    # prev_seq_name = None
    
    
    try: 
        for (frame,batch)  in enumerate(test_loader_mf):
            segment = batch['frame_id'][0].rsplit("_", 1)[0]
            if segment not in common_segments:
                continue
            
            if segment in completed:
                continue
                
            if current_segment is None:
                current_segment = segment
                target_frames, target_frames_anno, segment_spoof_annos = setup_segment_spoofs(current_segment, target_frames_global, global_to_segment, 
                         seg_localidx_to_pose, traces, args.seed)
            #dump data for past segment
            if(segment != current_segment):
                # print(segment)
                save_segment(
                    save_dir_dataset,
                    save_dir_annos,
                    current_segment,
                    segment_dataset,
                    segment_spoof_annos
                )
                
                log.info("Saved: %s", current_segment)

                segment_dataset = [] 
                segment_spoof_annos = []
                current_segment = segment
                
                target_frames = None
                target_frames_anno = None
           
                target_frames, target_frames_anno, segment_spoof_annos = setup_segment_spoofs(current_segment, target_frames_global, global_to_segment, 
                         seg_localidx_to_pose, traces, args.seed)
            #get spoofs for current segment
            
            #build datase
            # frame_seq = global_to_segment[frame][1]
            # perturbation = 'global_position_fixed'
            pts = dataset_sf[mf_to_sf_global[frame]]['points']
            points = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)])  # Add lag=0 column
            frame_rng = np.random.default_rng(int(args.seed + frame))
            frame_points_spoof_rc, gt_box, n_spoof_r, n_spoof_k = spoof_frame(target_frames[frame], points, target_frames_anno[frame], frame_rng)
            frame_points_spoof_rc = frame_points_spoof_rc.astype(np.float32, copy=False)
            dict_mf_mod = dataset_mf[frame].copy()
            for k in ['voxels', 'voxel_coords', 'voxel_num_points']:
                dict_mf_mod.pop(k, None)
            dict_mf_mod['points'] = frame_points_spoof_rc
            dict_mf_mod['pose'] = dict_mf_mod['poses'][:4]
            dict_mf_mod['spoof_gt'] = gt_box
            dict_mf_mod['n_spoof_r'] = n_spoof_r
            dict_mf_mod['n_spoof_k'] = n_spoof_k
            
            # print(frame_data)

            
            
            segment_dataset.append(dict_mf_mod)
            
    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise  

    if segment_dataset and current_segment is not None:
        save_segment(
                    save_dir_dataset,
                    save_dir_annos,
                    current_segment,
                    segment_dataset,
                    segment_spoof_annos
        )
        log.info('Saved: %s', current_segment)

    return

    
    
if __name__ == "__main__":
    main()
