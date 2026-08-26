#!/usr/bin/env python
"""A-HFR point-removal (occlusion) adversarial split for nuScenes.

nuScenes analogue of generate_data_removal.py. Tracks one real object across a
contiguous window and removes LiDAR points in an azimuth wedge aimed at it,
simulating an Adaptive High-Frequency Removal attack. The geometry lives in the
dataset-agnostic removal.py; only the tracking/target-selection wiring changes
for nuScenes (class label comes from gt_names, not a trailing box column).

Example:
  python generate_data_removal_nuscenes.py \
      --cfg_file ../tools/cfgs/dataset_configs/nuscenes_dataset_mini.yaml \
      --save_dir_dataset <out>/removal --save_dir_annos <out>/removal_annos \
      --az_width 20 --log_file removal.log --seed 42
"""
import argparse
import importlib
import logging
import sys
import traceback

import numpy as np

import removal
import nuscenes_atlas_common as C
importlib.reload(removal)

# nuScenes detection classes -> integer label used by removal.py's gates.
# We target 'car' (label 1) to mirror the Waymo vehicle attack.
TARGET_CLASS_NAME = "car"
TARGET_CLASS_INT = 1

# per-wedge removal probability (from the A-HFR paper), keyed by az width
PRR = {10: 0.97, 20: 1.00, 30: 0.99, 40: 0.98, 50: 0.90, 60: 0.90}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--save_dir_dataset", required=True)
    p.add_argument("--save_dir_annos", required=True)
    p.add_argument("--az_width", type=int, required=True, choices=sorted(PRR))
    p.add_argument("--log_file", required=True)
    p.add_argument("--seed", type=int, default=42)
    C.add_schedule_args(p)
    return p.parse_args()


def boxes_with_class(data_dict):
    """
    Return the prepared gt_boxes as (M, 10). After NuScenesDataset.prepare_data
    the final column is already the 1-indexed class label (car=1 ... = the
    TARGET_CLASS_INT removal.py reads via class_col=-1), and gt_names has been
    consumed. So we can hand the boxes straight through.
    """
    boxes = data_dict.get("gt_boxes")
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 10), dtype=np.float64)
    return boxes.astype(np.float64)


# nuScenes annotated keyframes are 2 Hz, NOT Waymo's 10 Hz. Two consequences for
# tracking, both of which silently truncate a track if ignored:
#   * the constant-velocity prediction must extrapolate over dt=0.5 s. With the
#     Waymo default (0.1) the prediction is 5x too short, so a car at 15 m/s has
#     a 6 m residual and falls outside any sane spatial gate.
#   * the per-frame displacement itself is 5x larger. Measured on nuScenes val,
#     tracked cars step a median 2.2 m and up to 6.3 m between keyframes, so the
#     Waymo gate of 6.0 m clips the tail.
NUSC_KEYFRAME_DT = 0.5
NUSC_MAX_STEP_M = 12.0


def track_sequence(dataset, indices, desired_range=40.0,
                   max_step=NUSC_MAX_STEP_M, dt=NUSC_KEYFRAME_DT):
    """Track one car across the window (list of dataset indices)."""
    T = len(indices)
    tracked = [None] * T

    init = boxes_with_class(dataset[indices[0]])
    init_idx, cur = removal.select_initial_target(
        init, desired_range=desired_range, target_class=TARGET_CLASS_INT,
        min_range=10.0, negative=True)
    if cur is None:
        return tracked, {"ok": False, "reason": "no_init_target",
                         "num_tracked": 0, "total_frames": T}

    tracked[0] = cur
    prev, prev_prev = cur, None
    ranges = [removal._box_range(cur)]
    lost_at = None

    for t in range(1, T):
        curr = boxes_with_class(dataset[indices[t]])
        new_box, _, d = removal.match_with_velocity(
            prev, prev_prev, curr, target_class=TARGET_CLASS_INT,
            max_step=max_step, dt=dt)
        if new_box is None:
            lost_at = t
            break
        tracked[t] = new_box
        ranges.append(removal._box_range(new_box))
        prev_prev, prev = prev, new_box

    diag = {"ok": True, "num_tracked": int(sum(b is not None for b in tracked)),
            "total_frames": T, "init_range_m": float(ranges[0]),
            "lost_at_offset": lost_at, "max_step_m": float(max_step),
            "dt_s": float(dt)}
    return tracked, diag


def main():
    args = parse_args()
    p_remove = PRR[args.az_width]

    log = logging.getLogger("spoof")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(args.log_file); fh.setFormatter(fmt); log.addHandler(fh)

    try:
        dataset, nusc, _ = C.build_nuscenes(args.cfg_file)
        scene_to_idx, idx_to_scene, idx_to_pose = C.build_scene_index(dataset, nusc)
        schedule, sched_meta, use_spans = C.resolve_schedule(
            scene_to_idx, args, log)

        import os
        os.makedirs(args.save_dir_dataset, exist_ok=True)
        os.makedirs(args.save_dir_annos, exist_ok=True)
        completed = {f[:-6] for f in os.listdir(args.save_dir_dataset)
                     if f.endswith("_d.pkl")}

        for scene, indices in scene_to_idx.items():
            if scene in completed:
                log.info("skip (done): %s", scene)
                continue

            marks = schedule[scene]
            # cache attacked points + target boxes per span
            attacked_pts = {}
            target_box = {}
            removal_annos = {}
            run_annos = []

            if use_spans:
                units = C.extract_attack_spans(marks)
            else:
                units = [(run, run) for run in C.extract_one_runs(marks)]

            for span, attacked in units:
                # Seed off the span start so continuous and intermittent at the
                # same L select the SAME victim vehicle.
                run_seed = int(args.seed + min(span))
                attacked_set = set(attacked)
                # Track over the FULL span, including gap frames. Tracking only
                # the attacked frames would step >max_step across a gap and the
                # tracker would drop the target (or re-init onto a different car
                # on the next run), which would make an intermittent attack a
                # sequence of unrelated attacks instead of one flickering attack.
                tracked, diag = track_sequence(dataset, span)
                run_annos.append({
                    "window": [idx_to_scene[min(span)][1], idx_to_scene[max(span)][1]],
                    "perturbation": "ahfr_point_removal",
                    "seed": run_seed, "az_width_deg": int(args.az_width),
                    "p_remove": float(p_remove), "tracking_diagnostics": diag,
                    "n_span_frames": len(span),
                    "n_scheduled_attacks": len(attacked),
                    "gap_frames": [idx_to_scene[g][1]
                                   for g in span if g not in attacked_set],
                })
                state = None
                for offset, gidx in enumerate(span):
                    box = tracked[offset] if offset < len(tracked) else None
                    if box is None:
                        continue
                    points6, _ = C.load_scene_points(dataset, gidx)
                    frame_rng = np.random.default_rng(int(run_seed + gidx))
                    atk, state, meta = removal.spoof_frame_ahfr_roadside(
                        points=points6, target_box=box, state=state,
                        p_remove=p_remove, az_width_deg=args.az_width, rng=frame_rng)
                    # `state` is advanced on EVERY span frame so the spoofer's
                    # aim point slews continuously through the gaps, but the
                    # perturbed cloud is only kept where the schedule fires.
                    meta["scheduled_attack"] = gidx in attacked_set
                    if gidx in attacked_set:
                        attacked_pts[gidx] = atk.astype(np.float32, copy=False)
                        target_box[gidx] = box.copy()
                    removal_annos[idx_to_scene[gidx][1]] = meta

            # build the per-frame dataset (attacked where scheduled, clean elsewhere)
            segment_dataset = []
            for gidx in indices:
                if gidx in attacked_pts:
                    points6 = attacked_pts[gidx]
                    gt_spoof = target_box[gidx]
                else:
                    points6, _ = C.load_scene_points(dataset, gidx)
                    gt_spoof = None
                _, data_dict = C.load_scene_points(dataset, gidx)
                data_dict["points"] = points6
                data_dict["pose"] = idx_to_pose[gidx]
                data_dict["spoof_gt"] = gt_spoof  # the box we tried to erase
                data_dict["scheduled_attack"] = bool(marks.get(gidx) == 1)
                segment_dataset.append(data_dict)

            C.save_segment(args.save_dir_dataset, args.save_dir_annos, scene,
                           segment_dataset, removal_annos, extra_annos=run_annos,
                           schedule_meta=(sched_meta[scene] if sched_meta else None))
            log.info("Saved: %s (%d frames, %d scheduled attacks, %d applied)",
                     scene, len(segment_dataset),
                     sum(1 for g in indices if marks.get(g) == 1),
                     len(attacked_pts))

    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
