#!/usr/bin/env python
"""
Shared nuScenes machinery for the ATLAS adversarial-split generators.

This is the nuScenes analogue of the Waymo-specific glue that lives inside
generate_data_global.py / generate_data_relative.py / generate_data_removal.py.
The physics (ground anchoring, range-conditioned intensity, occlusion-aware
injection) is preserved; the pieces that were Waymo-only are replaced:

  * beam geometry            : 32-beam HDL-32E inclinations + 0.33 deg azimuth
                               step (built by build_nuscenes_assets.py) instead
                               of the 64-beam Waymo inclinations / 0.1358 deg.
  * dual returns             : nuScenes is single-return, so P_two = 0.
  * point feature col 4      : timestamp (NOT elongation) -> never overwritten.
  * segment / pose           : scene token instead of `lidar_sequence`;
                               ref(lidar)->global built from `ref_from_car` and
                               `car_from_global` instead of `info['pose']`.
  * MF / SF dual loaders     : a single NuScenesDataset; sweeps are already
                               accumulated by get_lidar_with_sweeps.

Points convention used throughout this module:
  scene points are stored as (N, 6) = [x, y, z, intensity, time, lag]
  where `lag` is a trailing bookkeeping column (kept 0.0) so that
  point_injection.py's `points[:, :-1]` recovers the 5-feature nuScenes cloud.
"""
import os
import pickle as pkl
from collections import defaultdict

import numpy as np

# --------------------------------------------------------------------------
# Assets (built by build_nuscenes_assets.py, loaded once at import time)
# Override the asset dir with NUSCENES_ATLAS_ASSET_DIR to point at a different
# set (e.g. assets_trainval/ vs the default mini assets in this dir).
# --------------------------------------------------------------------------
_ASSET_DIR = os.environ.get("NUSCENES_ATLAS_ASSET_DIR",
                            os.path.dirname(os.path.abspath(__file__)))

ELEVATIONS = np.load(os.path.join(_ASSET_DIR, "nuscenes_beam_inclinations.npy"))
DTHETA = float(np.load(os.path.join(_ASSET_DIR, "nuscenes_azimuth_res.npy"))[0])
STATS = np.load(os.path.join(_ASSET_DIR, "nuscenes_range_conditioned_stats.npy"),
                allow_pickle=True).item()

# nuScenes LiDAR is single-return; disable the Waymo second-return branch.
P_TWO = 0.0


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def cart_2_spherical(points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x) % (2 * np.pi)
    r_safe = np.where(r == 0, 1e-9, r)
    phi = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))
    return r, theta, phi


def avg_spoof_range(points):
    return float(np.mean(np.sqrt(np.sum(points[:, :3] ** 2, axis=1))))


def get_stats_for_range(r, stats=STATS):
    for (rmin, rmax), values in stats.items():
        if rmin <= r < rmax:
            return values
    return None


# --------------------------------------------------------------------------
# Trace placement
# --------------------------------------------------------------------------
def center_trace(selected_trace):
    """Move a trace to the origin, ego-aligned (yaw 0)."""
    pts = selected_trace["points"][:, :2]
    box = selected_trace["box"]

    centered = pts - box[:2]
    yaw = box[6]
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    rotated = centered @ R.T

    box_new = box.copy()
    box_new[0] = box_new[1] = 0.0
    box_new[6] = 0.0
    return rotated, box_new


def place_trace_relative(traces, window_min, window_max, seed):
    """Ego-relative phantom: fixed offset in front of the ego, reused per frame."""
    rng = np.random.default_rng(seed)
    trace_idx = int(rng.integers(len(traces)))
    trace = traces[trace_idx]

    offset = np.array([rng.uniform(15, 20), rng.uniform(-3, 3)], dtype=np.float32)
    centered_trace, centered_box = center_trace(trace)

    yaw_new = float(rng.uniform(-0.2, 0.2))
    c, s = np.cos(yaw_new), np.sin(yaw_new)
    centered_trace = centered_trace @ np.array([[c, -s], [s, c]]).T
    centered_box[6] = yaw_new

    planar = centered_trace + offset
    xyz = np.concatenate([planar, trace["points"][:, 2].reshape(-1, 1)], axis=1)
    spoof_points = np.hstack((xyz, trace["points"][:, 3:5]))

    box = centered_box.copy()
    box[0] += offset[0]
    box[1] += offset[1]

    anno = {
        "window": [window_min, window_max],
        "perturbation": "relative_position_fixed",
        "seed": int(seed),
        "trace_idx": trace_idx,
        "relative_position_offset": offset.tolist(),
        "yaw": yaw_new,
    }
    return spoof_points, anno, box


def place_trace_global(traces, window_min, window_max, seed, anchor_pose):
    """
    Global-fixed phantom: placed in the anchor frame's ego-relative coords,
    then lifted to global coordinates so every frame in the run sees it at the
    same world location. `anchor_pose` is ref(lidar)->global (4x4).
    """
    rng = np.random.default_rng(seed)
    trace_idx = int(rng.integers(len(traces)))
    trace = traces[trace_idx]

    offset = np.array([rng.uniform(15, 20), rng.uniform(-3, 3)], dtype=np.float32)
    centered_trace, centered_box = center_trace(trace)

    yaw_new = float(rng.uniform(-0.2, 0.2))
    c, s = np.cos(yaw_new), np.sin(yaw_new)
    centered_trace = centered_trace @ np.array([[c, -s], [s, c]]).T
    centered_box[6] = yaw_new

    planar = centered_trace + offset
    front_near_trace = np.concatenate(
        [planar, trace["points"][:, 2].reshape(-1, 1)], axis=1
    )
    box = centered_box.copy()
    box[0] += offset[0]
    box[1] += offset[1]

    # ego(ref) -> global
    trace_global = (front_near_trace @ anchor_pose[:3, :3].T) + anchor_pose[:3, 3]
    spoof_points = np.hstack((trace_global, trace["points"][:, 3:5]))

    box_xyz = np.array([box[0], box[1], box[2]])
    box_xyz_g = (box_xyz @ anchor_pose[:3, :3].T) + anchor_pose[:3, 3]
    box[0], box[1], box[2] = box_xyz_g
    box[6] = box[6] + np.arctan2(anchor_pose[1, 0], anchor_pose[0, 0])

    anno = {
        "window": [window_min, window_max],
        "perturbation": "global_position_fixed",
        "seed": int(seed),
        "trace_idx": trace_idx,
        "relative_position_offset": offset.tolist(),
        "yaw": yaw_new,
    }
    return spoof_points, anno, box


def transform_spoof_to_ego_frame(spoof_points_global, gt_box_global, target_pose):
    """Global -> a target frame's ref(lidar) coords. target_pose is ref->global."""
    inv = np.linalg.inv(target_pose)
    homo = np.hstack([spoof_points_global[:, :3],
                      np.ones((spoof_points_global.shape[0], 1))])
    ego = homo @ inv.T
    spoof_ego = np.hstack([ego[:, :3], spoof_points_global[:, 3:5]])

    box = gt_box_global.copy()
    box_xyz = np.array([box[0], box[1], box[2], 1.0]) @ inv.T
    box[0], box[1], box[2] = box_xyz[:3]
    box[6] = gt_box_global[6] - np.arctan2(target_pose[1, 0], target_pose[0, 0])
    return spoof_ego, box


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
def ground_anchor_spoof(points, spoof, gt, radius=2.0):
    gt_box = gt.copy()
    spf = spoof.copy()
    scene_xyz = points[:, :3]

    center_xy = spf[:, :2].mean(axis=0)
    dists = np.linalg.norm(scene_xyz[:, :2] - center_xy, axis=1)
    local = dists < radius
    if np.sum(local) > 20:
        ground_z = np.percentile(scene_xyz[local, 2], 5)
        dz = ground_z - np.min(spf[:, 2])
        spf[:, 2] += dz
        gt_box[2] += dz
    return spf, gt_box


def spoof_frame(spoof_points, points, gt_box, rng):
    """
    Inject an occlusion-consistent phantom into a nuScenes scene cloud.

    points : (N, 6) scene = [x, y, z, intensity, time, lag]
    spoof_points : (M, 5) phantom in ref(lidar) coords = [x, y, z, intensity, time]
    Returns (fused_points (N', 6), gt_box, n_spoof_raw, n_spoof_kept).
    """
    if spoof_points is None:
        return points, gt_box, 0, 0

    n_spoof_r = spoof_points.shape[0]
    spf, gt_box = ground_anchor_spoof(points, spoof_points, gt_box)

    # range-conditioned intensity (col 3). col 4 = time is left untouched
    # because nuScenes has no elongation channel.
    r_avg = avg_spoof_range(spf)
    band = get_stats_for_range(r_avg)
    if band is not None:
        I = rng.lognormal(band["log_mu"], band["log_sigma"], size=spf.shape[0])
        I = np.clip(I, band["intensity_p1"], band["intensity_p99"])
        spf[:, 3] = I

    r_trace, theta_trace, phi_trace = cart_2_spherical(spf)
    r_scene, theta_scene, phi_scene = cart_2_spherical(points[:, :3])

    az_idx_scene = np.floor(np.mod(theta_scene, 2 * np.pi) / DTHETA).astype(np.int32)
    beam_idx_scene = np.abs(ELEVATIONS[:, None] - phi_scene).argmin(axis=0)

    order = np.lexsort((r_scene, beam_idx_scene, az_idx_scene))
    az_idx_scene = az_idx_scene[order]
    beam_idx_scene = beam_idx_scene[order]
    r_scene = r_scene[order]

    keys = az_idx_scene * 1000 + beam_idx_scene
    unique_keys, group_start = np.unique(keys, return_index=True)
    group_end = np.r_[group_start[1:], len(keys)]
    key_to_slice = {k: (s, e) for k, s, e in zip(unique_keys, group_start, group_end)}

    remove_spoof, remove_scene = [], []
    for idx, (r, theta, phi) in enumerate(zip(r_trace, theta_trace, phi_trace)):
        az_idx = int(np.floor((theta % (2 * np.pi)) / DTHETA))
        beam_idx = int(np.argmin(np.abs(ELEVATIONS - phi)))
        key = az_idx * 1000 + beam_idx
        if key not in key_to_slice:
            continue
        start, end = key_to_slice[key]
        r_group = r_scene[start:end]
        idx_group = order[start:end]
        if len(r_group) == 0:
            continue

        if r_group[0] < r:
            # phantom is behind the nearest real return -> occluded
            remove_spoof.append(idx)
        elif r < r_group[0]:
            # phantom is the first return; single-return -> occlude everything behind
            if rng.random() < P_TWO:
                remove_scene.extend(idx_group[1:].tolist())
            else:
                remove_scene.extend(idx_group.tolist())
        else:
            k = np.searchsorted(r_group, r, side="right")
            if k < len(r_group):
                if rng.random() < P_TWO:
                    remove_scene.extend(idx_group[k + 1:].tolist())
                else:
                    remove_scene.extend(idx_group[k:].tolist())

    mask_spoof = np.ones(spf.shape[0], dtype=bool)
    mask_spoof[remove_spoof] = False
    spoof_kept = spf[mask_spoof]

    mask_scene = np.ones(points.shape[0], dtype=bool)
    mask_scene[remove_scene] = False
    scene_kept = points[mask_scene]

    # append the trailing lag/bookkeeping column so fused stays (*, 6)
    lag_value = points[0, -1]
    lag_col = np.full((spoof_kept.shape[0], 1), lag_value, dtype=np.float32)
    spoof_kept6 = np.hstack([spoof_kept, lag_col])

    fused = np.concatenate([scene_kept, spoof_kept6], axis=0)
    return fused, gt_box, n_spoof_r, spoof_kept.shape[0]


# --------------------------------------------------------------------------
# Scene grouping + pose (nuScenes)
# --------------------------------------------------------------------------
def build_scene_index(dataset, nusc):
    """
    Group dataset indices by scene and expose the per-frame ref(lidar)->global
    pose. Returns:
        scene_to_idx   : {scene_name: [dataset_idx, ...]}  (time-ordered)
        idx_to_scene   : {dataset_idx: (scene_name, local_idx)}
        idx_to_pose    : {dataset_idx: (4,4) ref->global}
    """
    tok2scene = {}
    for sc in nusc.scene:
        t = sc["first_sample_token"]
        while t:
            tok2scene[t] = sc["name"]
            t = nusc.get("sample", t)["next"]

    scene_to_idx = defaultdict(list)
    idx_to_scene = {}
    idx_to_pose = {}

    for i, info in enumerate(dataset.infos):
        scene = tok2scene[info["token"]]
        local = len(scene_to_idx[scene])
        scene_to_idx[scene].append(i)
        idx_to_scene[i] = (scene, local)

        ref_from_car = np.array(info["ref_from_car"])
        car_from_global = np.array(info["car_from_global"])
        global_to_ref = ref_from_car @ car_from_global
        idx_to_pose[i] = np.linalg.inv(global_to_ref)  # ref -> global

    return dict(scene_to_idx), idx_to_scene, idx_to_pose


def build_target_schedule(scene_to_idx, period=32):
    """
    Toggle target windows every `period` frames within each scene (faithful to
    the Waymo generator). Returns {scene: {dataset_idx: 1 or None}}.
    """
    schedule = {}
    for scene, indices in scene_to_idx.items():
        toggle = False
        marks = {}
        for local, gidx in enumerate(indices):
            if local == 0:
                marks[gidx] = None
                continue
            if local % period == 0:
                toggle = not toggle
            marks[gidx] = 1 if toggle else None
        schedule[scene] = marks
    return schedule


def _scene_seed(seed, scene):
    """Stable per-scene seed.

    Python's builtin hash() is salted per process (PYTHONHASHSEED), so it would
    give a different schedule on every run and silently break the "same L across
    all variants" guarantee. Use a content hash instead.
    """
    import hashlib
    h = hashlib.sha256(f"{int(seed)}|{scene}".encode()).digest()
    return int.from_bytes(h[:8], "little") % (2 ** 32)


def build_single_attack_schedule(scene_to_idx, mode, seed=42, min_len=2,
                                 max_len=32, duty=0.5, clean_prefix=4):
    """ONE attack per scene, of a length that varies across scenes.

    Unlike build_target_schedule (which toggles every `period` frames and, on
    ~40-frame nuScenes scenes, degenerates to a single 8-frame window pinned to
    the tail), this gives every scene exactly one attack whose length L varies
    scene to scene.

    mode='continuous'   -> L contiguous attacked frames.
    mode='intermittent' -> the SAME attack toggling on and off: L attacked
                           frames spread over a longer span with clean frames
                           between them. Attacked-frame count is held equal to
                           the continuous case at the same L, so the two modes
                           differ only in continuity.

    L and the span placement are derived from a per-scene RNG seeded by
    (seed, scene_name), so they are IDENTICAL across every base variant and
    across both modes. That is what makes removal-vs-injection a paired
    per-sequence comparison.

    Duty cannot be fixed: intermittent needs span = L/duty, and a ~40-frame
    scene minus the clean prefix leaves only ~36 usable frames, so L=32 at
    duty=0.5 would need 64. We therefore use the requested duty where it fits
    and degrade towards 1.0 as L approaches the available span. Scenes where L
    fills the span (no room for a gap) are flagged `degenerate` so the analysis
    can exclude them rather than silently counting them as intermittent.

    Returns (schedule, meta):
      schedule : {scene: {dataset_idx: 1 | None}}  -- same contract as
                 build_target_schedule, so existing consumers keep working.
      meta     : {scene: {...}} recording L, span, start, realised duty, the
                 gap pattern and the degenerate flag.
    """
    if mode not in ("continuous", "intermittent"):
        raise ValueError(f"mode must be continuous|intermittent, got {mode!r}")

    schedule, meta = {}, {}
    for scene, indices in scene_to_idx.items():
        n = len(indices)
        rng = np.random.default_rng(_scene_seed(seed, scene))

        # frames [clean_prefix, n) are available to the attack
        avail = max(1, n - clean_prefix)
        L = int(rng.integers(min_len, max_len + 1))
        L = min(L, avail)

        if mode == "continuous":
            span_len = L
        else:
            # widen the span to interleave clean frames, capped by what fits
            span_len = min(avail, int(np.ceil(L / max(duty, 1e-9))))
        span_len = max(span_len, L)

        start_lo = clean_prefix
        start_hi = n - span_len            # inclusive
        start = (int(rng.integers(start_lo, start_hi + 1))
                 if start_hi > start_lo else start_lo)

        span_local = list(range(start, start + span_len))
        if mode == "continuous":
            attacked_local = list(span_local)
            degenerate = False
        else:
            degenerate = (span_len <= L)   # no room for any gap
            if degenerate:
                attacked_local = list(span_local)
            else:
                # keep the span's endpoints attacked so the span is well defined,
                # then spread the remaining attacked frames evenly inside it.
                interior = span_local[1:-1]
                k = L - 2
                if k <= 0:
                    attacked_local = [span_local[0], span_local[-1]][:L]
                else:
                    pick = np.linspace(0, len(interior) - 1, k)
                    pick = sorted({int(round(p)) for p in pick})
                    # linspace rounding can collide; backfill to reach exactly k
                    if len(pick) < k:
                        for cand in range(len(interior)):
                            if cand not in pick:
                                pick.append(cand)
                                if len(pick) == k:
                                    break
                        pick = sorted(pick)
                    attacked_local = sorted(
                        [span_local[0], span_local[-1]]
                        + [interior[i] for i in pick[:k]])

        attacked_set = set(attacked_local)
        marks = {}
        for local, gidx in enumerate(indices):
            marks[gidx] = 1 if local in attacked_set else None
        schedule[scene] = marks

        gaps = [l for l in span_local if l not in attacked_set]
        meta[scene] = {
            "mode": mode,
            "n_frames": n,
            "attack_len_L": len(attacked_local),
            "requested_L": L,
            "span_start_local": start,
            "span_len": span_len,
            "attacked_local": attacked_local,
            "gap_local": gaps,
            "n_gaps": len(gaps),
            "realised_duty": len(attacked_local) / span_len,
            "degenerate": bool(degenerate),
            "clean_prefix": clean_prefix,
            "sched_seed": int(seed),
        }
    return schedule, meta


def extract_one_runs(marks):
    """Contiguous runs of target==1, keyed by (ordered) dataset index."""
    runs, current = [], []
    for gidx in sorted(marks.keys()):
        if marks[gidx] == 1:
            current.append(gidx)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def extract_attack_spans(marks):
    """Attack SPANS, as [(span_indices, attacked_subset), ...].

    A span runs contiguously from the first to the last attacked frame of one
    attack, and therefore INCLUDES any interior gap frames. This is the unit an
    intermittent attack must be built on: `extract_one_runs` would split such an
    attack into several disjoint runs, and every consumer that keys off a run
    (trace placement, target selection, aim-point state) would then restart --
    producing a different phantom, or a different victim car, after each gap
    instead of one attack flickering.

    Gap frames are included in `span_indices` so tracking/state can be advanced
    across them, and excluded from `attacked_subset` so no perturbation is
    applied there.

    With a legacy (contiguous) schedule this is equivalent to extract_one_runs,
    since a contiguous run has no interior gaps. Runs separated by a gap are
    treated as ONE span, which is exactly the intended semantics here: every
    scene carries a single attack.
    """
    attacked = sorted(g for g, v in marks.items() if v == 1)
    if not attacked:
        return []
    ordered = sorted(marks.keys())
    lo, hi = ordered.index(attacked[0]), ordered.index(attacked[-1])
    span = ordered[lo:hi + 1]
    return [(span, list(attacked))]


# --------------------------------------------------------------------------
# Loading points from the dataset in the (N, 6) scene convention
# --------------------------------------------------------------------------
def load_scene_points(dataset, index):
    """
    Return the accumulated nuScenes cloud as (N, 6) = [x,y,z,intensity,time,lag].
    Voxel keys are dropped; a zero `lag` column is appended for storage compat.
    """
    data_dict = dataset[index]
    pts = data_dict["points"]  # (N, 5): x,y,z,intensity,time
    lag = np.zeros((pts.shape[0], 1), dtype=pts.dtype)
    points6 = np.hstack([pts, lag])
    for k in ("voxels", "voxel_coords", "voxel_num_points"):
        data_dict.pop(k, None)
    return points6, data_dict


def save_segment(save_dir_dataset, save_dir_annos, segment,
                 segment_dataset, spoof_annos, extra_annos=None,
                 schedule_meta=None):
    annos = {"segment": segment, "spoof_annos": spoof_annos}
    if extra_annos is not None:
        annos["run_annos"] = extra_annos
    if schedule_meta is not None:
        annos["schedule_meta"] = schedule_meta
    with open(f"{save_dir_dataset}/{segment}_d.pkl", "wb") as f:
        pkl.dump(segment_dataset, f)
    with open(f"{save_dir_annos}/{segment}_a.pkl", "wb") as f:
        pkl.dump(annos, f)


# --------------------------------------------------------------------------
# Dataset construction + shared injection driver (global / relative)
# --------------------------------------------------------------------------
def build_nuscenes(cfg_file, logger=None):
    """Build a single (non-multiframe) NuScenesDataset from a dataset cfg yaml."""
    import yaml
    from easydict import EasyDict
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.utils import common_utils
    from nuscenes.nuscenes import NuScenes

    dataset_cfg = EasyDict(yaml.safe_load(open(cfg_file)))
    root = _resolve_dataroot(dataset_cfg)
    if logger is None:
        logger = common_utils.create_logger()

    dataset = NuScenesDataset(
        dataset_cfg=dataset_cfg,
        class_names=dataset_cfg.get("CLASS_NAMES", None),
        root_path=root,
        logger=logger,
        training=False,
    )
    nusc = NuScenes(version=dataset_cfg.VERSION, dataroot=str(root / dataset_cfg.VERSION),
                    verbose=False)
    return dataset, nusc, dataset_cfg


def _resolve_dataroot(dataset_cfg):
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent
    return ROOT_DIR / "data" / "nuscenes"


def setup_scene_spoofs(scene, schedule, idx_to_scene, idx_to_pose, traces, seed,
                       mode, use_spans=False):
    """
    Precompute per-frame phantom points + boxes for a scene. `mode` selects the
    placement policy: 'global' (world-fixed) or 'relative' (ego-fixed).
    Returns {dataset_idx: spoof_points|None}, {dataset_idx: box|None}, [annos].

    `use_spans=True` switches the unit of work from contiguous runs to attack
    SPANS (see extract_attack_spans). This matters for intermittent schedules:
    with runs, each contiguous chunk would draw its own trace_idx / offset / yaw
    and a DIFFERENT phantom car would appear after every gap. With spans, one
    placement is derived per attack and merely withheld on gap frames, so the
    same phantom flickers in and out.
    """
    marks = schedule[scene]
    frame_pts = dict(marks)
    frame_box = dict(marks)
    scene_annos = []

    if use_spans:
        units = [(span, attacked) for span, attacked in extract_attack_spans(marks)]
    else:
        units = [(run, run) for run in extract_one_runs(marks)]

    for span, attacked in units:
        window_min = idx_to_scene[min(span)][1]
        window_max = idx_to_scene[max(span)][1]
        # seed off the span start so continuous and intermittent at the same L
        # select the SAME phantom
        run_seed = int(seed + min(span))

        if mode == "global":
            anchor_pose = idx_to_pose[max(span)]
            spoof_global, anno, box_global = place_trace_global(
                traces, window_min, window_max, run_seed, anchor_pose)
            for gidx in attacked:
                pts_ego, box_ego = transform_spoof_to_ego_frame(
                    spoof_global, box_global, idx_to_pose[gidx])
                frame_pts[gidx] = pts_ego
                frame_box[gidx] = box_ego.copy()
        else:  # relative
            spoof_pts, anno, box = place_trace_relative(
                traces, window_min, window_max, run_seed)
            for gidx in attacked:
                frame_pts[gidx] = spoof_pts
                frame_box[gidx] = box.copy()

        anno["span_window"] = [window_min, window_max]
        anno["n_span_frames"] = len(span)
        anno["n_attacked_frames"] = len(attacked)
        anno["gap_frames"] = [idx_to_scene[g][1]
                              for g in span if g not in set(attacked)]
        scene_annos.append(anno)

    return frame_pts, frame_box, scene_annos


def resolve_schedule(scene_to_idx, args, log=None):
    """Pick the frame schedule from --schedule and friends.

    Returns (schedule, sched_meta, use_spans). `legacy` reproduces the original
    alternating-window behaviour exactly, so existing splits stay bit-identical.
    """
    kind = getattr(args, "schedule", "legacy")
    if kind == "legacy":
        if log:
            log.info("schedule: legacy alternating windows (period=32)")
        return build_target_schedule(scene_to_idx), None, False

    schedule, meta = build_single_attack_schedule(
        scene_to_idx, mode=kind,
        seed=getattr(args, "sched_seed", 42),
        min_len=getattr(args, "min_len", 2),
        max_len=getattr(args, "max_len", 32),
        duty=getattr(args, "duty", 0.5),
        clean_prefix=getattr(args, "clean_prefix", 4))
    if log:
        Ls = [m["attack_len_L"] for m in meta.values()]
        ndeg = sum(m["degenerate"] for m in meta.values())
        log.info("schedule: %s | one attack/scene | L med %.1f (min %d max %d) "
                 "| degenerate %d/%d", kind, float(np.median(Ls)),
                 min(Ls), max(Ls), ndeg, len(meta))
    return schedule, meta, True


def add_schedule_args(parser):
    """Shared CLI flags for the temporal attack schedule."""
    parser.add_argument("--schedule", default="legacy",
                        choices=["legacy", "continuous", "intermittent"],
                        help="legacy = original alternating 32-frame windows; "
                             "continuous/intermittent = ONE attack per scene "
                             "with a scene-varying length")
    parser.add_argument("--sched_seed", type=int, default=42)
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--duty", type=float, default=0.5,
                        help="target attacked-frame fraction inside the span "
                             "(intermittent only; adapts upward when L is large)")
    parser.add_argument("--clean_prefix", type=int, default=4,
                        help="frames at the start of each scene kept clean")
    return parser


def run_injection(args, mode, log):
    """Shared driver for the global / relative injection generators."""
    dataset, nusc, _ = build_nuscenes(args.cfg_file)
    scene_to_idx, idx_to_scene, idx_to_pose = build_scene_index(dataset, nusc)
    schedule, sched_meta, use_spans = resolve_schedule(scene_to_idx, args, log)

    os.makedirs(args.save_dir_dataset, exist_ok=True)
    os.makedirs(args.save_dir_annos, exist_ok=True)
    completed = {f[:-6] for f in os.listdir(args.save_dir_dataset)
                 if f.endswith("_d.pkl")}

    with open(args.trace_file, "rb") as f:
        traces = pkl.load(f)

    for scene, indices in scene_to_idx.items():
        if scene in completed:
            log.info("skip (done): %s", scene)
            continue

        frame_pts, frame_box, scene_annos = setup_scene_spoofs(
            scene, schedule, idx_to_scene, idx_to_pose, traces, args.seed, mode,
            use_spans=use_spans)

        marks = schedule[scene]
        segment_dataset = []
        for gidx in indices:
            points6, data_dict = load_scene_points(dataset, gidx)
            frame_rng = np.random.default_rng(int(args.seed + gidx))
            fused, gt_box, n_r, n_k = spoof_frame(
                frame_pts[gidx], points6, frame_box[gidx], frame_rng)

            data_dict["points"] = fused.astype(np.float32, copy=False)
            data_dict["pose"] = idx_to_pose[gidx]
            data_dict["spoof_gt"] = gt_box
            data_dict["n_spoof_r"] = n_r
            data_dict["n_spoof_k"] = n_k
            # scheduled != realised: occlusion filtering can drop the phantom
            # entirely (n_spoof_k == 0) on a frame the schedule marked attacked.
            # Record both so a dropout is never mistaken for a scheduled gap.
            data_dict["scheduled_attack"] = bool(marks.get(gidx) == 1)
            segment_dataset.append(data_dict)

        extra = {"run_annos": scene_annos}
        if sched_meta is not None:
            extra["schedule_meta"] = sched_meta[scene]
        save_segment(args.save_dir_dataset, args.save_dir_annos, scene,
                     segment_dataset, scene_annos,
                     extra_annos=extra.get("run_annos"),
                     schedule_meta=extra.get("schedule_meta"))
        log.info("Saved: %s (%d frames, %d scheduled attacks)", scene,
                 len(segment_dataset),
                 sum(1 for g in indices if marks.get(g) == 1))
