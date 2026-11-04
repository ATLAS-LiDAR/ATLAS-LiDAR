# =============================================================================
# generate_data_ablation.py  --  temporal / spoof-source ablations for ATLAS.
# Patched copy of generate_data_global_rebuttal.py (the generator behind
# global_easy/hard_seed42). Everything above the ABLATION EXTENSIONS block is
# that script unchanged except: (1) --data_path override (shared/waymo* gone),
# (2) the manifest records the ablation config. New switches:
#   --positioning   global | relative   object fixed in the world (anchor pose,
#                                       as global_*) or fixed in ego coords
#   --spoof_source  atlas | gt_paste    ATLAS trace + physical ray filter, or a
#                                       val-object-bank instance pasted the way
#                                       OpenPCDet GT sampling does (all points
#                                       kept, scene points inside the box removed)
#   --object_mode   fixed | per_frame   same object for the whole run, or a new
#                                       object every frame ("random objects")
#   --position_mode fixed | per_frame   same placement, or a new placement each
#                                       frame ("random position")
#   --random p                          (existing) flicker: each frame fires w.p. p
# Placement draws (offset 15-20 m ahead, +-3 m lateral, yaw +-0.2) come from the
# SAME per-run seed stream as the original, so every variant shares the
# global_* placements. With defaults (global/atlas/fixed/fixed) it must
# reproduce global_easy/hard exactly (see abl_repro_test.sh).
# =============================================================================
#!/usr/bin/env python
# CVPR copy (cvpr_scripts/): original + I = np.tanh(I) after intensity sampling.
"""
ATLAS global-injection generator with structured rebuttal audit logging.

Reviewer coverage:
  MuS2 Q1 / Weakness 1:
    Ground placement, scan-lattice residuals, ray collisions, raycasting
    outcomes, background occlusion ordering, intensity/elongation summaries.
  MuS2 Q2:
    A manifest records every sensor/attack parameter that must be recalibrated
    when transferring the generator.
  MuS2 Q3 / Weakness 3:
    Stable frame/run IDs and spoof GT fields support joining target-level
    detector metrics and selecting qualitative examples.
  MuS2 Q5:
    The manifest records code/input hashes, runtime, and generated size.
  Uh4V Q3 / Weakness 3:
    The same audits establish simulator-internal consistency; they deliberately
    do not claim hardware validation.
  Uh4V Q5 / Dataset Assessment:
    Per-variant manifests and per-segment tables identify complete artifacts.
  fftM Q2:
    Frame/window identifiers expose the exact fixed schedule and window phase.
  fftM Q4:
    Candidate, retained, discarded, and scene-occlusion counts report the
    effective attack budget after raycasting and visibility filtering.

The runtime text log remains for humans. CSV sidecars are the canonical
notebook-facing audit output.
"""
import os
import sys
import argparse
import importlib
import pickle as pkl
import logging
import csv
import json
import hashlib
import time
import socket
import platform
import subprocess
from pathlib import Path
import traceback
from tqdm import tqdm
import numpy as np
import random

# import PTT.neurips.helpers_ptt as helpers_ptt
import helpers_ptt
importlib.reload(helpers_ptt)
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import defaultdict

elevations = np.load('waymo_top_lidar_inclinations.npy')
stats = np.load("range_conditioned_stats.npy", allow_pickle=True).item()

# CVPR intensity modes (env ATLAS_INTENSITY_MODE). The spoof_frame() draws are made in EVERY mode,
# so the random stream, raycasting and geometry are identical across modes; only what is written
# into the spoof's intensity/elongation columns differs.
#   adaptive   : per frame, values drawn from the band of the spoof's current range (tanh-fixed)
#   trace      : the trace's own recorded intensity/elongation, unchanged in every frame
#   calibrated : drawn ONCE per attack window from the band of the spoof's range in the anchor
#                frame (tanh-fixed, separate seeded stream), then fixed for all frames
INTENSITY_MODE = os.environ.get("ATLAS_INTENSITY_MODE", "adaptive")
if INTENSITY_MODE not in ("adaptive", "trace", "calibrated", "saturated", "attenuated"):
    raise ValueError(f"ATLAS_INTENSITY_MODE={INTENSITY_MODE!r}")


def _calibrate_spoof_features(pts, pts_for_range, *seed_parts):
    """calibrated mode: overwrite pts[:, 3:5] in place with one draw for the whole window."""
    band = helpers_ptt.get_stats_for_range(helpers_ptt.avg_spoof_range(pts_for_range[:, :3]), stats)
    if band is None:      # anchor-frame spoof beyond the stats' bands: keep the trace values
        return None
    payload = ":".join(["calibrated_intensity", *map(str, seed_parts)]).encode("utf-8")
    crng = np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:8], "little"))
    I = crng.lognormal(band["log_mu"], band["log_sigma"], size=pts.shape[0])
    I = np.clip(I, band["intensity_p1"], band["intensity_p99"])
    pts[:, 3] = np.tanh(I)
    pts[:, 4] = crng.choice(band["elong_pool"], size=pts.shape[0])
    return band


# Hardware-motivated constant intensities (2026-10-30; Sato et al. NDSS'24, Sako et al. NDSS'24
# poster). Elongation keeps the trace values. Set once per attack in the setup, like calibrated.
#   saturated  : every spoof point = ATLAS_INTENSITY_VALUE (default 1.0 = top of the loader's tanh
#                scale; prior chosen-pattern injection saturates VLP-16 intensity at 250-255)
#   attenuated : every spoof point = quantile ATLAS_INTENSITY_Q (default 0.9) of the real-vehicle
#                intensities in the loaded trace library (Sako's ICPI reaches 60-75 vs cars' 0-100),
#                or ATLAS_INTENSITY_VALUE if given
CONST_I = None


# CVPR spoof position error (Sato et al., NDSS'24, Sec. IV-B3): errors lie ALONG each laser ray
# (spoofer timing -> range). r_i += d_inner_i + d_inter, d_inner_i ~ N(0, ATLAS_JITTER_INNER) per point,
# d_inter ~ N(0, ATLAS_JITTER_INTER) one per frame (the whole pattern moves along the line of sight).
# Applied after ground anchoring and before raycasting (occlusion uses the jittered ranges). Drawn
# from a separate hash-seeded stream keyed by (seed, segment, frame), so every other draw is unchanged
# and jittered sets stay frame-paired with the unjittered ones. spoof_gt = realized box (centre moved
# by d_inter along its line of sight); the aimed box is kept as spoof_gt_intended.
JITTER_INNER = float(os.environ.get("ATLAS_JITTER_INNER", "0"))
JITTER_INTER = float(os.environ.get("ATLAS_JITTER_INTER", "0"))
JITTER_KEY = None          # set by the caller before each spoof_frame()
LAST_JITTER = (None, 0.0)  # (intended box, d_inter) of the last spoofed frame


def _jitter_spoof(spf, gt_box):
    global LAST_JITTER
    if JITTER_INNER <= 0 and JITTER_INTER <= 0:
        LAST_JITTER = (None, 0.0)
        return spf, gt_box
    h = hashlib.sha256(f"atlas_jitter:{JITTER_KEY}".encode("utf-8")).digest()
    jr = np.random.default_rng(int.from_bytes(h[:8], "little"))
    d_inter = float(jr.normal(0.0, JITTER_INTER)) if JITTER_INTER > 0 else 0.0
    d = jr.normal(0.0, JITTER_INNER, spf.shape[0]) if JITTER_INNER > 0 else np.zeros(spf.shape[0])
    xyz = spf[:, :3].astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    out = spf.copy()
    out[:, :3] = (xyz * ((r + d + d_inter) / np.maximum(r, 1e-6))[:, None]).astype(spf.dtype)
    box = np.asarray(gt_box, dtype=np.float64).copy()
    c = np.linalg.norm(box[:3])
    box[:3] = box[:3] * (c + d_inter) / max(c, 1e-6)
    LAST_JITTER = (np.asarray(gt_box).copy(), d_inter)
    return out, box.astype(np.asarray(gt_box).dtype)


def set_constant_level(traces):
    """Fix CONST_I for the constant modes from the env and the loaded trace library."""
    global CONST_I
    if INTENSITY_MODE not in ("saturated", "attenuated"):
        return None
    v = os.environ.get("ATLAS_INTENSITY_VALUE")
    if v is not None:
        CONST_I = float(v)
    elif INTENSITY_MODE == "saturated":
        CONST_I = 1.0
    else:
        q = float(os.environ.get("ATLAS_INTENSITY_Q", "0.9"))
        CONST_I = float(np.quantile(np.concatenate([np.asarray(t["points"])[:, 3] for t in traces]), q))
    return CONST_I


def _constant_spoof_features(pts):
    pts[:, 3] = CONST_I



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
    parser.add_argument(
        "--metrics_dir",
        required=True,
        help="Directory for notebook-friendly run/frame/point audit tables"
    )
    parser.add_argument(
        "--variant_name",
        required=True,
        help="Stable dataset variant label, e.g. global_1000"
    )
    parser.add_argument("--schema_version", default="atlas_global_rebuttal_v2")
    parser.add_argument(
        "--save_point_audit",
        action="store_true",
        help="Write one audit row per candidate spoof point (large)"
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=None,
        help=(
            "Number of common Waymo driving sequences to generate. "
            "Default: all available sequences."
        ),
    )
    parser.add_argument(
        "--sequence_selection",
        choices=("random", "first"),
        default="random",
        help=(
            "How to choose --num_sequences from sorted common sequences. "
            "'random' is reproducible using --seed."
        ),
    )
    parser.add_argument(
        "--max_returns_per_ray",
        type=int,
        default=2,
        help="Maximum permitted final returns per affected discretized ray.",
    )
    parser.add_argument(
        "--no_save_audit_arrays",
        action="store_true",
        help=(
            "Disable compressed per-segment candidate/removed-point arrays. "
            "These are saved by default to avoid needing regeneration."
        ),
    )
    parser.add_argument(
        "--radial_bin_width_m",
        type=float,
        default=1.0,
        help="Width in metres of ego-centred radial density annuli.",
    )
    parser.add_argument(
        "--radial_max_m",
        type=float,
        default=None,
        help=(
            "Maximum planar radius for radial tables. Default: farthest "
            "XY corner of POINT_CLOUD_RANGE rounded up to a complete bin."
        ),
    )
    parser.add_argument(
        "--random",
        type=float,
        default=None,
        help=(
            "Enable FLICKER mode. Per-frame probability in (0, 1] that the "
            "spoofer successfully fires within an ATTACK block. The full "
            "32-frame block is intended, but each frame (including the anchor) "
            "independently drops its spoof with probability 1 - random, "
            "simulating an imperfect spoofer. When omitted, the generator "
            "instead uses CONTINUOUS mode: one random contiguous sub-window "
            "per block is spoofed."
        ),
    )
    parser.add_argument(
        "--continuous_min_len",
        type=int,
        default=8,
        help=(
            "CONTINUOUS mode only: minimum length (inclusive) of the randomly "
            "placed contiguous spoof sub-window inside each 32-frame block."
        ),
    )
    parser.add_argument(
        "--continuous_max_len",
        type=int,
        default=16,
        help=(
            "CONTINUOUS mode only: maximum length (inclusive) of the randomly "
            "placed contiguous spoof sub-window inside each 32-frame block."
        ),
    )


    return parser.parse_args()


def _python_scalar(value):
    """Convert NumPy scalars and non-finite floats to JSON/CSV-safe values."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summary(prefix, values):
    """Flat numeric summary suitable for a CSV row."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_p05": None,
            f"{prefix}_p50": None,
            f"{prefix}_p95": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p50": float(np.percentile(values, 50)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def _atomic_write_csv(path, rows):
    """Write a complete per-segment table atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: _python_scalar(v) for k, v in row.items()} for row in rows]
    fieldnames = sorted({key for row in rows for key in row})
    tmp_path = path.with_suffix(".tmp.csv")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_python_scalar)
    os.replace(tmp_path, path)


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _optional_sha256(path):
    path = Path(path)
    return _sha256(path) if path.exists() and path.is_file() else None


def _git_commit(repo_path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _circular_difference(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _point_cloud_range_mask(xyz, point_cloud_range):
    xyz = np.asarray(xyz)
    pcr = np.asarray(point_cloud_range, dtype=np.float64)
    return (
        (xyz[:, 0] >= pcr[0]) & (xyz[:, 0] <= pcr[3])
        & (xyz[:, 1] >= pcr[1]) & (xyz[:, 1] <= pcr[4])
        & (xyz[:, 2] >= pcr[2]) & (xyz[:, 2] <= pcr[5])
    )


def _normalized_box_coordinates(xyz, box):
    """Return point coordinates in the oriented box frame, normalized by L/W/H."""
    xyz = np.asarray(xyz, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    relative = xyz[:, :3] - box[:3]
    yaw = float(box[6])
    c, s = np.cos(-yaw), np.sin(-yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    local = relative.copy()
    local[:, :2] = relative[:, :2] @ rotation.T
    dimensions = np.maximum(np.abs(box[3:6]), 1e-12)
    normalized = local / dimensions
    inside = np.all(np.abs(normalized) <= 0.5 + 1e-6, axis=1)
    return normalized, inside


def _population_bin_statistics(points, radii, lower, upper):
    """Counts and feature summaries for one half-open radial annulus."""
    mask = (radii >= lower) & (radii < upper)
    selected = np.asarray(points)[mask]
    result = {"count": int(np.sum(mask))}
    if selected.shape[0] == 0:
        result.update({
            "intensity_mean": None,
            "intensity_std": None,
            "intensity_median": None,
            "elongation_mean": None,
            "elongation_std": None,
            "elongation_median": None,
        })
    else:
        result.update({
            "intensity_mean": float(np.mean(selected[:, 3])),
            "intensity_std": float(np.std(selected[:, 3])),
            "intensity_median": float(np.median(selected[:, 3])),
            "elongation_mean": float(np.mean(selected[:, 4])),
            "elongation_std": float(np.std(selected[:, 4])),
            "elongation_median": float(np.median(selected[:, 4])),
        })
    return result


def _radial_density_rows(
    points_before,
    surviving_scene_points,
    removed_scene_points,
    retained_spoof_points,
    points_after,
    gt_box,
    radial_edges,
):
    """
    Create one numerical row per ego-centred planar annulus.

    Density is points per square metre of XY annulus. Counts are also saved,
    allowing alternative normalizations to be calculated without regeneration.
    """
    populations = {
        "before": np.asarray(points_before),
        "surviving_scene_after": np.asarray(surviving_scene_points),
        "removed_background": np.asarray(removed_scene_points),
        "spoof": np.asarray(retained_spoof_points),
        "total_after": np.asarray(points_after),
    }
    radii = {
        name: np.linalg.norm(array[:, :2], axis=1)
        for name, array in populations.items()
    }
    spoof_center_radius = float(np.linalg.norm(np.asarray(gt_box)[:2]))
    spoof_point_radii = radii["spoof"]
    spoof_radius_min = (
        float(np.min(spoof_point_radii)) if spoof_point_radii.size else None
    )
    spoof_radius_median = (
        float(np.median(spoof_point_radii)) if spoof_point_radii.size else None
    )
    spoof_radius_max = (
        float(np.max(spoof_point_radii)) if spoof_point_radii.size else None
    )

    rows = []
    for bin_index, (lower, upper) in enumerate(
        zip(radial_edges[:-1], radial_edges[1:])
    ):
        lower = float(lower)
        upper = float(upper)
        width = upper - lower
        annulus_area = float(np.pi * (upper ** 2 - lower ** 2))
        row = {
            "radial_bin_index": int(bin_index),
            "radius_bin_start_m": lower,
            "radius_bin_end_m": upper,
            "radius_bin_center_m": float(0.5 * (lower + upper)),
            "radial_bin_width_m": float(width),
            "annulus_area_m2": annulus_area,
            "spoof_center_radius_m": spoof_center_radius,
            "spoof_point_radius_min_m": spoof_radius_min,
            "spoof_point_radius_median_m": spoof_radius_median,
            "spoof_point_radius_max_m": spoof_radius_max,
        }
        if upper <= spoof_center_radius:
            row["position_relative_to_spoof_center"] = "in_front"
        elif lower > spoof_center_radius:
            row["position_relative_to_spoof_center"] = "behind"
        else:
            row["position_relative_to_spoof_center"] = "contains_spoof_center"
        if spoof_radius_min is None or spoof_radius_max is None:
            row["position_relative_to_spoof_extent"] = None
        elif upper <= spoof_radius_min:
            row["position_relative_to_spoof_extent"] = "in_front_of_spoof"
        elif lower >= spoof_radius_max:
            row["position_relative_to_spoof_extent"] = "behind_spoof"
        else:
            row["position_relative_to_spoof_extent"] = "overlaps_spoof"

        for population_name, population_points in populations.items():
            stats_row = _population_bin_statistics(
                population_points,
                radii[population_name],
                lower,
                upper,
            )
            for key, value in stats_row.items():
                row[f"{population_name}_{key}"] = value
            row[f"{population_name}_density_points_per_m2"] = float(
                stats_row["count"] / max(annulus_area, 1e-12)
            )
            row[f"{population_name}_linear_density_points_per_m"] = float(
                stats_row["count"] / max(width, 1e-12)
            )

        row["net_point_change"] = int(
            row["total_after_count"] - row["before_count"]
        )
        row["density_change_points_per_m2"] = float(
            row["total_after_density_points_per_m2"]
            - row["before_density_points_per_m2"]
        )
        row["density_change_fraction"] = (
            float(
                row["total_after_density_points_per_m2"]
                / row["before_density_points_per_m2"]
                - 1.0
            )
            if row["before_density_points_per_m2"] > 0
            else None
        )
        row["background_density_reduction_fraction"] = (
            float(
                row["removed_background_density_points_per_m2"]
                / row["before_density_points_per_m2"]
            )
            if row["before_density_points_per_m2"] > 0
            else None
        )
        rows.append(row)
    return rows


def _atomic_write_audit_arrays(path, records):
    """Save ragged per-frame audit arrays in one compressed NPZ per segment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.npz")

    frame_global_indices = np.asarray(
        [r["global_frame_index"] for r in records], dtype=np.int64
    )
    frame_local_indices = np.asarray(
        [r["local_frame_index"] for r in records], dtype=np.int32
    )
    frame_ids = np.asarray([r["frame_id"] for r in records], dtype=np.str_)

    candidate_lengths = np.asarray(
        [len(r["candidate_points"]) for r in records], dtype=np.int64
    )
    candidate_offsets = np.r_[0, np.cumsum(candidate_lengths)]
    removed_lengths = np.asarray(
        [len(r["removed_scene_indices"]) for r in records], dtype=np.int64
    )
    removed_offsets = np.r_[0, np.cumsum(removed_lengths)]

    def concat(name, shape_tail, dtype):
        arrays = [np.asarray(r[name], dtype=dtype) for r in records if len(r[name])]
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.empty((0,) + tuple(shape_tail), dtype=dtype)

    np.savez_compressed(
        tmp_path,
        frame_global_indices=frame_global_indices,
        frame_local_indices=frame_local_indices,
        frame_ids=frame_ids,
        candidate_offsets=candidate_offsets,
        candidate_points=concat("candidate_points", (5,), np.float32),
        candidate_kept=concat("candidate_kept", (), np.bool_),
        candidate_outcome_code=concat("candidate_outcome_code", (), np.int8),
        candidate_beam_index=concat("candidate_beam_index", (), np.int16),
        candidate_azimuth_bin=concat("candidate_azimuth_bin", (), np.int32),
        candidate_ray_key=concat("candidate_ray_key", (), np.int64),
        candidate_azimuth_residual_deg=concat(
            "candidate_azimuth_residual_deg", (), np.float32
        ),
        candidate_elevation_residual_deg=concat(
            "candidate_elevation_residual_deg", (), np.float32
        ),
        candidate_scene_returns_on_ray=concat(
            "candidate_scene_returns_on_ray", (), np.int32
        ),
        candidate_nearest_scene_range=concat(
            "candidate_nearest_scene_range", (), np.float32
        ),
        candidate_scene_points_removed=concat(
            "candidate_scene_points_removed", (), np.int32
        ),
        candidate_minimum_removed_scene_range=concat(
            "candidate_minimum_removed_scene_range", (), np.float32
        ),
        candidate_occlusion_order_violation=concat(
            "candidate_occlusion_order_violation", (), np.bool_
        ),
        removed_offsets=removed_offsets,
        removed_scene_indices=concat("removed_scene_indices", (), np.int64),
        removed_scene_points=concat("removed_scene_points", (6,), np.float32),
    )
    os.replace(tmp_path, path)


def _aggregate_saved_metrics(metrics_dir, selected_segments):
    """Aggregate reviewer-facing integrity totals from saved segment CSVs."""
    sum_fields = (
        "spoof_candidate_count",
        "spoof_kept_count",
        "spoof_discarded_count",
        "background_removed_unique_count",
        "count_balance_error",
        "occlusion_order_violation_count",
        "final_return_order_violation_count",
        "preexisting_return_limit_violation_count",
        "final_return_limit_violation_count",
        "new_return_limit_violation_count",
        "candidate_nonfinite_point_count",
        "retained_nonfinite_point_count",
        "final_nonfinite_point_count",
        "candidate_points_outside_point_cloud_range",
        "retained_points_outside_point_cloud_range",
        "spoof_ray_collision_count",
        "ground_anchor_applied",
    )
    totals = {f"total_{field}": 0 for field in sum_fields}
    total_frames = 0
    attacked_frames = 0
    clean_frames = 0
    attack_windows = 0
    saved_segments = 0

    for segment in selected_segments:
        frame_path = (
            Path(metrics_dir) / "frames" / f"{segment}_frames.csv"
        )
        run_path = Path(metrics_dir) / "runs" / f"{segment}_runs.csv"
        if not frame_path.exists() or not run_path.exists():
            continue
        saved_segments += 1
        with open(frame_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                total_frames += 1
                is_attack = str(row.get("attack_active", "")).lower() == "true"
                attacked_frames += int(is_attack)
                clean_frames += int(not is_attack)
                for field in sum_fields:
                    value = row.get(field, "")
                    if value not in ("", None):
                        if str(value).lower() in ("true", "false"):
                            numeric_value = int(str(value).lower() == "true")
                        else:
                            numeric_value = int(float(value))
                        totals[f"total_{field}"] += numeric_value
        with open(run_path, newline="", encoding="utf-8") as handle:
            attack_windows += sum(1 for _ in csv.DictReader(handle))

    totals.update({
        "saved_selected_segment_count": int(saved_segments),
        "saved_frame_count": int(total_frames),
        "saved_attacked_frame_count": int(attacked_frames),
        "saved_clean_frame_count": int(clean_frames),
        "saved_attack_window_count": int(attack_windows),
    })
    return totals


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

def place_trace(
    traces, perturbation, window_seg_min, window_seg_max, seed,
    anchor_pose=None, variant_name=None, segment_name=None,
    anchor_local_index=None, anchor_global_index=None, trace_file=None
):
    """
    For global_position_fixed:
      anchor_pose is the (4,4) ego pose of the last frame in the spoof run.
      The trace is placed in ego-relative coords of that anchor frame,
      then converted to world/global coordinates. Every historical frame
      will then transform from global back into its own ego frame inside
      process_4frames.
    """
    rng = np.random.default_rng(seed)
    trace_idx = int(rng.integers(0, len(traces)))
    trace = traces[trace_idx]
    relative_position_offset = np.array(
        [rng.uniform(15, 20), rng.uniform(-3, 3)]
    )
    source_box = np.asarray(trace["box"]).copy()

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
        'seed': seed,
        'trace_id': trace_idx,
    }
    run_id = (
        f"{variant_name}:{segment_name}:"
        f"{window_seg_min:04d}-{window_seg_max:04d}"
    )
    run_record = {
        "variant_name": variant_name,
        "segment_id": segment_name,
        "run_id": run_id,
        "run_seed": int(seed),
        "window_start_local": int(window_seg_min),
        "window_end_local": int(window_seg_max),
        "window_length": int(window_seg_max - window_seg_min + 1),
        "anchor_local_index": int(anchor_local_index),
        "anchor_global_index": int(anchor_global_index),
        "trace_id": trace_idx,
        "trace_file": str(trace_file),
        "source_trace_point_count": int(trace["points"].shape[0]),
        "source_box_x": float(source_box[0]),
        "source_box_y": float(source_box[1]),
        "source_box_z": float(source_box[2]),
        "source_box_length": float(source_box[3]),
        "source_box_width": float(source_box[4]),
        "source_box_height": float(source_box[5]),
        "source_box_yaw": float(source_box[6]),
        "source_box_range": float(np.linalg.norm(source_box[:2])),
        "sampled_relative_x": float(relative_position_offset[0]),
        "sampled_relative_y": float(relative_position_offset[1]),
        "sampled_relative_yaw": float(yaw_new),
        "placed_global_box_x": float(front_near_box[0]),
        "placed_global_box_y": float(front_near_box[1]),
        "placed_global_box_z": float(front_near_box[2]),
        "placed_global_box_length": float(front_near_box[3]),
        "placed_global_box_width": float(front_near_box[4]),
        "placed_global_box_height": float(front_near_box[5]),
        "placed_global_box_yaw": float(front_near_box[6]),
        "candidate_point_count": int(spoof_points.shape[0]),
    }
    annotation.update(run_record)

    return spoof_points, annotation, front_near_box, run_record

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

    neighbor_count = int(np.sum(local_mask))
    spoof_bottom_z_before = float(np.min(spf_cur[:, 2]))
    ground_record = {
        "ground_anchor_attempted": True,
        "ground_anchor_applied": False,
        "ground_neighbor_count": neighbor_count,
        "ground_radius_m": float(radius),
        "ground_percentile": 5.0,
        "estimated_ground_z": None,
        "spoof_bottom_z_before": spoof_bottom_z_before,
        "vertical_shift_dz": 0.0,
        "spoof_bottom_z_after": spoof_bottom_z_before,
        "ground_anchor_failure_reason": "insufficient_neighbors",
        "local_ground_z_p05": None,
        "local_ground_z_p50": None,
        "local_ground_z_std": None,
        "box_bottom_z_after": float(gt_box[2] - 0.5 * gt_box[5]),
        "box_ground_gap_after": None,
    }

    if neighbor_count > 20:
        local_points = scene_xyz[local_mask]

        ground_z = np.percentile(local_points[:, 2], 5)
        ground_z_p50 = np.percentile(local_points[:, 2], 50)
        ground_z_std = np.std(local_points[:, 2])
        spoof_bottom_z = np.min(spf_cur[:, 2])

        dz = ground_z - spoof_bottom_z
        
        # print(f"dz: {dz}")
        spf_cur[:, 2] += dz
        gt_box[2] += dz
        ground_record.update({
            "ground_anchor_applied": True,
            "estimated_ground_z": float(ground_z),
            "vertical_shift_dz": float(dz),
            "spoof_bottom_z_after": float(np.min(spf_cur[:, 2])),
            "ground_anchor_failure_reason": None,
            "local_ground_z_p05": float(ground_z),
            "local_ground_z_p50": float(ground_z_p50),
            "local_ground_z_std": float(ground_z_std),
            "box_bottom_z_after": float(gt_box[2] - 0.5 * gt_box[5]),
            "box_ground_gap_after": float(
                gt_box[2] - 0.5 * gt_box[5] - ground_z
            ),
        })

    return spf_cur, gt_box, ground_record


def spoof_frame(
    spoof_points,
    points,
    gt_box,
    rng,
    point_cloud_range,
    radial_edges,
    max_returns_per_ray=2,
    save_point_audit=False,
):
    global elevations, stats
    if spoof_points is None:
        audit = {
            "attack_active": False,
            "scene_point_count_before": int(points.shape[0]),
            "spoof_candidate_count": 0,
            "spoof_kept_count": 0,
            "spoof_discarded_count": 0,
            "background_removed_event_count": 0,
            "background_removed_unique_count": 0,
            "final_point_count": int(points.shape[0]),
            "count_balance_error": 0,
            "retention_rate": None,
            "final_spoof_start_index": int(points.shape[0]),
            "final_spoof_end_index": int(points.shape[0]),
            "occlusion_order_violation_count": 0,
            "final_return_order_violation_count": 0,
            "preexisting_return_limit_violation_count": 0,
            "final_return_limit_violation_count": 0,
            "new_return_limit_violation_count": 0,
            "spoof_ray_collision_count": 0,
            "ground_anchor_applied": False,
            "candidate_nonfinite_point_count": 0,
            "retained_nonfinite_point_count": 0,
            "final_nonfinite_point_count": int(
                np.sum(~np.all(np.isfinite(points), axis=1))
            ),
            "candidate_points_outside_point_cloud_range": 0,
            "retained_points_outside_point_cloud_range": 0,
        }
        recovery = {
            "candidate_points": np.empty((0, 5), dtype=np.float32),
            "candidate_kept": np.empty((0,), dtype=bool),
            "candidate_outcome_code": np.empty((0,), dtype=np.int8),
            "candidate_beam_index": np.empty((0,), dtype=np.int16),
            "candidate_azimuth_bin": np.empty((0,), dtype=np.int32),
            "candidate_ray_key": np.empty((0,), dtype=np.int64),
            "candidate_azimuth_residual_deg": np.empty((0,), dtype=np.float32),
            "candidate_elevation_residual_deg": np.empty((0,), dtype=np.float32),
            "candidate_scene_returns_on_ray": np.empty((0,), dtype=np.int32),
            "candidate_nearest_scene_range": np.empty((0,), dtype=np.float32),
            "candidate_scene_points_removed": np.empty((0,), dtype=np.int32),
            "candidate_minimum_removed_scene_range": np.empty(
                (0,), dtype=np.float32
            ),
            "candidate_occlusion_order_violation": np.empty((0,), dtype=bool),
            "removed_scene_indices": np.empty((0,), dtype=np.int64),
            "removed_scene_points": np.empty(
                (0, points.shape[1]), dtype=np.float32
            ),
        }
        return points, gt_box, audit, [], recovery, []
    
    frame_start = time.perf_counter()
    n_spoof_r = spoof_points.shape[0]
    spf_cur, gt_box, ground_record = ground_anchor_spoof(
        points, spoof_points, gt_box
    )
    spf_cur, gt_box = _jitter_spoof(spf_cur, gt_box)
    intensity_before = spf_cur[:, 3].copy()
    elongation_before = spf_cur[:, 4].copy()
        
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
        # CVPR fix: range_conditioned_stats.npy is fit on RAW Waymo intensity, but every scene
        # point arrives through pcdet get_lidar(), which applies tanh. Map the sampled spoof
        # intensities to the same scale (applied after the draws: RNG stream unchanged).
        I = np.tanh(I)
        E = rng.choice(band_stats["elong_pool"], size=spf_cur.shape[0])
        if INTENSITY_MODE == "adaptive":   # trace/calibrated: keep the columns set at setup
            spf_cur[:, 3] = I
            spf_cur[:, 4] = E
    intensity_after = spf_cur[:, 3].copy()
    elongation_after = spf_cur[:, 4].copy()
        
    dtheta = np.deg2rad(0.1358)

    r_trace, theta_trace, phi_trace = helpers_ptt.cart_2_spherical(spf_cur)
    scene_xyz = points[:, :3]
    r_scene, theta_scene, phi_scene = helpers_ptt.cart_2_spherical(scene_xyz)
    r_scene_by_original_index = np.asarray(r_scene).copy()

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
    key_stride = int(len(elevations))
    keys = az_idx_scene * key_stride + beam_idx_scene
    unique_keys, group_start = np.unique(keys, return_index=True)
    group_end = np.r_[group_start[1:], len(keys)]

    # Build fast lookup map
    key_to_slice = {
        k: (start, end)
        for k, start, end in zip(unique_keys, group_start, group_end)
    }

    az_trace = np.mod(theta_trace, 2 * np.pi)
    az_idx_trace = np.floor(az_trace / dtheta).astype(np.int32)
    beam_idx_trace = np.abs(
        elevations[:, None] - phi_trace
    ).argmin(axis=0).astype(np.int32)
    trace_keys = az_idx_trace * key_stride + beam_idx_trace
    az_center_trace = (az_idx_trace.astype(np.float64) + 0.5) * dtheta
    az_residual_deg = np.rad2deg(
        _circular_difference(az_trace, az_center_trace)
    )
    elevation_residual_deg = np.rad2deg(
        phi_trace - elevations[beam_idx_trace]
    )

    trace_key_counts = {
        int(key): int(count)
        for key, count in zip(*np.unique(trace_keys, return_counts=True))
    }
    spoof_collision_mask = np.array(
        [trace_key_counts[int(key)] > 1 for key in trace_keys], dtype=bool
    )

    outcome = np.full(n_spoof_r, "unclassified", dtype=object)
    scene_returns_on_ray = np.zeros(n_spoof_r, dtype=np.int32)
    scene_removed_by_candidate = np.zeros(n_spoof_r, dtype=np.int32)
    nearest_scene_range = np.full(n_spoof_r, np.nan, dtype=np.float64)
    minimum_removed_scene_range = np.full(
        n_spoof_r, np.nan, dtype=np.float64
    )
    occlusion_order_violation = np.zeros(n_spoof_r, dtype=bool)
    dual_return_trial = np.zeros(n_spoof_r, dtype=bool)
    dual_return_success = np.zeros(n_spoof_r, dtype=bool)
    remove_spoof = []
    remove_scene_events = []
    occlusion_depth_events = []

    for idx, (r, theta, phi) in enumerate(zip(r_trace, theta_trace, phi_trace)):
        key = int(trace_keys[idx])

        if key not in key_to_slice:
            outcome[idx] = "kept_empty_scene_ray"
            continue

        start, end = key_to_slice[key]
        r_group = r_scene[start:end]
        idx_group = scene_indices_sorted[start:end]
        scene_returns_on_ray[idx] = int(len(r_group))

        if len(r_group) == 0:
            outcome[idx] = "kept_empty_scene_ray"
            continue
        nearest_scene_range[idx] = float(r_group[0])

        if r_group[0] < r:
            remove_spoof.append(idx)
            outcome[idx] = "discarded_behind_first_real_return"
        
        elif r < r_group[0]:
            dual_return_trial[idx] = True
            if rng.random() < P_two:
                dual_return_success[idx] = True
                outcome[idx] = "kept_with_scene_second_return"
                if len(idx_group) > 0:
                    removed = idx_group[1:].tolist()
                    removed_ranges = r_group[1:]
                    remove_scene_events.extend(removed)
                    scene_removed_by_candidate[idx] = len(removed)
                    if len(removed_ranges):
                        minimum_removed_scene_range[idx] = float(
                            np.min(removed_ranges)
                        )
                        occlusion_depth_events.extend(
                            (removed_ranges - float(r)).tolist()
                        )
            else:
                outcome[idx] = "kept_in_front_of_scene"
                removed = idx_group.tolist()
                remove_scene_events.extend(removed)
                scene_removed_by_candidate[idx] = len(removed)
                minimum_removed_scene_range[idx] = float(np.min(r_group))
                occlusion_depth_events.extend(
                    (r_group - float(r)).tolist()
                )

        # Equality is rare; it is logged separately rather than silently
        # presented as a general "between returns" case.
        else:
            outcome[idx] = "kept_equal_first_return"

        if scene_removed_by_candidate[idx] > 0:
            occlusion_order_violation[idx] = (
                minimum_removed_scene_range[idx] < float(r) - 1e-6
            )

    mask_spoof = np.ones(spf_cur.shape[0], dtype=bool)
    mask_spoof[remove_spoof] = False
    spoof_points_rc = spf_cur[mask_spoof]
    kept_mask = mask_spoof
    
    remove_scene_unique = sorted(set(remove_scene_events))
    removed_unique = len(remove_scene_unique)
    kept_spoof = spoof_points_rc.shape[0]

    mask_scene = np.ones(points.shape[0], dtype=bool)
    mask_scene[remove_scene_unique] = False
    scene_points_rc = points[mask_scene]

    lag_value = points[0, -1]
    lag_col = np.full((spoof_points_rc.shape[0], 1), lag_value, dtype=np.float32)    
    spf_pts_rc_lag = np.hstack([spoof_points_rc, lag_col])

    frame_points_spoof_rc = np.concatenate([scene_points_rc, spf_pts_rc_lag], axis=0)
    expected_final = points.shape[0] - removed_unique + kept_spoof
    final_spoof_start_index = int(scene_points_rc.shape[0])
    final_spoof_end_index = int(frame_points_spoof_rc.shape[0])

    candidate_in_range = _point_cloud_range_mask(
        spf_cur[:, :3], point_cloud_range
    )
    retained_in_range = _point_cloud_range_mask(
        spoof_points_rc[:, :3], point_cloud_range
    )
    box_center_in_range = bool(
        _point_cloud_range_mask(
            np.asarray(gt_box[:3], dtype=np.float64).reshape(1, 3),
            point_cloud_range,
        )[0]
    )
    candidate_norm_box, candidate_inside_box = _normalized_box_coordinates(
        spf_cur[:, :3], gt_box
    )
    retained_norm_box, retained_inside_box = _normalized_box_coordinates(
        spoof_points_rc[:, :3], gt_box
    )

    # Validate the final composition of every ray affected by spoof candidates.
    remaining_scene_xyz = scene_points_rc[:, :3]
    (
        final_scene_r,
        final_scene_theta,
        final_scene_phi,
    ) = helpers_ptt.cart_2_spherical(remaining_scene_xyz)
    final_scene_az = np.mod(final_scene_theta, 2 * np.pi)
    final_scene_az_idx = np.floor(final_scene_az / dtheta).astype(np.int32)
    final_scene_beam_idx = np.abs(
        elevations[:, None] - final_scene_phi
    ).argmin(axis=0).astype(np.int32)
    final_scene_keys = (
        final_scene_az_idx * key_stride + final_scene_beam_idx
    )

    final_spoof_r = r_trace[kept_mask]
    final_spoof_keys = trace_keys[kept_mask]
    affected_keys = np.unique(trace_keys)
    final_count_by_ray = {}
    preexisting_count_by_ray = {}
    final_rays_with_zero = 0
    final_rays_with_one = 0
    final_rays_with_two = 0
    final_rays_with_more_than_two = 0
    final_return_order_violations = 0
    final_nearest_spoof = 0
    final_nearest_scene = 0

    for key in affected_keys:
        key = int(key)
        preexisting_count = int(np.sum(keys == key))
        scene_ranges = final_scene_r[final_scene_keys == key]
        spoof_ranges = final_spoof_r[final_spoof_keys == key]
        final_count = int(len(scene_ranges) + len(spoof_ranges))
        preexisting_count_by_ray[key] = preexisting_count
        final_count_by_ray[key] = final_count

        if final_count == 0:
            final_rays_with_zero += 1
        elif final_count == 1:
            final_rays_with_one += 1
        elif final_count == 2:
            final_rays_with_two += 1
        else:
            final_rays_with_more_than_two += 1

        if len(spoof_ranges) and len(scene_ranges):
            if np.min(scene_ranges) < np.min(spoof_ranges) - 1e-6:
                final_return_order_violations += 1
            if np.min(spoof_ranges) <= np.min(scene_ranges):
                final_nearest_spoof += 1
            else:
                final_nearest_scene += 1
        elif len(spoof_ranges):
            final_nearest_spoof += 1
        elif len(scene_ranges):
            final_nearest_scene += 1

    preexisting_return_limit_violations = sum(
        count > max_returns_per_ray for count in preexisting_count_by_ray.values()
    )
    final_return_limit_violations = sum(
        count > max_returns_per_ray for count in final_count_by_ray.values()
    )
    new_return_limit_violations = sum(
        final_count_by_ray[key] > max_returns_per_ray
        and preexisting_count_by_ray[key] <= max_returns_per_ray
        for key in final_count_by_ray
    )
    rays_with_background_removal = int(
        np.unique(trace_keys[scene_removed_by_candidate > 0]).size
    )
    removed_scene_ranges_unique = r_scene_by_original_index[
        remove_scene_unique
    ] if remove_scene_unique else np.empty((0,), dtype=np.float64)

    if np.any(outcome == "unclassified"):
        raise RuntimeError("At least one spoof candidate was not classified")
    count_balance_error = int(frame_points_spoof_rc.shape[0] - expected_final)
    if count_balance_error != 0:
        raise RuntimeError(
            f"Point accounting failed: balance error={count_balance_error}"
        )

    candidate_az_span = float(
        np.rad2deg(np.ptp(np.unwrap(theta_trace)))
    ) if n_spoof_r > 1 else 0.0
    candidate_el_span = float(
        np.rad2deg(np.ptp(phi_trace))
    ) if n_spoof_r > 1 else 0.0
    kept_az_span = float(
        np.rad2deg(np.ptp(np.unwrap(theta_trace[kept_mask])))
    ) if kept_spoof > 1 else 0.0
    kept_el_span = float(
        np.rad2deg(np.ptp(phi_trace[kept_mask]))
    ) if kept_spoof > 1 else 0.0

    audit = {
        "attack_active": True,
        "scene_point_count_before": int(points.shape[0]),
        "spoof_candidate_count": int(n_spoof_r),
        "spoof_kept_count": int(kept_spoof),
        "spoof_discarded_count": int(n_spoof_r - kept_spoof),
        "retention_rate": float(kept_spoof / max(n_spoof_r, 1)),
        "background_removed_event_count": int(len(remove_scene_events)),
        "background_removed_unique_count": int(removed_unique),
        "rays_with_background_removal": rays_with_background_removal,
        "background_removed_per_affected_ray_mean": float(
            removed_unique / max(len(affected_keys), 1)
        ),
        "background_removed_per_removal_ray_mean": float(
            removed_unique / max(rays_with_background_removal, 1)
        ),
        "background_removed_per_candidate_max": int(
            np.max(scene_removed_by_candidate)
        ) if n_spoof_r else 0,
        "final_point_count": int(frame_points_spoof_rc.shape[0]),
        "count_balance_error": count_balance_error,
        "final_spoof_start_index": final_spoof_start_index,
        "final_spoof_end_index": final_spoof_end_index,
        "affected_ray_count": int(np.unique(trace_keys).size),
        "candidate_unique_ray_count": int(np.unique(trace_keys).size),
        "kept_unique_ray_count": int(np.unique(trace_keys[kept_mask]).size),
        "candidate_unique_beam_count": int(np.unique(beam_idx_trace).size),
        "kept_unique_beam_count": int(
            np.unique(beam_idx_trace[kept_mask]).size
        ),
        "spoof_ray_collision_count": int(
            sum(count > 1 for count in trace_key_counts.values())
        ),
        "spoof_points_on_colliding_rays": int(np.sum(spoof_collision_mask)),
        "maximum_spoof_points_per_ray": int(
            max(trace_key_counts.values(), default=0)
        ),
        "max_returns_per_ray_allowed": int(max_returns_per_ray),
        "preexisting_return_limit_violation_count": int(
            preexisting_return_limit_violations
        ),
        "final_return_limit_violation_count": int(
            final_return_limit_violations
        ),
        "new_return_limit_violation_count": int(
            new_return_limit_violations
        ),
        "final_max_returns_per_affected_ray": int(
            max(final_count_by_ray.values(), default=0)
        ),
        "final_rays_with_zero_returns": int(final_rays_with_zero),
        "final_rays_with_one_return": int(final_rays_with_one),
        "final_rays_with_two_returns": int(final_rays_with_two),
        "final_rays_with_more_than_two_returns": int(
            final_rays_with_more_than_two
        ),
        "final_return_order_violation_count": int(
            final_return_order_violations
        ),
        "final_nearest_return_spoof_ray_count": int(final_nearest_spoof),
        "final_nearest_return_scene_ray_count": int(final_nearest_scene),
        "occlusion_order_violation_count": int(
            np.sum(occlusion_order_violation)
        ),
        "dual_return_trial_count": int(np.sum(dual_return_trial)),
        "dual_return_success_count": int(np.sum(dual_return_success)),
        "n_kept_empty_scene_ray": int(
            np.sum(outcome == "kept_empty_scene_ray")
        ),
        "n_kept_in_front_of_scene": int(
            np.sum(outcome == "kept_in_front_of_scene")
        ),
        "n_kept_with_scene_second_return": int(
            np.sum(outcome == "kept_with_scene_second_return")
        ),
        "n_kept_equal_first_return": int(
            np.sum(outcome == "kept_equal_first_return")
        ),
        "n_discarded_behind_real_return": int(
            np.sum(outcome == "discarded_behind_first_real_return")
        ),
        "average_spoof_range": float(r_avg),
        "feature_stats_found": bool(band_stats is not None),
        "azimuth_resolution_deg": float(np.rad2deg(dtheta)),
        "dual_return_probability": float(P_two),
        "candidate_range_mean": float(np.mean(r_trace)),
        "candidate_range_median": float(np.median(r_trace)),
        "kept_range_mean": (
            float(np.mean(r_trace[kept_mask])) if kept_spoof else None
        ),
        "kept_range_median": (
            float(np.median(r_trace[kept_mask])) if kept_spoof else None
        ),
        "candidate_azimuth_span_deg": candidate_az_span,
        "candidate_elevation_span_deg": candidate_el_span,
        "kept_azimuth_span_deg": kept_az_span,
        "kept_elevation_span_deg": kept_el_span,
        "candidate_points_per_angular_area": float(
            n_spoof_r / max(candidate_az_span * candidate_el_span, 1e-12)
        ),
        "kept_points_per_angular_area": float(
            kept_spoof / max(kept_az_span * kept_el_span, 1e-12)
        ),
        "raycasting_wall_seconds": float(time.perf_counter() - frame_start),
        "spoof_box_in_point_cloud_range": box_center_in_range,
        "candidate_points_outside_point_cloud_range": int(
            np.sum(~candidate_in_range)
        ),
        "retained_points_outside_point_cloud_range": int(
            np.sum(~retained_in_range)
        ),
        "candidate_points_behind_ego": int(np.sum(spf_cur[:, 0] < 0)),
        "retained_points_behind_ego": int(
            np.sum(spoof_points_rc[:, 0] < 0)
        ),
        "candidate_nonfinite_point_count": int(
            np.sum(~np.all(np.isfinite(spf_cur), axis=1))
        ),
        "retained_nonfinite_point_count": int(
            np.sum(~np.all(np.isfinite(spoof_points_rc), axis=1))
        ),
        "final_nonfinite_point_count": int(
            np.sum(
                ~np.all(
                    np.isfinite(frame_points_spoof_rc),
                    axis=1,
                )
            )
        ),
        "candidate_points_inside_box": int(np.sum(candidate_inside_box)),
        "candidate_fraction_inside_box": float(
            np.mean(candidate_inside_box)
        ) if n_spoof_r else None,
        "retained_points_inside_box": int(np.sum(retained_inside_box)),
        "retained_fraction_inside_box": float(
            np.mean(retained_inside_box)
        ) if kept_spoof else None,
        "retained_elevation_error_over_0p01_deg": int(
            np.sum(np.abs(elevation_residual_deg[kept_mask]) > 0.01)
        ),
        "retained_elevation_error_over_0p05_deg": int(
            np.sum(np.abs(elevation_residual_deg[kept_mask]) > 0.05)
        ),
        "retained_azimuth_error_over_quarter_bin": int(
            np.sum(
                np.abs(az_residual_deg[kept_mask])
                > 0.25 * np.rad2deg(dtheta)
            )
        ),
        "retained_azimuth_error_over_half_bin": int(
            np.sum(
                np.abs(az_residual_deg[kept_mask])
                > 0.5 * np.rad2deg(dtheta)
            )
        ),
    }
    if band_stats is not None:
        audit.update({
            "feature_log_mu": float(band_stats["log_mu"]),
            "feature_log_sigma": float(band_stats["log_sigma"]),
            "feature_intensity_clip_p1": float(
                band_stats["intensity_p1"]
            ),
            "feature_intensity_clip_p99": float(
                band_stats["intensity_p99"]
            ),
            "feature_elongation_pool_size": int(
                len(band_stats["elong_pool"])
            ),
        })
    audit.update(ground_record)
    audit.update(_summary("intensity_before", intensity_before))
    audit.update(_summary("intensity_after", intensity_after))
    audit.update(_summary("elongation_before", elongation_before))
    audit.update(_summary("elongation_after", elongation_after))
    audit.update(_summary("azimuth_residual_deg", az_residual_deg))
    audit.update(_summary("elevation_residual_deg", elevation_residual_deg))
    audit.update(_summary(
        "retained_azimuth_residual_deg",
        az_residual_deg[kept_mask],
    ))
    audit.update(_summary(
        "retained_elevation_residual_deg",
        elevation_residual_deg[kept_mask],
    ))
    audit.update(_summary("occlusion_depth", occlusion_depth_events))
    audit.update(_summary(
        "removed_scene_range",
        removed_scene_ranges_unique,
    ))
    for axis, axis_name in enumerate(("x", "y", "z")):
        audit.update(_summary(
            f"candidate_normalized_box_{axis_name}",
            candidate_norm_box[:, axis],
        ))
        audit.update(_summary(
            f"retained_normalized_box_{axis_name}",
            retained_norm_box[:, axis],
        ))

    point_audit = []
    if save_point_audit:
        for idx in range(n_spoof_r):
            point_audit.append({
                "candidate_index": int(idx),
                "candidate_x": float(spf_cur[idx, 0]),
                "candidate_y": float(spf_cur[idx, 1]),
                "candidate_z": float(spf_cur[idx, 2]),
                "candidate_range": float(r_trace[idx]),
                "candidate_retained": bool(kept_mask[idx]),
                "candidate_inside_box": bool(candidate_inside_box[idx]),
                "candidate_in_point_cloud_range": bool(
                    candidate_in_range[idx]
                ),
                "normalized_box_x": float(candidate_norm_box[idx, 0]),
                "normalized_box_y": float(candidate_norm_box[idx, 1]),
                "normalized_box_z": float(candidate_norm_box[idx, 2]),
                "intensity_before": float(intensity_before[idx]),
                "intensity_after": float(intensity_after[idx]),
                "elongation_before": float(elongation_before[idx]),
                "elongation_after": float(elongation_after[idx]),
                "beam_index": int(beam_idx_trace[idx]),
                "azimuth_bin": int(az_idx_trace[idx]),
                "ray_key": int(trace_keys[idx]),
                "azimuth_residual_deg": float(az_residual_deg[idx]),
                "elevation_residual_deg": float(elevation_residual_deg[idx]),
                "spoof_candidates_on_same_ray": int(
                    trace_key_counts[int(trace_keys[idx])]
                ),
                "preexisting_returns_on_ray": int(
                    preexisting_count_by_ray[int(trace_keys[idx])]
                ),
                "final_returns_on_ray": int(
                    final_count_by_ray[int(trace_keys[idx])]
                ),
                "scene_returns_on_ray": int(scene_returns_on_ray[idx]),
                "nearest_scene_range": (
                    float(nearest_scene_range[idx])
                    if np.isfinite(nearest_scene_range[idx]) else None
                ),
                "scene_points_removed": int(scene_removed_by_candidate[idx]),
                "minimum_removed_scene_range": (
                    float(minimum_removed_scene_range[idx])
                    if np.isfinite(minimum_removed_scene_range[idx]) else None
                ),
                "occlusion_order_violation": bool(
                    occlusion_order_violation[idx]
                ),
                "dual_return_trial": bool(dual_return_trial[idx]),
                "dual_return_success": bool(dual_return_success[idx]),
                "outcome": str(outcome[idx]),
            })
    outcome_code_map = {
        "kept_empty_scene_ray": 0,
        "kept_in_front_of_scene": 1,
        "kept_with_scene_second_return": 2,
        "kept_equal_first_return": 3,
        "discarded_behind_first_real_return": 4,
    }
    recovery = {
        "candidate_points": np.asarray(spf_cur[:, :5], dtype=np.float32),
        "candidate_kept": np.asarray(kept_mask, dtype=bool),
        "candidate_outcome_code": np.asarray(
            [outcome_code_map[str(value)] for value in outcome],
            dtype=np.int8,
        ),
        "candidate_beam_index": np.asarray(beam_idx_trace, dtype=np.int16),
        "candidate_azimuth_bin": np.asarray(az_idx_trace, dtype=np.int32),
        "candidate_ray_key": np.asarray(trace_keys, dtype=np.int64),
        "candidate_azimuth_residual_deg": np.asarray(
            az_residual_deg, dtype=np.float32
        ),
        "candidate_elevation_residual_deg": np.asarray(
            elevation_residual_deg, dtype=np.float32
        ),
        "candidate_scene_returns_on_ray": np.asarray(
            scene_returns_on_ray, dtype=np.int32
        ),
        "candidate_nearest_scene_range": np.asarray(
            nearest_scene_range, dtype=np.float32
        ),
        "candidate_scene_points_removed": np.asarray(
            scene_removed_by_candidate, dtype=np.int32
        ),
        "candidate_minimum_removed_scene_range": np.asarray(
            minimum_removed_scene_range, dtype=np.float32
        ),
        "candidate_occlusion_order_violation": np.asarray(
            occlusion_order_violation, dtype=bool
        ),
        "removed_scene_indices": np.asarray(
            remove_scene_unique, dtype=np.int64
        ),
        "removed_scene_points": np.asarray(
            points[remove_scene_unique], dtype=np.float32
        ) if remove_scene_unique else np.empty(
            (0, points.shape[1]), dtype=np.float32
        ),
    }
    removed_scene_points = (
        points[remove_scene_unique]
        if remove_scene_unique
        else np.empty((0, points.shape[1]), dtype=points.dtype)
    )
    radial_rows = _radial_density_rows(
        points_before=points,
        surviving_scene_points=scene_points_rc,
        removed_scene_points=removed_scene_points,
        retained_spoof_points=spf_pts_rc_lag,
        points_after=frame_points_spoof_rc,
        gt_box=gt_box,
        radial_edges=radial_edges,
    )

    return (
        frame_points_spoof_rc,
        gt_box,
        audit,
        point_audit,
        recovery,
        radial_rows,
    )


def _stable_seed(master_seed, *parts):
    """Deterministic 64-bit seed from the master seed and string parts.

    Uses hashlib (not the process-randomized built-in hash) so that a resumed
    run reconstructs identical per-segment schedules and per-frame flicker
    draws for any not-yet-completed segment.
    """
    payload = ":".join([str(master_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _on_block_ranges(n_frames, block_length):
    """
    Yield (start_local, end_local_inclusive) for every ATTACK block under the
    original alternating schedule.

    Local frame 0 starts a CLEAN block and blocks alternate CLEAN/ATTACK every
    ``block_length`` frames, so the ATTACK blocks are the 2nd, 4th, ... blocks
    (odd 0-based block index): [block_length, 2*block_length-1], etc. The final
    block may be shorter than ``block_length`` at the end of a segment.
    """
    ranges = []
    block_index = 1
    while block_index * block_length < n_frames:
        if block_index % 2 == 1:
            start = block_index * block_length
            end = min((block_index + 1) * block_length - 1, n_frames - 1)
            ranges.append((start, end))
        block_index += 1
    return ranges


def build_target_schedule_global(
    segment_to_global,
    block_length,
    schedule_mode,
    seed,
    continuous_min_len,
    continuous_max_len,
):
    """
    Build the {global_frame_index: 1 or None} target schedule per segment.

    The 32-on/32-off alternation is retained only as an execution-time scaffold
    that defines each ATTACK block. Which frames inside a block are actually
    marked as targets (value 1) depends on ``schedule_mode``:

      "continuous": within every ATTACK block, a single contiguous run of
        length L in [continuous_min_len, continuous_max_len] (clamped to the
        block length) is marked; its offset within the block is random.
        extract_one_runs() then recovers exactly that run, so the global anchor
        (max index of the run) is the LAST frame of the marked sub-window.

      "flicker": every frame of the ATTACK block is marked (all intended). The
        per-frame drop is applied later in setup_segment_spoofs(), so the run
        stays the full block and the anchor stays the LAST frame of the block.

    Only frames that end up with an actual spoof are reported as targets:
    dropped frames are set back to None downstream and read as clean.
    """
    schedules = {}
    for segment, global_indices in segment_to_global.items():
        n_frames = len(global_indices)
        schedule = {g: None for g in global_indices}
        seg_rng = np.random.default_rng(_stable_seed(seed, segment, "schedule"))
        for start_local, end_local in _on_block_ranges(n_frames, block_length):
            block_len = end_local - start_local + 1
            if schedule_mode == "flicker":
                on_locals = range(start_local, end_local + 1)
            else:
                span = int(seg_rng.integers(
                    continuous_min_len, continuous_max_len + 1
                ))
                span = min(span, block_len)
                offset = int(seg_rng.integers(0, block_len - span + 1))
                on_locals = range(
                    start_local + offset, start_local + offset + span
                )
            for local_index in on_locals:
                schedule[global_indices[local_index]] = 1
        schedules[segment] = schedule
    return schedules


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
                         original_mf_indices, seg_localidx_to_pose, traces, seed,
                         variant_name, trace_file,
                         schedule_mode="continuous", flicker_prob=None):
    
    target_frames = target_frames_global[segment].copy()
    target_frames_anno = target_frames_global[segment].copy()
    segment_spoof_annos = []
    target_frame_context = {}
    segment_run_records = []
    
    runs_global = extract_one_runs(target_frames)
    for run in runs_global:
        subset_g = {k: target_frames[k] for k in run}
        subset_g_anno = {k: target_frames_anno[k] for k in run}
        window_min = global_to_segment[min(subset_g)][1]
        window_max = global_to_segment[max(subset_g)][1]
        
        anchor_filtered_global_idx = max(subset_g.keys())
        anchor_original_global_idx = int(
            original_mf_indices[anchor_filtered_global_idx]
        )
        anchor_local_idx = global_to_segment[anchor_filtered_global_idx][1]
        anchor_pose = seg_localidx_to_pose[(segment, anchor_local_idx)]

        run_start_filtered_global_idx = min(subset_g)
        run_start_original_global_idx = int(
            original_mf_indices[run_start_filtered_global_idx]
        )
        # Preserve the original generator's seed stream even though the working
        # dataset has been filtered to the selected segments.
        run_seed = int(seed + run_start_original_global_idx)
        # The trace is placed globally at the anchor pose (max index of the
        # run). In flicker mode the run is the full ATTACK block, so the anchor
        # stays the last frame of the block; placement is computed here and is
        # independent of the per-frame flicker draws below (the anchor frame's
        # spoof may itself be dropped without moving the placement).
        spoof_points_global, spoof_annotation, box_global, run_record = place_trace(
            traces, 'global_position_fixed',
            window_min, window_max, run_seed,
            anchor_pose=anchor_pose,
            variant_name=variant_name,
            segment_name=segment,
            anchor_local_index=anchor_local_idx,
            anchor_global_index=anchor_original_global_idx,
            trace_file=trace_file,
        )
        if INTENSITY_MODE == "calibrated":   # range of the spoof in the anchor frame
            _anchor_pts, _ = transform_spoof_to_ego_frame(spoof_points_global, box_global, anchor_pose)
            _calibrate_spoof_features(spoof_points_global, _anchor_pts, seed, segment, run_start_original_global_idx)
        elif INTENSITY_MODE in ("saturated", "attenuated"):
            _constant_spoof_features(spoof_points_global)

        flicker_rng = None
        if schedule_mode == "flicker":
            flicker_rng = np.random.default_rng(
                _stable_seed(
                    seed, segment, run_start_original_global_idx, "flicker"
                )
            )

        for key in sorted(subset_g.keys()):
            # FLICKER mode: each intended frame (including the anchor) fires
            # only with probability flicker_prob; otherwise no spoof is emitted
            # and the frame reads as clean.
            if flicker_rng is not None and not (flicker_rng.random() < flicker_prob):
                subset_g[key] = None
                subset_g_anno[key] = None
                continue

            target_segment, target_local_idx = global_to_segment[key]
            target_pose = seg_localidx_to_pose[(target_segment, target_local_idx)]
            
            spoof_points_ego, box_ego = transform_spoof_to_ego_frame(
                spoof_points_global, box_global, target_pose
            )
            
            subset_g[key] = spoof_points_ego      # New array ref (safe)
            subset_g_anno[key] = box_ego.copy()   # CRITICAL: .copy() here
            target_frame_context[key] = {
                "run_id": run_record["run_id"],
                "trace_id": run_record["trace_id"],
                "run_seed": run_seed,
                "window_start_local": window_min,
                "window_end_local": window_max,
                "window_length": window_max - window_min + 1,
                "window_position": target_local_idx - window_min,
                "anchor_local_index": anchor_local_idx,
                "anchor_global_index": anchor_original_global_idx,
                "anchor_filtered_global_index": anchor_filtered_global_idx,
            }
        
        segment_spoof_annos.append(spoof_annotation)
        segment_run_records.append(run_record)
        target_frames.update(subset_g)
        target_frames_anno.update(subset_g_anno)
    
    return (
        target_frames,
        target_frames_anno,
        segment_spoof_annos,
        target_frame_context,
        segment_run_records,
    )

def save_segment(save_dir_dataset, save_dir_annos, segment, segment_dataset,
                 segment_spoof_annos, metrics_dir, segment_frame_records,
                 segment_run_records, segment_point_records,
                 segment_audit_array_records, save_audit_arrays,
                 save_point_audit, segment_radial_records):
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
    _atomic_write_csv(
        Path(metrics_dir) / "frames" / f"{segment}_frames.csv",
        segment_frame_records,
    )
    _atomic_write_csv(
        Path(metrics_dir) / "runs" / f"{segment}_runs.csv",
        segment_run_records,
    )
    if save_point_audit:
        _atomic_write_csv(
            Path(metrics_dir) / "points" / f"{segment}_points.csv",
            segment_point_records,
        )
    if save_audit_arrays:
        _atomic_write_audit_arrays(
            Path(metrics_dir) / "arrays" / f"{segment}_audit_arrays.npz",
            segment_audit_array_records,
        )
    _atomic_write_csv(
        Path(metrics_dir) / "radial" / f"{segment}_radial.csv",
        segment_radial_records,
    )
    return



def main():
    global JITTER_KEY, LAST_JITTER

    args = parse_args()

    # Resolve the temporal-attack schedule mode. --random enables the flickering
    # (imperfect-spoofer) mode; otherwise we use the continuous sub-window mode.
    if args.random is not None:
        schedule_mode = "flicker"
        flicker_prob = float(args.random)
        if not (0.0 < flicker_prob <= 1.0):
            raise ValueError("--random must be in (0, 1]")
    else:
        schedule_mode = "continuous"
        flicker_prob = None
    if args.continuous_min_len < 1 or args.continuous_max_len < args.continuous_min_len:
        raise ValueError(
            "require 1 <= --continuous_min_len <= --continuous_max_len"
        )

    script_start = time.perf_counter()
    run_started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metrics_dir / "run_manifest.json"
    manifest = {
        "schema_version": args.schema_version,
        "generator_name": "generate_data_ablation", "ablation": {k: v for k, v in ABL.items() if k not in ("bank", "bank_range")},
        "positioning_mode": "global_position_fixed",
        "variant_name": args.variant_name,
        "master_seed": int(args.seed),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "trace_file": str(Path(args.trace_file).resolve()),
        "trace_file_sha256": _sha256(args.trace_file),
        "inclination_file": str(Path("waymo_top_lidar_inclinations.npy").resolve()),
        "inclination_file_sha256": _sha256("waymo_top_lidar_inclinations.npy"),
        "range_stats_file": str(Path("range_conditioned_stats.npy").resolve()),
        "range_stats_file_sha256": _sha256("range_conditioned_stats.npy"),
        "cp_mf_cfg": str(Path(args.cp_mf_cfg).resolve()),
        "cp_mf_cfg_sha256": _sha256(args.cp_mf_cfg),
        "cp_sf_cfg": str(Path(args.cp_sf_cfg).resolve()),
        "cp_sf_cfg_sha256": _sha256(args.cp_sf_cfg),
        "azimuth_resolution_deg": 0.1358,
        "dual_return_probability": 0.01,
        "placement_x_min": 15.0,
        "placement_x_max": 20.0,
        "placement_y_min": -3.0,
        "placement_y_max": 3.0,
        "placement_yaw_min_rad": -0.2,
        "placement_yaw_max_rad": 0.2,
        "schedule_type": "alternating_clean_attack",
        "clean_window_length": 32,
        "attack_window_length": 32,
        "schedule_mode": schedule_mode,
        "flicker_fire_probability": flicker_prob,
        "continuous_min_len": int(args.continuous_min_len),
        "continuous_max_len": int(args.continuous_max_len),
        "save_point_audit": bool(args.save_point_audit),
        "save_audit_arrays": bool(not args.no_save_audit_arrays),
        "requested_num_sequences": args.num_sequences,
        "sequence_selection": args.sequence_selection,
        "sequence_selection_seed": int(args.seed),
        "max_returns_per_ray": int(args.max_returns_per_ray),
        "radial_distance_definition": "planar_xy_distance_from_ego",
        "radial_density_definition": "points_per_square_metre_of_xy_annulus",
        "radial_bin_width_m": float(args.radial_bin_width_m),
        "requested_radial_max_m": args.radial_max_m,
        "helpers_ptt_path": str(Path(helpers_ptt.__file__).resolve()),
        "helpers_ptt_sha256": _optional_sha256(helpers_ptt.__file__),
        "numpy_version": np.__version__,
        "torch_version": getattr(
            sys.modules.get("torch"), "__version__", None
        ),
        "cp_mf_git_commit": _git_commit(args.cp_mf_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "started_utc": run_started_utc,
        "status": "running",
    }
    _atomic_write_json(manifest_path, manifest)

    log = logging.getLogger("spoof")
    log.setLevel(logging.INFO)
    log.handlers.clear()

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
    import pcdet as _pcdet_mod  # CVPR: record which pcdet the loaders come from
    log.info("pcdet loaded from %s", _pcdet_mod.__file__)
    log.info("intensity mode: %s", INTENSITY_MODE)
    if not os.path.realpath(_pcdet_mod.__file__).startswith(os.path.realpath(args.cp_mf_root)):
        raise RuntimeError(f"pcdet not from --cp_mf_root: {_pcdet_mod.__file__}")

    # cfg_cp.TAG = 'centerpoint_multiframe'
    # cfg_cp.EXP_GROUP_PATH = 'waymo_multiframe'
    cfg_cp.clear()
    cfg_from_yaml_file_cp(args.cp_mf_cfg, cfg_cp)
    if ABL.get('data_path'):
        cfg_cp.DATA_CONFIG.DATA_PATH = ABL['data_path']

    logger = common_utils.create_logger()
    logger.info(f'Loaded cfg from {args.cp_mf_cfg}')
    dataset_mf, _, _ = build_cp_loader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=8,
        logger = logger,
        training=False
    )
    point_cloud_range = np.asarray(
        cfg_cp.DATA_CONFIG.POINT_CLOUD_RANGE,
        dtype=np.float64,
    ).tolist()
    manifest["point_cloud_range"] = point_cloud_range
    if args.radial_bin_width_m <= 0:
        raise ValueError("--radial_bin_width_m must be positive")
    if args.radial_max_m is None:
        radial_max_m = float(np.ceil(
            np.hypot(
                max(abs(point_cloud_range[0]), abs(point_cloud_range[3])),
                max(abs(point_cloud_range[1]), abs(point_cloud_range[4])),
            ) / args.radial_bin_width_m
        ) * args.radial_bin_width_m)
    else:
        radial_max_m = float(args.radial_max_m)
    if radial_max_m <= 0:
        raise ValueError("--radial_max_m must be positive")
    radial_edges = np.arange(
        0.0,
        radial_max_m + args.radial_bin_width_m * 0.5,
        args.radial_bin_width_m,
        dtype=np.float64,
    )
    if radial_edges[-1] < radial_max_m:
        radial_edges = np.r_[radial_edges, radial_max_m]
    manifest.update({
        "radial_max_m": radial_max_m,
        "radial_bin_count": int(len(radial_edges) - 1),
        "radial_edges_m": radial_edges.tolist(),
    })
    
    cfg_cp.clear()
    
    cfg_from_yaml_file_cp(args.cp_sf_cfg, cfg_cp)
    if ABL.get('data_path'):
        cfg_cp.DATA_CONFIG.DATA_PATH = ABL['data_path']

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

    # Keep the full validation metadata for segment selection and for preserving
    # the original global frame indices after the working datasets are filtered.
    original_mf_infos = list(dataset_mf.infos)
    original_sf_infos = list(dataset_sf.infos)
    original_mf_index_by_id = {
        id(info): index for index, info in enumerate(original_mf_infos)
    }

    frame_ids = [info['frame_id'] for info in original_sf_infos]

    # Split the frame_id to get only the segment name (everything before the last underscore)
    unique_segments_sf = sorted(list(set(fid.rsplit('_', 1)[0] for fid in frame_ids)))

    mf_segments = set(info['point_cloud']['lidar_sequence']
                  for info in original_mf_infos)
    all_common_segments = sorted(
        mf_segments.intersection(unique_segments_sf)
    )
    if args.num_sequences is not None:
        if args.num_sequences <= 0:
            raise ValueError("--num_sequences must be a positive integer")
        if args.num_sequences > len(all_common_segments):
            raise ValueError(
                f"Requested {args.num_sequences} sequences, but only "
                f"{len(all_common_segments)} common sequences are available"
            )
        if args.sequence_selection == "random":
            selection_rng = np.random.default_rng(args.seed)
            selected_segments = sorted(
                selection_rng.choice(
                    all_common_segments,
                    size=args.num_sequences,
                    replace=False,
                ).tolist()
            )
        else:
            selected_segments = all_common_segments[:args.num_sequences]
    else:
        selected_segments = all_common_segments
    common_segments = set(selected_segments)

    # Restrict both working datasets before constructing the DataLoader. This
    # prevents --num_sequences N from loading and collating frames belonging to
    # the other validation segments. The order of retained frames is unchanged.
    dataset_mf.infos = [
        info for info in original_mf_infos
        if info['point_cloud']['lidar_sequence'] in common_segments
    ]
    dataset_sf.infos = [
        info for info in original_sf_infos
        if info['frame_id'].rsplit('_', 1)[0] in common_segments
    ]
    original_mf_indices = [
        original_mf_index_by_id[id(info)] for info in dataset_mf.infos
    ]

    test_loader_mf = DataLoader(
        dataset_mf,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_mf.collate_batch,
    )

    manifest.update({
        "available_common_segment_count": int(len(all_common_segments)),
        "selected_segment_count": int(len(selected_segments)),
        "selected_frame_count": int(len(dataset_mf.infos)),
        "selected_segments": selected_segments,
    })
    _atomic_write_json(
        metrics_dir / "selected_segments.json",
        {
            "variant_name": args.variant_name,
            "selection_method": args.sequence_selection,
            "selection_seed": int(args.seed),
            "segments": selected_segments,
        },
    )
    _atomic_write_json(manifest_path, manifest)
    log.info(
        "Selected %d of %d common sequences using %s selection; "
        "working DataLoader contains %d frames from selected sequences only.",
        len(selected_segments),
        len(all_common_segments),
        args.sequence_selection,
        len(dataset_mf.infos),
    )

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


    # 32-on/32-off remains the execution-time scaffold. In continuous mode only
    # a random contiguous sub-window (len 8..16) of each ATTACK block is marked;
    # in flicker mode the full block is marked and dropped per-frame later.
    target_frames_global = build_target_schedule_global(
        segment_to_global,
        block_length=32,
        schedule_mode=schedule_mode,
        seed=args.seed,
        continuous_min_len=args.continuous_min_len,
        continuous_max_len=args.continuous_max_len,
    )
    # Local-index mirror kept for parity/debugging (not consumed downstream).
    target_frames_seg = {
        seg: {global_to_segment[g][1]: value for g, value in schedule.items()}
        for seg, schedule in target_frames_global.items()
    }


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
    segment_frame_records = []
    segment_run_records = []
    segment_point_records = []
    segment_audit_array_records = []
    segment_radial_records = []
    save_dir_dataset = args.save_dir_dataset
    os.makedirs(save_dir_dataset, exist_ok=True)
    save_dir_annos = args.save_dir_annos
    os.makedirs(save_dir_annos, exist_ok=True)
    for subdir in ("frames", "runs", "points", "arrays", "radial"):
        (metrics_dir / subdir).mkdir(parents=True, exist_ok=True)

    completed = set()
    for dataset_file in Path(save_dir_dataset).glob("*.pkl"):
        segment_name = dataset_file.stem
        if (
            (Path(save_dir_annos) / f"{segment_name}_a.pkl").exists()
            and (
                metrics_dir / "frames" / f"{segment_name}_frames.csv"
            ).exists()
            and (
                metrics_dir / "runs" / f"{segment_name}_runs.csv"
            ).exists()
            and (
                args.no_save_audit_arrays
                or (
                    metrics_dir / "arrays"
                    / f"{segment_name}_audit_arrays.npz"
                ).exists()
            )
            and (
                not args.save_point_audit
                or (
                    metrics_dir / "points"
                    / f"{segment_name}_points.csv"
                ).exists()
            )
            and (
                metrics_dir / "radial"
                / f"{segment_name}_radial.csv"
            ).exists()
        ):
            completed.add(segment_name)
    
    
    traces_file = args.trace_file
    with open(traces_file, 'rb') as file:
        traces = pkl.load(file)
    if set_constant_level(traces) is not None:
        log.info("constant spoof intensity (%s): %.4f", INTENSITY_MODE, CONST_I)
    manifest["trace_count"] = int(len(traces))
    manifest["common_segment_count"] = int(len(common_segments))
    manifest["completed_segment_count_at_start"] = int(len(completed))
    _atomic_write_json(manifest_path, manifest)

    target_frames = None
    current_segment = None
    target_frames_anno = None
    target_frame_context = None
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
                (
                    target_frames,
                    target_frames_anno,
                    segment_spoof_annos,
                    target_frame_context,
                    segment_run_records,
                ) = setup_segment_spoofs(
                    current_segment, target_frames_global, global_to_segment,
                    original_mf_indices, seg_localidx_to_pose, traces, args.seed,
                    args.variant_name, args.trace_file,
                    schedule_mode=schedule_mode, flicker_prob=flicker_prob
                )
            #dump data for past segment
            if(segment != current_segment):
                # print(segment)
                save_segment(
                    save_dir_dataset,
                    save_dir_annos,
                    current_segment,
                    segment_dataset,
                    segment_spoof_annos,
                    args.metrics_dir,
                    segment_frame_records,
                    segment_run_records,
                    segment_point_records,
                    segment_audit_array_records,
                    not args.no_save_audit_arrays,
                    args.save_point_audit,
                    segment_radial_records,
                )
                
                log.info("Saved: %s", current_segment)

                segment_dataset = [] 
                segment_spoof_annos = []
                segment_frame_records = []
                segment_run_records = []
                segment_point_records = []
                segment_audit_array_records = []
                segment_radial_records = []
                current_segment = segment
                
                target_frames = None
                target_frames_anno = None
                target_frame_context = None
           
                (
                    target_frames,
                    target_frames_anno,
                    segment_spoof_annos,
                    target_frame_context,
                    segment_run_records,
                ) = setup_segment_spoofs(
                    current_segment, target_frames_global, global_to_segment,
                    original_mf_indices, seg_localidx_to_pose, traces, args.seed,
                    args.variant_name, args.trace_file,
                    schedule_mode=schedule_mode, flicker_prob=flicker_prob
                )
            #get spoofs for current segment
            
            #build datase
            # frame_seq = global_to_segment[frame][1]
            # perturbation = 'global_position_fixed'
            pts = dataset_sf[mf_to_sf_global[frame]]['points']
            points = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=pts.dtype)])  # Add lag=0 column
            original_global_index = int(original_mf_indices[frame])
            frame_seed = int(args.seed + original_global_index)
            frame_rng = np.random.default_rng(frame_seed)
            JITTER_KEY = f"{args.seed}:{current_segment}:{original_global_index}"
            LAST_JITTER = (None, 0.0)
            (
                frame_points_spoof_rc,
                gt_box,
                frame_audit,
                point_audit,
                recovery,
                radial_rows,
            ) = spoof_frame(  # JITTER_KEY set just above
                target_frames[frame],
                points,
                target_frames_anno[frame],
                frame_rng,
                point_cloud_range=point_cloud_range,
                radial_edges=radial_edges,
                max_returns_per_ray=args.max_returns_per_ray,
                save_point_audit=args.save_point_audit,
            )
            frame_points_spoof_rc = frame_points_spoof_rc.astype(np.float32, copy=False)
            local_frame_index = global_to_segment[frame][1]
            frame_id = str(batch["frame_id"][0])
            context = target_frame_context.get(frame, {})
            frame_record = {
                "schema_version": args.schema_version,
                "variant_name": args.variant_name,
                "segment_id": segment,
                "frame_id": frame_id,
                # Keep global_frame_index backward-compatible with the
                # unfiltered full-validation index and expose the working index
                # separately.
                "global_frame_index": original_global_index,
                "original_global_frame_index": original_global_index,
                "filtered_global_frame_index": int(frame),
                "local_frame_index": int(local_frame_index),
                "frame_seed": frame_seed,
                "run_id": context.get("run_id"),
                "trace_id": context.get("trace_id"),
                "run_seed": context.get("run_seed"),
                "window_start_local": context.get("window_start_local"),
                "window_end_local": context.get("window_end_local"),
                "window_length": context.get("window_length"),
                "window_position": context.get("window_position"),
                "anchor_local_index": context.get("anchor_local_index"),
                "anchor_global_index": context.get("anchor_global_index"),
                "anchor_filtered_global_index": context.get(
                    "anchor_filtered_global_index"
                ),
            }
            frame_record.update(frame_audit)
            if gt_box is not None:
                frame_record.update({
                    "spoof_box_x": float(gt_box[0]),
                    "spoof_box_y": float(gt_box[1]),
                    "spoof_box_z": float(gt_box[2]),
                    "spoof_box_length": float(gt_box[3]),
                    "spoof_box_width": float(gt_box[4]),
                    "spoof_box_height": float(gt_box[5]),
                    "spoof_box_yaw": float(gt_box[6]),
                    "spoof_box_range": float(np.linalg.norm(gt_box[:2])),
                })
            segment_frame_records.append(frame_record)
            for radial_row in radial_rows:
                radial_row.update({
                    "schema_version": args.schema_version,
                    "variant_name": args.variant_name,
                    "segment_id": segment,
                    "frame_id": frame_id,
                    "global_frame_index": original_global_index,
                    "original_global_frame_index": original_global_index,
                    "filtered_global_frame_index": int(frame),
                    "local_frame_index": int(local_frame_index),
                    "run_id": context.get("run_id"),
                    "trace_id": context.get("trace_id"),
                })
                segment_radial_records.append(radial_row)
            recovery.update({
                "global_frame_index": original_global_index,
                "original_global_frame_index": original_global_index,
                "filtered_global_frame_index": int(frame),
                "local_frame_index": int(local_frame_index),
                "frame_id": frame_id,
            })
            segment_audit_array_records.append(recovery)

            for point_record in point_audit:
                point_record.update({
                    "schema_version": args.schema_version,
                    "variant_name": args.variant_name,
                    "segment_id": segment,
                    "frame_id": frame_id,
                    "global_frame_index": original_global_index,
                    "original_global_frame_index": original_global_index,
                    "filtered_global_frame_index": int(frame),
                    "local_frame_index": int(local_frame_index),
                    "run_id": context.get("run_id"),
                    "trace_id": context.get("trace_id"),
                })
                segment_point_records.append(point_record)

            dict_mf_mod = dataset_mf[frame].copy()
            for k in ['voxels', 'voxel_coords', 'voxel_num_points']:
                dict_mf_mod.pop(k, None)
            dict_mf_mod['points'] = frame_points_spoof_rc
            dict_mf_mod['pose'] = dict_mf_mod['poses'][:4]
            dict_mf_mod['spoof_gt'] = gt_box
            if JITTER_INNER > 0 or JITTER_INTER > 0:
                dict_mf_mod['spoof_gt_intended'] = LAST_JITTER[0]
                dict_mf_mod['jitter_inter'] = LAST_JITTER[1]
            dict_mf_mod['n_spoof_r'] = frame_audit["spoof_candidate_count"]
            dict_mf_mod['n_spoof_k'] = frame_audit["spoof_kept_count"]
            dict_mf_mod['spoof_candidate_count'] = frame_audit[
                "spoof_candidate_count"
            ]
            dict_mf_mod['spoof_kept_count'] = frame_audit["spoof_kept_count"]
            dict_mf_mod['attack_run_id'] = context.get("run_id")
            dict_mf_mod['final_spoof_start_index'] = frame_audit.get(
                "final_spoof_start_index"
            )
            dict_mf_mod['final_spoof_end_index'] = frame_audit.get(
                "final_spoof_end_index"
            )
            
            # print(frame_data)

            
            
            segment_dataset.append(dict_mf_mod)
            
    except Exception:
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        manifest.update({
            "status": "failed",
            "ended_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "wall_time_seconds": float(time.perf_counter() - script_start),
            "exception": traceback.format_exc(),
        })
        _atomic_write_json(manifest_path, manifest)
        raise  

    if segment_dataset and current_segment is not None:
        save_segment(
                    save_dir_dataset,
                    save_dir_annos,
                    current_segment,
                    segment_dataset,
                    segment_spoof_annos,
                    args.metrics_dir,
                    segment_frame_records,
                    segment_run_records,
                    segment_point_records,
                    segment_audit_array_records,
                    not args.no_save_audit_arrays,
                    args.save_point_audit,
                    segment_radial_records,
        )
        log.info('Saved: %s', current_segment)

    output_size_bytes = 0
    for output_root in (
        Path(args.save_dir_dataset),
        Path(args.save_dir_annos),
        metrics_dir,
    ):
        output_size_bytes += sum(
            path.stat().st_size
            for path in output_root.rglob("*")
            if path.is_file()
        )
    manifest.update(_aggregate_saved_metrics(
        metrics_dir,
        selected_segments,
    ))
    manifest.update({
        "status": "completed",
        "ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_time_seconds": float(time.perf_counter() - script_start),
        "output_size_bytes": int(output_size_bytes),
    })
    _atomic_write_json(manifest_path, manifest)

    return

    
    

# =============================================================================
# ABLATION EXTENSIONS
# =============================================================================
ABL = {}
_orig_parse_args = parse_args
_atlas_spoof_frame = spoof_frame
_orig_setup_segment_spoofs = setup_segment_spoofs


def parse_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    p.add_argument("--positioning", choices=("global", "relative"), default="global")
    p.add_argument("--spoof_source", choices=("atlas", "gt_paste"), default="atlas")
    p.add_argument("--object_mode", choices=("fixed", "per_frame"), default="fixed")
    p.add_argument("--position_mode", choices=("fixed", "per_frame"), default="fixed")
    p.add_argument("--object_bank", default=None,
                   help="val object bank pkl (build_val_object_bank.py) for gt_paste")
    p.add_argument("--target_class", choices=("Vehicle", "Cyclist"), default="Vehicle")
    p.add_argument("--paste_range_tol", type=float, default=2.0,
                   help="gt_paste: bank object's original range must be within this of the target range (m)")
    p.add_argument("--paste_min_points", type=int, default=5)
    p.add_argument("--data_path", default=None)
    known, rest = p.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest
    args = _orig_parse_args()
    ABL.update(vars(known))
    if known.spoof_source == "gt_paste":
        if not known.object_bank:
            raise ValueError("--spoof_source gt_paste requires --object_bank")
        with open(known.object_bank, "rb") as f:
            bank = pkl.load(f)
        bank = [o for o in bank if o["class"] == known.target_class
                and o["num_points"] >= known.paste_min_points]
        if not bank:
            raise ValueError(f"no {known.target_class} objects in {known.object_bank}")
        ABL["bank"] = bank
        ABL["bank_range"] = np.array([o["range"] for o in bank], dtype=np.float64)
        ABL["object_bank_sha256"] = _sha256(known.object_bank)
        ABL["object_bank_size"] = len(bank)
    return args


def _draw_placement(rng, n_traces):
    """Exactly the draw order of place_trace(): trace index, offset, yaw."""
    trace_idx = int(rng.integers(0, n_traces))
    off = np.array([rng.uniform(15, 20), rng.uniform(-3, 3)])
    yaw = float(rng.uniform(-0.2, 0.2))
    return trace_idx, off, yaw


def _atlas_object_ego(trace, off, yaw):
    """place_trace()'s ego-relative construction for one trace."""
    centered, cbox = center_trace(trace)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    xy = centered @ R.T + off
    pts = np.concatenate([xy, trace["points"][:, 2:3], trace["points"][:, 3:5]], axis=1)
    box = np.asarray(cbox, dtype=np.float64).copy()
    box[6] = yaw
    box[0] += off[0]
    box[1] += off[1]
    return pts, box


def _paste_object_ego(rng, off):
    """GT-sampling-style object: a val bank instance whose ORIGINAL range matches
    the target range, rotated about the ego z-axis to the target bearing, so its
    density and scan pattern stay those of a real object at that distance."""
    target_r = float(np.hypot(off[0], off[1]))
    target_b = float(np.arctan2(off[1], off[0]))
    tol = ABL["paste_range_tol"]
    cand = np.where(np.abs(ABL["bank_range"] - target_r) <= tol)[0]
    while cand.size == 0:
        tol *= 2.0
        cand = np.where(np.abs(ABL["bank_range"] - target_r) <= tol)[0]
    idx = int(cand[rng.integers(0, cand.size)])
    o = ABL["bank"][idx]
    box = np.asarray(o["box"], dtype=np.float64).copy()
    d = target_b - float(np.arctan2(box[1], box[0]))
    c, s = np.cos(d), np.sin(d)
    R = np.array([[c, -s], [s, c]])
    pts = np.asarray(o["points"], dtype=np.float64).copy()
    pts[:, :2] = pts[:, :2] @ R.T
    box[:2] = box[:2] @ R.T
    box[6] = box[6] + d
    return pts, box, idx


def _to_world(pts, box, anchor_pose):
    """place_trace()'s anchor-ego -> world conversion."""
    w = pts.copy()
    w[:, :3] = (pts[:, :3] @ anchor_pose[:3, :3].T) + anchor_pose[:3, 3]
    b = box.copy()
    b[:3] = (box[:3] @ anchor_pose[:3, :3].T) + anchor_pose[:3, 3]
    b[6] = b[6] + np.arctan2(anchor_pose[1, 0], anchor_pose[0, 0])
    return w, b


def setup_segment_spoofs(segment, target_frames_global, global_to_segment,
                         original_mf_indices, seg_localidx_to_pose, traces, seed,
                         variant_name, trace_file,
                         schedule_mode="continuous", flicker_prob=None):
    baseline = (ABL.get("positioning", "global") == "global"
                and ABL.get("spoof_source", "atlas") == "atlas"
                and ABL.get("object_mode", "fixed") == "fixed"
                and ABL.get("position_mode", "fixed") == "fixed")
    if not baseline and INTENSITY_MODE == "calibrated":
        raise NotImplementedError("calibrated intensity is implemented for the baseline ATLAS global path only")
    if baseline:   # byte-identical to the original generator
        return _orig_setup_segment_spoofs(
            segment, target_frames_global, global_to_segment, original_mf_indices,
            seg_localidx_to_pose, traces, seed, variant_name, trace_file,
            schedule_mode=schedule_mode, flicker_prob=flicker_prob)

    target_frames = target_frames_global[segment].copy()
    target_frames_anno = target_frames_global[segment].copy()
    segment_spoof_annos, segment_run_records, target_frame_context = [], [], {}
    for run in extract_one_runs(target_frames):
        subset_g = {k: target_frames[k] for k in run}
        subset_g_anno = {k: target_frames_anno[k] for k in run}
        window_min = global_to_segment[min(subset_g)][1]
        window_max = global_to_segment[max(subset_g)][1]
        anchor_filtered_global_idx = max(subset_g.keys())
        anchor_original_global_idx = int(original_mf_indices[anchor_filtered_global_idx])
        anchor_local_idx = global_to_segment[anchor_filtered_global_idx][1]
        anchor_pose = seg_localidx_to_pose[(segment, anchor_local_idx)]
        run_start_original_global_idx = int(original_mf_indices[min(subset_g)])
        run_seed = int(seed + run_start_original_global_idx)

        # Same annotation/run-record schema (and same placement draws) as the original.
        _, spoof_annotation, _, run_record = place_trace(
            traces, "global_position_fixed", window_min, window_max, run_seed,
            anchor_pose=anchor_pose, variant_name=variant_name, segment_name=segment,
            anchor_local_index=anchor_local_idx,
            anchor_global_index=anchor_original_global_idx, trace_file=trace_file)
        trace_idx, off, yaw = _draw_placement(np.random.default_rng(run_seed), len(traces))
        assert trace_idx == run_record["trace_id"], "placement stream diverged"

        flicker_rng = None
        if schedule_mode == "flicker":
            flicker_rng = np.random.default_rng(
                _stable_seed(seed, segment, run_start_original_global_idx, "flicker"))

        used_objects = []
        for key in sorted(subset_g.keys()):
            if flicker_rng is not None and not (flicker_rng.random() < flicker_prob):
                subset_g[key] = None
                subset_g_anno[key] = None
                continue
            target_segment, target_local_idx = global_to_segment[key]
            target_pose = seg_localidx_to_pose[(target_segment, target_local_idx)]

            f_off, f_yaw = off, yaw
            if ABL["position_mode"] == "per_frame":
                prng = np.random.default_rng(_stable_seed(
                    seed, segment, run_start_original_global_idx, "position", target_local_idx))
                f_off = np.array([prng.uniform(15, 20), prng.uniform(-3, 3)])
                f_yaw = float(prng.uniform(-0.2, 0.2))
            obj_key = target_local_idx if ABL["object_mode"] == "per_frame" else -1
            orng = np.random.default_rng(_stable_seed(
                seed, segment, run_start_original_global_idx, "object", obj_key))

            if ABL["spoof_source"] == "atlas":
                tidx = int(orng.integers(0, len(traces))) if ABL["object_mode"] == "per_frame" else trace_idx
                obj_pts, obj_box = _atlas_object_ego(traces[tidx], f_off, f_yaw)
                obj_id = tidx
            else:
                obj_pts, obj_box, obj_id = _paste_object_ego(orng, f_off)
            used_objects.append(int(obj_id))

            if ABL["positioning"] == "global":
                w_pts, w_box = _to_world(obj_pts, obj_box, anchor_pose)
                spoof_points_ego, box_ego = transform_spoof_to_ego_frame(w_pts, w_box, target_pose)
            else:
                spoof_points_ego, box_ego = obj_pts.copy(), obj_box.copy()

            if INTENSITY_MODE in ("saturated", "attenuated") and ABL["spoof_source"] == "atlas":
                _constant_spoof_features(spoof_points_ego)   # same constant as the baseline path
            subset_g[key] = spoof_points_ego
            subset_g_anno[key] = box_ego.copy()
            target_frame_context[key] = {
                "run_id": run_record["run_id"],
                "trace_id": int(obj_id),
                "run_seed": run_seed,
                "window_start_local": window_min,
                "window_end_local": window_max,
                "window_length": window_max - window_min + 1,
                "window_position": target_local_idx - window_min,
                "anchor_local_index": anchor_local_idx,
                "anchor_global_index": anchor_original_global_idx,
                "anchor_filtered_global_index": anchor_filtered_global_idx,
            }
        ablation_fields = {
            "positioning": ABL["positioning"], "spoof_source": ABL["spoof_source"],
            "object_mode": ABL["object_mode"], "position_mode": ABL["position_mode"],
            "target_class": ABL["target_class"],
            "distinct_objects_in_run": len(set(used_objects)),
            "object_ids_in_run": used_objects,
        }
        spoof_annotation.update(ablation_fields)
        run_record.update({k: v for k, v in ablation_fields.items() if k != "object_ids_in_run"})
        segment_spoof_annos.append(spoof_annotation)
        segment_run_records.append(run_record)
        target_frames.update(subset_g)
        target_frames_anno.update(subset_g_anno)
    return (target_frames, target_frames_anno, segment_spoof_annos,
            target_frame_context, segment_run_records)


def _points_in_box(xyz, box):
    c, s = np.cos(-box[6]), np.sin(-box[6])
    local = xyz - np.asarray(box[:3], dtype=np.float64)
    lx = local[:, 0] * c - local[:, 1] * s
    ly = local[:, 0] * s + local[:, 1] * c
    return ((np.abs(lx) <= box[3] / 2) & (np.abs(ly) <= box[4] / 2)
            & (np.abs(local[:, 2]) <= box[5] / 2))


def _paste_frame(spoof_points, points, gt_box, rng, point_cloud_range, radial_edges,
                 max_returns_per_ray=2, save_point_audit=False):
    """OpenPCDet GT-sampling insertion: drop scene points inside the object box
    (REMOVE_EXTRA_WIDTH = 0 for Waymo), append ALL object points unchanged
    (original intensity/elongation, no ray/occlusion filtering, no ground
    anchoring). Returns the same 6-tuple schema as spoof_frame()."""
    _, _, audit, _, recovery, _ = _atlas_spoof_frame(
        None, points, gt_box, rng, point_cloud_range, radial_edges,
        max_returns_per_ray=max_returns_per_ray, save_point_audit=False)
    sp = np.asarray(spoof_points, dtype=np.float32)
    inside = _points_in_box(points[:, :3].astype(np.float64), gt_box)
    removed_idx = np.nonzero(inside)[0]
    scene = points[~inside]
    lag_col = np.full((sp.shape[0], 1), points[0, -1], dtype=np.float32)
    sp_lag = np.hstack([sp[:, :points.shape[1] - 1], lag_col]).astype(points.dtype)
    final = np.concatenate([scene, sp_lag], axis=0)
    n = int(sp.shape[0])
    audit.update({
        "attack_active": True,
        "spoof_candidate_count": n,
        "spoof_kept_count": n,
        "spoof_discarded_count": 0,
        "background_removed_event_count": int(removed_idx.size),
        "background_removed_unique_count": int(removed_idx.size),
        "final_point_count": int(final.shape[0]),
        "count_balance_error": int(final.shape[0] - (points.shape[0] - removed_idx.size + n)),
        "retention_rate": 1.0 if n else None,
        "final_spoof_start_index": int(scene.shape[0]),
        "final_spoof_end_index": int(final.shape[0]),
        "insertion_mode": "gt_paste",
    })
    recovery.update({
        "candidate_points": sp,
        "candidate_kept": np.ones(n, dtype=bool),
        "candidate_outcome_code": np.full(n, -1, dtype=np.int8),
        "candidate_beam_index": np.full(n, -1, dtype=np.int16),
        "candidate_azimuth_bin": np.full(n, -1, dtype=np.int32),
        "candidate_ray_key": np.full(n, -1, dtype=np.int64),
        "candidate_azimuth_residual_deg": np.full(n, np.nan, dtype=np.float32),
        "candidate_elevation_residual_deg": np.full(n, np.nan, dtype=np.float32),
        "candidate_scene_returns_on_ray": np.full(n, -1, dtype=np.int32),
        "candidate_nearest_scene_range": np.full(n, np.nan, dtype=np.float32),
        "candidate_scene_points_removed": np.zeros(n, dtype=np.int32),
        "candidate_minimum_removed_scene_range": np.full(n, np.nan, dtype=np.float32),
        "candidate_occlusion_order_violation": np.zeros(n, dtype=bool),
        "removed_scene_indices": removed_idx.astype(np.int64),
        "removed_scene_points": points[removed_idx].astype(np.float32),
    })
    return final, gt_box, audit, [], recovery, []


def spoof_frame(spoof_points, points, gt_box, rng, point_cloud_range, radial_edges,
                max_returns_per_ray=2, save_point_audit=False):
    if spoof_points is None or ABL.get("spoof_source", "atlas") == "atlas":
        return _atlas_spoof_frame(spoof_points, points, gt_box, rng,
                                  point_cloud_range=point_cloud_range,
                                  radial_edges=radial_edges,
                                  max_returns_per_ray=max_returns_per_ray,
                                  save_point_audit=save_point_audit)
    return _paste_frame(spoof_points, points, gt_box, rng, point_cloud_range,
                        radial_edges, max_returns_per_ray, save_point_audit)


if __name__ == "__main__":
    main()
