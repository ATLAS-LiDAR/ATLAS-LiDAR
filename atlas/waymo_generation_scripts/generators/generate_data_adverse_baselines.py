#!/usr/bin/env python
"""
ATLAS adverse-condition point-removal baselines (three modes, one script).

These baselines are non-adversarial point-loss controls for the ATLAS
point-removal attack. They share ATLAS's temporal schedule, target
selection, tracking, and (for matched mode) firing logic, but the
corruption itself is always applied SCENE-WIDE and unstructured. Matched
mode isolates ATLAS's spatial targeting at the same realized point budget;
uniform and ring modes provide fixed-severity contextual controls under the
same temporal and target-valid opportunity.

Three modes (select with --mode):

  uniform   Random scene-wide point dropout.
            Argument: --drop_prob P            (e.g. 0.50 or 0.90)
            Every point is deleted independently with probability P on
            each scheduled-on frame while the target stays trackable.

  ring      Random elevation-layer dropout.
            Argument: --n_rings N              (e.g. 16 or 32)
            N elevation layers are selected from one deterministic random
            ordering per 32-frame corrupted block. Ring-16 is therefore a
            subset of Ring-32 for the same seed and block. All points assigned
            to those layers are deleted while the target stays trackable.
            Without explicit sensor/channel IDs this is an elevation-layer
            approximation, not a claim of exact hardware beam identity.

  matched   Matched-count uniform removal paired to ATLAS.
            Argument: --az_width W             (same as ATLAS, e.g. 50)
            The azimuth width sets the ATLAS point-removal rate (PRR).
            We run ATLAS's sector construction, firing gate, and per-point
            removal draw ONLY to count K_t = how many points ATLAS would
            delete this frame. We then delete exactly K_t points sampled
            uniformly at random from the WHOLE scene -- not the sector.
            Firing opportunity, temporal support, and realized point budget
            match ATLAS; only spatial placement changes.

All modes share:
  * the 32-frame alternating clean/corrupt schedule (--attack_block_length),
  * ATLAS target selection + velocity tracking (one vehicle per run),
  * the SAME distance-dependent A-HFR firing gate as ATLAS, evaluated on the
    tracked target's range (removal.ahfr_gate_prob). Corruption is applied
    only when schedule-on AND target trackable AND the attack fires, so the
    baselines attack on the same frames ATLAS does. This is not a weather
    model -- it exists purely for an apples-to-apples temporal comparison.
    With --atlas_firing_reference the realized firing decisions (and matched
    K) are replayed from a paired ATLAS run for exact frame pairing.
  * a deterministic per-(segment, frame, vehicle) target manifest,
  * per-frame / per-run / radial / removed-point CSVs, audit-index arrays,
    a run manifest, SHA-256 hashing, and per-segment resume.

Output layout matches the ATLAS removal rebuttal generator so downstream
notebooks can join on frame_id / run_id across attack and baseline.
"""

import argparse
import csv
import hashlib
import importlib
import json
import logging
import os
import pickle as pkl
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

import removal as removal
import adverse_baselines as baselines

importlib.reload(removal)
importlib.reload(baselines)


P_REMOVE_BY_AZ_WIDTH = {
    10: 0.97,
    20: 1.00,
    30: 0.99,
    40: 0.98,
    50: 0.90,
    60: 0.90,
}


# ==========================================================================
# Arguments
# ==========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ATLAS adverse-condition removal baselines."
    )
    parser.add_argument("--cp_mf_root", required=True)
    parser.add_argument("--cp_mf_cfg", required=True)
    parser.add_argument("--cp_sf_cfg", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--save_dir_dataset", required=True)
    parser.add_argument("--save_dir_annos", required=True)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--seed", type=int, required=True)

    parser.add_argument(
        "--mode",
        required=True,
        choices=("uniform", "ring", "matched"),
        help="Which baseline corruption to generate.",
    )

    # --- mode-specific arguments (validated in main by mode) ---
    parser.add_argument(
        "--drop_prob",
        type=float,
        default=None,
        help="uniform mode: per-point deletion probability, e.g. 0.50.",
    )
    parser.add_argument(
        "--n_rings",
        type=int,
        default=None,
        help="ring mode: number of elevation rings to delete, e.g. 16.",
    )
    parser.add_argument(
        "--az_width",
        type=int,
        default=None,
        choices=tuple(sorted(P_REMOVE_BY_AZ_WIDTH)),
        help="matched mode: ATLAS azimuth width; sets the matched PRR.",
    )

    # --- ring binning source ---
    parser.add_argument(
        "--inclination_file",
        default=None,
        help=(
            "ring mode: optional .npy of Waymo top-LiDAR beam inclinations "
            "(radians). If given, layer boundaries align with the nominal "
            "inclinations; exact beam identity still requires channel IDs. "
            "Otherwise a uniform elevation partition is used."
        ),
    )
    parser.add_argument(
        "--n_elevation_bins",
        type=int,
        default=64,
        help="ring mode: ring count when no inclination file is supplied.",
    )

    # --- firing gate pairing ---
    parser.add_argument(
        "--atlas_firing_reference",
        default=None,
        help=(
            "Optional metrics_dir of a paired ATLAS removal run. When given, "
            "the realized per-frame firing decision (active_this_frame) is "
            "replayed exactly, and for matched mode the realized removal "
            "count (n_points_removed) is used as K. This is the plan-faithful "
            "'same realized firing decision as ATLAS' path. When omitted, the "
            "firing gate is drawn stochastically with the same distance-vs-"
            "probability table, shared across all baseline modes so they fire "
            "on identical frames as each other."
        ),
    )
    parser.add_argument(
        "--pairing_box_atol",
        type=float,
        default=1e-5,
        help=(
            "Absolute tolerance used when verifying that the independently "
            "tracked baseline target box matches the paired ATLAS target box."
        ),
    )

    # --- sequence selection (identical semantics to ATLAS generator) ---
    parser.add_argument("--num_sequences", type=int, default=None)
    parser.add_argument(
        "--sequence_selection",
        choices=("random", "first"),
        default="random",
    )

    parser.add_argument(
        "--schema_version", default="atlas_baseline_v2"
    )
    parser.add_argument("--attack_block_length", type=int, default=32)

    # --- ATLAS-shared tracking / sector parameters (matched needs these) ---
    parser.add_argument("--el_width_deg", type=float, default=16.0)
    parser.add_argument("--max_az_step_deg", type=float, default=15.0)
    parser.add_argument("--center_jitter_std_deg", type=float, default=0.05)
    parser.add_argument("--min_target_range_m", type=float, default=6.0)
    parser.add_argument("--desired_target_range_m", type=float, default=60.0)
    parser.add_argument("--tracking_max_step_m", type=float, default=6.0)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--front_targets_only", action="store_true")
    parser.add_argument(
        "--allow_close_range_extrapolation", action="store_true"
    )

    # --- diagnostics ---
    parser.add_argument("--radial_bin_width_m", type=float, default=1.0)
    parser.add_argument("--radial_max_m", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data_path", default=None, help="override DATA_CONFIG.DATA_PATH of both cfgs")
    parser.add_argument(
        "--save_point_audit",
        action="store_true",
        help="Also save one CSV row per removed point; potentially large.",
    )
    parser.add_argument("--no_save_audit_arrays", action="store_true")
    return parser.parse_args()


# ==========================================================================
# Small shared utilities (mirrors the ATLAS rebuttal generator)
# ==========================================================================

def unload_pcdet():
    for key in list(sys.modules):
        if key.startswith("pcdet"):
            del sys.modules[key]


def _python_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_div(num, den):
    if den == 0:
        return None
    return float(num) / float(den)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(master_seed, *parts):
    payload = ":".join([str(master_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True,
                  default=_python_scalar)
    os.replace(temp, path)


def _atomic_write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", newline="", encoding="utf-8") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {key: _python_scalar(row.get(key)) for key in fields}
                )
    os.replace(temp, path)


def _concat_with_offsets(records, key, dtype):
    arrays = [np.asarray(record[key], dtype=dtype) for record in records]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for index, array in enumerate(arrays):
        offsets[index + 1] = offsets[index] + len(array)
    if not arrays or offsets[-1] == 0:
        return np.empty((0,), dtype=dtype), offsets
    return np.concatenate(arrays, axis=0), offsets


def _atomic_write_audit_arrays(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp.npz")
    removed_indices, removed_offsets = _concat_with_offsets(
        records, "removed_indices", np.int64
    )
    target_indices, target_offsets = _concat_with_offsets(
        records, "target_indices", np.int64
    )
    removed_points = [
        np.asarray(record["removed_points"], dtype=np.float32)
        for record in records
    ]
    point_offsets = np.zeros(len(removed_points) + 1, dtype=np.int64)
    for index, array in enumerate(removed_points):
        point_offsets[index + 1] = point_offsets[index] + len(array)
    if removed_points and point_offsets[-1]:
        removed_points_concat = np.concatenate(removed_points, axis=0)
    else:
        removed_points_concat = np.empty((0, 6), dtype=np.float32)
    np.savez_compressed(
        temp,
        frame_id=np.asarray(
            [str(record["frame_id"]) for record in records], dtype="U128"
        ),
        local_frame_index=np.asarray(
            [record["local_frame_index"] for record in records],
            dtype=np.int64,
        ),
        original_global_frame_index=np.asarray(
            [record["original_global_frame_index"] for record in records],
            dtype=np.int64,
        ),
        removed_indices=removed_indices,
        removed_indices_offsets=removed_offsets,
        target_indices=target_indices,
        target_indices_offsets=target_offsets,
        removed_points=removed_points_concat,
        removed_points_offsets=point_offsets,
    )
    os.replace(temp, path)


def _cart_to_spherical(points_xyz):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    planar = np.hypot(x, y)
    radius = np.sqrt(planar ** 2 + z ** 2)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.maximum(planar, 1e-12))
    return radius, azimuth, elevation


def _points_in_oriented_box(points_xyz, box):
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)
    delta = points_xyz[:, :3] - box[None, :3]
    cosine = np.cos(box[6])
    sine = np.sin(box[6])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    local_z = delta[:, 2]
    epsilon = 1e-6
    return (
        (np.abs(local_x) <= box[3] / 2.0 + epsilon)
        & (np.abs(local_y) <= box[4] / 2.0 + epsilon)
        & (np.abs(local_z) <= box[5] / 2.0 + epsilon)
    )


# ==========================================================================
# ATLAS firing-reference loader (optional exact pairing)
# ==========================================================================

def _optional_bool(value):
    if value in (None, "", "None"):
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _optional_int(value):
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _optional_float(value):
    if value in (None, "", "None"):
        return None
    return float(value)


def _target_box_from_reference_row(row):
    names = (
        "target_box_x", "target_box_y", "target_box_z",
        "target_box_length", "target_box_width", "target_box_height",
        "target_box_yaw",
    )
    values = [_optional_float(row.get(name)) for name in names]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        missing = [name for name, value in zip(names, values) if value is None]
        raise ValueError(
            "ATLAS reference has an incomplete target box; missing "
            + ", ".join(missing)
        )
    return np.asarray(values, dtype=np.float64)


def load_atlas_reference_manifest(atlas_metrics_dir):
    manifest_path = Path(atlas_metrics_dir) / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Exact ATLAS pairing requires run_manifest.json in the reference "
            f"metrics directory: {manifest_path}"
        )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return manifest, manifest_path


def _values_equal(expected, actual):
    if isinstance(expected, float) or isinstance(actual, float):
        return bool(np.isclose(float(expected), float(actual), rtol=0.0, atol=1e-12))
    return expected == actual


def validate_atlas_reference_manifest(manifest, args, selected_segments):
    """Fail early if the paired ATLAS run used a different evaluation setup."""
    mismatches = []

    if manifest.get("generator_name") != "generate_data_removal_rebuttal":
        mismatches.append(
            "generator_name: expected 'generate_data_removal_rebuttal', got "
            f"{manifest.get('generator_name')!r}"
        )

    top_level = {
        "master_seed": int(args.seed),
        "sequence_selection": args.sequence_selection,
        "requested_num_sequences": args.num_sequences,
    }
    for key, expected in top_level.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"{key}: ATLAS={actual!r}, baseline={expected!r}")

    actual_segments = manifest.get("selected_sequences")
    if actual_segments is None:
        mismatches.append("selected_sequences missing from ATLAS manifest")
    elif list(actual_segments) != list(selected_segments):
        mismatches.append("selected_sequences differ from the baseline selection")

    attack_config = manifest.get("attack_configuration", {})
    if attack_config.get("mode") != "ahfr_point_removal":
        mismatches.append(
            "attack_configuration.mode: expected 'ahfr_point_removal', got "
            f"{attack_config.get('mode')!r}"
        )
    expected_attack_config = {
        "attack_block_length": int(args.attack_block_length),
        "desired_target_range_m": float(args.desired_target_range_m),
        "tracking_max_step_m": float(args.tracking_max_step_m),
        "target_class": int(args.target_class),
        "front_targets_only": bool(args.front_targets_only),
        "min_target_range_m": float(args.min_target_range_m),
        "allow_close_range_extrapolation": bool(
            args.allow_close_range_extrapolation
        ),
    }
    if args.mode == "matched":
        expected_attack_config.update({
            "az_width_deg": int(args.az_width),
            "el_width_deg": float(args.el_width_deg),
            "max_az_step_deg": float(args.max_az_step_deg),
            "center_jitter_std_deg": float(args.center_jitter_std_deg),
        })
    for key, expected in expected_attack_config.items():
        actual = attack_config.get(key)
        if actual is None or not _values_equal(expected, actual):
            mismatches.append(
                f"attack_configuration.{key}: "
                f"ATLAS={actual!r}, baseline={expected!r}"
            )

    if manifest.get("status") != "completed":
        mismatches.append(
            f"ATLAS reference status is {manifest.get('status')!r}, not 'completed'"
        )

    if mismatches:
        raise ValueError(
            "ATLAS reference is not exactly paired with this baseline run:\n  - "
            + "\n  - ".join(mismatches)
        )


def validate_atlas_reference_frame(
    ref,
    *,
    segment,
    local_index,
    tracking_record,
    target_box,
    n_points_before=None,
    box_atol=1e-5,
):
    """Verify frame identity, target identity/box, tracking, and point count."""
    errors = []
    if ref.get("segment_id") != segment:
        errors.append(
            f"segment_id ATLAS={ref.get('segment_id')!r}, baseline={segment!r}"
        )
    if ref.get("local_frame_index") != int(local_index):
        errors.append(
            "local_frame_index "
            f"ATLAS={ref.get('local_frame_index')!r}, baseline={local_index!r}"
        )

    if ref.get("attack_scheduled") is not True:
        errors.append(
            f"attack_scheduled ATLAS={ref.get('attack_scheduled')!r}, expected True"
        )

    n_removed = int(ref.get("n_removed", 0))
    if n_removed < 0:
        errors.append(f"negative ATLAS removal count: {n_removed}")
    if not bool(ref.get("fired", False)) and n_removed != 0:
        errors.append(
            f"ATLAS non-firing frame has n_removed={n_removed}, expected 0"
        )

    baseline_tracking_ok = target_box is not None
    if ref.get("tracking_ok") is None:
        errors.append("tracking_ok missing from ATLAS frame record")
    elif bool(ref["tracking_ok"]) != baseline_tracking_ok:
        errors.append(
            f"tracking_ok ATLAS={ref['tracking_ok']!r}, "
            f"baseline={baseline_tracking_ok!r}"
        )

    if baseline_tracking_ok:
        baseline_target_index = tracking_record.get("target_box_index")
        if ref.get("target_box_index") != baseline_target_index:
            errors.append(
                "target_box_index "
                f"ATLAS={ref.get('target_box_index')!r}, "
                f"baseline={baseline_target_index!r}"
            )
        ref_box = ref.get("target_box")
        if ref_box is None:
            errors.append("target box missing from ATLAS frame record")
        elif not np.allclose(
            np.asarray(ref_box, dtype=np.float64),
            np.asarray(target_box[:7], dtype=np.float64),
            rtol=0.0,
            atol=box_atol,
        ):
            errors.append(
                "target box differs: "
                f"ATLAS={np.asarray(ref_box).tolist()}, "
                f"baseline={np.asarray(target_box[:7]).tolist()}"
            )
    else:
        if bool(ref.get("fired", False)) or int(ref.get("n_removed", 0)) != 0:
            errors.append(
                "ATLAS reference marks a tracking-lost frame as active or modified"
            )

    if n_points_before is not None:
        ref_count = ref.get("n_points_before")
        if ref_count is None:
            errors.append("n_points_before missing from ATLAS frame record")
        elif int(ref_count) != int(n_points_before):
            errors.append(
                f"n_points_before ATLAS={ref_count}, baseline={n_points_before}"
            )

    if errors:
        raise ValueError("ATLAS frame-pairing validation failed: " + "; ".join(errors))

def load_atlas_firing_reference(atlas_metrics_dir):
    """Load strict per-frame metadata from a paired ATLAS removal run."""
    frames_dir = Path(atlas_metrics_dir) / "frames"
    if not frames_dir.is_dir():
        raise FileNotFoundError(
            f"ATLAS firing reference has no frames/ dir: {frames_dir}"
        )
    csv_paths = sorted(frames_dir.glob("*_frames.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"ATLAS firing reference contains no *_frames.csv files: {frames_dir}"
        )

    reference = {}
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame_id = row.get("frame_id")
                if not frame_id:
                    raise ValueError(f"Missing frame_id in {csv_path}")
                frame_id = str(frame_id)
                if frame_id in reference:
                    raise ValueError(
                        f"Duplicate frame_id {frame_id!r} in ATLAS reference"
                    )

                removed_raw = row.get("n_points_removed")
                if removed_raw in (None, "", "None"):
                    removed_raw = row.get("n_points_removed_total")
                n_removed = _optional_int(removed_raw)
                if n_removed is None:
                    n_removed = 0

                reference[frame_id] = {
                    "segment_id": row.get("segment_id"),
                    "local_frame_index": _optional_int(
                        row.get("local_frame_index")
                    ),
                    "target_box_index": _optional_int(
                        row.get("target_box_index")
                    ),
                    "tracking_ok": _optional_bool(row.get("tracking_ok")),
                    "attack_scheduled": _optional_bool(
                        row.get("attack_scheduled")
                    ),
                    "fired": bool(
                        _optional_bool(row.get("active_this_frame")) or False
                    ),
                    "n_removed": int(n_removed),
                    "n_points_before": _optional_int(
                        row.get("n_points_before")
                    ),
                    "target_box": _target_box_from_reference_row(row),
                    "source_csv": str(csv_path),
                }
    return reference


# ==========================================================================
# Per-frame corruption + audit
# ==========================================================================

def _radial_rows(points, remove_mask, target_mask, target_box, radial_edges):
    """
    Scene-wide + target-restricted removal density per ego-centred annulus.

    Unlike the ATLAS radial table there is no sector column, because these
    baselines have no sector. We keep the scene and target columns so the
    "how much of the target vs background is lost per range" comparison is
    still available frame-by-frame.
    """
    planar_radius = np.linalg.norm(points[:, :2], axis=1)
    target_planar_range = float(np.linalg.norm(np.asarray(target_box[:2])))
    rows = []
    for index in range(len(radial_edges) - 1):
        lower = float(radial_edges[index])
        upper = float(radial_edges[index + 1])
        if index == len(radial_edges) - 2:
            radial_mask = (planar_radius >= lower) & (planar_radius <= upper)
        else:
            radial_mask = (planar_radius >= lower) & (planar_radius < upper)
        removed_radial = radial_mask & remove_mask
        target_radial = radial_mask & target_mask
        target_removed = target_radial & remove_mask
        annular_area = np.pi * (upper ** 2 - lower ** 2)
        before_count = int(np.sum(radial_mask))
        removed_count = int(np.sum(removed_radial))
        target_before = int(np.sum(target_radial))
        target_removed_count = int(np.sum(target_removed))
        if upper <= target_planar_range:
            relation = "in_front_of_target"
        elif lower > target_planar_range:
            relation = "behind_target"
        else:
            relation = "contains_target_center"
        rows.append({
            "radial_bin_index": int(index),
            "radius_min_m": lower,
            "radius_max_m": upper,
            "annular_area_m2": float(annular_area),
            "target_planar_range_m": target_planar_range,
            "radial_relation_to_target": relation,
            "scene_count_before": before_count,
            "scene_count_removed": removed_count,
            "scene_count_after": before_count - removed_count,
            "scene_density_before_points_per_m2": _safe_div(
                before_count, annular_area
            ),
            "scene_density_removed_points_per_m2": _safe_div(
                removed_count, annular_area
            ),
            "scene_removal_fraction": _safe_div(removed_count, before_count),
            "target_count_before": target_before,
            "target_count_removed": target_removed_count,
            "target_count_after": target_before - target_removed_count,
            "target_removal_fraction": _safe_div(
                target_removed_count, target_before
            ),
            "scene_count_identity_error": int(
                (before_count - removed_count)
                - int(np.sum(radial_mask & ~remove_mask))
            ),
        })
    return rows


def audited_baseline_frame(
    points,
    target_box,
    args,
    mode,
    state,
    rng,
    firing_rng,
    ring_context,
    radial_edges,
    p_remove,
    firing_override=None,
    atlas_k=None,
):
    """
    Apply one baseline corruption frame and return the same audit shape the
    ATLAS removal generator produces (minus sector-only fields).

    The firing gate is computed first and shared across all modes so the
    baselines attack on the same frames as ATLAS (and as each other):
      * ``firing_override`` (bool) replays a paired ATLAS run's realized
        decision when a reference is supplied; otherwise the gate is drawn
        stochastically with ATLAS's distance-vs-probability table using the
        mode-independent ``firing_rng``.
      * On a non-firing frame the point cloud is returned untouched, exactly
        as ATLAS does, so no unmatched mAP degradation is introduced.

    ``ring_context`` carries the precomputed per-block ring choice + the
    elevation edges, used only in ring mode. ``atlas_k`` (matched + reference)
    is ATLAS's realized removal count for this frame.
    """
    points = np.asarray(points)
    target_box = np.asarray(target_box).copy()
    target_mask = _points_in_oriented_box(points[:, :3], target_box)
    target_range_3d = float(np.linalg.norm(target_box[:3]))

    new_state = state

    # ---- shared firing gate (same distance-vs-probability as ATLAS) ----
    if firing_override is not None:
        fired = bool(firing_override)
        gate_meta = {
            "scan_gate_probability": float(
                removal.ahfr_gate_prob(target_range_3d)
            ),
            "scan_gate_draw": None,
            "scan_gate_passed": fired,
            "range_gate_passed": True,
            "inactive_reason": None if fired else "scan_gate",
            "firing_source": "atlas_reference",
        }
    else:
        fired, gate_meta = baselines.compute_firing_gate(
            target_range_3d,
            firing_rng,
            min_target_range_m=args.min_target_range_m,
            allow_close_range_extrapolation=(
                args.allow_close_range_extrapolation
            ),
        )
        gate_meta["firing_source"] = "stochastic_gate"

    mode_info = {}

    if mode == "matched":
        # matched advances its aim state every tracked frame; it removes
        # points only when fired.
        keep_mask, remove_mask, new_state, mode_info = baselines.matched_frame(
            points=points,
            target_box=target_box,
            state=state,
            fired=fired,
            rng=rng,
            p_remove=p_remove,
            az_width_deg=args.az_width,
            el_width_deg=args.el_width_deg,
            max_az_step_deg=args.max_az_step_deg,
            center_jitter_std_deg=args.center_jitter_std_deg,
            atlas_k=atlas_k,
        )
    elif not fired:
        # uniform / ring on a non-firing frame: leave the cloud untouched.
        keep_mask = np.ones(len(points), dtype=bool)
        remove_mask = np.zeros(len(points), dtype=bool)
        mode_info = {"mode": mode}
    elif mode == "uniform":
        keep_mask, remove_mask, mode_info = baselines.uniform_dropout(
            points, args.drop_prob, rng
        )
    elif mode == "ring":
        ring_index = baselines.assign_ring_index(
            points[:, :3], ring_context["elevation_edges"]
        )
        keep_mask, remove_mask, mode_info = baselines.ring_dropout(
            points, ring_index, ring_context["chosen_rings"]
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    attacked_points = points[keep_mask].copy()
    removed_indices = np.flatnonzero(remove_mask)
    target_indices = np.flatnonzero(target_mask)

    number_before = int(len(points))
    number_after = int(len(attacked_points))
    number_removed = int(np.sum(remove_mask))
    number_target = int(np.sum(target_mask))
    number_target_removed = int(np.sum(target_mask & remove_mask))
    number_background_removed = int(np.sum((~target_mask) & remove_mask))

    audit = {
        "corruption_mode": mode,
        "attack_scheduled": True,
        "tracking_ok": True,
        # active = the attack FIRED this frame (matches ATLAS semantics);
        # modified = points were actually removed (empty sector / K=0 -> False).
        "active_this_frame": bool(fired),
        "modified_this_frame": bool(number_removed > 0),
        "inactive_reason": gate_meta.get("inactive_reason"),
        "target_box_x": float(target_box[0]),
        "target_box_y": float(target_box[1]),
        "target_box_z": float(target_box[2]),
        "target_box_length": float(target_box[3]),
        "target_box_width": float(target_box[4]),
        "target_box_height": float(target_box[5]),
        "target_box_yaw": float(target_box[6]),
        "target_range_3d_m": target_range_3d,
        "target_range_m": target_range_3d,           # compatibility alias
        "n_points_before": number_before,
        "n_points_after": number_after,
        "n_points_removed": number_removed,
        "n_points_removed_total": number_removed,
        "n_target_points_before": number_target,
        "n_target_points_removed": number_target_removed,
        "n_target_points_after": number_target - number_target_removed,
        "target_removal_rate": _safe_div(number_target_removed, number_target),
        "n_background_points_removed": number_background_removed,
        "removed_points_target_fraction": _safe_div(
            number_target_removed, number_removed
        ),
        "collateral_removal_fraction": _safe_div(
            number_background_removed, number_removed
        ),
        "point_balance_error": int(
            number_before - number_after - number_removed
        ),
        "output_subset_check_passed": bool(
            np.array_equal(attacked_points, points[keep_mask])
        ),
        "added_point_count": 0,
    }
    # Fold in the firing-gate fields, then the mode-specific fields
    # (drop_prob / rings / matched budget).
    for key, value in gate_meta.items():
        audit.setdefault(key, value)
    for key, value in mode_info.items():
        audit.setdefault(key, value)

    radial_rows = _radial_rows(
        points, remove_mask, target_mask, target_box, radial_edges
    )
    arrays = {
        "removed_indices": removed_indices,
        "target_indices": target_indices,
        "removed_points": points[remove_mask].astype(np.float32, copy=False),
    }

    point_rows = []
    if args.save_point_audit and number_removed:
        radius, azimuth, elevation = _cart_to_spherical(points[:, :3])
        radial_bin = np.searchsorted(
            radial_edges, np.linalg.norm(points[:, :2], axis=1), side="right"
        ) - 1
        for original_index in removed_indices:
            point = points[original_index]
            point_rows.append({
                "original_point_index": int(original_index),
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "intensity": float(point[3]) if point.shape[0] > 3 else None,
                "elongation": float(point[4]) if point.shape[0] > 4 else None,
                "range_3d_m": float(radius[original_index]),
                "planar_range_m": float(np.linalg.norm(point[:2])),
                "azimuth_rad": float(azimuth[original_index]),
                "elevation_rad": float(elevation[original_index]),
                "radial_bin_index": int(radial_bin[original_index]),
                "inside_target_box": bool(target_mask[original_index]),
            })

    return attacked_points, new_state, audit, arrays, radial_rows, point_rows


# ==========================================================================
# Scheduling + tracking (identical semantics to the ATLAS generator)
# ==========================================================================

def extract_one_runs(schedule):
    runs, current = [], []
    for index in sorted(schedule):
        if schedule[index] == 1:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def track_sequence(dataset, start, end, desired_range, target_class,
                   max_step, allow_behind):
    length = end - start + 1
    boxes = [None] * length
    records = []
    initial_boxes = dataset[start]["gt_boxes"]
    initial_index, current_box = removal.select_initial_target(
        initial_boxes,
        desired_range=desired_range,
        target_class=target_class,
        negative=allow_behind,
    )
    if current_box is None:
        records = [
            {
                "window_position": int(offset),
                "tracking_status": "no_initial_target",
                "target_box_index": None,
                "tracking_match_distance_m": None,
                "tracking_step_distance_m": None,
                "target_speed_mps": None,
            }
            for offset in range(length)
        ]
        return boxes, records, {
            "tracking_ok": False,
            "tracking_failure_reason": "no_initial_target",
            "scheduled_frame_count": int(length),
            "tracked_frame_count": 0,
            "tracking_completion_fraction": 0.0,
            "initial_target_range_m": None,
            "target_range_min_m": None,
            "target_range_median_m": None,
            "target_range_max_m": None,
            "mean_tracking_match_distance_m": None,
            "max_tracking_match_distance_m": None,
        }
    boxes[0] = current_box
    records.append({
        "window_position": 0,
        "tracking_status": "initial_target",
        "target_box_index": int(initial_index),
        "tracking_match_distance_m": None,
        "tracking_step_distance_m": None,
        "target_speed_mps": float(np.linalg.norm(current_box[7:9])),
    })
    previous_box = current_box
    previous_previous_box = None
    for offset in range(1, length):
        current_boxes = dataset[start + offset]["gt_boxes"]
        new_box, new_index, match_distance = removal.match_with_velocity(
            previous_box, previous_previous_box, current_boxes,
            target_class=target_class, max_step=max_step,
        )
        if new_box is None:
            for remaining in range(offset, length):
                records.append({
                    "window_position": int(remaining),
                    "tracking_status": "lost",
                    "target_box_index": None,
                    "tracking_match_distance_m": None,
                    "tracking_step_distance_m": None,
                    "target_speed_mps": None,
                })
            break
        boxes[offset] = new_box
        records.append({
            "window_position": int(offset),
            "tracking_status": "matched",
            "target_box_index": int(new_index),
            "tracking_match_distance_m": float(match_distance),
            "tracking_step_distance_m": float(np.linalg.norm(
                new_box[:3] - previous_box[:3]
            )),
            "target_speed_mps": float(np.linalg.norm(new_box[7:9])),
        })
        previous_previous_box = previous_box
        previous_box = new_box
    while len(records) < length:
        records.append({
            "window_position": int(len(records)),
            "tracking_status": "lost",
            "target_box_index": None,
            "tracking_match_distance_m": None,
            "tracking_step_distance_m": None,
            "target_speed_mps": None,
        })
    valid_boxes = [box for box in boxes if box is not None]
    ranges = [float(np.linalg.norm(box[:3])) for box in valid_boxes]
    match_distances = [
        record["tracking_match_distance_m"] for record in records
        if record["tracking_match_distance_m"] is not None
    ]
    summary = {
        "tracking_ok": True,
        "tracking_failure_reason": (
            None if len(valid_boxes) == length else "tracking_lost"
        ),
        "scheduled_frame_count": int(length),
        "tracked_frame_count": int(len(valid_boxes)),
        "tracking_completion_fraction": _safe_div(len(valid_boxes), length),
        "initial_target_range_m": ranges[0] if ranges else None,
        "target_range_min_m": float(np.min(ranges)) if ranges else None,
        "target_range_median_m": float(np.median(ranges)) if ranges else None,
        "target_range_max_m": float(np.max(ranges)) if ranges else None,
        "mean_tracking_match_distance_m": (
            float(np.mean(match_distances)) if match_distances else None
        ),
        "max_tracking_match_distance_m": (
            float(np.max(match_distances)) if match_distances else None
        ),
    }
    return boxes, records, summary


def build_target_schedule(segment_to_global, block_length):
    schedules = {}
    for segment, global_indices in segment_to_global.items():
        toggle = False
        schedule = {}
        for local_index, global_index in enumerate(global_indices):
            if local_index == 0:
                schedule[global_index] = None
                continue
            if local_index % block_length == 0:
                toggle = not toggle
            schedule[global_index] = 1 if toggle else None
        schedules[segment] = schedule
    return schedules


def _frame_id_from_info(info, fallback):
    if "frame_id" in info:
        return str(info["frame_id"])
    point_cloud = info.get("point_cloud", {})
    sample_index = point_cloud.get("sample_idx")
    segment = point_cloud.get("lidar_sequence")
    if segment is not None and sample_index is not None:
        return f"{segment}_{sample_index}"
    return str(fallback)


# ==========================================================================
# Per-segment generation
# ==========================================================================

def setup_segment(
    segment, dataset_mf, dataset_sf, target_frames_global, global_to_segment,
    mf_to_sf_global, original_mf_indices, args, mode, p_remove,
    radial_edges, elevation_edges, atlas_reference,
):
    target_frames = target_frames_global[segment].copy()
    removal_cache = {}
    removal_annos = {}
    run_records = []
    array_records = []
    radial_records = []
    point_records = []

    for run in extract_one_runs(target_frames):
        window_start = min(run)
        window_end = max(run)
        start_local = global_to_segment[window_start][1]
        end_local = global_to_segment[window_end][1]
        run_seed = _stable_seed(args.seed, segment, start_local, end_local)
        pairing_run_id = f"{segment}:{start_local:04d}-{end_local:04d}"
        run_id = f"{args.variant_name}:{pairing_run_id}"

        # Ring choice is fixed once per corrupted block (ring mode only).
        ring_context = {"elevation_edges": elevation_edges, "chosen_rings": []}
        if mode == "ring":
            n_rings_total = len(elevation_edges) - 1
            block_rng = np.random.default_rng(
                _stable_seed(args.seed, segment, start_local, "ringchoice")
            )
            k = min(args.n_rings, n_rings_total)
            # Use a single deterministic permutation and take its prefix so
            # Ring-16 is a strict subset of Ring-32 for the same seed/block.
            ring_order = block_rng.permutation(n_rings_total)
            ring_context["chosen_rings"] = sorted(
                ring_order[:k].tolist()
            )

        tracked_boxes, tracking_records, tracking_summary = track_sequence(
            dataset_mf, start=window_start, end=window_end,
            desired_range=args.desired_target_range_m,
            target_class=args.target_class,
            max_step=args.tracking_max_step_m,
            allow_behind=not args.front_targets_only,
        )
        if len(tracked_boxes) != len(run) or len(tracking_records) != len(run):
            raise RuntimeError(
                f"Tracking output length mismatch for {segment}: "
                f"run={len(run)}, boxes={len(tracked_boxes)}, "
                f"records={len(tracking_records)}"
            )
        run_record = {
            "schema_version": args.schema_version,
            "variant_name": args.variant_name,
            "corruption_mode": mode,
            "segment_id": segment,
            "run_id": run_id,
            "pairing_run_id": pairing_run_id,
            "run_seed": int(run_seed),
            "window_start_local": int(start_local),
            "window_end_local": int(end_local),
            "window_length": int(len(run)),
            "attack_block_length": int(args.attack_block_length),
            "desired_target_range_m": float(args.desired_target_range_m),
            "target_class": int(args.target_class),
            "tracking_max_step_m": float(args.tracking_max_step_m),
            "front_targets_only": bool(args.front_targets_only),
        }
        if mode == "uniform":
            run_record["configured_drop_prob"] = float(args.drop_prob)
        elif mode == "ring":
            run_record["n_rings_removed"] = int(len(
                ring_context["chosen_rings"]
            ))
            run_record["removed_ring_indices"] = json.dumps(
                ring_context["chosen_rings"]
            )
        elif mode == "matched":
            run_record["az_width_deg"] = int(args.az_width)
            run_record["el_width_deg"] = float(args.el_width_deg)
            run_record["configured_point_removal_probability"] = float(
                p_remove
            )
        run_record.update(tracking_summary)

        active_frames = 0
        modified_frames = 0
        total_removed = 0
        total_target_removed = 0
        total_background_removed = 0

        state = None
        for offset, global_index in enumerate(sorted(run)):
            local_index = global_to_segment[global_index][1]
            original_global_index = int(original_mf_indices[global_index])
            frame_id = _frame_id_from_info(
                dataset_mf.infos[global_index], global_index
            )
            box = tracked_boxes[offset]
            tracking_record = tracking_records[offset]

            base_context = {
                "schema_version": args.schema_version,
                "variant_name": args.variant_name,
                "corruption_mode": mode,
                "segment_id": segment,
                "frame_id": frame_id,
                "filtered_global_frame_index": int(global_index),
                "original_global_frame_index": original_global_index,
                "local_frame_index": int(local_index),
                "run_id": run_id,
                "pairing_run_id": pairing_run_id,
                "run_seed": int(run_seed),
                "window_start_local": int(start_local),
                "window_end_local": int(end_local),
                "window_length": int(len(run)),
                "window_position": int(offset),
                **tracking_record,
            }

            ref = None
            if atlas_reference is not None:
                ref = atlas_reference.get(frame_id)
                if ref is None:
                    raise KeyError(
                        f"Scheduled frame {frame_id!r} is absent from the "
                        "ATLAS reference; exact pairing cannot be claimed."
                    )
                validate_atlas_reference_frame(
                    ref,
                    segment=segment,
                    local_index=local_index,
                    tracking_record=tracking_record,
                    target_box=box,
                    box_atol=args.pairing_box_atol,
                )

            if box is None:
                target_frames[global_index] = None
                removal_annos[local_index] = {
                    **base_context,
                    "attack_scheduled": True,
                    "tracking_ok": False,
                    "active_this_frame": False,
                    "modified_this_frame": False,
                    "corruption_applied": False,
                    "inactive_reason": "tracking_lost",
                    "target_valid_reason": "track_ended_or_missing_GT",
                    "n_points_removed": 0,
                    "n_points_removed_total": 0,
                    "n_target_points_removed": 0,
                    "n_background_points_removed": 0,
                }
                continue

            sf_index = mf_to_sf_global[global_index]
            raw_points = dataset_sf[sf_index]["points"].copy()
            points = np.hstack([
                raw_points,
                np.zeros((raw_points.shape[0], 1), dtype=raw_points.dtype),
            ])
            if ref is not None:
                validate_atlas_reference_frame(
                    ref,
                    segment=segment,
                    local_index=local_index,
                    tracking_record=tracking_record,
                    target_box=box,
                    n_points_before=len(points),
                    box_atol=args.pairing_box_atol,
                )
            frame_seed = _stable_seed(
                args.seed, segment, local_index, mode
            )
            frame_rng = np.random.default_rng(frame_seed)
            # Firing RNG is mode-INDEPENDENT so uniform/ring/matched fire on
            # identical frames as each other under the stochastic gate.
            firing_seed = _stable_seed(args.seed, segment, local_index, "firing")
            firing_rng = np.random.default_rng(firing_seed)

            # Optional exact pairing with a paired ATLAS removal run.
            firing_override = None
            atlas_k = None
            if ref is not None:
                firing_override = bool(ref["fired"])
                if mode == "matched":
                    atlas_k = int(ref["n_removed"])

            (
                attacked_points,
                state,
                audit,
                arrays,
                radial_rows,
                removed_point_rows,
            ) = audited_baseline_frame(
                points=points,
                target_box=box,
                args=args,
                mode=mode,
                state=state,
                rng=frame_rng,
                firing_rng=firing_rng,
                ring_context=ring_context,
                radial_edges=radial_edges,
                p_remove=p_remove,
                firing_override=firing_override,
                atlas_k=atlas_k,
            )
            attacked_points = attacked_points.astype(np.float32, copy=False)
            removal_cache[global_index] = attacked_points
            target_frames[global_index] = box.copy()
            audit.update(base_context)
            audit["frame_seed"] = int(frame_seed)
            audit["target_valid_reason"] = "valid"
            audit["corruption_applied"] = bool(audit["modified_this_frame"])
            removal_annos[local_index] = audit

            common = {
                key: base_context[key] for key in (
                    "schema_version", "variant_name", "corruption_mode",
                    "segment_id", "frame_id",
                    "original_global_frame_index", "local_frame_index",
                    "run_id", "pairing_run_id",
                )
            }
            for row in radial_rows:
                row.update(common)
                radial_records.append(row)
            if args.save_point_audit:
                for row in removed_point_rows:
                    row.update(common)
                    point_records.append(row)

            arrays.update({
                "frame_id": frame_id,
                "local_frame_index": int(local_index),
                "original_global_frame_index": original_global_index,
            })
            array_records.append(arrays)

            active_frames += int(audit["active_this_frame"])
            modified_frames += int(audit["modified_this_frame"])
            total_removed += int(audit["n_points_removed_total"])
            total_target_removed += int(audit["n_target_points_removed"])
            total_background_removed += int(
                audit["n_background_points_removed"]
            )

        run_record.update({
            "active_frame_count": int(active_frames),
            "modified_frame_count": int(modified_frames),
            "total_removed_point_count": int(total_removed),
            "total_target_removed_point_count": int(total_target_removed),
            "total_background_removed_point_count": int(
                total_background_removed
            ),
            "aggregate_removed_points_target_fraction": _safe_div(
                total_target_removed, total_removed
            ),
        })
        run_records.append(run_record)

    return (
        target_frames, removal_cache, removal_annos, run_records,
        array_records, radial_records, point_records,
    )


def save_segment(
    args, segment, segment_dataset, removal_annos, run_records,
    frame_records, radial_records, array_records, point_records,
):
    dataset_path = Path(args.save_dir_dataset) / f"{segment}.pkl"
    annotation_path = Path(args.save_dir_annos) / f"{segment}_a.pkl"
    with open(dataset_path, "wb") as handle:
        pkl.dump(segment_dataset, handle)
    with open(annotation_path, "wb") as handle:
        pkl.dump({
            "segment": segment,
            "spoof_annos": removal_annos,
            "run_annos": run_records,
        }, handle)
    metrics = Path(args.metrics_dir)
    _atomic_write_csv(
        metrics / "frames" / f"{segment}_frames.csv", frame_records
    )
    _atomic_write_csv(
        metrics / "runs" / f"{segment}_runs.csv", run_records
    )
    _atomic_write_csv(
        metrics / "radial" / f"{segment}_radial.csv", radial_records
    )
    if not args.no_save_audit_arrays:
        _atomic_write_audit_arrays(
            metrics / "arrays" / f"{segment}_baseline_arrays.npz",
            array_records,
        )
    if args.save_point_audit:
        _atomic_write_csv(
            metrics / "points" / f"{segment}_removed_points.csv",
            point_records,
        )


def _segment_is_complete(args, segment):
    metrics = Path(args.metrics_dir)
    frame_csv = metrics / "frames" / f"{segment}_frames.csv"
    run_csv = metrics / "runs" / f"{segment}_runs.csv"
    required = [
        Path(args.save_dir_dataset) / f"{segment}.pkl",
        Path(args.save_dir_annos) / f"{segment}_a.pkl",
        frame_csv,
        run_csv,
        metrics / "radial" / f"{segment}_radial.csv",
    ]
    if not args.no_save_audit_arrays:
        required.append(metrics / "arrays" / f"{segment}_baseline_arrays.npz")
    if args.save_point_audit:
        required.append(metrics / "points" / f"{segment}_removed_points.csv")
    if not all(path.exists() for path in required):
        return False

    # Do not resume from outputs generated by an older schema or another
    # variant. This is important after changing ring assignment/pairing rules.
    try:
        for csv_path in (frame_csv, run_csv):
            with open(csv_path, newline="", encoding="utf-8") as handle:
                first = next(csv.DictReader(handle), None)
            if first is None:
                return False
            if first.get("schema_version") != args.schema_version:
                return False
            if first.get("variant_name") != args.variant_name:
                return False
    except (OSError, csv.Error):
        return False
    return True


def _build_manifest(args, selected_segments, mode, p_remove, radial_max_m,
                    elevation_edges):
    attack_config = {"mode": mode}
    if mode == "uniform":
        attack_config["drop_prob"] = float(args.drop_prob)
    elif mode == "ring":
        attack_config["n_rings"] = int(args.n_rings)
        attack_config["n_elevation_bins"] = int(len(elevation_edges) - 1)
        attack_config["inclination_file"] = args.inclination_file
        attack_config["ring_assignment"] = "elevation_layer_approximation"
        attack_config["out_of_range_ring_index"] = -1
        attack_config["nested_severity_by_prefix"] = True
    elif mode == "matched":
        attack_config.update({
            "az_width_deg": int(args.az_width),
            "el_width_deg": float(args.el_width_deg),
            "configured_point_removal_probability": float(p_remove),
            "max_az_step_deg": float(args.max_az_step_deg),
            "center_jitter_std_deg": float(args.center_jitter_std_deg),
            "min_target_range_m": float(args.min_target_range_m),
        })
    attack_config.update({
        "attack_block_length": int(args.attack_block_length),
        "desired_target_range_m": float(args.desired_target_range_m),
        "tracking_max_step_m": float(args.tracking_max_step_m),
        "target_class": int(args.target_class),
        "front_targets_only": bool(args.front_targets_only),
    })
    return {
        "schema_version": args.schema_version,
        "generator_name": "generate_data_adverse_baselines",
        "corruption_mode": mode,
        "variant_name": args.variant_name,
        "master_seed": int(args.seed),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "baselines_module_sha256": _sha256(
            Path(baselines.__file__).resolve()
        ),
        "removal_module_sha256": _sha256(Path(removal.__file__).resolve()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "selected_sequence_count": int(len(selected_segments)),
        "selected_sequences": list(selected_segments),
        "sequence_selection": args.sequence_selection,
        "requested_num_sequences": args.num_sequences,
        "attack_configuration": attack_config,
        "audit_configuration": {
            "radial_bin_width_m": float(args.radial_bin_width_m),
            "radial_max_m": float(radial_max_m),
            "save_audit_arrays": not args.no_save_audit_arrays,
            "save_point_audit": bool(args.save_point_audit),
            "pairing_box_atol": float(args.pairing_box_atol),
        },
        "paths": {
            "cp_mf_root": args.cp_mf_root,
            "cp_mf_cfg": args.cp_mf_cfg,
            "cp_sf_cfg": args.cp_sf_cfg,
            "save_dir_dataset": args.save_dir_dataset,
            "save_dir_annos": args.save_dir_annos,
            "metrics_dir": args.metrics_dir,
            "log_file": args.log_file,
        },
    }


# ==========================================================================
# Main
# ==========================================================================

def _validate_mode_args(args):
    if args.mode == "uniform":
        if args.drop_prob is None:
            raise ValueError("uniform mode requires --drop_prob")
        if not (0.0 < args.drop_prob < 1.0):
            raise ValueError("--drop_prob must be in (0, 1)")
    elif args.mode == "ring":
        if args.n_rings is None:
            raise ValueError("ring mode requires --n_rings")
        if args.n_rings <= 0:
            raise ValueError("--n_rings must be positive")
    elif args.mode == "matched":
        if args.az_width is None:
            raise ValueError("matched mode requires --az_width")


def main():
    args = parse_args()
    _validate_mode_args(args)
    if args.attack_block_length <= 0:
        raise ValueError("--attack_block_length must be positive")
    if args.radial_bin_width_m <= 0:
        raise ValueError("--radial_bin_width_m must be positive")
    if args.pairing_box_atol < 0:
        raise ValueError("--pairing_box_atol must be nonnegative")

    np.random.seed(args.seed)
    random.seed(args.seed)
    mode = args.mode
    p_remove = (
        P_REMOVE_BY_AZ_WIDTH[args.az_width] if mode == "matched" else None
    )

    Path(args.save_dir_dataset).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir_annos).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    for directory in ("frames", "runs", "radial", "arrays", "points"):
        (metrics_dir / directory).mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("atlas_baseline")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.log_file)
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)

    unload_pcdet()
    sys.path.insert(0, args.cp_mf_root)
    from pcdet.config import cfg as cfg_cp
    from pcdet.config import cfg_from_yaml_file as load_cfg
    from pcdet.datasets import build_dataloader
    from pcdet.utils import common_utils
    from torch.utils.data import DataLoader
    sys.path.pop(0)

    cfg_cp.clear()
    load_cfg(args.cp_mf_cfg, cfg_cp)
    if args.data_path:
        cfg_cp.DATA_CONFIG.DATA_PATH = args.data_path
    mf_logger = common_utils.create_logger()
    mf_logger.info("Loaded cfg from %s", args.cp_mf_cfg)
    dataset_mf, _, _ = build_dataloader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1, dist=False, workers=args.workers,
        logger=mf_logger, training=False,
    )
    point_cloud_range = np.asarray(
        cfg_cp.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float64
    )

    cfg_cp.clear()
    load_cfg(args.cp_sf_cfg, cfg_cp)
    if args.data_path:
        cfg_cp.DATA_CONFIG.DATA_PATH = args.data_path
    sf_logger = common_utils.create_logger()
    sf_logger.info("Loaded cfg from %s", args.cp_sf_cfg)
    dataset_sf, _, _ = build_dataloader(
        dataset_cfg=cfg_cp.DATA_CONFIG,
        class_names=cfg_cp.CLASS_NAMES,
        batch_size=1, dist=False, workers=args.workers,
        logger=sf_logger, training=False,
    )

    original_mf_infos = list(dataset_mf.infos)
    original_sf_infos = list(dataset_sf.infos)
    original_mf_index_by_id = {
        id(info): index for index, info in enumerate(original_mf_infos)
    }
    mf_segments = {
        info["point_cloud"]["lidar_sequence"] for info in original_mf_infos
    }
    sf_segments = {
        info["frame_id"].rsplit("_", 1)[0] for info in original_sf_infos
    }
    all_common_segments = sorted(mf_segments & sf_segments)
    if args.num_sequences is None:
        selected_segments = all_common_segments
    else:
        if args.num_sequences <= 0:
            raise ValueError("--num_sequences must be positive")
        if args.num_sequences > len(all_common_segments):
            raise ValueError(
                f"Requested {args.num_sequences} sequences, but only "
                f"{len(all_common_segments)} are common."
            )
        if args.sequence_selection == "first":
            selected_segments = all_common_segments[:args.num_sequences]
        else:
            rng = np.random.default_rng(args.seed)
            selected_segments = sorted(rng.choice(
                all_common_segments, size=args.num_sequences, replace=False,
            ).tolist())
    selected_set = set(selected_segments)

    # IMPORTANT: same seed + same sequence_selection as the ATLAS removal run
    # yields the SAME 50 sequences, so baseline and attack are paired.
    dataset_mf.infos = [
        info for info in original_mf_infos
        if info["point_cloud"]["lidar_sequence"] in selected_set
    ]
    dataset_sf.infos = [
        info for info in original_sf_infos
        if info["frame_id"].rsplit("_", 1)[0] in selected_set
    ]
    original_mf_indices = [
        original_mf_index_by_id[id(info)] for info in dataset_mf.infos
    ]
    test_loader_mf = DataLoader(
        dataset_mf, batch_size=1, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=False,
        collate_fn=dataset_mf.collate_batch,
    )
    log.info(
        "Mode=%s. Selected %d of %d common sequences using %s selection.",
        mode, len(selected_segments), len(all_common_segments),
        args.sequence_selection,
    )

    # Radial edges.
    if args.radial_max_m is None:
        radial_max_m = float(np.ceil(
            np.hypot(
                max(abs(point_cloud_range[0]), abs(point_cloud_range[3])),
                max(abs(point_cloud_range[1]), abs(point_cloud_range[4])),
            ) / args.radial_bin_width_m
        ) * args.radial_bin_width_m)
    else:
        radial_max_m = float(args.radial_max_m)
    radial_edges = np.arange(
        0.0, radial_max_m + args.radial_bin_width_m * 0.5,
        args.radial_bin_width_m, dtype=np.float64,
    )
    if radial_edges[-1] < radial_max_m:
        radial_edges = np.r_[radial_edges, radial_max_m]

    # Elevation ring edges (ring mode). Built once, reused for every block.
    elevation_edges = None
    if mode == "ring":
        inclinations = None
        if args.inclination_file:
            inclinations = np.load(args.inclination_file)
        elevation_edges = baselines.build_elevation_bins(
            inclinations=inclinations, n_bins=args.n_elevation_bins
        )
        if args.n_rings > len(elevation_edges) - 1:
            raise ValueError(
                f"--n_rings {args.n_rings} exceeds available rings "
                f"{len(elevation_edges) - 1}"
            )

    # Optional exact firing/target pairing with a paired ATLAS removal run.
    atlas_reference = None
    atlas_reference_manifest = None
    atlas_reference_manifest_path = None
    if args.atlas_firing_reference:
        atlas_reference = load_atlas_firing_reference(
            args.atlas_firing_reference
        )
        (
            atlas_reference_manifest,
            atlas_reference_manifest_path,
        ) = load_atlas_reference_manifest(args.atlas_firing_reference)
        validate_atlas_reference_manifest(
            atlas_reference_manifest, args, selected_segments
        )
        log.info(
            "Loaded and validated ATLAS reference: %d frames from %s",
            len(atlas_reference), args.atlas_firing_reference,
        )

    manifest_path = metrics_dir / "run_manifest.json"
    manifest = _build_manifest(
        args, selected_segments, mode, p_remove, radial_max_m, elevation_edges
        if elevation_edges is not None else np.zeros(1)
    )
    manifest["firing_gate"] = {
        "distance_probability_table": "removal.ahfr_gate_prob",
        "mode_independent_stochastic_gate": args.atlas_firing_reference is None,
        "atlas_firing_reference": args.atlas_firing_reference,
        "atlas_reference_frame_count": (
            len(atlas_reference) if atlas_reference is not None else None
        ),
        "atlas_reference_manifest_path": (
            str(atlas_reference_manifest_path)
            if atlas_reference_manifest_path is not None else None
        ),
        "atlas_reference_manifest_sha256": (
            _sha256(atlas_reference_manifest_path)
            if atlas_reference_manifest_path is not None else None
        ),
        "atlas_reference_generator": (
            atlas_reference_manifest.get("generator_name")
            if atlas_reference_manifest is not None else None
        ),
        "atlas_reference_variant_name": (
            atlas_reference_manifest.get("variant_name")
            if atlas_reference_manifest is not None else None
        ),
        "atlas_reference_attack_configuration": (
            atlas_reference_manifest.get("attack_configuration")
            if atlas_reference_manifest is not None else None
        ),
        "strict_frame_target_validation": atlas_reference is not None,
        "pairing_box_atol": float(args.pairing_box_atol),
        "min_target_range_m": float(args.min_target_range_m),
        "allow_close_range_extrapolation": bool(
            args.allow_close_range_extrapolation
        ),
    }
    _atomic_write_json(manifest_path, manifest)

    segment_to_global = defaultdict(list)
    global_to_segment = {}
    for global_index, info in enumerate(dataset_mf.infos):
        segment = info["point_cloud"]["lidar_sequence"]
        local_index = len(segment_to_global[segment])
        segment_to_global[segment].append(global_index)
        global_to_segment[global_index] = (segment, local_index)

    sf_segment_to_global = defaultdict(list)
    for sf_index, info in enumerate(dataset_sf.infos):
        segment = info["frame_id"].rsplit("_", 1)[0]
        sf_segment_to_global[segment].append(sf_index)

    mf_to_sf_global = {}
    for mf_index, (segment, local_index) in global_to_segment.items():
        if local_index >= len(sf_segment_to_global[segment]):
            raise RuntimeError(
                f"Missing SF frame for {segment} local index {local_index}"
            )
        sf_index = sf_segment_to_global[segment][local_index]
        mf_frame_id = _frame_id_from_info(dataset_mf.infos[mf_index], mf_index)
        sf_frame_id = str(dataset_sf.infos[sf_index]["frame_id"])
        if mf_frame_id != sf_frame_id:
            raise RuntimeError(
                "MF/SF frame-ID mismatch for positional mapping: "
                f"MF={mf_frame_id!r}, SF={sf_frame_id!r}, "
                f"segment={segment!r}, local_index={local_index}"
            )
        mf_to_sf_global[mf_index] = sf_index
    for segment in selected_segments:
        if len(segment_to_global[segment]) != len(
            sf_segment_to_global[segment]
        ):
            raise RuntimeError(
                f"MF/SF length mismatch for {segment}: "
                f"MF={len(segment_to_global[segment])}, "
                f"SF={len(sf_segment_to_global[segment])}"
            )

    target_frames_global = build_target_schedule(
        segment_to_global, args.attack_block_length
    )
    completed = {
        segment for segment in selected_segments
        if _segment_is_complete(args, segment)
    }
    if completed:
        log.info("Resuming with %d complete sequences skipped.", len(completed))

    segment_dataset = []
    frame_records = []
    current_segment = None
    current_payload = None

    def start_segment(segment):
        log.info("Starting: %s", segment)
        return setup_segment(
            segment=segment, dataset_mf=dataset_mf, dataset_sf=dataset_sf,
            target_frames_global=target_frames_global,
            global_to_segment=global_to_segment,
            mf_to_sf_global=mf_to_sf_global,
            original_mf_indices=original_mf_indices,
            args=args, mode=mode, p_remove=p_remove,
            radial_edges=radial_edges, elevation_edges=elevation_edges,
            atlas_reference=atlas_reference,
        )

    def flush_segment(segment, payload):
        (_tf, _cache, removal_annos, run_records,
         array_records, radial_records, point_records) = payload
        save_segment(
            args=args, segment=segment, segment_dataset=segment_dataset,
            removal_annos=removal_annos, run_records=run_records,
            frame_records=frame_records, radial_records=radial_records,
            array_records=array_records, point_records=point_records,
        )
        log.info("Saved: %s", segment)

    try:
        for filtered_global_index, batch in enumerate(test_loader_mf):
            segment = batch["frame_id"][0].rsplit("_", 1)[0]
            if segment in completed:
                continue
            if current_segment is None:
                current_segment = segment
                current_payload = start_segment(segment)
            elif segment != current_segment:
                flush_segment(current_segment, current_payload)
                segment_dataset = []
                frame_records = []
                current_segment = segment
                current_payload = start_segment(segment)

            (target_frames, removal_cache, removal_annos,
             _rr, _ar, _rad, _pr) = current_payload
            local_index = global_to_segment[filtered_global_index][1]
            original_global_index = int(
                original_mf_indices[filtered_global_index]
            )
            frame_id = str(batch["frame_id"][0])

            if filtered_global_index in removal_cache:
                single_frame_points = removal_cache[filtered_global_index]
            else:
                sf_index = mf_to_sf_global[filtered_global_index]
                raw_points = dataset_sf[sf_index]["points"]
                single_frame_points = np.hstack([
                    raw_points,
                    np.zeros((raw_points.shape[0], 1), dtype=raw_points.dtype),
                ])
            single_frame_points = single_frame_points.astype(
                np.float32, copy=False
            )

            modified = dataset_mf[filtered_global_index].copy()
            for key in ("voxels", "voxel_coords", "voxel_num_points"):
                modified.pop(key, None)
            modified["points"] = single_frame_points
            modified["pose"] = modified["poses"][:4]

            attack_meta = removal_annos.get(local_index)
            if attack_meta is None:
                frame_record = {
                    "schema_version": args.schema_version,
                    "variant_name": args.variant_name,
                    "corruption_mode": mode,
                    "segment_id": segment,
                    "frame_id": frame_id,
                    "filtered_global_frame_index": int(filtered_global_index),
                    "original_global_frame_index": original_global_index,
                    "local_frame_index": int(local_index),
                    "attack_scheduled": False,
                    "active_this_frame": False,
                    "modified_this_frame": False,
                    "corruption_applied": False,
                    "inactive_reason": "clean_schedule",
                    "target_valid_reason": "clean_schedule",
                    "n_points_before": int(len(single_frame_points)),
                    "n_points_after": int(len(single_frame_points)),
                    "n_points_removed": 0,
                    "n_points_removed_total": 0,
                    "point_balance_error": 0,
                }
                modified["spoof_gt"] = None
                modified["removal_meta"] = None
            else:
                frame_record = dict(attack_meta)
                frame_record.setdefault(
                    "n_points_before", int(len(single_frame_points))
                )
                frame_record.setdefault(
                    "n_points_after", int(len(single_frame_points))
                )
                frame_record.setdefault("n_points_removed", 0)
                frame_record.setdefault("n_points_removed_total", 0)
                frame_record.setdefault("point_balance_error", 0)
                attack_active = bool(
                    attack_meta.get("active_this_frame", False)
                )
                target_box = target_frames.get(filtered_global_index)
                modified["spoof_gt"] = (
                    target_box.copy()
                    if attack_active and target_box is not None else None
                )
                compact_keys = (
                    "corruption_mode", "active_this_frame",
                    "modified_this_frame", "corruption_applied",
                    "inactive_reason", "n_points_before", "n_points_removed",
                    "n_points_after", "target_range_m", "target_valid_reason",
                )
                modified["removal_meta"] = {
                    key: attack_meta.get(key) for key in compact_keys
                }
                if target_box is not None:
                    modified["removal_meta"]["target_box"] = target_box.copy()

            frame_records.append(frame_record)
            segment_dataset.append(modified)

        if current_segment is not None and segment_dataset:
            flush_segment(current_segment, current_payload)

        manifest["status"] = "completed"
        manifest["completed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        manifest["completed_sequence_count"] = int(len(selected_segments))
        _atomic_write_json(manifest_path, manifest)
        log.info("Completed %d selected sequences.", len(selected_segments))
    except Exception:
        manifest["status"] = "failed"
        manifest["failed_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
        manifest["exception"] = traceback.format_exc()
        _atomic_write_json(manifest_path, manifest)
        log.error("===== EXCEPTION =====")
        log.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
