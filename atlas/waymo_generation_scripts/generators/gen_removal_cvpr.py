#!/usr/bin/env python
# CVPR copy (cvpr_scripts/) of PTT/generate_data_removal_rebuttal_comparison.py; only change: --data_path override
# (the cfgs' shared/waymo* DATA_PATH was emptied in Sep 2026). All RNG streams are keyed by (seed, segment, local frame).
"""
Generate a paired ATLAS A-HFR point-removal subset for rebuttal evaluation.

This is the original ATLAS angular-sector removal algorithm with deterministic
experiment-level pairing added around it. With the same data/configuration and
seed as generate_data_adverse_baselines.py, it uses the same:

  * randomly selected sequence subset (for example, 50 segments),
  * alternating 32-frame clean/corrupt schedule,
  * initial target and velocity-based target track,
  * target-valid scheduled frames, and
  * realized distance-dependent A-HFR firing decisions.

The attack itself remains ATLAS A-HFR removal: a tracked, servo-limited angular
sector is formed around the target and points inside that sector are removed
independently according to the azimuth-width-specific PRR.

Random streams are intentionally separated:

  * selection: np.random.default_rng(master_seed), identical to baselines;
  * firing: stable_seed(seed, segment, local_frame, "firing"), identical to
    every non-adversarial baseline mode;
  * aim/removal: stable_seed(seed, segment, local_frame, "matched"), identical
    to the matched-count baseline's ATLAS-counting stream. This stream is
    independent of azimuth width, so az20 and az50 share aim jitter while the
    actual sector and PRR remain width-specific.

An optional --pairing_reference_metrics path validates segment selection,
schedule, target identity/box, target validity, and firing decisions against an
already-generated non-adversarial baseline. Generation fails on any mismatch.

Outputs are compatible with the audited CenterPoint, MSF, and PTT evaluation
scripts:

  dataset/<segment>.pkl
  annos/<segment>_a.pkl
  metrics/run_manifest.json
  metrics/frames/<segment>_frames.csv
  metrics/runs/<segment>_runs.csv
  metrics/radial/<segment>_radial.csv
  metrics/arrays/<segment>_removal_arrays.npz
"""

from __future__ import annotations

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
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    import removal
except ImportError:  # Supports invocation from one directory above neurips.
    import PTT.neurips.removal as removal

importlib.reload(removal)


P_REMOVE_BY_AZ_WIDTH = {
    10: 0.97,
    20: 1.00,
    30: 0.99,
    40: 0.98,
    50: 0.90,
    60: 0.90,
}


# ============================================================================
# Arguments and generic utilities
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a baseline-paired ATLAS A-HFR removal subset."
    )
    parser.add_argument("--cp_mf_root", required=True)
    parser.add_argument("--cp_mf_cfg", required=True)
    parser.add_argument("--cp_sf_cfg", required=True)
    parser.add_argument("--log_file", required=True)
    parser.add_argument("--save_dir_dataset", required=True)
    parser.add_argument("--save_dir_annos", required=True)
    parser.add_argument("--metrics_dir", required=True)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--az_width", type=int, required=True,
                        choices=tuple(sorted(P_REMOVE_BY_AZ_WIDTH)))
    parser.add_argument("--seed", type=int, required=True)

    # Exact subset-selection semantics used by the adverse baselines.
    parser.add_argument("--num_sequences", type=int, default=50)
    parser.add_argument(
        "--sequence_selection", choices=("random", "first"), default="random"
    )

    # Scheduling/tracking/attack defaults match the existing generators.
    parser.add_argument("--attack_block_length", type=int, default=32)
    parser.add_argument("--desired_target_range_m", type=float, default=60.0)
    parser.add_argument("--tracking_max_step_m", type=float, default=6.0)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--front_targets_only", action="store_true")
    parser.add_argument("--el_width_deg", type=float, default=16.0)
    parser.add_argument("--max_az_step_deg", type=float, default=15.0)
    parser.add_argument("--center_jitter_std_deg", type=float, default=0.05)
    parser.add_argument("--min_target_range_m", type=float, default=6.0)
    parser.add_argument("--allow_close_range_extrapolation", action="store_true")

    # Optional strict comparison against one generated baseline variant.
    parser.add_argument(
        "--pairing_reference_metrics",
        default=None,
        help=(
            "metrics/ directory from a paired adverse-baseline run. When "
            "provided, selection, schedule, target tracking, and firing are "
            "validated frame-by-frame and generation fails on a mismatch."
        ),
    )
    parser.add_argument("--pairing_box_atol", type=float, default=1e-5)

    parser.add_argument("--radial_bin_width_m", type=float, default=1.0)
    parser.add_argument("--radial_max_m", type=float, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data_path", default=None,
                        help="override DATA_CONFIG.DATA_PATH of both cfgs (Cedar Waymo copy)")
    parser.add_argument("--schema_version", default="atlas_removal_rebuttal_v1")
    parser.add_argument("--no_save_audit_arrays", action="store_true")
    parser.add_argument(
        "--save_point_audit",
        action="store_true",
        help="Write one CSV row per removed point; potentially very large.",
    )
    return parser.parse_args()


def unload_pcdet() -> None:
    for key in list(sys.modules):
        if key.startswith("pcdet"):
            del sys.modules[key]


def _stable_seed(master_seed: int, *parts: Any) -> int:
    payload = ":".join([str(master_seed), *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_div(num: int | float, den: int | float) -> float | None:
    return None if den == 0 else float(num) / float(den)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=_python_scalar)
    os.replace(temp, path)


def _atomic_write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
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
                writer.writerow({key: _python_scalar(row.get(key)) for key in fields})
    os.replace(temp, path)


def _concat_with_offsets(
    records: list[dict[str, Any]], key: str, dtype: np.dtype
) -> tuple[np.ndarray, np.ndarray]:
    arrays = [np.asarray(record[key], dtype=dtype) for record in records]
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for index, array in enumerate(arrays):
        offsets[index + 1] = offsets[index] + len(array)
    if not arrays or offsets[-1] == 0:
        return np.empty((0,), dtype=dtype), offsets
    return np.concatenate(arrays), offsets


def _atomic_write_audit_arrays(
    path: str | Path, records: list[dict[str, Any]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp.npz")

    removed_indices, removed_offsets = _concat_with_offsets(
        records, "removed_indices", np.int64
    )
    target_indices, target_offsets = _concat_with_offsets(
        records, "target_indices", np.int64
    )
    sector_indices, sector_offsets = _concat_with_offsets(
        records, "sector_indices", np.int64
    )
    removed_points = [
        np.asarray(record["removed_points"], dtype=np.float32) for record in records
    ]
    point_offsets = np.zeros(len(removed_points) + 1, dtype=np.int64)
    for index, array in enumerate(removed_points):
        point_offsets[index + 1] = point_offsets[index] + len(array)
    if removed_points and point_offsets[-1]:
        removed_points_concat = np.concatenate(removed_points)
    else:
        removed_points_concat = np.empty((0, 6), dtype=np.float32)

    np.savez_compressed(
        temp,
        frame_id=np.asarray([str(r["frame_id"]) for r in records], dtype="U128"),
        local_frame_index=np.asarray(
            [r["local_frame_index"] for r in records], dtype=np.int64
        ),
        original_global_frame_index=np.asarray(
            [r["original_global_frame_index"] for r in records], dtype=np.int64
        ),
        removed_indices=removed_indices,
        removed_indices_offsets=removed_offsets,
        target_indices=target_indices,
        target_indices_offsets=target_offsets,
        sector_indices=sector_indices,
        sector_indices_offsets=sector_offsets,
        removed_points=removed_points_concat,
        removed_points_offsets=point_offsets,
    )
    os.replace(temp, path)


def _frame_id_from_info(info: dict[str, Any], fallback: int) -> str:
    if "frame_id" in info:
        return str(info["frame_id"])
    point_cloud = info.get("point_cloud", {})
    sample_index = point_cloud.get("sample_idx")
    segment = point_cloud.get("lidar_sequence")
    if segment is not None and sample_index is not None:
        return f"{segment}_{sample_index}"
    return str(fallback)


def _optional_bool(value: Any) -> bool | None:
    if value in (None, "", "None"):
        return None
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    raise ValueError(f"Cannot parse boolean value {value!r}")


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    return int(float(value))


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    return float(value)


# ============================================================================
# Optional exact pairing validation against an adverse-baseline run
# ============================================================================


def _target_box_from_csv(row: dict[str, str]) -> np.ndarray | None:
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
        raise ValueError("Incomplete target box in pairing reference: " + ", ".join(missing))
    return np.asarray(values, dtype=np.float64)


def load_pairing_reference(metrics_dir: str | Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metrics_dir = Path(metrics_dir)
    manifest_path = metrics_dir / "run_manifest.json"
    frames_dir = metrics_dir / "frames"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)

    reference: dict[str, dict[str, Any]] = {}
    csv_paths = sorted(frames_dir.glob("*_frames.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No *_frames.csv files in {frames_dir}")
    for csv_path in csv_paths:
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                frame_id = str(row.get("frame_id", ""))
                if not frame_id:
                    raise ValueError(f"Missing frame_id in {csv_path}")
                if frame_id in reference:
                    raise ValueError(f"Duplicate frame_id {frame_id!r} in reference")
                reference[frame_id] = {
                    "segment_id": row.get("segment_id"),
                    "local_frame_index": _optional_int(row.get("local_frame_index")),
                    "attack_scheduled": _optional_bool(row.get("attack_scheduled")),
                    "tracking_ok": _optional_bool(row.get("tracking_ok")),
                    "target_box_index": _optional_int(row.get("target_box_index")),
                    "target_box": _target_box_from_csv(row),
                    "active_this_frame": bool(
                        _optional_bool(row.get("active_this_frame")) or False
                    ),
                    "source_csv": str(csv_path),
                }
    return manifest, reference


def validate_pairing_manifest(
    manifest: dict[str, Any], args: argparse.Namespace, selected_segments: list[str]
) -> None:
    errors: list[str] = []
    if manifest.get("generator_name") != "generate_data_adverse_baselines":
        errors.append(
            "reference generator_name is not 'generate_data_adverse_baselines': "
            f"{manifest.get('generator_name')!r}"
        )
    expected_top = {
        "master_seed": int(args.seed),
        "sequence_selection": args.sequence_selection,
        "requested_num_sequences": args.num_sequences,
    }
    for key, expected in expected_top.items():
        if manifest.get(key) != expected:
            errors.append(f"{key}: reference={manifest.get(key)!r}, ATLAS={expected!r}")
    if list(manifest.get("selected_sequences", [])) != list(selected_segments):
        errors.append("selected_sequences differ")

    cfg = manifest.get("attack_configuration", {})
    expected_shared = {
        "attack_block_length": int(args.attack_block_length),
        "desired_target_range_m": float(args.desired_target_range_m),
        "tracking_max_step_m": float(args.tracking_max_step_m),
        "target_class": int(args.target_class),
        "front_targets_only": bool(args.front_targets_only),
    }
    for key, expected in expected_shared.items():
        actual = cfg.get(key)
        if actual is None or not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
            errors.append(f"attack_configuration.{key}: reference={actual!r}, ATLAS={expected!r}")
    firing = manifest.get("firing_gate", {})
    ref_min_range = firing.get("min_target_range_m", cfg.get("min_target_range_m"))
    ref_extrapolate = firing.get(
        "allow_close_range_extrapolation", cfg.get("allow_close_range_extrapolation")
    )
    if ref_min_range is not None and not np.isclose(
        float(ref_min_range), float(args.min_target_range_m), rtol=0.0, atol=1e-12
    ):
        errors.append(
            f"min_target_range_m: reference={ref_min_range!r}, ATLAS={args.min_target_range_m!r}"
        )
    if ref_extrapolate is not None and bool(ref_extrapolate) != bool(args.allow_close_range_extrapolation):
        errors.append(
            "allow_close_range_extrapolation: "
            f"reference={ref_extrapolate!r}, ATLAS={args.allow_close_range_extrapolation!r}"
        )
    if manifest.get("status") != "completed":
        errors.append(f"reference status is {manifest.get('status')!r}, not 'completed'")
    if errors:
        raise ValueError("Pairing-reference manifest mismatch:\n  - " + "\n  - ".join(errors))


def validate_pairing_frame(
    reference: dict[str, dict[str, Any]],
    *,
    frame_id: str,
    segment: str,
    local_index: int,
    scheduled: bool,
    tracking_record: dict[str, Any] | None,
    target_box: np.ndarray | None,
    fired: bool | None = None,
    box_atol: float,
) -> None:
    ref = reference.get(frame_id)
    if ref is None:
        raise KeyError(f"Frame {frame_id!r} is absent from pairing reference")
    errors: list[str] = []
    if ref["segment_id"] != segment:
        errors.append(f"segment reference={ref['segment_id']!r}, ATLAS={segment!r}")
    if ref["local_frame_index"] != int(local_index):
        errors.append(
            f"local index reference={ref['local_frame_index']!r}, ATLAS={local_index!r}"
        )
    if bool(ref["attack_scheduled"]) != bool(scheduled):
        errors.append(
            f"scheduled reference={ref['attack_scheduled']!r}, ATLAS={scheduled!r}"
        )
    if scheduled:
        tracking_ok = target_box is not None
        if bool(ref["tracking_ok"]) != tracking_ok:
            errors.append(
                f"tracking_ok reference={ref['tracking_ok']!r}, ATLAS={tracking_ok!r}"
            )
        if tracking_ok:
            assert tracking_record is not None
            if ref["target_box_index"] != tracking_record.get("target_box_index"):
                errors.append(
                    "target_box_index reference="
                    f"{ref['target_box_index']!r}, ATLAS={tracking_record.get('target_box_index')!r}"
                )
            if ref["target_box"] is None:
                errors.append("reference target box is missing")
            elif not np.allclose(
                np.asarray(ref["target_box"], dtype=np.float64),
                np.asarray(target_box[:7], dtype=np.float64),
                rtol=0.0,
                atol=box_atol,
            ):
                errors.append("target box differs")
        if fired is not None and bool(ref["active_this_frame"]) != bool(fired):
            errors.append(
                f"firing reference={ref['active_this_frame']!r}, ATLAS={bool(fired)!r}"
            )
    if errors:
        raise ValueError(
            f"Pairing validation failed for {frame_id}: " + "; ".join(errors)
        )


# ============================================================================
# Scheduling and target tracking: copied from the paired baseline semantics
# ============================================================================


def extract_one_runs(schedule: dict[int, int | None]) -> list[list[int]]:
    runs: list[list[int]] = []
    current: list[int] = []
    for index in sorted(schedule):
        if schedule[index] == 1:
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def track_sequence(
    dataset: Any,
    start: int,
    end: int,
    desired_range: float,
    target_class: int,
    max_step: float,
    allow_behind: bool,
) -> tuple[list[np.ndarray | None], list[dict[str, Any]], dict[str, Any]]:
    length = end - start + 1
    boxes: list[np.ndarray | None] = [None] * length
    records: list[dict[str, Any]] = []

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
            previous_box,
            previous_previous_box,
            current_boxes,
            target_class=target_class,
            max_step=max_step,
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
            "tracking_step_distance_m": float(
                np.linalg.norm(new_box[:3] - previous_box[:3])
            ),
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
        record["tracking_match_distance_m"]
        for record in records
        if record["tracking_match_distance_m"] is not None
    ]
    summary = {
        "tracking_ok": True,
        "tracking_failure_reason": None if len(valid_boxes) == length else "tracking_lost",
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


def build_target_schedule(
    segment_to_global: dict[str, list[int]], block_length: int
) -> dict[str, dict[int, int | None]]:
    schedules: dict[str, dict[int, int | None]] = {}
    for segment, global_indices in segment_to_global.items():
        toggle = False
        schedule: dict[int, int | None] = {}
        for local_index, global_index in enumerate(global_indices):
            if local_index == 0:
                schedule[global_index] = None
                continue
            if local_index % block_length == 0:
                toggle = not toggle
            schedule[global_index] = 1 if toggle else None
        schedules[segment] = schedule
    return schedules


# ============================================================================
# Original ATLAS A-HFR frame algorithm with paired RNG streams
# ============================================================================


def _points_in_oriented_box(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
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


def _radial_rows(
    points: np.ndarray,
    sector_mask: np.ndarray,
    remove_mask: np.ndarray,
    target_mask: np.ndarray,
    target_box: np.ndarray,
    radial_edges: np.ndarray,
) -> list[dict[str, Any]]:
    planar_radius = np.linalg.norm(points[:, :2], axis=1)
    target_planar_range = float(np.linalg.norm(np.asarray(target_box[:2])))
    rows: list[dict[str, Any]] = []
    for index in range(len(radial_edges) - 1):
        lower = float(radial_edges[index])
        upper = float(radial_edges[index + 1])
        if index == len(radial_edges) - 2:
            radial_mask = (planar_radius >= lower) & (planar_radius <= upper)
        else:
            radial_mask = (planar_radius >= lower) & (planar_radius < upper)
        before_count = int(np.sum(radial_mask))
        sector_count = int(np.sum(radial_mask & sector_mask))
        removed_count = int(np.sum(radial_mask & remove_mask))
        target_before = int(np.sum(radial_mask & target_mask))
        target_removed = int(np.sum(radial_mask & target_mask & remove_mask))
        annular_area = np.pi * (upper ** 2 - lower ** 2)
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
            "sector_count_before": sector_count,
            "scene_count_removed": removed_count,
            "scene_count_after": before_count - removed_count,
            "scene_removal_fraction": _safe_div(removed_count, before_count),
            "sector_removal_fraction": _safe_div(removed_count, sector_count),
            "target_count_before": target_before,
            "target_count_removed": target_removed,
            "target_count_after": target_before - target_removed,
            "target_removal_fraction": _safe_div(target_removed, target_before),
            "scene_density_before_points_per_m2": _safe_div(before_count, annular_area),
            "scene_density_removed_points_per_m2": _safe_div(removed_count, annular_area),
        })
    return rows


def audited_atlas_frame(
    *,
    points: np.ndarray,
    target_box: np.ndarray,
    state: dict[str, float] | None,
    args: argparse.Namespace,
    p_remove: float,
    attack_rng: np.random.Generator,
    firing_rng: np.random.Generator,
    radial_edges: np.ndarray,
) -> tuple[
    np.ndarray,
    dict[str, float],
    dict[str, Any],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Apply the original ATLAS roadside A-HFR removal algorithm.

    The geometric/state/removal sequence is the same as
    removal.spoof_frame_ahfr_roadside. Only the random streams are split so the
    frame-level firing draw can be paired exactly with the adverse baselines.
    """
    points = np.asarray(points)
    target_box = np.asarray(target_box).copy()
    if state is None:
        state = {}

    raw_az, raw_el = removal._box_center_to_az_el(target_box)
    target_range_m = float(np.linalg.norm(target_box[:3]))

    if "az_center" not in state:
        az_center = raw_az
    else:
        az_center = removal._clamp_angle_step(
            state["az_center"],
            raw_az,
            max_step_rad=np.deg2rad(args.max_az_step_deg),
        )
    if "el_center" not in state:
        el_center = raw_el
    else:
        # Original roadside attack uses fix_elevation=True.
        el_center = state["el_center"]

    jitter_deg = float(attack_rng.normal(0.0, args.center_jitter_std_deg))
    az_center = removal._wrap_to_pi(az_center + np.deg2rad(jitter_deg))
    new_state = {"az_center": float(az_center), "el_center": float(el_center)}

    target_mask = _points_in_oriented_box(points[:, :3], target_box)
    sector_mask = np.zeros(len(points), dtype=bool)
    remove_mask = np.zeros(len(points), dtype=bool)
    scan_gate_probability = float(removal.ahfr_gate_prob(target_range_m))
    scan_gate_draw: float | None = None
    fired = False
    inactive_reason: str | None = None
    range_gate_passed = True

    if target_range_m < args.min_target_range_m and not args.allow_close_range_extrapolation:
        range_gate_passed = False
        inactive_reason = "target_too_close"
    else:
        scan_gate_draw = float(firing_rng.random())
        fired = bool(scan_gate_draw < scan_gate_probability)
        if not fired:
            inactive_reason = "scan_gate"
        else:
            sector_mask = removal._sector_mask_from_center(
                points[:, :3],
                az_center=az_center,
                el_center=el_center,
                az_width_deg=args.az_width,
                el_width_deg=args.el_width_deg,
            )
            sector_indices = np.flatnonzero(sector_mask)
            if sector_indices.size == 0:
                inactive_reason = "empty_sector"
            else:
                # This is the original independent within-sector PRR draw.
                remove_draw = attack_rng.random(sector_indices.size) < p_remove
                remove_indices = sector_indices[remove_draw]
                remove_mask[remove_indices] = True

    keep_mask = ~remove_mask
    attacked_points = points[keep_mask].copy()
    removed_indices = np.flatnonzero(remove_mask)
    sector_indices = np.flatnonzero(sector_mask)
    target_indices = np.flatnonzero(target_mask)

    number_before = int(len(points))
    number_after = int(len(attacked_points))
    number_removed = int(len(removed_indices))
    number_sector = int(len(sector_indices))
    number_target = int(np.sum(target_mask))
    number_target_removed = int(np.sum(target_mask & remove_mask))
    number_background_removed = int(np.sum((~target_mask) & remove_mask))

    audit: dict[str, Any] = {
        "corruption_mode": "ahfr_point_removal",
        "attack_scheduled": True,
        "tracking_ok": True,
        "active_this_frame": bool(fired),
        "modified_this_frame": bool(number_removed > 0),
        "corruption_applied": bool(number_removed > 0),
        "inactive_reason": inactive_reason,
        "firing_source": "paired_stable_seed",
        "scan_gate_probability": scan_gate_probability,
        "scan_gate_draw": scan_gate_draw,
        "scan_gate_passed": bool(fired),
        "range_gate_passed": bool(range_gate_passed),
        "extrapolated_close_range": bool(
            target_range_m < args.min_target_range_m
            and args.allow_close_range_extrapolation
        ),
        "az_center_rad": float(az_center),
        "el_center_rad": float(el_center),
        "raw_target_az_rad": float(raw_az),
        "raw_target_el_rad": float(raw_el),
        "center_jitter_draw_deg": jitter_deg,
        "az_width_deg": int(args.az_width),
        "el_width_deg": float(args.el_width_deg),
        "max_az_step_deg": float(args.max_az_step_deg),
        "fix_elevation": True,
        "p_remove": float(p_remove),
        "target_box_x": float(target_box[0]),
        "target_box_y": float(target_box[1]),
        "target_box_z": float(target_box[2]),
        "target_box_length": float(target_box[3]),
        "target_box_width": float(target_box[4]),
        "target_box_height": float(target_box[5]),
        "target_box_yaw": float(target_box[6]),
        "target_range_3d_m": target_range_m,
        "target_range_m": target_range_m,
        "n_points_before": number_before,
        "n_points_in_sector": number_sector,
        "n_points_removed": number_removed,
        "n_points_removed_total": number_removed,
        "n_points_after": number_after,
        "n_target_points_before": number_target,
        "n_target_points_removed": number_target_removed,
        "n_target_points_after": number_target - number_target_removed,
        "target_removal_rate": _safe_div(number_target_removed, number_target),
        "n_background_points_removed": number_background_removed,
        "removed_points_target_fraction": _safe_div(number_target_removed, number_removed),
        "collateral_removal_fraction": _safe_div(number_background_removed, number_removed),
        "point_balance_error": int(number_before - number_after - number_removed),
        "output_subset_check_passed": bool(
            np.array_equal(attacked_points, points[keep_mask])
        ),
        "added_point_count": 0,
        "target_valid_reason": "valid",
    }

    arrays = {
        "removed_indices": removed_indices,
        "target_indices": target_indices,
        "sector_indices": sector_indices,
        "removed_points": points[remove_mask].astype(np.float32, copy=False),
    }
    radial_rows = _radial_rows(
        points, sector_mask, remove_mask, target_mask, target_box, radial_edges
    )

    point_rows: list[dict[str, Any]] = []
    if args.save_point_audit and number_removed:
        xyz = np.asarray(points[:, :3], dtype=np.float64)
        planar = np.linalg.norm(xyz[:, :2], axis=1)
        radius = np.linalg.norm(xyz, axis=1)
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        elevation = np.arctan2(xyz[:, 2], np.maximum(planar, 1e-12))
        radial_bin = np.searchsorted(radial_edges, planar, side="right") - 1
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
                "planar_range_m": float(planar[original_index]),
                "azimuth_rad": float(azimuth[original_index]),
                "elevation_rad": float(elevation[original_index]),
                "radial_bin_index": int(radial_bin[original_index]),
                "inside_target_box": bool(target_mask[original_index]),
                "inside_attack_sector": bool(sector_mask[original_index]),
            })

    return attacked_points, new_state, audit, arrays, radial_rows, point_rows


# ============================================================================
# Per-segment generation and persistence
# ============================================================================


def setup_segment(
    *,
    segment: str,
    dataset_mf: Any,
    dataset_sf: Any,
    target_frames_global: dict[str, dict[int, int | None]],
    global_to_segment: dict[int, tuple[str, int]],
    mf_to_sf_global: dict[int, int],
    original_mf_indices: list[int],
    args: argparse.Namespace,
    p_remove: float,
    radial_edges: np.ndarray,
    pairing_reference: dict[str, dict[str, Any]] | None,
) -> tuple[
    dict[int, np.ndarray | None],
    dict[int, np.ndarray],
    dict[int, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    target_frames = target_frames_global[segment].copy()
    removal_cache: dict[int, np.ndarray] = {}
    removal_annos: dict[int, dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []
    array_records: list[dict[str, Any]] = []
    radial_records: list[dict[str, Any]] = []
    point_records: list[dict[str, Any]] = []

    for run in extract_one_runs(target_frames):
        window_start = min(run)
        window_end = max(run)
        start_local = global_to_segment[window_start][1]
        end_local = global_to_segment[window_end][1]
        run_seed = _stable_seed(args.seed, segment, start_local, end_local)
        pairing_run_id = f"{segment}:{start_local:04d}-{end_local:04d}"
        run_id = f"{args.variant_name}:{pairing_run_id}"

        tracked_boxes, tracking_records, tracking_summary = track_sequence(
            dataset_mf,
            start=window_start,
            end=window_end,
            desired_range=args.desired_target_range_m,
            target_class=args.target_class,
            max_step=args.tracking_max_step_m,
            allow_behind=not args.front_targets_only,
        )
        if len(tracked_boxes) != len(run) or len(tracking_records) != len(run):
            raise RuntimeError("Tracking output length mismatch")

        run_record: dict[str, Any] = {
            "schema_version": args.schema_version,
            "variant_name": args.variant_name,
            "corruption_mode": "ahfr_point_removal",
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
            "az_width_deg": int(args.az_width),
            "el_width_deg": float(args.el_width_deg),
            "configured_point_removal_probability": float(p_remove),
            "frame_attack_rng_namespace": "matched",
            "firing_rng_namespace": "firing",
        }
        run_record.update(tracking_summary)

        active_frames = 0
        modified_frames = 0
        total_removed = 0
        total_target_removed = 0
        total_background_removed = 0

        state: dict[str, float] | None = None
        for offset, global_index in enumerate(sorted(run)):
            local_index = global_to_segment[global_index][1]
            original_global_index = int(original_mf_indices[global_index])
            frame_id = _frame_id_from_info(dataset_mf.infos[global_index], global_index)
            box = tracked_boxes[offset]
            tracking_record = tracking_records[offset]

            base_context: dict[str, Any] = {
                "schema_version": args.schema_version,
                "variant_name": args.variant_name,
                "corruption_mode": "ahfr_point_removal",
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

            if pairing_reference is not None:
                validate_pairing_frame(
                    pairing_reference,
                    frame_id=frame_id,
                    segment=segment,
                    local_index=local_index,
                    scheduled=True,
                    tracking_record=tracking_record,
                    target_box=box,
                    fired=None,
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

            # EXACT namespaces used by generate_data_adverse_baselines.py.
            frame_seed = _stable_seed(args.seed, segment, local_index, "matched")
            firing_seed = _stable_seed(args.seed, segment, local_index, "firing")
            attack_rng = np.random.default_rng(frame_seed)
            firing_rng = np.random.default_rng(firing_seed)

            (
                attacked_points,
                state,
                audit,
                arrays,
                radial_rows,
                removed_point_rows,
            ) = audited_atlas_frame(
                points=points,
                target_box=box,
                state=state,
                args=args,
                p_remove=p_remove,
                attack_rng=attack_rng,
                firing_rng=firing_rng,
                radial_edges=radial_edges,
            )

            if pairing_reference is not None:
                validate_pairing_frame(
                    pairing_reference,
                    frame_id=frame_id,
                    segment=segment,
                    local_index=local_index,
                    scheduled=True,
                    tracking_record=tracking_record,
                    target_box=box,
                    fired=bool(audit["active_this_frame"]),
                    box_atol=args.pairing_box_atol,
                )

            attacked_points = attacked_points.astype(np.float32, copy=False)
            removal_cache[global_index] = attacked_points
            target_frames[global_index] = box.copy()
            audit.update(base_context)
            audit["frame_seed"] = int(frame_seed)
            audit["firing_seed"] = int(firing_seed)
            removal_annos[local_index] = audit

            common = {
                key: base_context[key]
                for key in (
                    "schema_version", "variant_name", "corruption_mode",
                    "segment_id", "frame_id", "original_global_frame_index",
                    "local_frame_index", "run_id", "pairing_run_id",
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
            total_background_removed += int(audit["n_background_points_removed"])

        run_record.update({
            "active_frame_count": int(active_frames),
            "modified_frame_count": int(modified_frames),
            "total_removed_point_count": int(total_removed),
            "total_target_removed_point_count": int(total_target_removed),
            "total_background_removed_point_count": int(total_background_removed),
            "aggregate_removed_points_target_fraction": _safe_div(
                total_target_removed, total_removed
            ),
        })
        run_records.append(run_record)

    return (
        target_frames,
        removal_cache,
        removal_annos,
        run_records,
        array_records,
        radial_records,
        point_records,
    )


def save_segment(
    *,
    args: argparse.Namespace,
    segment: str,
    segment_dataset: list[dict[str, Any]],
    removal_annos: dict[int, dict[str, Any]],
    run_records: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    radial_records: list[dict[str, Any]],
    array_records: list[dict[str, Any]],
    point_records: list[dict[str, Any]],
) -> None:
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
    _atomic_write_csv(metrics / "frames" / f"{segment}_frames.csv", frame_records)
    _atomic_write_csv(metrics / "runs" / f"{segment}_runs.csv", run_records)
    _atomic_write_csv(metrics / "radial" / f"{segment}_radial.csv", radial_records)
    if not args.no_save_audit_arrays:
        _atomic_write_audit_arrays(
            metrics / "arrays" / f"{segment}_removal_arrays.npz", array_records
        )
    if args.save_point_audit:
        _atomic_write_csv(
            metrics / "points" / f"{segment}_removed_points.csv", point_records
        )


def _segment_is_complete(args: argparse.Namespace, segment: str) -> bool:
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
        required.append(metrics / "arrays" / f"{segment}_removal_arrays.npz")
    if args.save_point_audit:
        required.append(metrics / "points" / f"{segment}_removed_points.csv")
    if not all(path.exists() for path in required):
        return False
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


def build_manifest(
    args: argparse.Namespace,
    selected_segments: list[str],
    p_remove: float,
    radial_max_m: float,
    pairing_reference_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": args.schema_version,
        "generator_name": "generate_data_removal_rebuttal",
        "variant_name": args.variant_name,
        "master_seed": int(args.seed),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "removal_module_path": str(Path(removal.__file__).resolve()),
        "removal_module_sha256": _sha256(Path(removal.__file__).resolve()),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "running",
        "selected_sequence_count": int(len(selected_segments)),
        "selected_sequences": list(selected_segments),
        "sequence_selection": args.sequence_selection,
        "requested_num_sequences": args.num_sequences,
        "attack_configuration": {
            "mode": "ahfr_point_removal",
            "az_width_deg": int(args.az_width),
            "el_width_deg": float(args.el_width_deg),
            "configured_point_removal_probability": float(p_remove),
            "max_az_step_deg": float(args.max_az_step_deg),
            "center_jitter_std_deg": float(args.center_jitter_std_deg),
            "min_target_range_m": float(args.min_target_range_m),
            "allow_close_range_extrapolation": bool(
                args.allow_close_range_extrapolation
            ),
            "attack_block_length": int(args.attack_block_length),
            "desired_target_range_m": float(args.desired_target_range_m),
            "tracking_max_step_m": float(args.tracking_max_step_m),
            "target_class": int(args.target_class),
            "front_targets_only": bool(args.front_targets_only),
            "fix_elevation": True,
        },
        "pairing_configuration": {
            "selection_rng": "np.random.default_rng(master_seed)",
            "firing_rng_seed": "stable_seed(seed, segment, local_index, 'firing')",
            "attack_rng_seed": "stable_seed(seed, segment, local_index, 'matched')",
            "firing_width_independent": True,
            "attack_rng_width_independent": True,
            "pairing_reference_metrics": args.pairing_reference_metrics,
            "pairing_reference_generator": (
                pairing_reference_manifest.get("generator_name")
                if pairing_reference_manifest is not None else None
            ),
            "strict_frame_pairing_validation": pairing_reference_manifest is not None,
            "pairing_box_atol": float(args.pairing_box_atol),
        },
        "audit_configuration": {
            "radial_bin_width_m": float(args.radial_bin_width_m),
            "radial_max_m": float(radial_max_m),
            "save_audit_arrays": not args.no_save_audit_arrays,
            "save_point_audit": bool(args.save_point_audit),
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


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    if args.attack_block_length <= 0:
        raise ValueError("--attack_block_length must be positive")
    if args.num_sequences is not None and args.num_sequences <= 0:
        raise ValueError("--num_sequences must be positive")
    if args.radial_bin_width_m <= 0:
        raise ValueError("--radial_bin_width_m must be positive")
    if args.pairing_box_atol < 0:
        raise ValueError("--pairing_box_atol must be nonnegative")

    np.random.seed(args.seed)
    random.seed(args.seed)
    p_remove = P_REMOVE_BY_AZ_WIDTH[args.az_width]

    Path(args.save_dir_dataset).mkdir(parents=True, exist_ok=True)
    Path(args.save_dir_annos).mkdir(parents=True, exist_ok=True)
    Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    for directory in ("frames", "runs", "radial", "arrays", "points"):
        (metrics_dir / directory).mkdir(parents=True, exist_ok=True)

    log = logging.getLogger("atlas_removal_rebuttal")
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
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=mf_logger,
        training=False,
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
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=sf_logger,
        training=False,
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
    elif args.num_sequences > len(all_common_segments):
        raise ValueError(
            f"Requested {args.num_sequences} sequences, but only "
            f"{len(all_common_segments)} are common"
        )
    elif args.sequence_selection == "first":
        selected_segments = all_common_segments[:args.num_sequences]
    else:
        # This is byte-for-byte the adverse-baseline selection rule.
        selection_rng = np.random.default_rng(args.seed)
        selected_segments = sorted(selection_rng.choice(
            all_common_segments, size=args.num_sequences, replace=False
        ).tolist())
    selected_set = set(selected_segments)

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
        dataset_mf,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=dataset_mf.collate_batch,
    )
    log.info(
        "Selected %d of %d common segments using %s selection, seed=%d",
        len(selected_segments), len(all_common_segments),
        args.sequence_selection, args.seed,
    )

    pairing_manifest = None
    pairing_reference = None
    if args.pairing_reference_metrics:
        pairing_manifest, pairing_reference = load_pairing_reference(
            args.pairing_reference_metrics
        )
        validate_pairing_manifest(pairing_manifest, args, selected_segments)
        log.info(
            "Loaded strict baseline pairing reference: %d frames from %s",
            len(pairing_reference), args.pairing_reference_metrics,
        )

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
        0.0,
        radial_max_m + args.radial_bin_width_m * 0.5,
        args.radial_bin_width_m,
        dtype=np.float64,
    )
    if radial_edges[-1] < radial_max_m:
        radial_edges = np.r_[radial_edges, radial_max_m]

    manifest_path = metrics_dir / "run_manifest.json"
    manifest = build_manifest(
        args, selected_segments, p_remove, radial_max_m, pairing_manifest
    )
    _atomic_write_json(manifest_path, manifest)

    segment_to_global: dict[str, list[int]] = defaultdict(list)
    global_to_segment: dict[int, tuple[str, int]] = {}
    for global_index, info in enumerate(dataset_mf.infos):
        segment = info["point_cloud"]["lidar_sequence"]
        local_index = len(segment_to_global[segment])
        segment_to_global[segment].append(global_index)
        global_to_segment[global_index] = (segment, local_index)

    sf_segment_to_global: dict[str, list[int]] = defaultdict(list)
    for sf_index, info in enumerate(dataset_sf.infos):
        segment = info["frame_id"].rsplit("_", 1)[0]
        sf_segment_to_global[segment].append(sf_index)

    mf_to_sf_global: dict[int, int] = {}
    for mf_index, (segment, local_index) in global_to_segment.items():
        if local_index >= len(sf_segment_to_global[segment]):
            raise RuntimeError(f"Missing SF frame for {segment}[{local_index}]")
        sf_index = sf_segment_to_global[segment][local_index]
        mf_frame_id = _frame_id_from_info(dataset_mf.infos[mf_index], mf_index)
        sf_frame_id = str(dataset_sf.infos[sf_index]["frame_id"])
        if mf_frame_id != sf_frame_id:
            raise RuntimeError(
                "MF/SF frame-ID mismatch: "
                f"MF={mf_frame_id!r}, SF={sf_frame_id!r}, "
                f"segment={segment!r}, local_index={local_index}"
            )
        mf_to_sf_global[mf_index] = sf_index
    for segment in selected_segments:
        if len(segment_to_global[segment]) != len(sf_segment_to_global[segment]):
            raise RuntimeError(
                f"MF/SF length mismatch for {segment}: "
                f"MF={len(segment_to_global[segment])}, "
                f"SF={len(sf_segment_to_global[segment])}"
            )

    target_frames_global = build_target_schedule(
        segment_to_global, args.attack_block_length
    )
    completed = {
        segment for segment in selected_segments if _segment_is_complete(args, segment)
    }
    if completed:
        log.info("Resuming with %d complete segments skipped", len(completed))

    segment_dataset: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    current_segment: str | None = None
    current_payload = None

    def start_segment(segment: str):
        log.info("Starting: %s", segment)
        return setup_segment(
            segment=segment,
            dataset_mf=dataset_mf,
            dataset_sf=dataset_sf,
            target_frames_global=target_frames_global,
            global_to_segment=global_to_segment,
            mf_to_sf_global=mf_to_sf_global,
            original_mf_indices=original_mf_indices,
            args=args,
            p_remove=p_remove,
            radial_edges=radial_edges,
            pairing_reference=pairing_reference,
        )

    def flush_segment(segment: str, payload: Any) -> None:
        (
            _target_frames,
            _removal_cache,
            removal_annos,
            run_records,
            array_records,
            radial_records,
            point_records,
        ) = payload
        save_segment(
            args=args,
            segment=segment,
            segment_dataset=segment_dataset,
            removal_annos=removal_annos,
            run_records=run_records,
            frame_records=frame_records,
            radial_records=radial_records,
            array_records=array_records,
            point_records=point_records,
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

            (
                target_frames,
                removal_cache,
                removal_annos,
                _run_records,
                _array_records,
                _radial_records,
                _point_records,
            ) = current_payload

            local_index = global_to_segment[filtered_global_index][1]
            original_global_index = int(original_mf_indices[filtered_global_index])
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
            single_frame_points = single_frame_points.astype(np.float32, copy=False)

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
                    "corruption_mode": "ahfr_point_removal",
                    "segment_id": segment,
                    "frame_id": frame_id,
                    "filtered_global_frame_index": int(filtered_global_index),
                    "original_global_frame_index": original_global_index,
                    "local_frame_index": int(local_index),
                    "attack_scheduled": False,
                    "tracking_ok": None,
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
                if pairing_reference is not None:
                    validate_pairing_frame(
                        pairing_reference,
                        frame_id=frame_id,
                        segment=segment,
                        local_index=local_index,
                        scheduled=False,
                        tracking_record=None,
                        target_box=None,
                        fired=None,
                        box_atol=args.pairing_box_atol,
                    )
            else:
                frame_record = dict(attack_meta)
                frame_record.setdefault("n_points_before", int(len(single_frame_points)))
                frame_record.setdefault("n_points_after", int(len(single_frame_points)))
                frame_record.setdefault("n_points_removed", 0)
                frame_record.setdefault("n_points_removed_total", 0)
                frame_record.setdefault("point_balance_error", 0)

                attack_active = bool(attack_meta.get("active_this_frame", False))
                target_box = target_frames.get(filtered_global_index)
                modified["spoof_gt"] = (
                    target_box.copy()
                    if attack_active and target_box is not None else None
                )
                compact_keys = (
                    "corruption_mode", "active_this_frame", "modified_this_frame",
                    "corruption_applied", "inactive_reason", "n_points_before",
                    "n_points_in_sector", "n_points_removed", "n_points_after",
                    "target_range_m", "target_valid_reason", "az_width_deg",
                    "el_width_deg", "p_remove",
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
        log.info(
            "Completed %d segments. Pairing validation=%s",
            len(selected_segments), pairing_reference is not None,
        )
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
