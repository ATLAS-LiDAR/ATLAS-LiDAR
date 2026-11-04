#!/usr/bin/env python3
"""ATLAS point-removal *eval-only* metrics.

This script does **NO inference**. It consumes cached per-segment predictions
(spoofed / attacked), a clean full-validation result.pkl, and the point-removal
annos files, and computes victim-centric diagnostics: hiding ASR, victim recall
vs score threshold, victim-box IoU distribution, confidence degradation,
victim-box drift, target-region false positives (stray/fragmented detections),
and attack-strength-conditioned recall using the per-frame removal telemetry.

It builds NO models/dataloaders/checkpoints and needs NO GPU. 3D IoU is the same
exact numba rotated-box kernel used by the injection eval.

Discovery differs from injection: the victim (target) box comes directly from
each removal record's ``target_box`` in ``{segment}_a.pkl`` -> ``spoof_annos``
(keyed by frame index), NOT from a ``spoof_gt`` field on the dataset frame. A
frame is a target frame when its removal record is active.
"""

import argparse
import math
import json
import logging
import os
import pickle as pkl
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# small utilities (kept byte-for-byte compatible with the original scripts)    #
# --------------------------------------------------------------------------- #
def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def to_numpy(value, dtype=None):
    if value is None:
        return None
    try:
        import torch
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        else:
            value = np.asarray(value)
    except ImportError:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def prediction_arrays(frame_pred):
    """Identical extraction to the original scripts."""
    boxes = frame_pred.get('boxes_lidar', frame_pred.get('pred_boxes'))
    labels = frame_pred.get('pred_labels')
    scores = frame_pred.get('score', frame_pred.get('pred_scores'))
    boxes = to_numpy(boxes, np.float32)
    labels = to_numpy(labels, np.int64)
    scores = to_numpy(scores, np.float32)
    if boxes is None:
        boxes = np.empty((0, 7), dtype=np.float32)
    if labels is None:
        labels = np.empty((0,), dtype=np.int64)
    if scores is None:
        scores = np.empty((0,), dtype=np.float32)
    return boxes, labels, scores


def vehicle_predictions(frame_pred, vehicle_class_id):
    """Vehicle-class boxes+scores from a prediction anno dict.

    Preserves the original label convention (pred_labels == 1 -> Vehicle) with a
    string-name fallback when pred_labels are absent (e.g. a leaner clean
    result.pkl). Returns (boxes[:, :7], scores) with matching first dim.
    """
    boxes, labels, scores = prediction_arrays(frame_pred)
    if len(boxes) == 0:
        return boxes[:, :7] if boxes.ndim == 2 else np.empty((0, 7), np.float32), scores
    if labels is not None and labels.shape[0] == boxes.shape[0] and labels.size > 0:
        mask = labels == vehicle_class_id
    else:
        names = frame_pred.get('name')
        if names is not None and len(names) == boxes.shape[0]:
            mask = np.asarray([str(n).lower() == 'vehicle' for n in names], dtype=bool)
        else:
            mask = np.ones(boxes.shape[0], dtype=bool)
    return boxes[mask][:, :7].astype(np.float32, copy=False), scores[mask].astype(np.float32, copy=False)


def real_gt_vehicle_boxes(frame, vehicle_class_id):
    """Real GT vehicle boxes (N,7) for a frame.

    Primary source is ``frame['gt_boxes']`` (N,8: [x,y,z,l,w,h,yaw,class] in the
    detector coordinate frame, class 1-based). Falls back to a Waymo-style
    ``frame['annos']`` dict if gt_boxes is absent.
    """
    gt = frame.get('gt_boxes')
    if gt is None:
        annos = frame.get('annos')
        if isinstance(annos, dict):
            boxes = annos.get('gt_boxes_lidar', annos.get('boxes_lidar'))
            names = annos.get('name')
            boxes = to_numpy(boxes, np.float32)
            if boxes is None or len(boxes) == 0:
                return np.empty((0, 7), np.float32)
            if names is not None and len(names) == len(boxes):
                mask = np.asarray([str(n).lower() == 'vehicle' for n in names], dtype=bool)
                return boxes[mask][:, :7]
            return boxes[:, :7]
        return np.empty((0, 7), np.float32)
    gt = to_numpy(gt, np.float32)
    if gt.ndim != 2 or gt.shape[0] == 0:
        return np.empty((0, 7), np.float32)
    if gt.shape[1] >= 8:
        labels = gt[:, -1].astype(np.int64)
        gt = gt[labels == vehicle_class_id]
    return gt[:, :7].astype(np.float32, copy=False)


def spoof_box_7(frame):
    """7-dim spoof box, or None on a clean frame."""
    spoof = frame.get('spoof_gt')
    if spoof is None:
        return None
    return to_numpy(spoof, np.float32).reshape(-1)[:7]


# --------------------------------------------------------------------------- #
# CPU rotated-3D IoU — numba JIT, exact (Sutherland-Hodgman + height overlap)   #
# --------------------------------------------------------------------------- #
# Exact rotated 3D IoU: BEV rotated-rectangle intersection area via
# Sutherland-Hodgman polygon clipping, times z-axis height overlap. Same math
# as pcdet's boxes_iou3d_gpu; no GPU. numba compiles it to parallel machine
# code (~400x faster than a Shapely python double loop).
import numba


@numba.njit(cache=True, fastmath=True)
def _corners_2d(cx, cy, l, w, yaw):
    cos = math.cos(yaw); sin = math.sin(yaw)
    dx = l / 2.0; dy = w / 2.0
    out = np.empty((4, 2), np.float64)
    out[0, 0] =  dx * cos - dy * sin + cx; out[0, 1] =  dx * sin + dy * cos + cy
    out[1, 0] = -dx * cos - dy * sin + cx; out[1, 1] = -dx * sin + dy * cos + cy
    out[2, 0] = -dx * cos + dy * sin + cx; out[2, 1] = -dx * sin - dy * cos + cy
    out[3, 0] =  dx * cos + dy * sin + cx; out[3, 1] =  dx * sin - dy * cos + cy
    return out


@numba.njit(cache=True, fastmath=True)
def _line_intersect_xy(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    d1x = p2x - p1x; d1y = p2y - p1y
    d2x = p4x - p3x; d2y = p4y - p3y
    cross = d1x * d2y - d1y * d2x
    if abs(cross) < 1e-12:
        return p1x, p1y
    t = ((p3x - p1x) * d2y - (p3y - p1y) * d2x) / cross
    return p1x + t * d1x, p1y + t * d1y


@numba.njit(cache=True, fastmath=True)
def _clip_polygon_sh(subj, clip):
    # Sutherland-Hodgman clip of convex subj by convex clip. Two quads clip to
    # at most 8 vertices; 16-slot fixed buffers are safe.
    buf_a = np.empty((16, 2), np.float64)
    buf_b = np.empty((16, 2), np.float64)
    n_a = subj.shape[0]
    for i in range(n_a):
        buf_a[i, 0] = subj[i, 0]; buf_a[i, 1] = subj[i, 1]

    nc = clip.shape[0]
    use_a = True
    for i in range(nc):
        if n_a == 0:
            break
        if use_a:
            inp = buf_a; out = buf_b
        else:
            inp = buf_b; out = buf_a
        n_inp = n_a
        n_out = 0
        ex = clip[i, 0]; ey = clip[i, 1]
        fx = clip[(i + 1) % nc, 0]; fy = clip[(i + 1) % nc, 1]
        evx = fx - ex; evy = fy - ey
        for j in range(n_inp):
            cx_ = inp[j, 0]; cy_ = inp[j, 1]
            pj = (j - 1) % n_inp
            px = inp[pj, 0]; py = inp[pj, 1]
            c_inside = evx * (cy_ - ey) - evy * (cx_ - ex) >= 0
            p_inside = evx * (py - ey) - evy * (px - ex) >= 0
            if c_inside:
                if not p_inside:
                    ix, iy = _line_intersect_xy(px, py, cx_, cy_, ex, ey, fx, fy)
                    if n_out < 16:
                        out[n_out, 0] = ix; out[n_out, 1] = iy; n_out += 1
                if n_out < 16:
                    out[n_out, 0] = cx_; out[n_out, 1] = cy_; n_out += 1
            elif p_inside:
                ix, iy = _line_intersect_xy(px, py, cx_, cy_, ex, ey, fx, fy)
                if n_out < 16:
                    out[n_out, 0] = ix; out[n_out, 1] = iy; n_out += 1
        n_a = n_out
        use_a = not use_a

    if use_a:
        return buf_a[:n_a].copy()
    return buf_b[:n_a].copy()


@numba.njit(cache=True, fastmath=True)
def _poly_area(pts):
    n = pts.shape[0]
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        j = (i + 1) % n
        a += pts[i, 0] * pts[j, 1] - pts[j, 0] * pts[i, 1]
    return abs(a) * 0.5


@numba.njit(cache=True, parallel=True, fastmath=True)
def iou3d_rotated_cpu(boxes_a, boxes_b):
    """Exact rotated 3D IoU. boxes_*: (N,7) [x,y,z,l,w,h,yaw]. Returns (Na,Nb)."""
    na = boxes_a.shape[0]; nb = boxes_b.shape[0]
    out = np.zeros((na, nb), np.float32)
    for i in numba.prange(na):
        ca = _corners_2d(boxes_a[i, 0], boxes_a[i, 1],
                         boxes_a[i, 3], boxes_a[i, 4], boxes_a[i, 6])
        vol_a = boxes_a[i, 3] * boxes_a[i, 4] * boxes_a[i, 5]
        za_lo = boxes_a[i, 2] - boxes_a[i, 5] / 2.0
        za_hi = boxes_a[i, 2] + boxes_a[i, 5] / 2.0
        for j in range(nb):
            cb = _corners_2d(boxes_b[j, 0], boxes_b[j, 1],
                             boxes_b[j, 3], boxes_b[j, 4], boxes_b[j, 6])
            inter_pts = _clip_polygon_sh(ca, cb)
            inter_area = _poly_area(inter_pts)
            if inter_area <= 0.0:
                continue
            zb_lo = boxes_b[j, 2] - boxes_b[j, 5] / 2.0
            zb_hi = boxes_b[j, 2] + boxes_b[j, 5] / 2.0
            h_over = min(za_hi, zb_hi) - max(za_lo, zb_lo)
            if h_over <= 0.0:
                continue
            inter_3d = inter_area * h_over
            vol_b = boxes_b[j, 3] * boxes_b[j, 4] * boxes_b[j, 5]
            union = vol_a + vol_b - inter_3d
            if union > 0.0:
                out[i, j] = inter_3d / union
    return out


def make_iou_fn(logger):
    """iou_fn(boxes_a, boxes_b) -> (Na, Nb) float32 numpy via inlined numba
    rotated-3D IoU. Triggers JIT compilation once up front."""
    logger.info('Compiling numba 3D IoU kernel (first run only, ~30-60s) ...')
    _w = np.zeros((1, 7), dtype=np.float64)
    iou3d_rotated_cpu(_w, _w)
    logger.info('numba 3D IoU ready (CPU, parallel, exact).')

    def iou_fn(boxes_a, boxes_b):
        boxes_a = np.asarray(boxes_a, dtype=np.float64)
        boxes_b = np.asarray(boxes_b, dtype=np.float64)
        na = boxes_a.shape[0] if boxes_a.ndim == 2 else 0
        nb = boxes_b.shape[0] if boxes_b.ndim == 2 else 0
        if na == 0 or nb == 0:
            return np.zeros((na, nb), dtype=np.float32)
        return iou3d_rotated_cpu(np.ascontiguousarray(boxes_a[:, :7]),
                                 np.ascontiguousarray(boxes_b[:, :7]))

    return iou_fn

# --------------------------------------------------------------------------- #
# discovery / loading / clean alignment                                        #
# --------------------------------------------------------------------------- #
def dataset_file(dataset_dir, segment):
    dataset_dir = Path(dataset_dir)
    for path in (dataset_dir / f'{segment}.pkl', dataset_dir / f'{segment}_d.pkl'):
        if path.is_file():
            return path
    raise FileNotFoundError(f'No dataset pickle for segment {segment!r} in {dataset_dir}')


def pred_file(pred_dir, segment):
    """Return the cached-prediction pickle for a segment. Accepts either
    ``{segment}_p.pkl`` (neurips convention) or ``{segment}.pkl`` (flat)."""
    pred_dir = Path(pred_dir)
    for path in (pred_dir / f'{segment}_p.pkl', pred_dir / f'{segment}.pkl'):
        if path.is_file():
            return path
    raise FileNotFoundError(
        f'No prediction pickle for segment {segment!r} in {pred_dir} '
        f'(tried {segment}_p.pkl and {segment}.pkl).'
    )


def discover_selected_segments(dataset_dir, annos_dir, pred_dir):
    """Segments ATLAS evaluated = those with an ``*_a.pkl`` annotation file that
    also have a dataset pickle and a cached prediction pickle."""
    anno_paths = sorted(Path(annos_dir).glob('*_a.pkl'))
    if not anno_paths:
        raise RuntimeError(f'No *_a.pkl files found in {annos_dir}')
    segments = [p.stem[:-2] for p in anno_paths]
    missing_ds, missing_pred = [], []
    for seg in segments:
        try:
            dataset_file(dataset_dir, seg)
        except FileNotFoundError:
            missing_ds.append(seg)
        try:
            pred_file(pred_dir, seg)
        except FileNotFoundError:
            missing_pred.append(seg)
    if missing_ds:
        raise FileNotFoundError(
            f'{len(missing_ds)} ATLAS segment(s) lack a dataset pickle in '
            f'{dataset_dir}, e.g. ' + ', '.join(missing_ds[:5])
        )
    if missing_pred:
        raise FileNotFoundError(
            f'{len(missing_pred)} ATLAS segment(s) lack a cached prediction '
            f'pickle in {pred_dir}, e.g. ' + ', '.join(missing_pred[:5])
            + '. Tried {segment}_p.pkl and {segment}.pkl.'
        )
    return segments


def subsample_segments(all_segments, num_segments, seed, logger):
    """Return a random subset of segments. Deterministic given seed."""
    if num_segments is None or num_segments <= 0 or num_segments >= len(all_segments):
        logger.info('Using ALL %d discovered segments.', len(all_segments))
        return all_segments
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_segments), size=num_segments, replace=False)
    idx.sort()
    selected = [all_segments[i] for i in idx]
    logger.info('Subsampled %d / %d segments (seed=%d).',
                len(selected), len(all_segments), seed)
    return selected


def load_atlas_frames(dataset_dir, pred_dir, segments, logger):
    """Build the ordered ATLAS frame list.

    For each segment, the cached spoofed-prediction pickle and the dataset
    pickle are index-aligned (this is how the original scripts wrote them). We
    assert equal length and per-index frame_id agreement, and emit one record
    per frame with its spoofed prediction, spoof box, and real GT vehicles.
    """
    frames = []
    # Force torch to restore any tensor (even nested ones) to CPU, regardless
    # of what device string was baked into the pickle. Torch's own map_location
    # doesn't propagate through nested `_load_from_bytes` calls, so we patch
    # the restore hook itself.
    import torch
    import torch.serialization as _ts
    _orig_restore = _ts.default_restore_location
    def _cpu_restore(storage, location):
        return _orig_restore(storage, 'cpu')
    _ts.default_restore_location = _cpu_restore
    try:
        for si, seg in enumerate(segments):
            logger.info('[load] segment %d / %d: %s', si + 1, len(segments), seg)
            ds_path = dataset_file(dataset_dir, seg)
            try:
                with open(ds_path, 'rb') as f:
                    try:
                        seg_data = torch.load(f, map_location=torch.device('cpu'),
                                              weights_only=False)
                    except TypeError:
                        # Older torch (< 2.4) doesn't accept weights_only kwarg
                        f.seek(0)
                        seg_data = torch.load(f, map_location=torch.device('cpu'))
            except (RuntimeError, pkl.UnpicklingError):
                # Not a torch-saved file — plain pickle
                with open(ds_path, 'rb') as f:
                    seg_data = pkl.load(f)
            with open(pred_file(pred_dir, seg), 'rb') as f:
                seg_pred = pkl.load(f)
            if len(seg_data) != len(seg_pred):
                raise RuntimeError(
                    f'Segment {seg}: dataset has {len(seg_data)} frames but cached '
                    f'predictions have {len(seg_pred)}. Cache is stale/incomplete.'
                )
            for local_idx, (frame, pred) in enumerate(zip(seg_data, seg_pred)):
                fid_d = str(frame['frame_id'])
                fid_p = str(pred['frame_id'])
                if fid_d != fid_p:
                    raise AssertionError(
                        f'Frame-id mismatch in {seg}[{local_idx}]: dataset={fid_d} '
                        f'pred={fid_p}. Prediction cache is not aligned with dataset.'
                    )
                frames.append({
                    'segment': seg,
                    'local_idx': local_idx,
                    'frame_id': fid_d,
                    'pred': pred,
                    'spoof': frame,   # dataset frame (holds spoof_gt + gt_boxes)
                })
    finally:
        _ts.default_restore_location = _orig_restore
    logger.info('Assembled %d ATLAS frames across %d segments', len(frames), len(segments))
    return frames


def clean_frame_id(entry):
    fid = entry.get('frame_id')
    if fid is not None:
        return str(fid)
    meta = entry.get('metadata')
    if isinstance(meta, dict) and meta.get('frame_id') is not None:
        return str(meta['frame_id'])
    raise KeyError('Clean prediction entry has no frame_id (checked top-level and metadata).')


def build_clean_index(clean_path, atlas_frames, logger):
    """Load the full-val clean result.pkl and filter+reorder it to the ATLAS
    frames, keyed by frame_id. Asserts a strict 1:1 mapping."""
    logger.info('[load] Loading clean result.pkl: %s', clean_path)
    with open(clean_path, 'rb') as f:
        clean = pkl.load(f)
    if not isinstance(clean, (list, tuple)):
        raise TypeError(
            f'--clean must be a list of prediction anno dicts (OpenPCDet '
            f'result.pkl); got {type(clean).__name__}.'
        )
    index = {}
    duplicates = []
    for entry in clean:
        fid = clean_frame_id(entry)
        if fid in index:
            duplicates.append(fid)
        else:
            index[fid] = entry
    logger.info('Loaded clean result.pkl: %d entries (%d unique frame ids)',
                len(clean), len(index))

    wanted = [fr['frame_id'] for fr in atlas_frames]
    missing = [fid for fid in wanted if fid not in index]
    dup_in_atlas = sorted({fid for fid in duplicates if fid in set(wanted)})

    errors = []
    if missing:
        errors.append(
            f'{len(missing)} ATLAS frame(s) have NO clean prediction, e.g. '
            + ', '.join(missing[:8])
        )
    if dup_in_atlas:
        errors.append(
            f'{len(dup_in_atlas)} ATLAS frame id(s) appear MORE THAN ONCE in the '
            f'clean result.pkl, e.g. ' + ', '.join(dup_in_atlas[:8])
        )
    if errors:
        raise RuntimeError(
            'Clean/ATLAS frame alignment failed:\n  - ' + '\n  - '.join(errors)
            + '\nCheck that --clean is the clean full-validation result.pkl for '
              'the SAME detector and that frame ids share the Waymo convention.'
        )

    clean_by_fid = {fid: index[fid] for fid in wanted}
    logger.info('Clean predictions aligned to all %d ATLAS frames (1:1).', len(clean_by_fid))
    return clean_by_fid


# --------------------------------------------------------------------------- #
# statistics helpers                                                           #
# --------------------------------------------------------------------------- #
def stats_block(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {k: None for k in
                ('mean', 'std', 'min', 'q1', 'median', 'q3', 'max')} | {'count': 0}
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=0)),
        'min': float(np.min(values)),
        'q1': float(np.quantile(values, 0.25)),
        'median': float(np.median(values)),
        'q3': float(np.quantile(values, 0.75)),
        'max': float(np.max(values)),
    }


def score_statistics(scores):
    """Original scripts' fp-score stat block (kept for the preserved metrics)."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return {k: None for k in
                ('mean', 'std', 'min', 'q25', 'median', 'q75', 'max')} | {'count': 0}
    return {
        'count': int(values.size),
        'mean': float(np.mean(values)),
        'std': float(np.std(values, ddof=0)),
        'min': float(np.min(values)),
        'q25': float(np.quantile(values, 0.25)),
        'median': float(np.median(values)),
        'q75': float(np.quantile(values, 0.75)),
        'max': float(np.max(values)),
    }


# --------------------------------------------------------------------------- #
# removal-specific discovery / target extraction                               #
# --------------------------------------------------------------------------- #
def load_removal_annos(annos_dir, segment):
    """Load the removal annos pickle for a segment and return its
    ``spoof_annos`` dict: {frame_index -> record|None}. Each active record
    carries ``target_box`` (10-dim; [:7] = x,y,z,l,w,h,yaw), ``active_this_frame``,
    and point-removal telemetry (n_points_in_sector, n_points_removed,
    range_gate_passed, extrapolated_close_range, target_range_m, ...)."""
    path = Path(annos_dir) / f'{segment}_a.pkl'
    if not path.is_file():
        raise FileNotFoundError(f'No removal annos for segment {segment!r}: {path}')
    with open(path, 'rb') as f:
        anno = pkl.load(f)
    sa = anno.get('spoof_annos')
    if sa is None:
        raise KeyError(f'{path} has no spoof_annos')
    return sa


def target_box_7(record):
    """7-dim target (victim) box from a removal record, or None."""
    tb = record.get('target_box')
    if tb is None:
        return None
    return to_numpy(tb, np.float32).reshape(-1)[:7]


def discover_removal_segments(dataset_dir, annos_dir, pred_dir):
    """Segments with a removal annos file (*_a.pkl) that also have a dataset
    pickle and a cached prediction pickle."""
    anno_paths = sorted(Path(annos_dir).glob('*_a.pkl'))
    if not anno_paths:
        raise RuntimeError(f'No *_a.pkl files found in {annos_dir}')
    segments = [p.stem[:-2] for p in anno_paths]
    missing_ds, missing_pred = [], []
    for seg in segments:
        try:
            dataset_file(dataset_dir, seg)
        except FileNotFoundError:
            missing_ds.append(seg)
        try:
            pred_file(pred_dir, seg)
        except FileNotFoundError:
            missing_pred.append(seg)
    if missing_ds:
        raise FileNotFoundError(
            f'{len(missing_ds)} segment(s) lack a dataset pickle in {dataset_dir}, '
            'e.g. ' + ', '.join(missing_ds[:5]))
    if missing_pred:
        raise FileNotFoundError(
            f'{len(missing_pred)} segment(s) lack a prediction pickle in {pred_dir}, '
            'e.g. ' + ', '.join(missing_pred[:5]))
    return segments


def load_removal_frames(dataset_dir, pred_dir, annos_dir, segments, logger):
    """Build the ordered attacked-frame list for removal.

    For each segment: load the dataset pickle (for frame_id + real GT vehicles),
    the cached predictions (index-aligned with the dataset), and the removal
    annos. A frame is a *target* frame when its removal record is active. The
    victim box comes from the record's ``target_box`` (NOT from spoof_gt).
    """
    frames = []
    import torch
    import torch.serialization as _ts
    _orig_restore = _ts.default_restore_location
    def _cpu_restore(storage, location):
        return _orig_restore(storage, 'cpu')
    _ts.default_restore_location = _cpu_restore
    try:
        for si, seg in enumerate(segments):
            logger.info('[load] segment %d / %d: %s', si + 1, len(segments), seg)
            ds_path = dataset_file(dataset_dir, seg)
            try:
                with open(ds_path, 'rb') as f:
                    try:
                        seg_data = torch.load(f, map_location=torch.device('cpu'),
                                              weights_only=False)
                    except TypeError:
                        f.seek(0)
                        seg_data = torch.load(f, map_location=torch.device('cpu'))
            except (RuntimeError, pkl.UnpicklingError):
                with open(ds_path, 'rb') as f:
                    seg_data = pkl.load(f)
            with open(pred_file(pred_dir, seg), 'rb') as f:
                seg_pred = pkl.load(f)
            spoof_annos = load_removal_annos(annos_dir, seg)

            if len(seg_data) != len(seg_pred):
                raise RuntimeError(
                    f'Segment {seg}: dataset has {len(seg_data)} frames but cached '
                    f'predictions have {len(seg_pred)}. Cache is stale/incomplete.')

            for local_idx, (frame, pred) in enumerate(zip(seg_data, seg_pred)):
                fid_d = str(frame['frame_id'])
                fid_p = str(pred['frame_id'])
                if fid_d != fid_p:
                    raise AssertionError(
                        f'Frame-id mismatch in {seg}[{local_idx}]: dataset={fid_d} '
                        f'pred={fid_p}. Prediction cache not aligned with dataset.')
                frames.append({
                    'segment': seg,
                    'local_idx': local_idx,
                    'frame_id': fid_d,
                    'pred': pred,
                    'frame': frame,            # dataset frame (real GT vehicles)
                    'removal': spoof_annos.get(local_idx),  # record or None
                })
    finally:
        _ts.default_restore_location = _orig_restore
    logger.info('Assembled %d frames across %d segments', len(frames), len(segments))
    return frames


# --------------------------------------------------------------------------- #
# removal metrics                                                              #
# --------------------------------------------------------------------------- #
def _best_match_on_target(target7, veh_boxes, veh_scores, iou_target_col,
                          match_iou, score_floor):
    """Return (best_score, best_center, best_iou) for the vehicle prediction
    best matching the victim box (IoU>=match_iou, score>=score_floor), or None.
    iou_target_col is the precomputed (Nveh,) IoU vs the target box."""
    if len(veh_boxes) == 0:
        return None
    elig = veh_scores >= score_floor
    if not np.any(elig):
        return None
    ious = iou_target_col[elig]
    boxes = veh_boxes[elig]
    scores = veh_scores[elig]
    best = int(np.argmax(ious))
    if float(ious[best]) < match_iou:
        return None
    return (float(scores[best]), boxes[best, :3].astype(np.float64, copy=False),
            float(ious[best]))


def recall_and_hiding_asr(precomputed, score_thresholds, match_iou, asr_iou):
    """(i) Recall@score on the victim: fraction of target frames with a vehicle
    prediction matching the victim box at IoU>=asr_iou AND score>=s. ASR_hide(s)
    = 1 - Recall(s). (ii) Regular hiding ASR: fraction of frames with NO vehicle
    prediction matching the victim above asr_iou at ANY score (score floor 0).
    """
    n = len(precomputed)
    per_frame_maxiou = {s: np.zeros(n, dtype=np.float64) for s in score_thresholds}
    # max IoU at any score (score floor 0) for the regular hiding ASR
    per_frame_maxiou_anyscore = np.zeros(n, dtype=np.float64)

    for i, pc in enumerate(precomputed):
        ious = pc['iou_veh_target']      # (Nveh,) attacked vehicle preds vs victim
        scores = pc['veh_scores']
        if len(ious) == 0:
            continue
        per_frame_maxiou_anyscore[i] = float(np.max(ious))
        for s in score_thresholds:
            elig = scores >= s
            if np.any(elig):
                per_frame_maxiou[s][i] = float(np.max(ious[elig]))

    recall = {s: (float(np.mean(per_frame_maxiou[s] >= asr_iou)) if n else None)
              for s in score_thresholds}
    asr_hide = {s: (float(np.mean(per_frame_maxiou[s] < asr_iou)) if n else None)
                for s in score_thresholds}
    # regular hiding ASR: victim not recovered at any score
    regular_hiding_asr = float(np.mean(per_frame_maxiou_anyscore < asr_iou)) if n else None

    return {
        'asr_iou': float(asr_iou),
        'match_iou': float(match_iou),
        'recall_at_score': recall,
        'asr_hide_at_score': asr_hide,
        'regular_hiding_asr': regular_hiding_asr,
        'regular_hiding_asr_desc': ('fraction of target frames with no vehicle prediction '
                                    f'at IoU>={asr_iou} with the victim box, at ANY score'),
        '_per_frame_maxiou': {float(s): per_frame_maxiou[s] for s in score_thresholds},
        '_per_frame_maxiou_anyscore': per_frame_maxiou_anyscore,
    }


def target_iou_distribution(precomputed, score_thresholds, asr_iou):
    """Per target frame: max IoU(victim box, score-eligible vehicle preds), 0 if
    none. Reported as a distribution per score gate. Shows whether misses are
    total (IoU=0, victim gone) or partial (0<IoU<asr_iou, degraded box)."""
    n = len(precomputed)
    per_frame_maxiou = {s: np.zeros(n, dtype=np.float64) for s in score_thresholds}
    for i, pc in enumerate(precomputed):
        ious = pc['iou_veh_target']
        scores = pc['veh_scores']
        if len(ious) == 0:
            continue
        for s in score_thresholds:
            elig = scores >= s
            if np.any(elig):
                per_frame_maxiou[s][i] = float(np.max(ious[elig]))
    dist = {
        s: {**stats_block(per_frame_maxiou[s]),
            'frac_ge_asr_iou': (float(np.mean(per_frame_maxiou[s] >= asr_iou)) if n else None),
            'frac_zero': (float(np.mean(per_frame_maxiou[s] == 0.0)) if n else None)}
        for s in score_thresholds
    }
    base_s = min(score_thresholds)
    prim = per_frame_maxiou[base_s]
    primary = {
        'score_gate': float(base_s),
        'asr_iou': float(asr_iou),
        **stats_block(prim),
        'pct_ge_asr_iou': (float(np.mean(prim >= asr_iou)) if n else None),
        'pct_zero_iou': (float(np.mean(prim == 0.0)) if n else None),
    }
    return {
        'asr_iou': float(asr_iou),
        'target_iou_distribution_primary': primary,
        'target_iou_distribution_per_score_threshold': dist,
    }


def confidence_drop(precomputed, match_iou, score_floor):
    """Compare the score of the best clean-matched vs best attacked-matched
    vehicle prediction on the victim box. Reports the distribution of
    (score_attacked - score_clean), and how far confidence falls."""
    deltas = []
    clean_scores, attacked_scores = [], []
    n_clean_detected = 0
    n_both_detected = 0
    n_dropped_out = 0            # detected cleanly, gone under attack
    dropped_below = {}          # score threshold -> count that fell below it
    thresholds = (0.1, 0.3, 0.5, 0.7)
    for t in thresholds:
        dropped_below[t] = 0

    for pc in precomputed:
        cln = _best_match_on_target(
            pc['target7'], pc['cln_veh_boxes'], pc['cln_veh_scores'],
            pc['iou_cln_veh_target'], match_iou, score_floor)
        if cln is None:
            continue
        n_clean_detected += 1
        clean_score = cln[0]
        clean_scores.append(clean_score)
        att = _best_match_on_target(
            pc['target7'], pc['veh_boxes'], pc['veh_scores'],
            pc['iou_veh_target'], match_iou, score_floor)
        if att is None:
            n_dropped_out += 1
            for t in thresholds:
                if clean_score >= t:
                    dropped_below[t] += 1
            continue
        n_both_detected += 1
        att_score = att[0]
        attacked_scores.append(att_score)
        deltas.append(att_score - clean_score)
        for t in thresholds:
            if clean_score >= t and att_score < t:
                dropped_below[t] += 1

    return {
        'match_iou': float(match_iou),
        'score_floor': float(score_floor),
        'n_clean_detected': int(n_clean_detected),
        'n_both_detected': int(n_both_detected),
        'n_dropped_out_under_attack': int(n_dropped_out),
        'drop_out_rate': (float(n_dropped_out / n_clean_detected)
                          if n_clean_detected else None),
        'score_delta_stats': stats_block(deltas),
        'clean_score_stats': stats_block(clean_scores),
        'attacked_score_stats': stats_block(attacked_scores),
        'crossed_below_threshold': {float(t): int(c) for t, c in dropped_below.items()},
        'crossed_below_threshold_rate': {
            float(t): (float(c / n_clean_detected) if n_clean_detected else None)
            for t, c in dropped_below.items()},
    }


def victim_displacement(precomputed, match_iou, score_floor, eps=1e-6):
    """Box drift of the victim under attack. For frames where the victim is
    matched both cleanly and under attack, report ||Δc|| (center shift) and the
    IoU degradation. Unlike injection there is no spoof point to project toward,
    so magnitude and IoU change are the quantities of interest."""
    disp_mag, iou_clean, iou_att, iou_drop = [], [], [], []
    n_pairs = 0
    for pc in precomputed:
        cln = _best_match_on_target(
            pc['target7'], pc['cln_veh_boxes'], pc['cln_veh_scores'],
            pc['iou_cln_veh_target'], match_iou, score_floor)
        if cln is None:
            continue
        att = _best_match_on_target(
            pc['target7'], pc['veh_boxes'], pc['veh_scores'],
            pc['iou_veh_target'], match_iou, score_floor)
        if att is None:
            continue
        n_pairs += 1
        _, c_clean, iou_c = cln
        _, c_att, iou_a = att
        disp_mag.append(float(np.linalg.norm(c_att - c_clean)))
        iou_clean.append(iou_c)
        iou_att.append(iou_a)
        iou_drop.append(iou_c - iou_a)
    return {
        'match_iou': float(match_iou),
        'score_floor': float(score_floor),
        'n_matched_pairs': int(n_pairs),
        'displacement_magnitude_stats': stats_block(disp_mag),
        'iou_clean_stats': stats_block(iou_clean),
        'iou_attacked_stats': stats_block(iou_att),
        'iou_drop_stats': stats_block(iou_drop),
    }


def target_region_false_positives(precomputed, score_thresholds,
                                  target_iou_thresholds, match_iou):
    """Vehicle preds that overlap the victim region (IoU>=τ with target box) but
    do NOT match any real GT vehicle above match_iou. For removal this captures
    stray/fragmented detections near the attacked object (a removal failure
    mode where a partially-occluded vehicle splits into low-confidence pieces).
    Crossed over τ x score, with a clean baseline and attack-induced counts."""
    n = len(precomputed)
    results = {}
    for tau in target_iou_thresholds:
        for s in score_thresholds:
            per_frame_count = np.zeros(n, dtype=np.int64)
            fp_scores_list, fp_ious_list = [], []
            clean_per_frame_count = np.zeros(n, dtype=np.int64)
            attack_induced_count = np.zeros(n, dtype=np.int64)
            attack_induced_sub = np.zeros(n, dtype=np.int64)

            for i, pc in enumerate(precomputed):
                a_scores = pc['veh_scores']
                iou_a_tgt = pc['iou_veh_target']
                max_iou_a_gt = pc['max_iou_att_veh_gt']
                if len(a_scores) > 0:
                    a_mask = (a_scores >= s) & (iou_a_tgt >= tau) & (max_iou_a_gt < match_iou)
                    a_count = int(a_mask.sum())
                    per_frame_count[i] = a_count
                    if a_count:
                        fp_scores_list.extend(a_scores[a_mask].tolist())
                        fp_ious_list.extend(iou_a_tgt[a_mask].tolist())
                else:
                    a_count = 0

                c_scores = pc['cln_veh_scores']
                iou_c_tgt = pc['iou_cln_veh_target']
                max_iou_c_gt = pc['max_iou_cln_veh_gt']
                if len(c_scores) > 0:
                    c_mask = (c_scores >= s) & (iou_c_tgt >= tau) & (max_iou_c_gt < match_iou)
                    c_count = int(c_mask.sum())
                else:
                    c_count = 0
                clean_per_frame_count[i] = c_count
                attack_induced_count[i] = 0 if c_count > 0 else a_count
                attack_induced_sub[i] = max(0, a_count - c_count)

            block = {
                'target_iou_threshold': float(tau),
                'score_threshold': float(s),
                'n_target_frames': int(n),
                'total_fp': int(per_frame_count.sum()),
                'mean_fp_per_frame': (float(per_frame_count.mean()) if n else None),
                'stray_detection_rate': (float(np.mean(per_frame_count >= 1)) if n else None),
                'count_per_frame_stats': stats_block(per_frame_count),
                'fp_confidence_stats': {
                    'count': len(fp_scores_list),
                    'mean': (float(np.mean(fp_scores_list)) if fp_scores_list else None),
                    'median': (float(np.median(fp_scores_list)) if fp_scores_list else None),
                    'min': (float(np.min(fp_scores_list)) if fp_scores_list else None),
                    'max': (float(np.max(fp_scores_list)) if fp_scores_list else None),
                },
                'fp_iou_with_target_stats': {
                    'count': len(fp_ious_list),
                    'mean': (float(np.mean(fp_ious_list)) if fp_ious_list else None),
                    'median': (float(np.median(fp_ious_list)) if fp_ious_list else None),
                },
                'clean_total_fp': int(clean_per_frame_count.sum()),
                'clean_mean_fp_per_frame': (float(clean_per_frame_count.mean()) if n else None),
                'clean_stray_rate': (float(np.mean(clean_per_frame_count >= 1)) if n else None),
                'attack_induced_total_fp': int(attack_induced_count.sum()),
                'attack_induced_stray_rate': (float(np.mean(attack_induced_count >= 1)) if n else None),
                'attack_induced_total_fp_subtractive': int(attack_induced_sub.sum()),
            }
            results[f'tau{tau:g}_s{s:g}'] = block
    return results


def attack_strength_conditioned(precomputed, match_iou, asr_iou, score_floor):
    """Condition victim recall on removal telemetry from the annos records:
    n_points_removed buckets, target_range buckets, and the range_gate_passed
    subset. Answers "fine-grained attack effects" cheaply — no injection
    analogue exists because injection lacks per-frame attack telemetry."""
    # per-frame victim recovered? (max IoU at score_floor >= asr_iou)
    recovered = np.zeros(len(precomputed), dtype=bool)
    n_removed = np.full(len(precomputed), np.nan)
    tgt_range = np.full(len(precomputed), np.nan)
    gate_passed = np.zeros(len(precomputed), dtype=bool)
    have_meta = np.zeros(len(precomputed), dtype=bool)

    for i, pc in enumerate(precomputed):
        ious = pc['iou_veh_target']
        scores = pc['veh_scores']
        if len(ious) > 0:
            elig = scores >= score_floor
            if np.any(elig):
                recovered[i] = float(np.max(ious[elig])) >= asr_iou
        meta = pc.get('removal_meta') or {}
        if meta:
            have_meta[i] = True
            if meta.get('n_points_removed') is not None:
                n_removed[i] = float(meta['n_points_removed'])
            if meta.get('target_range_m') is not None:
                tgt_range[i] = float(meta['target_range_m'])
            gate_passed[i] = bool(meta.get('range_gate_passed', False))

    def _recall(mask):
        m = mask & np.isfinite(n_removed) if mask.dtype == bool else mask
        sel = recovered[mask]
        return (float(np.mean(sel)) if sel.size else None, int(mask.sum()))

    out = {'note': 'recall = victim recovered at IoU>=%.2f, score>=%.2f' % (asr_iou, score_floor)}

    # range_gate_passed subset
    r, c = _recall(gate_passed)
    out['recall_range_gate_passed'] = {'recall': r, 'n': c}
    r, c = _recall(~gate_passed & have_meta)
    out['recall_range_gate_failed'] = {'recall': r, 'n': c}

    # n_points_removed buckets (quartiles over frames with meta)
    valid = np.isfinite(n_removed)
    if valid.sum() >= 4:
        qs = np.quantile(n_removed[valid], [0.25, 0.5, 0.75])
        edges = [-np.inf, qs[0], qs[1], qs[2], np.inf]
        labels = ['q1_low', 'q2', 'q3', 'q4_high']
        buckets = {}
        for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
            mask = valid & (n_removed > lo) & (n_removed <= hi)
            sel = recovered[mask]
            buckets[lab] = {
                'range': [None if lo == -np.inf else float(lo),
                          None if hi == np.inf else float(hi)],
                'recall': (float(np.mean(sel)) if sel.size else None),
                'n': int(mask.sum()),
            }
        out['recall_by_n_points_removed_quartile'] = buckets

    # target_range buckets
    validr = np.isfinite(tgt_range)
    if validr.sum() >= 4:
        qs = np.quantile(tgt_range[validr], [0.33, 0.66])
        edges = [-np.inf, qs[0], qs[1], np.inf]
        labels = ['near', 'mid', 'far']
        buckets = {}
        for lab, lo, hi in zip(labels, edges[:-1], edges[1:]):
            mask = validr & (tgt_range > lo) & (tgt_range <= hi)
            sel = recovered[mask]
            buckets[lab] = {
                'range_m': [None if lo == -np.inf else float(lo),
                            None if hi == np.inf else float(hi)],
                'recall': (float(np.mean(sel)) if sel.size else None),
                'n': int(mask.sum()),
            }
        out['recall_by_target_range'] = buckets

    # telemetry summary
    out['telemetry'] = {
        'n_frames_with_meta': int(have_meta.sum()),
        'n_points_removed_stats': stats_block(n_removed[np.isfinite(n_removed)]),
        'target_range_m_stats': stats_block(tgt_range[np.isfinite(tgt_range)]),
        'range_gate_pass_rate': (float(np.mean(gate_passed[have_meta]))
                                 if have_meta.any() else None),
    }
    return out


# --------------------------------------------------------------------------- #
# rendering                                                                    #
# --------------------------------------------------------------------------- #
def _fmt(v, p=4):
    if v is None:
        return 'N/A'
    if isinstance(v, float):
        return f'{v:.{p}f}'
    return str(v)


def render_markdown(summary):
    L = []
    meta = summary['meta']
    L.append(f"# ATLAS removal eval-only metrics — {meta['detector']} / {meta['attack_variant']}")
    L.append('')
    L.append(f"- Segments: {meta['num_segments']}  |  frames: {meta['num_frames']}  "
             f"|  target (attacked) frames: {meta['num_target_frames']}")
    L.append(f"- IoU: exact CPU rotated-3D IoU (numba)  |  vehicle class id: {meta['vehicle_class_id']}  "
             f"|  real-GT match IoU: {meta['match_iou']}  |  ASR/recall IoU: {meta['asr_iou']}")
    L.append(f"- Clean predictions: `{meta['clean_path']}`")
    L.append('')

    rec = summary['recall_hiding']
    L.append('## Hiding ASR & victim recall')
    L.append('')
    L.append(f"- **Regular hiding ASR** (victim not recovered at IoU≥{meta['asr_iou']} at ANY score): "
             f"**{_fmt(rec['regular_hiding_asr'])}**")
    L.append('')
    L.append('| score threshold s | victim recall(s) | hiding ASR(s) = 1−recall |')
    L.append('|---|---|---|')
    for s in meta['score_thresholds']:
        L.append(f"| {s:g} | {_fmt(rec['recall_at_score'][s])} | {_fmt(rec['asr_hide_at_score'][s])} |")
    L.append('')

    td = summary['target_iou_distribution']
    L.append('## Victim-box IoU distribution (per-frame max IoU over score-eligible vehicle preds)')
    L.append('')
    ip = td['target_iou_distribution_primary']
    L.append(f"Primary (score gate ≥ {ip['score_gate']:g}): "
             f"min={_fmt(ip['min'])}, q1={_fmt(ip['q1'])}, median={_fmt(ip['median'])}, "
             f"mean={_fmt(ip['mean'])}, q3={_fmt(ip['q3'])}, max={_fmt(ip['max'])}  "
             f"|  %IoU≥{ip['asr_iou']:g} = {_fmt(ip['pct_ge_asr_iou'])}  |  %IoU=0 (fully hidden) = {_fmt(ip['pct_zero_iou'])}")
    L.append('')
    L.append('| score gate | min | q1 | median | mean | q3 | max | %IoU≥thr | %IoU=0 |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for s in meta['score_thresholds']:
        b = td['target_iou_distribution_per_score_threshold'][s]
        L.append(f"| {s:g} | {_fmt(b['min'])} | {_fmt(b['q1'])} | {_fmt(b['median'])} | "
                 f"{_fmt(b['mean'])} | {_fmt(b['q3'])} | {_fmt(b['max'])} | "
                 f"{_fmt(b['frac_ge_asr_iou'])} | {_fmt(b['frac_zero'])} |")
    L.append('')

    cd = summary['confidence_drop']
    L.append('## Confidence degradation on the victim')
    L.append('')
    L.append(f"- Victim detected cleanly: {cd['n_clean_detected']}  |  still detected under attack: "
             f"{cd['n_both_detected']}  |  dropped out entirely: {cd['n_dropped_out_under_attack']} "
             f"(rate {_fmt(cd['drop_out_rate'])})")
    sd = cd['score_delta_stats']
    L.append(f"- Score change (attacked − clean), matched pairs: n={sd['count']}, "
             f"mean={_fmt(sd['mean'])}, median={_fmt(sd['median'])}, min={_fmt(sd['min'])}, max={_fmt(sd['max'])}")
    cs, as_ = cd['clean_score_stats'], cd['attacked_score_stats']
    L.append(f"- Clean score: mean={_fmt(cs['mean'])} median={_fmt(cs['median'])}  |  "
             f"Attacked score: mean={_fmt(as_['mean'])} median={_fmt(as_['median'])}")
    L.append('')
    L.append('Victim confidence crossing below each threshold (clean≥t → attacked<t or gone):')
    L.append('')
    L.append('| threshold t | count crossed below | rate |')
    L.append('|---|---|---|')
    for t in (0.1, 0.3, 0.5, 0.7):
        c = cd['crossed_below_threshold'].get(t)
        r = cd['crossed_below_threshold_rate'].get(t)
        L.append(f"| {t:g} | {c} | {_fmt(r)} |")
    L.append('')

    vd = summary['victim_displacement']
    L.append('## Victim box drift (frames where the victim survives the attack)')
    L.append('')
    L.append(f"- Matched clean&attacked pairs: {vd['n_matched_pairs']}")
    dm = vd['displacement_magnitude_stats']
    L.append(f"- Displacement ‖Δc‖ (m): n={dm['count']}, median={_fmt(dm['median'])}, "
             f"mean={_fmt(dm['mean'])}, q3={_fmt(dm['q3'])}, max={_fmt(dm['max'])}")
    ids = vd['iou_drop_stats']
    L.append(f"- IoU drop (clean − attacked): median={_fmt(ids['median'])}, mean={_fmt(ids['mean'])}, "
             f"max={_fmt(ids['max'])}")
    L.append('')

    fp = summary['target_region_fp']
    L.append('## Target-region false positives (stray/fragmented detections)')
    L.append('')
    L.append('A vehicle prediction is a target-region FP when IoU3D(victim box) ≥ τ **and** it does '
             'not match any real GT vehicle above the Waymo matching IoU '
             f"({meta['match_iou']}). Attack-induced excludes strays already present in the clean frame.")
    L.append('')
    L.append('| τ_target | score s | total FP | mean/frame | stray rate | clean total | attack-induced total | attack-induced stray rate | mean conf | median IoU(target) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for tau in meta['target_iou_thresholds']:
        for s in meta['score_thresholds']:
            b = fp[f'tau{tau:g}_s{s:g}']
            L.append(
                f"| {tau:g} | {s:g} | {b['total_fp']} | {_fmt(b['mean_fp_per_frame'])} | "
                f"{_fmt(b['stray_detection_rate'])} | {b['clean_total_fp']} | "
                f"{b['attack_induced_total_fp']} | {_fmt(b['attack_induced_stray_rate'])} | "
                f"{_fmt(b['fp_confidence_stats']['mean'])} | {_fmt(b['fp_iou_with_target_stats']['median'])} |")
    L.append('')

    asc = summary.get('attack_strength')
    if asc:
        L.append('## Attack-strength conditioning (removal telemetry)')
        L.append('')
        tel = asc.get('telemetry', {})
        nps = tel.get('n_points_removed_stats', {})
        rng = tel.get('target_range_m_stats', {})
        L.append(f"- Frames with telemetry: {tel.get('n_frames_with_meta')}  |  "
                 f"range-gate pass rate: {_fmt(tel.get('range_gate_pass_rate'))}")
        L.append(f"- Points removed / frame: median={_fmt(nps.get('median'),0)}, "
                 f"mean={_fmt(nps.get('mean'),0)}, max={_fmt(nps.get('max'),0)}")
        L.append(f"- Target range (m): median={_fmt(rng.get('median'),1)}, "
                 f"min={_fmt(rng.get('min'),1)}, max={_fmt(rng.get('max'),1)}")
        gp = asc.get('recall_range_gate_passed', {})
        gf = asc.get('recall_range_gate_failed', {})
        L.append(f"- Victim recall | gate passed: {_fmt(gp.get('recall'))} (n={gp.get('n')})  |  "
                 f"gate failed: {_fmt(gf.get('recall'))} (n={gf.get('n')})")
        if 'recall_by_n_points_removed_quartile' in asc:
            L.append('')
            L.append('Recall by points-removed quartile:')
            L.append('')
            L.append('| quartile | points-removed range | recall | n |')
            L.append('|---|---|---|---|')
            for lab, b in asc['recall_by_n_points_removed_quartile'].items():
                rg = b['range']
                lo = 'min' if rg[0] is None else f"{rg[0]:.0f}"
                hi = 'max' if rg[1] is None else f"{rg[1]:.0f}"
                L.append(f"| {lab} | {lo}–{hi} | {_fmt(b['recall'])} | {b['n']} |")
            L.append('')
        if 'recall_by_target_range' in asc:
            L.append('Recall by target range:')
            L.append('')
            L.append('| bucket | range (m) | recall | n |')
            L.append('|---|---|---|---|')
            for lab, b in asc['recall_by_target_range'].items():
                rg = b['range_m']
                lo = 'min' if rg[0] is None else f"{rg[0]:.1f}"
                hi = 'max' if rg[1] is None else f"{rg[1]:.1f}"
                L.append(f"| {lab} | {lo}–{hi} | {_fmt(b['recall'])} | {b['n']} |")
            L.append('')
    return '\n'.join(L)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def strip_private(obj):
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith('_')}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def atomic_dump(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    writer(tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# driver                                                                       #
# --------------------------------------------------------------------------- #
def parse_thresholds(text):
    return [float(x) for x in str(text).replace(' ', '').split(',') if x != '']


def parse_args():
    p = argparse.ArgumentParser(description='ATLAS point-removal eval-only metrics (no inference).')
    p.add_argument('--pred_dir', required=True, help='Cached predictions dir ({segment}_p.pkl / {segment}.pkl).')
    p.add_argument('--dataset', required=True, help='Dataset dir ({segment}.pkl / {segment}_d.pkl).')
    p.add_argument('--annos_dir', required=True, help='Removal annos dir ({segment}_a.pkl).')
    p.add_argument('--clean', required=True, help='Full-validation clean result.pkl for the SAME detector.')
    p.add_argument('--out_dir', required=True, help='Output dir for detailed metrics.')
    p.add_argument('--detector', default='detector', help='Detector label for reporting.')
    p.add_argument('--attack_variant', default='variant', help='Attack variant / run name label.')
    p.add_argument('--device', default='cpu', help='Kept for compatibility; ignored (CPU numba IoU).')
    p.add_argument('--vehicle_class_id', type=int, default=1)
    p.add_argument('--match_iou', type=float, default=0.7, help='Waymo vehicle matching IoU.')
    p.add_argument('--asr_iou', type=float, default=0.7, help='Victim recovery / hiding IoU.')
    p.add_argument('--disp_score_floor', type=float, default=0.1,
                   help='Min score for a prediction to count as a matched detection.')
    p.add_argument('--score_thresholds', default='0.1,0.3,0.5,0.7')
    p.add_argument('--target_iou_thresholds', default='0.3,0.5,0.7')
    p.add_argument('--num_segments', type=int, default=0,
                   help='Randomly subsample this many segments (0 = use all).')
    p.add_argument('--seed', type=int, default=42, help='RNG seed for subsampling.')
    return p.parse_args()


def run(args, iou_fn, logger):
    score_thresholds = parse_thresholds(args.score_thresholds)
    target_iou_thresholds = parse_thresholds(args.target_iou_thresholds)

    segments = discover_removal_segments(args.dataset, args.annos_dir, args.pred_dir)
    segments = subsample_segments(segments, args.num_segments, args.seed, logger)
    all_frames = load_removal_frames(args.dataset, args.pred_dir, args.annos_dir, segments, logger)
    clean_by_fid = build_clean_index(args.clean, all_frames, logger)

    # target frames = active removal records with a target box
    target_frames = []
    for fr in all_frames:
        rec = fr['removal']
        if rec is None or not rec.get('active_this_frame'):
            continue
        tgt7 = target_box_7(rec)
        if tgt7 is None:
            continue
        fr = dict(fr)
        fr['target7'] = tgt7
        fr['removal_meta'] = {
            'n_points_in_sector': rec.get('n_points_in_sector'),
            'n_points_removed': rec.get('n_points_removed'),
            'n_points_before': rec.get('n_points_before'),
            'n_points_after': rec.get('n_points_after'),
            'p_remove': rec.get('p_remove'),
            'target_range_m': rec.get('target_range_m'),
            'range_gate_passed': rec.get('range_gate_passed'),
            'extrapolated_close_range': rec.get('extrapolated_close_range'),
            'mode': rec.get('mode'),
        }
        target_frames.append(fr)
    logger.info('Target (attacked) frames: %d / %d', len(target_frames), len(all_frames))
    if not target_frames:
        raise RuntimeError('No active removal target frames found. Nothing to evaluate.')

    import time as _time
    _t0 = _time.time()

    logger.info('[precompute] Computing all IoU matrices (%d frames) ...', len(target_frames))
    precomputed = []
    log_interval = max(1, len(target_frames) // 10)
    for i, fr in enumerate(target_frames):
        if i % log_interval == 0:
            logger.info('[precompute] frame %d / %d (%.1fs)',
                        i, len(target_frames), _time.time() - _t0)
        tgt7 = fr['target7']
        tgt_1x7 = tgt7.reshape(1, 7)
        gt_veh = real_gt_vehicle_boxes(fr['frame'], args.vehicle_class_id)

        veh_boxes, veh_scores = vehicle_predictions(fr['pred'], args.vehicle_class_id)
        if len(veh_boxes) > 0:
            iou_veh_target = iou_fn(veh_boxes, tgt_1x7)[:, 0]
            if len(gt_veh) > 0:
                max_iou_att_veh_gt = iou_fn(veh_boxes, gt_veh).max(axis=1)
            else:
                max_iou_att_veh_gt = np.zeros(len(veh_boxes), dtype=np.float32)
        else:
            iou_veh_target = np.empty(0, dtype=np.float32)
            max_iou_att_veh_gt = np.empty(0, dtype=np.float32)

        clean_pred = clean_by_fid[fr['frame_id']]
        cln_veh_boxes, cln_veh_scores = vehicle_predictions(clean_pred, args.vehicle_class_id)
        if len(cln_veh_boxes) > 0:
            iou_cln_veh_target = iou_fn(cln_veh_boxes, tgt_1x7)[:, 0]
            if len(gt_veh) > 0:
                max_iou_cln_veh_gt = iou_fn(cln_veh_boxes, gt_veh).max(axis=1)
            else:
                max_iou_cln_veh_gt = np.zeros(len(cln_veh_boxes), dtype=np.float32)
        else:
            iou_cln_veh_target = np.empty(0, dtype=np.float32)
            max_iou_cln_veh_gt = np.empty(0, dtype=np.float32)

        precomputed.append({
            'frame_id': fr['frame_id'],
            'segment': fr.get('segment', ''),
            'target7': tgt7,
            'removal_meta': fr.get('removal_meta'),
            'veh_boxes': veh_boxes,
            'veh_scores': veh_scores,
            'iou_veh_target': iou_veh_target,
            'max_iou_att_veh_gt': max_iou_att_veh_gt,
            'cln_veh_boxes': cln_veh_boxes,
            'cln_veh_scores': cln_veh_scores,
            'iou_cln_veh_target': iou_cln_veh_target,
            'max_iou_cln_veh_gt': max_iou_cln_veh_gt,
        })
    logger.info('[precompute] Done — all IoU cached in %.1fs', _time.time() - _t0)

    logger.info('[1/5] Recall + hiding ASR ...')
    rec = recall_and_hiding_asr(precomputed, score_thresholds, args.match_iou, args.asr_iou)
    logger.info('[2/5] Victim IoU distribution ...')
    td = target_iou_distribution(precomputed, score_thresholds, args.asr_iou)
    logger.info('[3/5] Confidence drop ...')
    cd = confidence_drop(precomputed, args.match_iou, args.disp_score_floor)
    logger.info('[4/5] Victim displacement ...')
    vd = victim_displacement(precomputed, args.match_iou, args.disp_score_floor)
    logger.info('[5/5] Target-region FPs + attack-strength conditioning ...')
    fp = target_region_false_positives(precomputed, score_thresholds,
                                       target_iou_thresholds, args.match_iou)
    asc = attack_strength_conditioned(precomputed, args.match_iou, args.asr_iou,
                                      args.disp_score_floor)
    logger.info('All metrics done (%.1fs elapsed)', _time.time() - _t0)

    summary = {
        'meta': {
            'detector': args.detector,
            'attack_variant': args.attack_variant,
            'attack_type': 'point_removal',
            'device': args.device,
            'vehicle_class_id': int(args.vehicle_class_id),
            'match_iou': float(args.match_iou),
            'asr_iou': float(args.asr_iou),
            'disp_score_floor': float(args.disp_score_floor),
            'score_thresholds': score_thresholds,
            'target_iou_thresholds': target_iou_thresholds,
            'num_segments': len(segments),
            'requested_num_segments': int(args.num_segments),
            'seed': int(args.seed),
            'selected_segments': list(segments),
            'num_frames': len(all_frames),
            'num_target_frames': len(target_frames),
            'dataset_dir': str(Path(args.dataset).resolve()),
            'annos_dir': str(Path(args.annos_dir).resolve()),
            'pred_dir': str(Path(args.pred_dir).resolve()),
            'clean_path': str(Path(args.clean).resolve()),
        },
        'recall_hiding': rec,
        'target_iou_distribution': td,
        'confidence_drop': cd,
        'victim_displacement': vd,
        'target_region_fp': fp,
        'attack_strength': asc,
    }

    out_dir = Path(args.out_dir)
    md_path = out_dir / 'detailed_metrics.md'
    json_path = out_dir / 'detailed_metrics.json'
    pkl_path = out_dir / 'detailed_summary.pkl'

    md = render_markdown(summary)
    atomic_dump(md_path, lambda t: t.write_text(md, encoding='utf-8'))
    json_view = strip_private(summary)
    atomic_dump(json_path, lambda t: t.write_text(
        json.dumps(json_view, indent=2, sort_keys=True, default=json_default), encoding='utf-8'))
    atomic_dump(pkl_path, lambda t: t.write_bytes(pkl.dumps(summary)))

    print(md)
    print(f'\nSaved markdown : {md_path}')
    print(f'Saved JSON     : {json_path}')
    print(f'Saved pickle   : {pkl_path}')
    return summary


def main():
    args = parse_args()
    logger = setup_logger('atlas_removal_eval_only')
    iou_fn = make_iou_fn(logger)
    run(args, iou_fn, logger)


if __name__ == '__main__':
    main()
