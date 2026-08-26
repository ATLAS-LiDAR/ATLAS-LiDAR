#!/usr/bin/env python
"""Qualitative BEV figures for ATLAS: two (clean, attacked, prediction) triplets.

Produces six images in one folder:

  injection_1_clean.png      unmodified cloud, GT boxes
  injection_2_injected.png   phantom points highlighted, phantom box marked
  injection_3_predicted.png  detector output -- a FALSE POSITIVE on the phantom

  removal_1_clean.png        unmodified cloud, victim vehicle marked
  removal_2_removed.png      wedge points deleted, victim now unsupported
  removal_3_predicted.png    detector output -- a FALSE NEGATIVE where the car was

Frames are chosen from the prediction cache so that the attack demonstrably
worked: the injection frame has a high-IoU, high-confidence phantom detection
absent from the clean prediction, and the removal frame has a vehicle confidently
detected in clean and entirely missed when attacked. Picking frames by eye, or by
attack metadata alone, risks illustrating a frame where the attack failed.

Views are bird's-eye, cropped around the attacked object, since a full 360-degree
nuScenes cloud at ~265k points renders as an undifferentiated disc.
"""
import argparse
import os
import pickle as pkl
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "atlas_generation"))

# muted background, saturated attack colours -- the attack must be the thing the
# eye lands on first
C_BG = "#9aa5b1"        # scene points
C_INJ = "#e8590c"       # injected phantom points
C_GT = "#495057"        # real ground-truth boxes
C_SPOOF = "#e8590c"     # phantom box
C_PRED = "#1c7ed6"      # ordinary predictions
C_FP = "#e03131"        # the false positive
C_FN = "#e8590c"        # where the false negative should have been
C_WEDGE = "#f08c00"     # removal wedge


def box_corners_bev(box):
    x, y = box[0], box[1]
    dx, dy, yaw = box[3], box[4], box[6]
    c, s = np.cos(yaw), np.sin(yaw)
    loc = np.array([[-dx / 2, -dy / 2], [dx / 2, -dy / 2],
                    [dx / 2, dy / 2], [-dx / 2, dy / 2]])
    return loc @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def draw_box(ax, box, color, label=None, lw=2.0, ls="-", alpha=1.0, z=5):
    ax.add_patch(MplPolygon(box_corners_bev(box), closed=True, fill=False,
                            edgecolor=color, linewidth=lw, linestyle=ls,
                            alpha=alpha, zorder=z, label=label))
    # heading tick, so orientation is readable
    x, y, yaw = box[0], box[1], box[6]
    dx = box[3] / 2
    ax.plot([x, x + dx * np.cos(yaw)], [y, y + dx * np.sin(yaw)],
            color=color, lw=lw * 0.7, alpha=alpha, zorder=z)


def setup_ax(ax, cx, cy, half, title):
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.15, lw=0.5)
    ax.set_facecolor("white")


def scatter(ax, pts, color, s=1.2, alpha=0.55, z=1, label=None):
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, alpha=alpha,
                   linewidths=0, zorder=z, label=label)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(path)}")


def ego_marker(ax):
    ax.plot([0], [0], marker="^", ms=9, color="#212529", zorder=10)
    ax.annotate("ego", (0, 0), textcoords="offset points", xytext=(6, -12),
                fontsize=8, color="#212529")


def injection_triplet(out, clean_pts, atk_pts, spoof_gt, gt_boxes, pred, fp_idx,
                      half=26.0):
    """clean / injected / predicted, cropped around the phantom."""
    cx, cy = float(spoof_gt[0]), float(spoof_gt[1])

    # Which points are the phantom? spoof_frame appends kept phantom points to the
    # end of the cloud, but rather than rely on that ordering we recover them
    # geometrically: points present in the attacked cloud inside the phantom box.
    inbox = points_in_box(atk_pts, spoof_gt, pad=1.05)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    setup_ax(ax, cx, cy, half, "Clean scene")
    scatter(ax, clean_pts, C_BG)
    for b in gt_boxes:
        draw_box(ax, b, C_GT, lw=1.6, alpha=0.85)
    draw_box(ax, spoof_gt, C_SPOOF, lw=1.8, ls=":", alpha=0.75)
    ax.annotate("no object here", (cx, cy), textcoords="offset points",
                xytext=(0, -34), ha="center", fontsize=9, color=C_SPOOF)
    ego_marker(ax)
    ax.legend(handles=[
        plt.Line2D([], [], color=C_GT, lw=1.6, label="ground-truth object"),
        plt.Line2D([], [], color=C_SPOOF, lw=1.8, ls=":", label="future phantom location"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out, "injection_1_clean.png"))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    setup_ax(ax, cx, cy, half, "Injected: phantom vehicle added")
    scatter(ax, atk_pts[~inbox], C_BG)
    scatter(ax, atk_pts[inbox], C_INJ, s=5.0, alpha=0.95, z=3)
    for b in gt_boxes:
        draw_box(ax, b, C_GT, lw=1.6, alpha=0.85)
    draw_box(ax, spoof_gt, C_SPOOF, lw=2.4)
    ax.annotate(f"{int(inbox.sum())} injected points", (cx, cy),
                textcoords="offset points", xytext=(0, -34), ha="center",
                fontsize=9, color=C_INJ, weight="bold")
    ego_marker(ax)
    ax.legend(handles=[
        plt.Line2D([], [], marker="o", ls="", color=C_INJ, ms=5, label="injected points"),
        plt.Line2D([], [], color=C_SPOOF, lw=2.4, label="phantom box"),
        plt.Line2D([], [], color=C_GT, lw=1.6, label="ground-truth object"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out, "injection_2_injected.png"))

    pb = np.asarray(pred["boxes_lidar"], np.float32)
    ps = np.asarray(pred["score"], np.float32)
    pn = np.asarray(pred["name"])
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    setup_ax(ax, cx, cy, half, "Detector output: false positive on the phantom")
    scatter(ax, atk_pts[~inbox], C_BG, alpha=0.4)
    scatter(ax, atk_pts[inbox], C_INJ, s=4.0, alpha=0.8, z=3)
    for i, b in enumerate(pb):
        if i == fp_idx or ps[i] < 0.2:
            continue
        draw_box(ax, b, C_PRED, lw=1.3, alpha=0.75, z=4)
    draw_box(ax, pb[fp_idx], C_FP, lw=2.8, z=8)
    ax.annotate(f"FALSE POSITIVE\n{pn[fp_idx]}  conf {ps[fp_idx]:.2f}",
                (pb[fp_idx][0], pb[fp_idx][1]), textcoords="offset points",
                xytext=(0, -40), ha="center", fontsize=9, color=C_FP,
                weight="bold")
    ego_marker(ax)
    ax.legend(handles=[
        plt.Line2D([], [], color=C_FP, lw=2.8, label="false positive (phantom detected)"),
        plt.Line2D([], [], color=C_PRED, lw=1.3, label="other detections"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out, "injection_3_predicted.png"))


def azimuth_only_removal(clean_pts, az_center, az_width_deg):
    """Points surviving an azimuth-ONLY wedge -- a display variant.

    The real A-HFR attack masks on azimuth AND elevation (`el_width_deg` = 16 in
    removal.py), so within ~10 m of the ego nothing is deleted: at close range the
    ground sits at -20 to -84 deg elevation, far outside a 16 deg band centred near
    horizontal. In a bird's-eye projection, which collapses elevation, that reads as
    an unexplained range cutoff.

    Dropping the elevation term makes the wedge a clean BEV sector all the way to
    the ego, which is easier to read as a figure. It removes ~3x more points than
    the real attack (2453 vs 789 on the example frame), so it does NOT depict the
    data any detector was evaluated on -- caption it as illustrative.
    """
    x, y = clean_pts[:, 0], clean_pts[:, 1]
    d = np.abs(_wrap_pi(np.arctan2(y, x) - az_center))
    return clean_pts[d > np.deg2rad(az_width_deg) / 2.0]


def _wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def removal_triplet(out, clean_pts, atk_pts, victim, gt_boxes, clean_pred,
                    atk_pred, clean_idx, half=26.0):
    """clean / removed / predicted, cropped around the victim vehicle."""
    cx, cy = float(victim[0]), float(victim[1])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    setup_ax(ax, cx, cy, half, "Clean scene")
    scatter(ax, clean_pts, C_BG)
    for b in gt_boxes:
        draw_box(ax, b, C_GT, lw=1.5, alpha=0.8)
    draw_box(ax, victim, C_WEDGE, lw=2.6)
    n_in = int(points_in_box(clean_pts, victim, pad=1.05).sum())
    ax.annotate(f"target vehicle\n{n_in} points", (cx, cy),
                textcoords="offset points", xytext=(0, -38), ha="center",
                fontsize=9, color=C_WEDGE, weight="bold")
    ego_marker(ax)
    ax.legend(handles=[
        plt.Line2D([], [], color=C_WEDGE, lw=2.6, label="target vehicle"),
        plt.Line2D([], [], color=C_GT, lw=1.5, label="ground-truth object"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out, "removal_1_clean.png"))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    setup_ax(ax, cx, cy, half, "Removed: wedge points deleted")
    # Points are genuinely ABSENT here -- nothing is drawn where they were. The
    # empty wedge is the attack.
    scatter(ax, atk_pts, C_BG)
    n_gone = len(clean_pts) - len(atk_pts)
    for b in gt_boxes:
        draw_box(ax, b, C_GT, lw=1.5, alpha=0.8)
    draw_box(ax, victim, C_WEDGE, lw=2.6, ls="--")
    n_left = int(points_in_box(atk_pts, victim, pad=1.05).sum())
    ax.annotate(f"{n_gone} points removed\n{n_left} left on target",
                (cx, cy), textcoords="offset points", xytext=(0, -38),
                ha="center", fontsize=9, color=C_FN, weight="bold")
    ego_marker(ax)
    ax.legend(handles=[
        plt.Line2D([], [], color=C_WEDGE, lw=2.6, ls="--", label="target vehicle (GT)"),
        plt.Line2D([], [], color=C_GT, lw=1.5, label="ground-truth object"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out, "removal_2_removed.png"))

    ab = np.asarray(atk_pred["boxes_lidar"], np.float32)
    asc = np.asarray(atk_pred["score"], np.float32)
    cb = np.asarray(clean_pred["boxes_lidar"], np.float32)
    cs = np.asarray(clean_pred["score"], np.float32)
    cn = np.asarray(clean_pred["name"])
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    setup_ax(ax, cx, cy, half, "Detector output: false negative at the target")
    scatter(ax, atk_pts, C_BG, alpha=0.5)
    for i, b in enumerate(ab):
        if asc[i] < 0.2:
            continue
        draw_box(ax, b, C_PRED, lw=1.3, alpha=0.75, z=4)
    # the detection that EXISTED in clean and is now gone
    draw_box(ax, cb[clean_idx], C_FN, lw=2.8, ls="--", z=8)
    ax.annotate(f"FALSE NEGATIVE\nclean: {cn[clean_idx]} conf {cs[clean_idx]:.2f}"
                f"\nattacked: no detection",
                (cb[clean_idx][0], cb[clean_idx][1]), textcoords="offset points",
                xytext=(0, -46), ha="center", fontsize=9, color=C_FN, weight="bold")
    ego_marker(ax)
    ax.legend(handles=[
        plt.Line2D([], [], color=C_FN, lw=2.8, ls="--",
                   label="detected when clean, missing now"),
        plt.Line2D([], [], color=C_PRED, lw=1.3, label="surviving detections"),
    ], loc="upper right", fontsize=8, framealpha=0.9)
    save(fig, os.path.join(out, "removal_3_predicted.png"))


def keyframe_only(pts, tol=1e-4):
    """Keep just the keyframe sweep (time == 0) of an accumulated cloud.

    ATLAS nuScenes clouds are 10-sweep accumulated (~265k points, column 4 is the
    per-point age in seconds: 0.00, 0.05, ... 0.45). Ten overlapping beam-ring
    patterns, each from a slightly different ego position, render as smeared or
    doubled structure that obscures the attack. Showing the t=0 sweep alone gives a
    clean single-revolution ring pattern at ~34k points.

    This is a DISPLAY choice only -- the detector consumes all ten sweeps.
    """
    if pts.shape[1] < 5:
        return pts
    return pts[np.abs(pts[:, 4]) < tol]


def points_in_box(pts, box, pad=1.0):
    if len(pts) == 0:
        return np.zeros(0, bool)
    d = pts[:, :3] - np.asarray(box[:3], np.float64)
    yaw = float(box[6])
    c, s = np.cos(-yaw), np.sin(-yaw)
    xr = d[:, 0] * c - d[:, 1] * s
    yr = d[:, 0] * s + d[:, 1] * c
    return ((np.abs(xr) <= box[3] / 2 * pad) & (np.abs(yr) <= box[4] / 2 * pad)
            & (np.abs(d[:, 2]) <= box[5] / 2 * pad))


def deleted_points(clean_pts, atk_pts, tol=1e-3):
    """Points present in clean but absent from attacked (the removed wedge)."""
    if len(clean_pts) == 0:
        return np.zeros((0, 3))
    q = np.round(atk_pts[:, :3] / tol).astype(np.int64)
    keep = set(map(tuple, q.tolist()))
    qc = np.round(clean_pts[:, :3] / tol).astype(np.int64)
    mask = np.fromiter((tuple(v) not in keep for v in qc.tolist()),
                       bool, len(qc))
    return clean_pts[mask]


def load_frame(split_dir, scene, frame_index):
    seg = pkl.load(open(os.path.join(split_dir, f"{scene}_d.pkl"), "rb"))
    return seg[frame_index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--inj_cache", required=True)
    ap.add_argument("--rem_cache", required=True)
    ap.add_argument("--inj_config", default="inject_relative_easy")
    ap.add_argument("--rem_config", default="removal_az40")
    ap.add_argument("--inj_scene", required=True)
    ap.add_argument("--inj_frame", type=int, required=True)
    ap.add_argument("--rem_scene", required=True)
    ap.add_argument("--rem_frame", type=int, required=True)
    ap.add_argument("--half", type=float, default=26.0)
    ap.add_argument("--all_sweeps", action="store_true",
                    help="render all 10 accumulated sweeps instead of the keyframe "
                         "sweep alone (default). The detector always sees all ten; "
                         "showing one makes the beam pattern legible.")
    ap.add_argument("--azimuth_only", action="store_true",
                    help="ILLUSTRATIVE removal figure: drop the elevation term so "
                         "the wedge is a clean BEV sector reaching the ego. Removes "
                         "~3x more points than the real attack, which also gates on "
                         "a 16 deg elevation band -- do not present as evaluated data.")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch
    from pcdet.ops.iou3d_nms import iou3d_nms_utils

    def iou(b, g):
        if len(b) == 0:
            return np.zeros(0, np.float32)
        return iou3d_nms_utils.boxes_iou3d_gpu(
            torch.from_numpy(np.ascontiguousarray(b[:, :7])).cuda().float(),
            torch.from_numpy(np.ascontiguousarray(g[:7]).reshape(1, 7)).cuda().float()
        ).cpu().numpy()[:, 0]

    # ---------------- injection
    print("injection triplet:")
    fr = load_frame(os.path.join(a.atlas_dir, a.inj_config), a.inj_scene, a.inj_frame)
    atk_pts = np.asarray(fr["points"], np.float64)
    spoof_gt = np.asarray(fr["spoof_gt"], np.float64)
    gtb = np.asarray(fr["gt_boxes"], np.float64)
    preds = {r["frame_index"]: r for r in
             pkl.load(open(os.path.join(a.inj_cache, a.inj_config,
                                        f"{a.inj_scene}_p.pkl"), "rb"))}
    pred = preds[a.inj_frame]
    pb = np.asarray(pred["boxes_lidar"], np.float32)
    fp_idx = int(iou(pb, spoof_gt).argmax())
    if not a.all_sweeps:
        n0 = len(atk_pts)
        atk_pts = keyframe_only(atk_pts)
        print(f"  keyframe sweep only: {n0} -> {len(atk_pts)} points")
    # clean cloud: the phantom points are the ones inside the phantom box
    inbox = points_in_box(atk_pts, spoof_gt, pad=1.05)
    clean_pts = atk_pts[~inbox]
    injection_triplet(a.out_dir, clean_pts, atk_pts, spoof_gt, gtb, pred, fp_idx,
                      half=a.half)

    # ---------------- removal
    print("removal triplet:")
    fr = load_frame(os.path.join(a.atlas_dir, a.rem_config), a.rem_scene, a.rem_frame)
    atk_pts = np.asarray(fr["points"], np.float64)
    victim = np.asarray(fr["spoof_gt"], np.float64)
    gtb = np.asarray(fr["gt_boxes"], np.float64)
    apreds = {r["frame_index"]: r for r in
              pkl.load(open(os.path.join(a.rem_cache, a.rem_config,
                                         f"{a.rem_scene}_p.pkl"), "rb"))}
    cpreds = {r["frame_index"]: r for r in
              pkl.load(open(os.path.join(a.rem_cache, "clean",
                                         f"{a.rem_scene}_p.pkl"), "rb"))}
    atk_pred, clean_pred = apreds[a.rem_frame], cpreds[a.rem_frame]
    cb = np.asarray(clean_pred["boxes_lidar"], np.float32)
    clean_idx = int(iou(cb, victim).argmax())
    an = pkl.load(open(os.path.join(a.atlas_dir, f"{a.rem_config}_annos",
                                    f"{a.rem_scene}_a.pkl"), "rb"))
    wedge = (an.get("spoof_annos") or {}).get(a.rem_frame) or {}
    # Clean cloud for this frame: take the injection config's frame and strip its
    # phantom. Same source scene and same accumulation, so the only difference from
    # the removal frame is the wedge itself.
    clean_dir = os.path.join(a.atlas_dir, a.inj_config)
    cfr = load_frame(clean_dir, a.rem_scene, a.rem_frame)
    cpts = np.asarray(cfr["points"], np.float64)
    if cfr.get("spoof_gt") is not None:
        cpts = cpts[~points_in_box(cpts, np.asarray(cfr["spoof_gt"], np.float64), 1.05)]
    if not a.all_sweeps:
        n0, n1 = len(cpts), len(atk_pts)
        cpts, atk_pts = keyframe_only(cpts), keyframe_only(atk_pts)
        print(f"  keyframe sweep only: clean {n0} -> {len(cpts)}, "
              f"attacked {n1} -> {len(atk_pts)} points")
    if a.azimuth_only and wedge:
        n1 = len(atk_pts)
        atk_pts = azimuth_only_removal(cpts, wedge["az_center_rad"],
                                       wedge["az_width_deg"])
        print(f"  DISPLAY azimuth-only wedge: {n1} -> {len(atk_pts)} points "
              f"(real attack keeps the 16 deg elevation band)")
    removal_triplet(a.out_dir, cpts, atk_pts, victim, gtb, clean_pred, atk_pred,
                    clean_idx, half=a.half)
    print(f"\n6 images -> {a.out_dir}")


if __name__ == "__main__":
    main()
