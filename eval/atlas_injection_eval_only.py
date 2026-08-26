#!/usr/bin/env python
"""ATLAS injection eval-only diagnostics.

Computes the five metric blocks specified in atlas_eval_metrics_summary.pdf over
CACHED predictions (see cache_injection_preds.py). No inference happens here, so
metrics can be re-derived, re-thresholded and debugged without touching the GPU
model path.

  1. Preserved ASR         -- original semantics, plus per-class breakdown
  2. Score-thresholded ASR -- ASR as the confidence bar rises
  3. IoU distribution      -- are misses near-misses or total whiffs
  4. Box displacement      -- is a real object's box dragged toward the phantom
  5. Spoof-region FPs      -- phantom detections, net of a clean baseline

IoU is exact rotated 3D IoU. The PDF specifies a numba CPU Sutherland-Hodgman
kernel; this uses pcdet's `boxes_iou3d_gpu`, which is the same quantity (BEV
polygon intersection x z-overlap) already compiled and used by the rest of the
codebase, so there is no second implementation to keep honest. Set
--force_cpu to fall back to a pure-numpy path (slower; intended as a cross-check).

Class labels follow the nuScenes head ordering used elsewhere in ATLAS:
1 = car (vehicle), 2 = truck, 3 = construction_vehicle, ... The PDF's
"vehicle/pedestrian/cyclist" grouping is expressed via --vehicle_labels etc. so the
mapping is explicit rather than assumed.
"""
import argparse
import glob
import json
import os
import pickle as pkl
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCORE_GATES = (0.1, 0.3, 0.5, 0.7)
SPOOF_TAUS = (0.3, 0.5, 0.7)


# --------------------------------------------------------------------- IoU
def _iou3d_gpu(a, b):
    import torch
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ta = torch.from_numpy(np.ascontiguousarray(a[:, :7])).float().cuda()
    tb = torch.from_numpy(np.ascontiguousarray(b[:, :7])).float().cuda()
    return iou3d_nms_utils.boxes_iou3d_gpu(ta, tb).cpu().numpy()


def _corners_bev(box):
    """4 BEV corners of [x,y,z,dx,dy,dz,heading], counter-clockwise."""
    x, y, _, dx, dy, _, yaw = box[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    hx, hy = dx / 2.0, dy / 2.0
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    R = np.array([[c, -s], [s, c]])
    return local @ R.T + np.array([x, y])


def _clip_poly(subject, clip):
    """Sutherland-Hodgman: intersect convex `subject` with convex `clip`."""
    out = list(subject)
    for i in range(len(clip)):
        if not out:
            return []
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        inside = lambda p: edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0]) >= -1e-12
        new = []
        for j in range(len(out)):
            p, q = out[j], out[(j + 1) % len(out)]
            ip, iq = inside(p), inside(q)
            if ip:
                new.append(p)
            if ip != iq:
                d = q - p
                den = edge[0] * d[1] - edge[1] * d[0]
                if abs(den) > 1e-18:
                    t = (edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])) / -den
                    new.append(p + t * d)
        out = new
    return out


def _poly_area(poly):
    if len(poly) < 3:
        return 0.0
    p = np.asarray(poly)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _iou3d_cpu(a, b):
    """Reference implementation: BEV polygon clip x z-overlap. Slow but dependency
    free -- used to cross-check the GPU kernel."""
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    if len(a) == 0 or len(b) == 0:
        return out
    ca = [_corners_bev(x) for x in a]
    cb = [_corners_bev(x) for x in b]
    for i, bi in enumerate(a):
        za0, za1 = bi[2] - bi[5] / 2, bi[2] + bi[5] / 2
        va = bi[3] * bi[4] * bi[5]
        for j, bj in enumerate(b):
            zb0, zb1 = bj[2] - bj[5] / 2, bj[2] + bj[5] / 2
            dz = min(za1, zb1) - max(za0, zb0)
            if dz <= 0:
                continue
            inter_bev = _poly_area(_clip_poly(ca[i], cb[j]))
            if inter_bev <= 0:
                continue
            inter = inter_bev * dz
            vb = bj[3] * bj[4] * bj[5]
            out[i, j] = inter / max(va + vb - inter, 1e-9)
    return out


# ------------------------------------------------------------------ loading
def load_cache(pred_dir):
    """{scene: [record, ...]} keyed by frame_index order."""
    out = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*_p.pkl"))):
        scene = os.path.basename(f)[:-len("_p.pkl")]
        try:
            out[scene] = pkl.load(open(f, "rb"))
        except Exception:
            continue
    return out


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return {k: None for k in ("n", "min", "q1", "median", "mean", "q3", "max")}
    return {"n": int(v.size), "min": float(v.min()),
            "q1": float(np.percentile(v, 25)), "median": float(np.median(v)),
            "mean": float(v.mean()), "q3": float(np.percentile(v, 75)),
            "max": float(v.max())}


# ------------------------------------------------------------------ metrics
def compute(attacked, clean, iou_fn, args):
    """All five metric blocks over one (model, config) pair of caches."""
    veh = set(args.vehicle_labels)
    ped = set(args.pedestrian_labels)
    cyc = set(args.cyclist_labels)

    # --- 1 preserved ASR
    n_target = 0
    n_zero_pred = 0
    hit_by_class = defaultdict(int)
    veh_fp_scores = []
    best_iou_all = []

    # --- 2/3 score-gated
    gate_maxiou = {s: [] for s in SCORE_GATES}

    # --- 4 displacement
    disp = []               # per matched (frame, gt) record
    n_clean_matched = 0
    n_pairs = 0
    n_disappear = 0
    primary = []

    # --- 5 spoof-region FPs
    fp = {(t, s): dict(count=[], scores=[], ious=[], clean_count=[],
                       induced_gated=[], induced_sub=[])
          for t in SPOOF_TAUS for s in SCORE_GATES}

    for scene, recs in attacked.items():
        crecs = {r["frame_index"]: r for r in clean.get(scene, [])}
        for r in recs:
            sg = np.asarray(r["spoof_gt"], dtype=np.float32).reshape(1, -1)
            if sg.shape[1] < 7:
                continue
            n_target += 1
            boxes = np.asarray(r["boxes_lidar"], dtype=np.float32)
            labels = np.asarray(r["pred_labels"], dtype=np.int64)
            scores = np.asarray(r["score"], dtype=np.float32)
            gtb = np.asarray(r.get("gt_boxes"), dtype=np.float32)
            gtb = gtb.reshape(-1, gtb.shape[-1]) if gtb.ndim == 2 else np.zeros((0, 10), np.float32)
            # real GT vehicles only (class column is last, 1-indexed)
            gt_veh = (gtb[gtb[:, -1].astype(int) == args.gt_vehicle_label][:, :7]
                      if len(gtb) else np.zeros((0, 7), np.float32))

            if len(boxes) == 0:
                n_zero_pred += 1
                best_iou_all.append(0.0)
                for s in SCORE_GATES:
                    gate_maxiou[s].append(0.0)
                continue

            # ---- 1: argmax over ALL classes, threshold depends on that class
            iou_all = iou_fn(boxes, sg)[:, 0]
            k = int(iou_all.argmax())
            best_iou_all.append(float(iou_all[k]))
            lab = int(labels[k])
            thr = (args.asr_iou if lab in veh else
                   args.asr_iou_small if (lab in ped or lab in cyc) else None)
            if thr is not None and iou_all[k] >= thr:
                grp = ("vehicle" if lab in veh else
                       "pedestrian" if lab in ped else "cyclist")
                hit_by_class[grp] += 1
                if grp == "vehicle":
                    veh_fp_scores.append(float(scores[k]))

            # ---- vehicle-only preds reused by 2,3,5
            vmask = np.isin(labels, list(veh))
            vboxes, vscores = boxes[vmask], scores[vmask]
            iou_v_sg = iou_fn(vboxes, sg)[:, 0] if len(vboxes) else np.zeros(0, np.float32)
            iou_v_gt = (iou_fn(vboxes, gt_veh) if len(vboxes) and len(gt_veh)
                        else np.zeros((len(vboxes), 0), np.float32))

            # ---- 2/3
            for s in SCORE_GATES:
                m = vscores >= s
                gate_maxiou[s].append(float(iou_v_sg[m].max()) if m.any() else 0.0)

            # ---- 5
            not_real = (iou_v_gt.max(axis=1) < args.match_iou
                        if iou_v_gt.shape[1] else np.ones(len(vboxes), bool))
            cr = crecs.get(r["frame_index"])
            if cr is not None:
                cb = np.asarray(cr["boxes_lidar"], dtype=np.float32)
                cl = np.asarray(cr["pred_labels"], dtype=np.int64)
                cs = np.asarray(cr["score"], dtype=np.float32)
                cvm = np.isin(cl, list(veh))
                cvb, cvs = cb[cvm], cs[cvm]
                ciou_sg = iou_fn(cvb, sg)[:, 0] if len(cvb) else np.zeros(0, np.float32)
                ciou_gt = (iou_fn(cvb, gt_veh) if len(cvb) and len(gt_veh)
                           else np.zeros((len(cvb), 0), np.float32))
                cnot_real = (ciou_gt.max(axis=1) < args.match_iou
                             if ciou_gt.shape[1] else np.ones(len(cvb), bool))
            else:
                cvs = np.zeros(0, np.float32); ciou_sg = np.zeros(0, np.float32)
                cnot_real = np.zeros(0, bool)

            for t in SPOOF_TAUS:
                for s in SCORE_GATES:
                    sel = (vscores >= s) & (iou_v_sg >= t) & not_real
                    n = int(sel.sum())
                    csel = (cvs >= s) & (ciou_sg >= t) & cnot_real
                    cn = int(csel.sum())
                    e = fp[(t, s)]
                    e["count"].append(n)
                    e["clean_count"].append(cn)
                    e["scores"].extend(vscores[sel].tolist())
                    e["ious"].extend(iou_v_sg[sel].tolist())
                    # two attribution rules, both reported (PDF section 5)
                    e["induced_gated"].append(0 if cn > 0 else n)
                    e["induced_sub"].append(max(0, n - cn))

            # ---- 4 displacement: needs the clean pass
            if cr is None or len(gt_veh) == 0:
                continue
            cb = np.asarray(cr["boxes_lidar"], dtype=np.float32)
            cl = np.asarray(cr["pred_labels"], dtype=np.int64)
            cs = np.asarray(cr["score"], dtype=np.float32)
            cvm = np.isin(cl, list(veh)) & (cs >= args.disp_score)
            cvb = cb[cvm]
            avm = vscores >= args.disp_score
            avb = vboxes[avm]
            if len(cvb) == 0:
                continue
            c_gt = iou_fn(cvb, gt_veh)
            a_gt = iou_fn(avb, gt_veh) if len(avb) else np.zeros((0, len(gt_veh)), np.float32)

            frame_entries = []
            for g in range(len(gt_veh)):
                ci = int(c_gt[:, g].argmax())
                if c_gt[ci, g] < args.match_iou:
                    continue           # real object not cleanly detected: unmeasurable
                n_clean_matched += 1
                if a_gt.shape[0] == 0:
                    n_disappear += 1
                    continue
                ai = int(a_gt[:, g].argmax())
                if a_gt[ai, g] < args.match_iou:
                    n_disappear += 1
                    continue
                n_pairs += 1
                c_ctr = cvb[ci, :3].astype(np.float64)
                a_ctr = avb[ai, :3].astype(np.float64)
                sep = sg[0, :3].astype(np.float64) - c_ctr
                nsep = float(np.linalg.norm(sep))
                if nsep < 1e-6:
                    continue
                dc = a_ctr - c_ctr
                d_toward = float(dc @ (sep / nsep))
                frame_entries.append({
                    "scene": scene, "frame_index": int(r["frame_index"]),
                    "gt_index": g, "d_toward": d_toward,
                    "rho": d_toward / nsep, "disp_mag": float(np.linalg.norm(dc)),
                    "separation": nsep})
            disp.extend(frame_entries)
            if frame_entries:
                # "primary target": clean pred nearest the spoof
                primary.append(min(frame_entries, key=lambda e: e["separation"]))

    # ------------------------------------------------------------ assemble
    n_veh_fp = hit_by_class["vehicle"]
    res = {
        "n_target_frames": n_target,
        "preserved_asr": {
            "asr": n_veh_fp / n_target if n_target else None,
            "vehicle_fp_count": n_veh_fp,
            "per_class_asr": {k: v / n_target for k, v in hit_by_class.items()} if n_target else {},
            "per_class_count": dict(hit_by_class),
            "zero_pred_frames": n_zero_pred,
            "zero_pred_rate": n_zero_pred / n_target if n_target else None,
            "vehicle_fp_score_stats": _stats(veh_fp_scores),
            "best_iou_stats": _stats(best_iou_all),
        },
        "score_thresholded_asr": {
            f"{s}": {"asr": float(np.mean(np.asarray(gate_maxiou[s]) >= args.asr_iou))
                     if gate_maxiou[s] else None,
                     "n": len(gate_maxiou[s])}
            for s in SCORE_GATES},
        "iou_distribution": {
            f"{s}": dict(_stats(gate_maxiou[s]),
                         frac_ge_asr_iou=float(np.mean(np.asarray(gate_maxiou[s]) >= args.asr_iou))
                         if gate_maxiou[s] else None)
            for s in SCORE_GATES},
        "displacement": {
            "n_clean_matched": n_clean_matched,
            "n_matched_pairs": n_pairs,
            "n_disappearances": n_disappear,
            "disappearance_rate": n_disappear / n_clean_matched if n_clean_matched else None,
            "frac_d_toward_positive": float(np.mean([e["d_toward"] > 0 for e in disp]))
            if disp else None,
            "d_toward": _stats([e["d_toward"] for e in disp]),
            "rho": _stats([e["rho"] for e in disp]),
            "disp_mag": _stats([e["disp_mag"] for e in disp]),
            "separation": _stats([e["separation"] for e in disp]),
            "primary_target": {
                "n": len(primary),
                "frac_d_toward_positive": float(np.mean([e["d_toward"] > 0 for e in primary]))
                if primary else None,
                "d_toward": _stats([e["d_toward"] for e in primary]),
                "rho": _stats([e["rho"] for e in primary]),
            },
        },
        "spoof_region_fp": {},
    }
    for (t, s), e in fp.items():
        cnt = np.asarray(e["count"])
        cc = np.asarray(e["clean_count"])
        res["spoof_region_fp"][f"tau{t}_score{s}"] = {
            "total_fp": int(cnt.sum()),
            "per_frame_mean": float(cnt.mean()) if cnt.size else None,
            "phantom_detection_rate": float(np.mean(cnt >= 1)) if cnt.size else None,
            "per_frame_stats": _stats(cnt),
            "fp_score_stats": _stats(e["scores"]),
            "fp_iou_stats": _stats(e["ious"]),
            "clean_total_fp": int(cc.sum()),
            "clean_detection_rate": float(np.mean(cc >= 1)) if cc.size else None,
            "attack_induced_clean_gated": int(np.sum(e["induced_gated"])),
            "attack_induced_subtractive": int(np.sum(e["induced_sub"])),
        }
    return res, disp


# ------------------------------------------------------------------ report
def to_markdown(all_res, meta):
    L = ["# ATLAS injection eval-only metrics", ""]
    L += [f"- detector(s): {', '.join(meta['models'])}",
          f"- configs: {', '.join(meta['configs'])}",
          f"- asr_iou {meta['asr_iou']} | asr_iou_small {meta['asr_iou_small']} | "
          f"match_iou {meta['match_iou']}", ""]
    L += ["## 1-3. ASR, score gates, IoU", "",
          "| model | config | frames | ASR | zero-pred | " +
          " | ".join(f"ASR@s{s}" for s in SCORE_GATES) + " | med IoU@0.1 |",
          "|---|---|---|---|---|" + "---|" * (len(SCORE_GATES) + 1)]
    def cell(v, spec=".3f"):
        return format(v, spec) if v is not None else "-"

    for (m, c), r in sorted(all_res.items()):
        p = r["preserved_asr"]
        gates = " | ".join(cell(r["score_thresholded_asr"][str(s)]["asr"])
                           for s in SCORE_GATES)
        med = r["iou_distribution"][str(SCORE_GATES[0])]["median"]
        L.append(f"| {m} | {c} | {r['n_target_frames']} | {cell(p['asr'], '.4f')} | "
                 f"{cell(p['zero_pred_rate'])} | {gates} | {cell(med)} |")

    def fmt(v, spec=".3f"):
        return format(v, spec) if v is not None else "-"

    L += ["", "## 4. Box displacement toward the spoof", "",
          "| model | config | clean-matched | pairs | disappear | frac d>0 | "
          "med d_toward | med rho |", "|---|---|---|---|---|---|---|---|"]
    for (m, c), r in sorted(all_res.items()):
        d = r["displacement"]
        L.append(f"| {m} | {c} | {d['n_clean_matched']} | {d['n_matched_pairs']} | "
                 f"{d['n_disappearances']} | {fmt(d['frac_d_toward_positive'])} | "
                 f"{fmt(d['d_toward']['median'])} | {fmt(d['rho']['median'])} |")

    L += ["", "## 5. Spoof-region false positives", "",
          "| model | config | tau | score | FPs | /frame | det rate | clean rate | "
          "induced (gated) | induced (sub) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (m, c), r in sorted(all_res.items()):
        for t in SPOOF_TAUS:
            for s in SCORE_GATES:
                e = r["spoof_region_fp"][f"tau{t}_score{s}"]
                L.append(f"| {m} | {c} | {t} | {s} | {e['total_fp']} | "
                         f"{e['per_frame_mean']:.3f} | {e['phantom_detection_rate']:.3f} | "
                         f"{e['clean_detection_rate']:.3f} | "
                         f"{e['attack_induced_clean_gated']} | "
                         f"{e['attack_induced_subtractive']} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_cache", required=True,
                    help="root written by cache_injection_preds.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--asr_iou", type=float, default=0.7)
    ap.add_argument("--asr_iou_small", type=float, default=0.5,
                    help="IoU gate for pedestrian/cyclist argmax hits")
    ap.add_argument("--match_iou", type=float, default=0.7)
    ap.add_argument("--disp_score", type=float, default=0.1,
                    help="score floor for displacement matching")
    # nuScenes 10-class head ordering, verified against cached `name`/`pred_labels`
    # pairs: 1 car, 2 truck, 3 construction_vehicle, 4 bus, 5 trailer, 6 barrier,
    # 7 motorcycle, 8 bicycle, 9 pedestrian, 10 traffic_cone.
    # The PDF's "vehicle" is car (the class the phantom is drawn from); its
    # "cyclist" maps onto nuScenes motorcycle+bicycle, which have no single label.
    ap.add_argument("--vehicle_labels", type=int, nargs="+", default=[1])
    ap.add_argument("--pedestrian_labels", type=int, nargs="+", default=[9])
    ap.add_argument("--cyclist_labels", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--gt_vehicle_label", type=int, default=1)
    ap.add_argument("--force_cpu", action="store_true",
                    help="use the numpy reference IoU instead of the GPU kernel")
    a = ap.parse_args()

    iou_fn = _iou3d_cpu if a.force_cpu else _iou3d_gpu
    os.makedirs(a.out_dir, exist_ok=True)

    models = a.models or sorted(
        d for d in os.listdir(a.pred_cache)
        if os.path.isdir(os.path.join(a.pred_cache, d)))
    all_res, all_disp = {}, {}
    for m in models:
        mdir = os.path.join(a.pred_cache, m)
        clean = load_cache(os.path.join(mdir, "clean"))
        confs = a.configs or sorted(
            d for d in os.listdir(mdir)
            if d != "clean" and os.path.isdir(os.path.join(mdir, d)))
        for c in confs:
            attacked = load_cache(os.path.join(mdir, c))
            if not attacked:
                print(f"  [skip] {m}/{c}: no cached preds")
                continue
            res, disp = compute(attacked, clean, iou_fn, a)
            all_res[(m, c)] = res
            all_disp[(m, c)] = disp
            p = res["preserved_asr"]
            print(f"  {m:22s} {c:24s} frames={res['n_target_frames']:5d} "
                  f"ASR={p['asr']:.4f} pairs={res['displacement']['n_matched_pairs']}",
                  flush=True)

    meta = {"models": models, "configs": sorted({c for _, c in all_res}),
            "asr_iou": a.asr_iou, "asr_iou_small": a.asr_iou_small,
            "match_iou": a.match_iou, "disp_score": a.disp_score,
            "score_gates": list(SCORE_GATES), "spoof_taus": list(SPOOF_TAUS),
            "iou_backend": "cpu_reference" if a.force_cpu else "boxes_iou3d_gpu",
            "pred_cache": os.path.abspath(a.pred_cache)}

    js = {"meta": meta,
          "results": {f"{m}|{c}": r for (m, c), r in all_res.items()}}
    # atomic writes
    for name, payload in (("detailed_metrics.json", json.dumps(js, indent=1)),
                          ("detailed_metrics.md", to_markdown(all_res, meta))):
        tmp = os.path.join(a.out_dir, name + ".tmp")
        open(tmp, "w").write(payload)
        os.replace(tmp, os.path.join(a.out_dir, name))
    tmp = os.path.join(a.out_dir, "detailed_summary.pkl.tmp")
    pkl.dump({"meta": meta, "results": all_res, "displacement_records": all_disp},
             open(tmp, "wb"))
    os.replace(tmp, os.path.join(a.out_dir, "detailed_summary.pkl"))
    print(f"\nwrote detailed_metrics.{{md,json}} + detailed_summary.pkl -> {a.out_dir}")


if __name__ == "__main__":
    main()
