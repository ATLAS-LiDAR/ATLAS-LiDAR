"""Test-time mitigations for the mitigation table (rows "Intensity-consistency filter", "Sector point-drop detection").
The paper names both but does not define them; the definitions below were chosen on 2026-11-06 and are calibrated on
CLEAN frames only (the 50 evaluation segments), never on attacked data.

IC  (intensity-consistency filter): for every predicted Vehicle box with >= MIN_PTS current-frame points inside,
    features = (mean, std) of the point intensities (tanh scale, as the detectors see them). Reference = robust
    mean/sigma of the same features over real (GT) vehicles in the same 10 m range band. z = max(|z_mean|, |z_std|).
    The box is dropped if z > k, with k = the 99th percentile of z over clean true-positive boxes of that detector
    (so ~1% of correct clean detections are removed). Mirrors the paper's intensity audit (SMD per range band).
SD  (sector point-drop detection + hold): per frame, a 1-deg azimuth histogram of points at 2-80 m range is compared
    with that of a reference frame (the last un-flagged frame, ego-motion compensated with the Waymo poses, at most
    MAX_AGE frames old). A bin is flagged if (n_cur + 5) / (n_ref + 5) < rho; a contiguous run of >= MIN_SECTOR
    flagged bins is a dropped sector. rho is the largest value whose flag rate on clean frames is <= 1%.
    Response: the reference frame's predicted boxes whose centres fall in a dropped sector (+-2 deg) and that have no
    current box within 2 m are re-inserted (pose-compensated, original score): a "hold last detection" defence.

  calib  --det cp_sf|cp4f --label gt_epoch30 --clean <clean_recall predictions.pkl> --out calib.json
  apply  --calib calib.json --mode run|clean|val --src ... --seqs i:j --out_dir ...   (both filters)
  merge  --mode clean|val --parts dir --out file
"""
import argparse, glob, json, os, pickle, time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

WAYMO = '/storage/cedar/cedar0/cedarp-gchou3-0/Waymo/waymo_processed_data_v0_5_0'
GEN = '/storage/scratch1/9/spanse30/cvpr_datasets'
MIN_PTS, BANDS = 5, np.arange(0, 80, 10)
NBIN, RMIN, RMAX, MIN_SECTOR, MAX_AGE, SMOOTH = 360, 2.0, 80.0, 5, 30, 5.0


# ---------------------------------------------------------------- data
def seq_infos(seq):
    return pickle.load(open(f'{WAYMO}/{seq}/{seq}.pkl', 'rb'))


def clean_points(seq, idx):
    a = np.load(f'{WAYMO}/{seq}/{idx:04d}.npy')
    a = a[a[:, 5] == -1]                      # DISABLE_NLZ_FLAG_ON_POINTS, as the detectors' get_lidar
    out = a[:, :4].astype(np.float32); out[:, 3] = np.tanh(out[:, 3]); return out


def split_fid(fid):
    s, i = str(fid).rsplit('_', 1); return s, int(i)


# ---------------------------------------------------------------- geometry
def box_point_sets(pts, tree, boxes):
    """indices of points inside each box (x,y,z,l,w,h,yaw)."""
    out = []
    for b in boxes:
        r = 0.5 * np.hypot(b[3], b[4]) + 0.1
        idx = np.asarray(tree.query_ball_point(b[:2], r), dtype=np.int64)
        if idx.size == 0: out.append(idx); continue
        p = pts[idx, :3] - b[:3]; c, s = np.cos(-b[6]), np.sin(-b[6])
        x = p[:, 0] * c - p[:, 1] * s; y = p[:, 0] * s + p[:, 1] * c
        m = (np.abs(x) <= b[3] / 2) & (np.abs(y) <= b[4] / 2) & (np.abs(p[:, 2]) <= b[5] / 2)
        out.append(idx[m])
    return out


def band(r):
    return np.clip(np.searchsorted(BANDS, r, side='right') - 1, 0, len(BANDS) - 1)


def feats(pts, sets):
    return np.array([[pts[i, 3].mean(), pts[i, 3].std()] if len(i) >= MIN_PTS else [np.nan, np.nan] for i in sets])


def az_hist(pts):
    r = np.hypot(pts[:, 0], pts[:, 1]); m = (r >= RMIN) & (r <= RMAX)
    a = (np.degrees(np.arctan2(pts[m, 1], pts[m, 0])) + 360.0) % 360.0
    return np.bincount(a.astype(int) % NBIN, minlength=NBIN).astype(np.float64)


def transform(pts, T_ref, T_cur):
    M = np.linalg.inv(T_cur) @ T_ref
    return pts[:, :3] @ M[:3, :3].T + M[:3, 3]


def flagged_bins(h_cur, h_ref, rho):
    f = (h_cur + SMOOTH) / (h_ref + SMOOTH) < rho
    if not f.any(): return np.zeros(NBIN, bool)
    out = np.zeros(NBIN, bool); ff = np.concatenate([f, f])      # circular runs
    i = 0
    while i < NBIN:
        if ff[i]:
            j = i
            while j < i + NBIN and ff[j]: j += 1
            if j - i >= MIN_SECTOR: out[np.arange(i, j) % NBIN] = True
            i = j
        else: i += 1
    return out


# ---------------------------------------------------------------- filters on one sequence
def ic_filter(pred, pts, tree, ref, k):
    """drop inconsistent Vehicle boxes; returns new pred dict."""
    keep = np.ones(len(pred['score']), bool)
    veh = np.flatnonzero(pred['name'] == 'Vehicle')
    if len(veh):
        B = pred['boxes_lidar'][veh]; F = feats(pts, box_point_sets(pts, tree, B))
        z = zscore(F, np.hypot(B[:, 0], B[:, 1]), ref)
        keep[veh[np.nan_to_num(z, nan=0.0) > k]] = False
    return {**pred, **{key: pred[key][keep] for key in ('name', 'score', 'boxes_lidar', 'pred_labels') if key in pred}}


def zscore(F, rng, ref):
    b = band(rng); mu = np.asarray(ref['mu'])[b]; sg = np.asarray(ref['sigma'])[b]
    return np.max(np.abs(F - mu) / sg, axis=1)


def sd_sequence(frames, rho):
    """frames: list of (pts, pose, pred). Returns list of new preds and the flag record."""
    out, rec = [], []
    ref = None                                    # (pts, pose, pred, age)
    for pts, pose, pred in frames:
        new = pred; nflag = 0
        if ref is not None:
            h_ref = az_hist(transform(ref[0], ref[1], pose)); h_cur = az_hist(pts)
            fb = flagged_bins(h_cur, h_ref, rho); nflag = int(fb.sum())
        if nflag and ref[3] < MAX_AGE:
            M = np.linalg.inv(pose) @ ref[1]
            rb = ref[2]['boxes_lidar'].copy()
            if len(rb):
                rb[:, :3] = rb[:, :3] @ M[:3, :3].T + M[:3, 3]
                rb[:, 6] = rb[:, 6] + np.arctan2(M[1, 0], M[0, 0])
                az = (np.degrees(np.arctan2(rb[:, 1], rb[:, 0])) + 360.0) % 360.0
                wide = fb | np.roll(fb, 1) | np.roll(fb, -1) | np.roll(fb, 2) | np.roll(fb, -2)
                sel = wide[az.astype(int) % NBIN]
                cur = pred['boxes_lidar']
                if len(cur) and sel.any():
                    d = np.linalg.norm(rb[sel][:, None, :2] - cur[None, :, :2], axis=2).min(1)
                    s2 = np.flatnonzero(sel)[d > 2.0]
                else:
                    s2 = np.flatnonzero(sel)
                if len(s2):
                    new = dict(pred)
                    new['boxes_lidar'] = np.concatenate([cur, rb[s2]]) if len(cur) else rb[s2]
                    for key in ('name', 'score', 'pred_labels'):
                        if key in pred: new[key] = np.concatenate([pred[key], ref[2][key][s2]])
            ref = (ref[0], ref[1], ref[2], ref[3] + 1)
        else:
            ref = (pts, pose, pred, 0)
        out.append(new); rec.append(nflag)
    return out, rec


# ---------------------------------------------------------------- calibration
def iter_clean_seq(det_annos):
    by = {}
    for d in det_annos:
        s, i = split_fid(d['frame_id']); by.setdefault(s, {})[i] = d
    for s in sorted(by):
        yield s, by[s]


def calib(a):
    C = pickle.load(open(a.clean, 'rb')); det, gts = C['det_annos'], {g['frame_id']: g for g in C['gts']}
    gf, gr, tz_parts, flags = [], [], [], []
    tp = []                                        # (features, range) of clean TP vehicle predictions
    rho_grid = np.round(np.arange(0.05, 0.61, 0.025), 3)
    nflag = {r: 0 for r in rho_grid}; nfr = 0
    for s, frames in iter_clean_seq(det):
        info = {i['point_cloud']['sample_idx']: i for i in seq_infos(s)}
        prev = None
        for idx in sorted(frames):
            pts = clean_points(s, idx); tree = cKDTree(pts[:, :2]); pose = info[idx]['pose']
            g = gts[frames[idx]['frame_id']]
            gv = g['boxes'][g['name'] == 'Vehicle'][:, :7] if len(g['name']) else np.zeros((0, 7))
            if len(gv):
                F = feats(pts, box_point_sets(pts, tree, gv)); ok = ~np.isnan(F[:, 0])
                gf.append(F[ok]); gr.append(np.hypot(gv[ok, 0], gv[ok, 1]))
            p = frames[idx]; m = (p['name'] == 'Vehicle') & (p['score'] >= 0.5)
            if m.any() and len(gv):
                B = p['boxes_lidar'][m]; dmin = np.linalg.norm(B[:, None, :2] - gv[None, :, :2], axis=2).min(1)
                B = B[dmin < 1.0]
                if len(B):
                    F = feats(pts, box_point_sets(pts, tree, B)); ok = ~np.isnan(F[:, 0])
                    tp.append((F[ok], np.hypot(B[ok, 0], B[ok, 1])))
            if prev is not None:                    # clean flag rate (reference = previous frame: nothing is flagged)
                h_ref = az_hist(transform(prev[0], prev[1], pose)); h_cur = az_hist(pts); nfr += 1
                for r in rho_grid:
                    if flagged_bins(h_cur, h_ref, r).any(): nflag[r] += 1
            prev = (pts, pose)
    GF, GR = np.concatenate(gf), np.concatenate(gr); b = band(GR)
    mu, sg = [], []
    for k in range(len(BANDS)):
        x = GF[b == k] if (b == k).sum() >= 50 else GF[b >= k - 1] if (b >= k - 1).sum() >= 50 else GF
        med = np.median(x, 0); mad = 1.4826 * np.median(np.abs(x - med), 0)
        mu.append(med.tolist()); sg.append(np.maximum(mad, 1e-3).tolist())
    ref = {'mu': mu, 'sigma': sg}
    TF = np.concatenate([t[0] for t in tp]); TR = np.concatenate([t[1] for t in tp])
    z = zscore(TF, TR, ref); k = float(np.percentile(z, 99))
    rate = {float(r): nflag[r] / max(nfr, 1) for r in rho_grid}
    ok = [r for r in rho_grid if rate[float(r)] <= 0.01]; rho = float(max(ok)) if ok else float(rho_grid[0])
    out = {'det': a.det, 'label': a.label, 'ic_ref': ref, 'ic_k': k, 'ic_n_gt_boxes': int(len(GF)),
           'ic_n_tp_boxes': int(len(TF)), 'sd_rho': rho, 'sd_flag_rate': rate, 'sd_n_frames': nfr,
           'params': {'MIN_PTS': MIN_PTS, 'BANDS': BANDS.tolist(), 'NBIN': NBIN, 'RMIN': RMIN, 'RMAX': RMAX,
                      'MIN_SECTOR': MIN_SECTOR, 'MAX_AGE': MAX_AGE, 'SMOOTH': SMOOTH}}
    json.dump(out, open(a.out, 'w'), indent=1); print(json.dumps({k_: v for k_, v in out.items() if k_ != 'ic_ref'}, indent=1))


# ---------------------------------------------------------------- apply
def apply(a):
    cal = json.load(open(a.calib)); ref, k, rho = cal['ic_ref'], cal['ic_k'], cal['sd_rho']
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'run':                            # attacked set: <run>/dataset/<seq>.pkl + preds <seq>_p.pkl
        seqs = sorted(Path(f'{GEN}/{a.src}/dataset').glob('*.pkl'))
    else:                                          # clean (50 segs) or val: one det_annos list
        D = pickle.load(open(a.src, 'rb')); D = D['det_annos'] if isinstance(D, dict) else D
        groups = dict(iter_clean_seq(D)); seqs = sorted(groups)
    i0, i1 = (int(x) for x in a.seqs.split(':')) if a.seqs else (0, len(seqs))
    t_ic = t_sd = n = 0
    for s in seqs[i0:i1]:
        if a.mode == 'run':
            name = Path(s).stem; ofile = out / 'ic' / f'{name}_p.pkl'
            if ofile.exists() and (out / 'sd' / f'{name}_p.pkl').exists(): continue
            F = pickle.load(open(s, 'rb')); P = pickle.load(open(f'{GEN}/{a.src}/evaluation/{a.pred_sub}/predictions/{name}_p.pkl', 'rb'))
            assert len(F) == len(P)
            frames = [(f['points'][:, :4].astype(np.float32), f['pose'], p) for f, p in zip(F, P)]
        else:
            name = s; ofile = out / 'ic' / f'{name}.pkl'
            if not os.path.isdir(f'{WAYMO}/{s}'):        # val predictions on 202 seqs; processed data / split has 195
                print('skip (no processed data):', s); continue
            if ofile.exists() and (out / 'sd' / f'{name}.pkl').exists(): continue
            info = {i['point_cloud']['sample_idx']: i for i in seq_infos(s)}
            frames = [(clean_points(s, i), info[i]['pose'], groups[s][i]) for i in sorted(groups[s])]
        t = time.time(); ic = [ic_filter(p, x, cKDTree(x[:, :2]), ref, k) for x, _, p in frames]; t_ic += time.time() - t
        t = time.time(); sd, rec = sd_sequence(frames, rho); t_sd += time.time() - t; n += len(frames)
        for sub, res in (('ic', ic), ('sd', sd)):
            (out / sub).mkdir(exist_ok=True)
            pickle.dump(res, open(out / sub / (f'{name}_p.pkl' if a.mode == 'run' else f'{name}.pkl'), 'wb'))
        json.dump({'flagged_bins': rec}, open(out / 'sd' / f'{name}.flags.json', 'w'))
    if n: json.dump({'frames': n, 'ic_ms_per_frame': 1000 * t_ic / n, 'sd_ms_per_frame': 1000 * t_sd / n},
                    open(out / f'timing_{i0}_{i1}.json', 'w'))
    print('done', a.mode, a.src, i0, i1, n, 'frames')


def merge(a):
    parts = sorted(Path(a.parts).glob('*.pkl')); res = []
    for p in parts: res += pickle.load(open(p, 'rb'))
    pickle.dump(res, open(a.out, 'wb')); print('merged', len(parts), 'seqs', len(res), 'frames ->', a.out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('calib'); c.add_argument('--det'); c.add_argument('--label'); c.add_argument('--clean'); c.add_argument('--out')
    p = sp.add_parser('apply'); p.add_argument('--calib'); p.add_argument('--mode', choices=['run', 'clean', 'val'])
    p.add_argument('--src'); p.add_argument('--pred_sub', default=''); p.add_argument('--seqs', default=''); p.add_argument('--out_dir')
    m = sp.add_parser('merge'); m.add_argument('--parts'); m.add_argument('--out')
    a = ap.parse_args(); {'calib': calib, 'apply': apply, 'merge': merge}[a.cmd](a)
