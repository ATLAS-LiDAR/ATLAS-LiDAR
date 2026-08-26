#!/usr/bin/env python3
"""REAL PRA removal characteristics, v3.

Fixes two bugs in v1/v2:
  1. np.median over a NaN-padded stack returns NaN -> ~55% of rays silently
     dropped out of every outcome bucket. Use nanmedian.
  2. "stable ray" was >0.9 occupancy, which admits NaN-reference rays. Require
     full occupancy in the reference phase.

Phases from saturated-return count + sector occupancy:
  A target present, attack off (clean reference)   B sector empty   C attack on
The sector has essentially no background (phase B is empty), so every stable ray
in the sector is a target ray -- the removal rate needs no target/background
disambiguation.

Removal, per the paper's mechanism, is a genuine echo being REPLACED on its own
ray (the spoof echo wins) or lost entirely -- not a missing packet.
"""
import sys, numpy as np
sys.path.insert(0, ".")
from extract_removal_real import frames_from_pcap, SAT_REFL

TOL = 0.5   # m; range change counted as a replacement

def run(path, az_res, tag):
    R, F = [], []
    for o, r, f in frames_from_pcap(path, az_res):
        R.append(r); F.append(f)
    R = np.stack(R); F = np.stack(F)
    sat = (F >= SAT_REFL) & (R > 0)
    satn = sat.sum(axis=(1, 2))
    C = satn >= 0.5 * np.percentile(satn, 90)
    _, ai, _ = np.nonzero(sat[C])
    lo, hi = int(np.percentile(ai, 5)), int(np.percentile(ai, 95))
    S = slice(lo, hi + 1)
    socc = (R[:, S, :] > 0).mean(axis=(1, 2))
    A = (~C) & (socc > 0.5)
    B = (~C) & (socc <= 0.05)

    Rm = np.where(R > 0, R, np.nan)
    refR = np.nanmedian(Rm[A], axis=0)                      # FIX: nanmedian
    stable = ((R[A] > 0).mean(axis=0) >= 1.0) & np.isfinite(refR)
    m = np.zeros_like(stable); m[S, :] = True
    stable &= m
    nch = len(np.unique(np.nonzero(stable)[1]))

    cur = R[C][:, stable]; ref = refR[stable]
    d = cur - ref
    absent  = cur == 0
    surv    = (~absent) & (np.abs(d) <= TOL)
    repl    = (~absent) & (np.abs(d) > TOL)
    removed = absent | repl
    per_frame = removed.mean(axis=1)

    print(f"\n=== {tag}  az_res={az_res:.1f}  frames={len(R)} ===")
    print(f"  wedge az {lo*az_res:.1f}-{(hi+1)*az_res:.1f} deg  "
          f"width {(hi-lo+1)*az_res:.2f}   A/B/C={A.sum()}/{B.sum()}/{C.sum()}")
    print(f"  background returns in wedge (phase B): {int((R[B][:,S,:]>0).sum())}"
          f"  -> wedge is free space beyond the target")
    print(f"  stable target rays {int(stable.sum())}  over {nch}/16 channels  "
          f"target range med {np.nanmedian(ref):.2f} m")
    print(f"  outcome fractions (sum {absent.mean()+surv.mean()+repl.mean():.3f}): "
          f"absent {absent.mean():.3f}  survived {surv.mean():.3f}  "
          f"replaced {repl.mean():.3f}")
    print(f"  REMOVAL RATE  med {np.median(per_frame):.3f}  "
          f"IQR [{np.percentile(per_frame,25):.3f},{np.percentile(per_frame,75):.3f}]  "
          f"max {per_frame.max():.3f}")
    # best-aimed frames: the paper reports per-attack success, not a time average
    top = np.sort(per_frame)[-max(1, len(per_frame)//10):]
    print(f"  removal rate, best-aimed decile of frames: {top.mean():.3f}")
    if repl.any():
        print(f"  replacement range med {np.nanmedian(cur[repl]):.2f} m")
    return dict(width=(hi-lo+1)*az_res, rate=float(np.median(per_frame)),
                best=float(top.mean()), nch=nch, nray=int(stable.sum()),
                rng=float(np.nanmedian(ref)))

out = {}
for p, t in (("Multimode Analysis/last_mode.pcap", "last_mode"),
             ("Multimode Analysis/dual_mode.pcap", "dual_mode")):
    for res in (0.4, 0.2):
        out[(t, res)] = run(p, res, t)

print("\n--- chord length (the paper's own normalisation: 2 d sin(dtheta/2)) ---")
for (t, res), v in out.items():
    if res != 0.4: continue
    print(f"  real {t}: {v['width']:.2f} deg at {v['rng']:.2f} m "
          f"-> chord {2*v['rng']*np.sin(np.deg2rad(v['width'])/2):.3f} m")
for w in (10, 20, 30, 40, 50, 60):
    for d in (20.0, 30.0):
        print(f"  ATLAS az{w}: {w} deg at {d:.0f} m -> chord "
              f"{2*d*np.sin(np.deg2rad(w)/2):.2f} m", end="   " if d==20 else "\n")
