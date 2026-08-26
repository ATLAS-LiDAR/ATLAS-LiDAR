#!/usr/bin/env python
"""Analyse the ATLAS schedule-variant splits (atlas_val_sched/).

Why this script exists
----------------------
The scheduled splits give each of the 150 nuScenes val sequences exactly ONE
attack whose length L varies per sequence (2-32 frames). The stock ASR is
frame-pooled -- sum(success)/sum(targets) over all frames -- so a 32-frame attack
carries 16x the weight of a 2-frame attack and the pooled number mostly describes
long attacks. Three metrics are therefore reported:

  1. frame-pooled ASR   sum(success)/sum(targets)         (continuity with the
                                                           existing results table)
  2. sequence-level ASR fraction of sequences with >=1 successful frame, and a
                        stricter >=50%-of-attacked-frames variant
  3. macro-averaged ASR mean over sequences of success/targets  (equal weight per
                                                                 sequence)

Denominator rule for 2 and 3: sequences where the attack never fired (no
trackable victim, or the phantom was fully occlusion-filtered) have no attacked
frames at all. They are EXCLUDED from the denominator and reported separately --
counting them as failures would conflate target availability with attack efficacy,
and dropping them silently would overstate coverage.

Primary comparison: removal vs injection WITHIN each continuity regime. L and the
span placement are keyed to the scene name, so they are identical across all 12
base variants -- this is a paired per-sequence comparison, reported as such.

Caveat that the paired comparison must respect: removal carries the A-HFR firing
gate (ahfr_gate_prob, 0.38-1.0 by target range), so at matched scheduled L removal
fires on systematically fewer frames than injection. Realised attacked-frame counts
are reported alongside every number for that reason.

WHAT THESE SPLITS CAN AND CANNOT SHOW
-------------------------------------
All nuScenes checkpoints evaluated here are SINGLE-FRAME detectors: each inference
consumes one accumulated sweep with no temporal state. Consequently:

  * Attack LENGTH cannot causally change per-frame detection, and neither can
    continuity (continuous vs intermittent). A frame is scored on its own content.
    The length breakdown printed below is descriptive bookkeeping only.
  * What the schedules DO buy is (a) a sequence-level unit of analysis -- one
    attack per sequence, so "did this attack work?" becomes answerable -- and
    (b) a much broader sample of phantom geometry than the fixed 8-frame tail
    window of the legacy splits, which were all sampled from one position in the
    sequence.
  * Genuine temporal claims (does a detector recover after a gap? does tracking
    smooth over an intermittent attack?) require a temporal or tracking-based
    model. These splits are the correct data for that experiment, but no model
    evaluated here can answer it.
"""
import argparse
import glob
import os
import pickle as pkl
from collections import defaultdict

import numpy as np


def load_per_scene(per_scene_dir):
    """{(model, split): {scene: {targets, success, scheduled}}}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(per_scene_dir, "*_per_scene.pkl"))):
        try:
            d = pkl.load(open(f, "rb"))
        except Exception:
            continue
        out[(d["model"], d["split"])] = d
    return out


def load_schedule_meta(atlas_dir):
    """{split: {scene: schedule_meta}} -- L, span, gaps, duty per sequence."""
    meta = {}
    for annos in sorted(glob.glob(os.path.join(atlas_dir, "*_annos"))):
        split = os.path.basename(annos)[:-len("_annos")]
        m = {}
        for af in sorted(glob.glob(os.path.join(annos, "*_a.pkl"))):
            try:
                an = pkl.load(open(af, "rb"))
            except Exception:
                continue
            sm = an.get("schedule_meta")
            if sm:
                m[an["segment"]] = sm
        if m:
            meta[split] = m
    return meta


def metrics(per_scene, strict_frac=0.5):
    """The three ASR metrics plus coverage bookkeeping."""
    tgt = np.array([v["targets"] for v in per_scene.values()], dtype=float)
    suc = np.array([v["success"] for v in per_scene.values()], dtype=float)
    fired = tgt > 0                      # attack actually reached the detector
    n_seq = len(tgt)
    n_fired = int(fired.sum())

    pooled = suc.sum() / tgt.sum() if tgt.sum() else float("nan")
    if n_fired:
        rate = suc[fired] / tgt[fired]
        seq_any = float((suc[fired] >= 1).mean())
        seq_strict = float((rate >= strict_frac).mean())
        macro = float(rate.mean())
    else:
        seq_any = seq_strict = macro = float("nan")

    return {
        "frame_pooled_asr": float(pooled),
        "seq_asr_any": seq_any,
        "seq_asr_50pct": seq_strict,
        "macro_asr": macro,
        "n_sequences": n_seq,
        "n_sequences_attacked": n_fired,
        "n_sequences_no_attack": n_seq - n_fired,
        "total_attacked_frames": int(tgt.sum()),
        "total_success_frames": int(suc.sum()),
        "median_attacked_frames": float(np.median(tgt[fired])) if n_fired else float("nan"),
    }


def fmt_split_table(data, atlas_dir, regime):
    sm = load_schedule_meta(atlas_dir)
    keys = sorted(k for k in data if k[1].endswith(regime))
    if not keys:
        print(f"  (no results for regime {regime})")
        return {}
    print(f"\n{'model':22s} {'split':30s} {'pool':>6s} {'seq>=1':>7s} "
          f"{'seq50':>6s} {'macro':>6s} {'seq':>4s} {'noatk':>6s} {'frm':>5s}")
    print("-" * 104)
    rows = {}
    for model, split in keys:
        m = metrics(data[(model, split)]["per_scene"])
        rows[(model, split)] = m
        print(f"{model:22s} {split:30s} {m['frame_pooled_asr']:6.3f} "
              f"{m['seq_asr_any']:7.3f} {m['seq_asr_50pct']:6.3f} {m['macro_asr']:6.3f} "
              f"{m['n_sequences_attacked']:4d} {m['n_sequences_no_attack']:6d} "
              f"{m['median_attacked_frames']:5.1f}")
    return rows


def paired_removal_vs_injection(data, regime, metric="macro_asr"):
    """Paired per-sequence comparison, within one continuity regime.

    Pairing is legitimate because L and the span placement are keyed to the scene
    name, so injection and removal attack the SAME sequence with the SAME schedule.
    """
    inj = [k for k in data if k[1].startswith("inject") and k[1].endswith(regime)]
    rem = [k for k in data if k[1].startswith("removal") and k[1].endswith(regime)]
    if not inj or not rem:
        return
    print(f"\n{'='*104}")
    print(f"PAIRED removal vs injection -- regime {regime} -- metric {metric}")
    print(f"{'='*104}")
    models = sorted({k[0] for k in inj} & {k[0] for k in rem})
    for model in models:
        i_scores, r_scores = [], []
        for k in [x for x in inj if x[0] == model]:
            i_scores.append(metrics(data[k]["per_scene"])[metric])
        for k in [x for x in rem if x[0] == model]:
            r_scores.append(metrics(data[k]["per_scene"])[metric])
        i_m, r_m = np.nanmean(i_scores), np.nanmean(r_scores)
        print(f"  {model:24s} injection {i_m:.3f} (n={len(i_scores)} splits)   "
              f"removal {r_m:.3f} (n={len(r_scores)})   diff {r_m - i_m:+.3f}")

    # per-sequence pairing on the shared scene set, averaged over splits
    print("\n  per-sequence paired detail (mean over splits, shared scenes):")
    for model in models:
        def seq_rate(keys):
            acc = defaultdict(list)
            for k in keys:
                for sc, v in data[k]["per_scene"].items():
                    if v["targets"] > 0:
                        acc[sc].append(v["success"] / v["targets"])
            return {sc: float(np.mean(v)) for sc, v in acc.items()}
        I = seq_rate([x for x in inj if x[0] == model])
        R = seq_rate([x for x in rem if x[0] == model])
        common = sorted(set(I) & set(R))
        if len(common) < 3:
            continue
        a = np.array([I[s] for s in common])
        b = np.array([R[s] for s in common])
        d = b - a
        try:
            from scipy.stats import wilcoxon
            stat, p = wilcoxon(a, b)
            ptxt = f"  wilcoxon p={p:.4f}"
        except Exception:
            ptxt = ""
        print(f"    {model:24s} n_common={len(common):3d}  inj {a.mean():.3f}  "
              f"rem {b.mean():.3f}  mean diff {d.mean():+.3f}  "
              f"r={np.corrcoef(a, b)[0, 1]:+.3f}{ptxt}")


def length_effect(data, atlas_dir, regime, metric_buckets=((2, 4), (5, 8), (9, 16), (17, 32))):
    """ASR broken down by scheduled attack length L -- DESCRIPTIVE ONLY.

    This must NOT be read as "longer attacks are harder to detect". Every model
    evaluated here is a SINGLE-FRAME detector: inference sees one accumulated
    sweep with no temporal state, so attack length cannot causally affect
    per-frame detection. A frame is detected on its own merits regardless of how
    many neighbouring frames are also attacked.

    Any trend across these buckets is a sampling artifact. Measured on
    inject_relative_easy_cont: phantom range is flat across L (median 16.8-17.9 m,
    r(L, range)=+0.14) because place_trace_relative pins a constant ego-relative
    offset, but phantom POINT DENSITY falls with L (median 62 pts at L2-4 vs
    33-36 at L9-32; frac<50pts 0.37 -> 0.56). Longer attacks simply sample more
    frames, including more occlusion-degraded ones, so the per-frame average
    drifts down.

    The table is kept because it documents that sampling imbalance, which matters
    when comparing these splits to the fixed-length legacy splits. It is not
    evidence about attack duration. Testing duration would need a temporal /
    tracking-based detector.
    """
    sm = load_schedule_meta(atlas_dir)
    print(f"\n{'='*104}")
    print(f"ASR by scheduled attack length L -- regime {regime}")
    print("DESCRIPTIVE ONLY: single-frame detectors cannot be sensitive to attack")
    print("length. Trends here reflect phantom-density sampling, not duration.")
    print(f"{'='*104}")
    for family in ("inject", "removal"):
        keys = sorted(k for k in data
                      if k[1].startswith(family) and k[1].endswith(regime))
        if not keys:
            continue
        acc = {b: [] for b in metric_buckets}
        for k in keys:
            meta = sm.get(k[1], {})
            for sc, v in data[k]["per_scene"].items():
                if v["targets"] <= 0 or sc not in meta:
                    continue
                L = meta[sc]["attack_len_L"]
                for b in metric_buckets:
                    if b[0] <= L <= b[1]:
                        acc[b].append(v["success"] / v["targets"])
                        break
        cells = "  ".join(
            f"L{b[0]}-{b[1]}: {np.mean(acc[b]):.3f} (n={len(acc[b])})"
            if acc[b] else f"L{b[0]}-{b[1]}: --"
            for b in metric_buckets)
        print(f"  {family:9s} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas_dir", required=True,
                    help="atlas_val_sched/ (holds the splits and *_annos)")
    ap.add_argument("--per_scene_dir", default=None,
                    help="default <atlas_dir>/per_scene")
    ap.add_argument("--metric", default="macro_asr",
                    choices=["macro_asr", "seq_asr_any", "seq_asr_50pct",
                             "frame_pooled_asr"])
    a = ap.parse_args()
    psd = a.per_scene_dir or os.path.join(a.atlas_dir, "per_scene")

    data = load_per_scene(psd)
    if not data:
        raise SystemExit(f"no per-scene dumps in {psd}; run run_sched_eval.sh first")
    print(f"loaded {len(data)} (model, split) results from {psd}")

    for regime, label in (("_cont", "CONTINUOUS"), ("_intermit", "INTERMITTENT")):
        print(f"\n\n{'#'*104}")
        print(f"# {label} regime")
        print(f"{'#'*104}")
        fmt_split_table(data, a.atlas_dir, regime)
        paired_removal_vs_injection(data, regime, a.metric)
        length_effect(data, a.atlas_dir, regime)

    print("\n\nNOTES")
    print("  pool  = frame-pooled ASR (long attacks dominate; kept for continuity)")
    print("  seq>=1= fraction of sequences with at least one successful frame")
    print("  seq50 = fraction of sequences successful on >=50% of attacked frames")
    print("  macro = mean of per-sequence success rates (equal weight per sequence)")
    print("  noatk = sequences where the attack never reached the detector")
    print("          (no trackable victim / phantom fully occluded) -- these are")
    print("          EXCLUDED from seq/macro denominators, not counted as failures")
    print("  frm   = median attacked frames per attacked sequence. Removal is lower")
    print("          than injection at matched L because of the A-HFR firing gate,")
    print("          so compare removal to injection at matched REALISED frames.")


if __name__ == "__main__":
    main()
