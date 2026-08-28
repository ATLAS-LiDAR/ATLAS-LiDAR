# Findings and limitations

## Findings

Two things surfaced that are worth carrying back to the pipeline.

**1. `osf_k6nf2/realizability.py` uses the wrong azimuth resolution, inflating
cell fill by 2.09×.** It assumes 0.40° for a 10 Hz VLP-16, but the datasheet
gives 0.192° at 10 Hz (300k points/sec over 16 channels; 0.40° corresponds to 20
Hz). Recomputed on `real_last.npz` + `real_dual.npz` (n=126 clusters):

| azimuth res | cell fill p95 (unique) | fill p95 (with dual) | fill max (dual) |
|---|---|---|---|
| 0.40° (as published) | 0.590 | 2.134 | **2.219** |
| 0.192° (datasheet) | 0.283 | 1.025 | 1.065 |

The decisive evidence is the last column: at 0.40° the measured dual-return fill
exceeds **2.219 returns per cell**, above the hard physical limit of 2 for a
dual-return sensor. At 0.192° it is 1.065. Because the published fill envelope is
~2× too permissive, synthetic phantoms compared against it look *more* realizable
than they are. Both sets of figures are kept in `realizability.js` so the
comparison stays explicit.

**2. Derived point rates independently corroborate the measured beam assets.**
Deriving `nBeams * azCellCount * hz` from the fitted azimuth resolutions lands on
the manufacturers' quoted single-return rates: HDL-32E 347,520 vs 350,000 quoted,
HDL-64E 1,333,760 vs 1.33M. That is a strong check that the `.npy` assets are
right.

## Known limitations

Genuine fidelity limits:

- **The scan lattice cannot be recovered exactly from released clouds.** Measured
  on `nuscenes-boston`: 1.63 returns per occupied cell with nearest-inclination
  binning, 1.53 using the dataset's own ring index, against an ideal 1.0. The
  residual is azimuth, not beam assignment — released clouds ship rectified
  float32 XYZ, so the original azimuth grid is gone before anything downstream
  runs (the same effect the validation doc reports as clean KITTI failing the
  azimuth-grid invariant at chance level). Consequence: occupied-cell counts are
  underestimated and cell fill slightly overestimated.
- **Intensity is a substantive gap in ATLAS and remains one here**, though v1 at
  least lets you switch between the in-distribution and saturated models to see
  the trade-off the validation doc quantifies (closing the gap collapses ASR by
  85–95%, because saturation is out-of-distribution in the top 1% tail rather
  than simply brighter).
- **The angular-leakage constants are fitted, not measured** — see *Calibration
  honesty* above.

Out of scope by decision (see **v1 scope** at the top), not oversights:

- **Single frames, no temporal dimension.** The schedule machinery in
  `nuscenes_atlas_common.py` — continuous vs intermittent attack spans — has no
  analogue here.
- **No downstream detector.** This shows what the attack does to the cloud, not
  whether a detector is fooled. Attack *success* remains a question for the
  offline pipeline.
- **No 3D or multi-ring view.** The lattice is modelled in full (azimuth × beam),
  but displayed as a BEV plus a range image rather than a 3D scene.

