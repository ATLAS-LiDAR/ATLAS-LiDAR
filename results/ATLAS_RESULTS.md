# ATLAS Adversarial Robustness Results

Evaluation of pretrained OpenPCDet LiDAR detectors against ATLAS adversarial
variants, on **nuScenes** (v1.0-trainval val, 150 scenes) and **KITTI**
(val split, 3769 frames).

- **Clean** columns are standard detection metrics (reproduce the OpenPCDet model zoo).
- **ASR** = Attack Success Rate (higher ⇒ more vulnerable), 3D IoU threshold 0.3.
  - *Injection* success = a phantom car is detected overlapping the spoofed box.
  - *Removal* success = a real tracked car is no longer detected (A-HFR occlusion),
    swept over azimuth-wedge widths 10°–60°.
- Trace difficulty (easy/medium/hard) = phantom object point density (dense→sparse).
  Attacker trace library built from the **train** split only (no train/val leakage).
- Models are sorted by clean performance.

---

## nuScenes

Clean = mAP / NDS. Injection has two placement modes: **global** (world-fixed
phantom across the scene) and **relative** (ego-fixed).

### Clean detection + injection ASR

| Model | mAP | NDS | inj·global·easy | ·global·med | ·global·hard | inj·rel·easy | ·rel·med | ·rel·hard |
|---|---|---|---|---|---|---|---|---|
| pp_multihead | 0.447 | 0.583 | 0.320 | 0.116 | 0.029 | 0.363 | 0.113 | 0.023 |
| dyn_pp_centerpoint | 0.500 | 0.607 | 0.400 | 0.146 | 0.045 | 0.453 | 0.162 | 0.031 |
| second_multihead | 0.505 | 0.623 | 0.363 | 0.130 | 0.036 | 0.418 | 0.148 | 0.029 |
| voxel0075_voxelnext | 0.605 | 0.667 | 0.418 | 0.194 | 0.051 | 0.485 | 0.198 | 0.047 |
| transfusion_lidar | 0.646 | 0.694 | 0.349 | 0.167 | 0.053 | 0.402 | 0.175 | 0.047 |

### Removal ASR (azimuth-wedge width, degrees)

| Model | mAP | az10 | az20 | az30 | az40 | az50 | az60 |
|---|---|---|---|---|---|---|---|
| pp_multihead | 0.447 | 0.663 | 0.812 | 0.785 | 0.738 | 0.508 | 0.504 |
| dyn_pp_centerpoint | 0.500 | 0.617 | 0.807 | 0.765 | 0.714 | 0.448 | 0.456 |
| second_multihead | 0.505 | 0.611 | 0.810 | 0.771 | 0.699 | 0.427 | 0.424 |
| voxel0075_voxelnext | 0.605 | 0.615 | 0.810 | 0.770 | 0.725 | 0.436 | 0.439 |
| transfusion_lidar | 0.646 | 0.537 | 0.797 | 0.723 | 0.653 | 0.352 | 0.335 |

*Excluded:* BEVFusion (camera+LiDAR, not runnable on LiDAR-only data); the two
voxel-CenterPoint models (dead OpenPCDet Google Drive links).

---

## KITTI

Clean = Car 3D AP@R40 (moderate). KITTI is single-frame (no sequences / ego
motion), so injection is a single mode (no global/relative distinction).

### Clean detection + injection ASR

| Model | Car3D | inj·easy | inj·med | inj·hard |
|---|---|---|---|---|
| pointpillar | 76.33 | 0.424 | 0.259 | 0.048 |
| pointrcnn | 79.13 | 0.642 | 0.315 | 0.042 |
| second | 79.34 | 0.703 | 0.496 | 0.168 |
| pointrcnn_iou | 79.66 | 0.682 | 0.375 | 0.062 |
| second_iou | 79.82 | 0.694 | 0.463 | 0.148 |
| PartA2 | 81.92 | 0.449 | 0.191 | 0.031 |
| pv_rcnn | 82.35 | 0.668 | 0.295 | 0.049 |
| voxel_rcnn_car | 82.54 | 0.458 | 0.238 | 0.032 |

### Removal ASR (azimuth-wedge width, degrees)

| Model | Car3D | az10 | az20 | az30 | az40 | az50 | az60 |
|---|---|---|---|---|---|---|---|
| pointpillar | 76.33 | 0.350 | 0.664 | 0.596 | 0.524 | 0.197 | 0.197 |
| pointrcnn | 79.13 | 0.377 | 0.686 | 0.613 | 0.497 | 0.248 | 0.322 |
| second | 79.34 | 0.261 | 0.653 | 0.433 | 0.321 | 0.133 | 0.132 |
| pointrcnn_iou | 79.66 | 0.348 | 0.686 | 0.580 | 0.450 | 0.220 | 0.298 |
| second_iou | 79.82 | 0.250 | 0.652 | 0.422 | 0.314 | 0.123 | 0.121 |
| PartA2 | 81.92 | 0.310 | 0.684 | 0.567 | 0.461 | 0.176 | 0.175 |
| pv_rcnn | 82.35 | 0.322 | 0.683 | 0.620 | 0.530 | 0.194 | 0.191 |
| voxel_rcnn_car | 82.54 | 0.358 | 0.682 | 0.619 | 0.516 | 0.189 | 0.194 |

*Excluded:* PartA2_free (dead Drive link); CaDDN and Focals-Conv-F (require camera).

---

## Observations

- **Removal** peaks at the az20 wedge on both datasets (the width whose paper-backed
  removal probability is 1.0), tapering at az50/60.
- **nuScenes:** removal ASR falls as clean mAP rises, and injection ASR broadly rises
  with clean mAP (transfusion_lidar, the only transformer/query detector, is an
  outlier — more injection-robust than its mAP predicts).
- **KITTI:** no clear correlation between clean AP and either attack
  (Pearson r ≈ −0.17 injection, +0.06 removal across 8 models). Vulnerability
  tracks **architecture family** instead — the SECOND anchor models are markedly the
  most injection-vulnerable regardless of AP. This suggests the nuScenes trend is
  partly confounded by architecture rather than a general performance→robustness law.
