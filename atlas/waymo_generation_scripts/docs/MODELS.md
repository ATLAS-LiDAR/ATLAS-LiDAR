# Models: training recipes and checkpoints (Waymo, OpenPCDet-style training)

Common to all: Waymo train split, adam_onecycle, augmentation random flip (x, y), rotation +-pi/4, scaling
0.95-1.05; voxel 0.1 x 0.1 x 0.15 m, <= 5 points/voxel, MeanVFE, VoxelResBackBone8x, CenterHead unless stated;
post-processing score >= 0.1, NMS IoU 0.7, <= 500 boxes. Point range x, y in [-75.2, 75.2], z in [-2, 4].

| Model | Config | GT run / no-GT run | Epochs, LR | Train frames | Total batch (logs) GT / no-GT |
|---|---|---|---|---|---|
| CP-SF | OpenPCDet `custom_models/centerpoint_singleframe_waymo.yaml` (features x,y,z,intensity,elongation) | `default/` / `no_gt_sampling/` | 30, 0.003 | every 5th frame | 16 / 16 |
| CP4F | OpenPCDet `custom_models/centerpoint_multiframe_waymo.yaml` (4 sweeps `SAMPLE_OFFSET [-3,0]`, + timestamp) | `default/` / `no_gt_sampling/` | 36, 0.003 | every frame | 8 / 16 |
| MSF-4 / MSF-8 | MSF `waymo_models/msf_{4,8}frames.yaml` (`SAMPLE_OFFSET [-3,0]` / `[-7,0]`) | `default/` / being trained (Nov 2026, NCSA Delta) | 6, 0.006 | every frame | 4 (MSF-4) |
| PTT-32 | PTT `waymo_models/ptt_32frames.yaml` (MPPNet + PTTHead, `SAMPLE_OFFSET [-31,0]`) / `ptt_32frames_nogt.yaml` | Feb 2026 run / Oct 2026 run | 6, 0.003 | every frame | see notes |

- **GT sampling** (OpenPCDet `gt_sampling`): pastes real database objects into each training frame (targets
  Vehicle 15, Pedestrian 10, Cyclist 10; objects with >= 5 points; whole-scene limit). Pasted objects keep real
  points and intensities and are positives. Measured paste:real ratio on val: ~0.06x vehicles, ~0.6x pedestrians,
  ~35x cyclists. The 4-frame DB stores each object's real 4-sweep history. **No-GT** = `DISABLE_AUG_LIST: ['gt_sampling']`.
- **Second stages**: MSF and PTT are trained on CP4F e36 predictions (`USE_PREDBOX`, `ROI_BOXES_PATH` = CP4F e36
  `result_fixed.pkl` on train/val). GT MSF/PTT use the GT CP4F; no-GT MSF/PTT use the no-GT CP4F e36 predictions
  (`result.pkl` -> `result_fixed.pkl` via OpenPCDet `fix_pkl.py`, which only adds `pred_boxes`/`pred_scores` keys).
- **PTT GT epochs**: e3, e6 from the Feb run (e1-3 a full 3-epoch schedule, e4-6 an extension; e4 ~25% of a pass);
  e1-2 from a separate 6-epoch GT run. No-GT PTT: 2 x L40S, batch 8, 6-epoch schedule; full-val L2 mAPH 73.28
  (GT e6: 73.63).
- Clean vehicle recall @ IoU 0.7 (final epochs, 50 study segments) GT / no-GT: CP-SF 70.4 / 71.0, CP4F 73.9 / 75.2,
  PTT 78.8 / 79.4. Cyclist recall @ 0.5: CP-SF 59.8 / 53.0, CP4F 66.7 / 63.0, PTT 67.8 / 64.2.

## Architecture facts the results depend on
- CP-SF: one sweep; intensity and elongation are 2 of 5 input features (averaged per voxel).
- CP4F: 4 sweeps in the current ego frame + timestamp, averaged per voxel (voxels can mix sweeps). A world-fixed
  ATLAS spoof replays the same trace points every sweep: 70-84% of its voxels hold all 4 sweeps (real static cars
  15-25%), and stacking adds few voxels (1.1x vs 1.7-2.0x for real cars).
- MSF: refines CP4F proposals; propagates each into the previous 3 / 7 frames (velocity x -0.1 per frame) and
  samples points from each of those frames.
- PTT: refines CP4F current-frame proposals with a 32-frame proposal track (constant velocity + IoU >= 0.5);
  current-frame points only; final score = sqrt(sigmoid(PTT) x CP4F score); cannot detect what CP4F did not propose.
