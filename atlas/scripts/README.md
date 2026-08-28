# Generation run records

The shell drivers that generated the ATLAS dataset variants. Kept verbatim — hashes in
`atlas/PROVENANCE.json` — as the record of what was actually run.

Like `eval/scripts/`, these encode the original machine: a conda env named `atlas`, and
`AG=/home/.../Polar-Hierarchical-Mamba/atlas_generation`, which is now simply `atlas/`.
Adjust `AG` and the config paths (`../tools/cfgs/dataset_configs/...`, now
`eval/cfgs/...`) before reusing one.

| | |
|---|---|
| `run_all_variants_trainval.sh` | every attack mode over the nuScenes trainval split |
| `run_sched_variants.sh` | the attack-schedule sweep — the runs `analyze_schedule_variants.py` reads |
| `regen_v1_removal.sh` | regenerates the v1 removal data, and documents the `dt` bug that made the original pass silently drop tracks |
