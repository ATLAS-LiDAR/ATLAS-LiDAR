# Where this code came from

Everything in this directory was copied **byte for byte** out of the paper's
pipeline repository (`Polar-Hierarchical-Mamba`, `atlas_generation/`). Not retyped,
not tidied, not refactored on the way in.

## It came from uncommitted work, and that matters

The copy is of that repository's **working tree**, not of a commit. Two files carried
uncommitted changes relative to its `HEAD` (`0a4a98e1c3fc`):

- **`nuscenes_atlas_common.py`** — +250/-13, adding the whole attack-scheduling
  feature: `build_single_attack_schedule`, `_scene_seed`, `extract_attack_spans`,
  `setup_scene_spoofs`, `resolve_schedule`, `add_schedule_args`.
- **`removal.py`** — +7/-2, adding the `dt` parameter to `match_with_velocity` and
  the docstring warning that passing Waymo's 0.1 for nuScenes keyframes makes the
  constant-velocity prediction 5x too short and silently drops the track.

The sandbox was developed against these versions — `src/core/schedule.js` is a port
of those scheduling functions, and the `dt` warning is one of the fidelity notes in
the README — so every 129/129 cross-validation result refers to them, and always
has.

The consequence is that **this extraction is not reproducible from the pipeline
repo's git history** until those changes are committed there. `check_provenance.py`
prints a warning on every run to keep that visible. Once they are committed, record
the commit here and drop the warning.

`PROVENANCE.json` records a sha256 for every file. `tools/check_provenance.py`
verifies them, and `--vs-phm` additionally diffs against the pipeline repo to catch
upstream drift.

## Why the hashes matter

The sandbox's central claim is that it agrees with ATLAS on every deterministic
outcome — 129 of 129 configurations, checked by `tools/crossvalidate.py`. While
ATLAS lived in a different repository, that was self-evidently a comparison against
an external reference.

Now that both halves live here, it would be trivially possible to "fix" a
disagreement by editing the thing being compared against. The hashes are what stop
that from happening silently: as long as they match, cross-validation is still
transitively a check against the published implementation.

Editing these files is allowed — it is how upstream fixes land — but it has to be
deliberate: change the file, re-run `tools/crossvalidate.py`, and only then re-record
with `tools/check_provenance.py --update`.

## Declared local patches

`PROVENANCE.json` records two hashes per file — `upstream` (what the pipeline repo
had) and `local` (what ships here). They are equal everywhere except the files listed
in its `patches` map. Currently there is exactly one:

- **`removal.py`** — deleted three module-level plotting imports
  (`matplotlib.pyplot as plt`, `plotly.graph_objects as go`, `plotly.io as pio`).
  All three were **unreferenced anywhere in the file**, verified by walking the AST
  rather than grepping. Their only effect was to make importing the A-HFR physics
  require a plotting stack, which is why `tools/crossvalidate.py` used to install
  fake `matplotlib` and `plotly` modules before it could import the reference
  implementation. That hack is gone. Cross-validation re-run after the change:
  129/129, unchanged.

## The two nuScenes calibration sets are not interchangeable

Both ship here, and they were measured over different amounts of data:

| | azimuth resolution | intensity band (0,10 m) `log_mu` | returns measured |
|---|---|---|---|
| `./` (mini) | 0.005787373 rad = 0.3316° | 2.350 | 6.0 M |
| `assets_trainval/` | 0.005801722 rad = 0.3324° | 2.742 | 9.4 M |

`nuscenes_atlas_common.py` defaults to the mini set (its `_ASSET_DIR` is the module's
own directory), and `src/core/sensor-assets.js` is generated from that same set, so
the sandbox and the reference implementation agree by construction.

The geometric difference is negligible — 0.25% in azimuth resolution. The intensity
difference is not: `exp(2.350) = 10.5` versus `exp(2.742) = 15.5` for the nearest
range band. Injected intensities therefore match a mini-scale calibration. A run
that wants trainval-scale statistics should set `NUSCENES_ATLAS_ASSET_DIR` to
`assets_trainval/` and regenerate the JS constants with
`tools/gen_sensor_assets.py --atlas ...`.

This is a scoping fact, not a defect. It is written down because nothing in the file
names hints that the two sets disagree.

## What deliberately did not come along

- **`trace_*.pkl`** — the extracted vehicle point clouds ATLAS injects, 159 MB
  across both datasets, with a single 97 MB file that is essentially at GitHub's
  100 MB per-file ceiling. The sandbox consumes the derived `.atlastrace` files in
  `assets/traces/` instead (6 MB). Regenerating those with
  `tools/export_traces.py` needs the pickles, so it stays a local-only tool.
- **`generate_data_*.py`, `point_injection*.py`, `eval_matrix_*.py`,
  `helpers_ptt.py`** — dataset generation, detector evaluation and plotting. These
  need OpenPCDet, torch, trained checkpoints and the raw dataset archives, and they
  stay in the pipeline repo. `.atlasattack` bundles are the interchange format.
