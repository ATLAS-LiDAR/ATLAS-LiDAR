# ATLAS

A LiDAR spoofing attack, and an interactive sandbox for it.

**[Try it in your browser →](https://atlas-lidar.github.io/ATLAS-LiDAR/)**

ATLAS injects a real vehicle's returns into a LiDAR scan, and deletes returns from a
sector to make a real vehicle disappear. This repository holds the attack, everything
needed to run and score it, and a visualizer that shows what it does to a scan as the
sensor spins.

## Layout

| | |
|---|---|
| `atlas/` | **The attack.** A-HFR sector removal and trace-based injection, ~1,400 lines, **numpy only** — no torch, no CUDA, no checkpoints. Plus the dataset generators. |
| `eval/` | Detector evaluation: attack success rate against clean mAP/NDS. Needs torch and `eval/phim`. |
| `validation/` | Real-world fidelity check against OSF deposit `k6nf2` — does ATLAS's output resemble an actual spoofer's? |
| `index.html`, `src/`, `assets/` | The sandbox. No build step, no Node. |
| `dev/` | Test and export harnesses. See [dev/README.md](dev/README.md). |
| `tools/` | Development and CI scripts. |

The split against the paper's pipeline repository: **the attack and its harness live
here; the LiDAR perception models live there.** PHiM, LOT, the OpenPCDet fork and the
checkpoints stay in `Polar-Hierarchical-Mamba`, vendored here as a submodule at
`eval/phim`.

## Quick start

```bash
# The sandbox — ES modules need a real origin, so file:// will not work
python3 -m http.server 8080
open http://127.0.0.1:8080/index.html

# The attack, on its own
pip install -e .                    # numpy is the only dependency
python3 -c "import sys; sys.path.insert(0,'atlas'); import removal"

# Scoring against real detectors
git submodule update --init eval/phim
pip install -e '.[eval]'
cd eval/phim && python setup.py develop     # builds pcdet; needs CUDA
```

Checkpoints and the raw KITTI/nuScenes archives are fetched separately.

## The fidelity claim

The sandbox is a JavaScript port of `atlas/`, so the obvious worry is that it drifts
into something that merely looks similar. Two checks stop that:

```bash
tools/crossvalidate.py      # 129 of 129 configurations agree exactly
./run-tests.sh              # 451 unit tests
```

Cross-validation runs the port and ATLAS's own Python on identical input and compares
integer outcomes. That means something only if `atlas/` is genuinely the published code,
so every file it came from is hash-pinned and verified against a commit in the pipeline
repo's history:

```bash
tools/check_provenance.py --vs-phm
# all 65 file(s) match: the extraction is reproducible from upstream history
```

Both run in CI. Details in [atlas/PROVENANCE.md](atlas/PROVENANCE.md).

## Data and licensing

`assets/` contains point clouds derived from KITTI (CC BY-NC-SA 4.0) and nuScenes,
committed so the sandbox runs from a clean clone. Both datasets' terms restrict
redistribution and commercial use. Regenerate them locally with
`tools/export_frames.py` instead if you would rather not rely on that.
