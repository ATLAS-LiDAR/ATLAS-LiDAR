# Standalone pages

The sandbox itself is `index.html` at the repo root. These are the other pages: four
harnesses that scripts in `tools/` drive headlessly, and one frozen figure.

They live here rather than at the root so the root is just the sandbox. All of them
reach back up for shared code (`../src/`, `../assets/`, `../test/`), so they must be
served over HTTP from the repo root — not opened as `file://`, which ES modules reject.

| page | driven by | what it does |
|---|---|---|
| `tests.html` | `run-tests.sh`, `tools/parse_dom.py` | the unit suite; CI greps its DOM for a pass marker |
| `crossvalidate.html` | `tools/crossvalidate.py` | runs the JS port on one config so Python can compare it against ATLAS — this is the page behind 129/129 |
| `export.html` | `tools/export_headless.py` | writes `.atlasattack` bundles without a browser in the loop |
| `bench.html` | — | timing harness for the compose/detect path; open it by hand |
| `lidar-spoof-sim.html` | — | the original v1 figure, self-contained and frozen. Kept because tag `v1` cites it |

```bash
python3 -m http.server 8080          # from the repo root
# http://127.0.0.1:8080/dev/bench.html
```
