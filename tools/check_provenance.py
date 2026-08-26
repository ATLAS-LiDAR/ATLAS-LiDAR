#!/usr/bin/env python3
"""Verify that atlas/ is still the code it claims to be.

WHY THIS EXISTS
While ATLAS lived in a separate repo, "the sandbox agrees with ATLAS on 129 of 129
configurations" was a claim about an external reference. Now that both halves live
here, that comparison is between two implementations in the same repository, and it
would quietly lose its meaning if someone edited atlas/ to make a disagreement go
away.

So the extraction is pinned instead. atlas/PROVENANCE.json records the sha256 of
every file copied out of the paper's pipeline repo, and this script checks them. As
long as they match, cross-validation is still transitively a check against the
published implementation rather than against a local edit.

    tools/check_provenance.py              # verify against the recorded hashes
    tools/check_provenance.py --vs-phm     # also diff against the pipeline repo

--vs-phm detects DRIFT: if the pipeline repo's ATLAS changes after this extraction,
the two are no longer the same attack and the copy here needs refreshing (which
means re-running cross-validation, not just re-recording a hash).

Editing atlas/ is not forbidden - it is how upstream fixes land - but it must be
deliberate: change the file, re-run tools/crossvalidate.py, and re-record the hash
with --update. A silent edit is what this catches.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "atlas", "PROVENANCE.json")
PHM = os.environ.get("PHM_ROOT", os.path.expanduser("~/rsch/Polar-Hierarchical-Mamba"))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vs-phm", action="store_true",
                    help="also compare against the pipeline repo, to detect upstream drift")
    ap.add_argument("--update", action="store_true",
                    help="re-record the hashes (only after re-running crossvalidate.py)")
    a = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    recorded = manifest["files"]
    rc = 0

    if a.update:
        for f in recorded:
            recorded[f] = sha256(os.path.join(ROOT, "atlas", f))
        json.dump(manifest, open(MANIFEST, "w"), indent=2, sort_keys=True)
        print("re-recorded %d hashes - re-run tools/crossvalidate.py to justify it" % len(recorded))
        return 0

    print("checking %d extracted files against atlas/PROVENANCE.json" % len(recorded))
    for f, want in sorted(recorded.items()):
        p = os.path.join(ROOT, "atlas", f)
        if not os.path.exists(p):
            print("  MISSING  %s" % f)
            rc = 1
            continue
        got = sha256(p)
        if got != want:
            print("  EDITED   %s" % f)
            print("           recorded %s" % want[:16])
            print("           on disk  %s" % got[:16])
            rc = 1

    if rc == 0:
        print("  all files match the extraction from %s@%s"
              % (manifest["source_repo"], manifest["source_commit"][:12]))

    if a.vs_phm:
        src = os.path.join(PHM, manifest["source_path"])
        print("\ncomparing against %s" % src)
        if not os.path.isdir(src):
            print("  pipeline repo not found - skipping drift check (set PHM_ROOT)")
            return rc
        drift = 0
        for f in sorted(recorded):
            up = os.path.join(src, f)
            if not os.path.exists(up):
                print("  gone upstream: %s" % f)
                drift += 1
            elif sha256(up) != recorded[f]:
                print("  DRIFTED: %s has changed upstream since extraction" % f)
                drift += 1
        if drift:
            print("\n%d file(s) drifted. Refresh the copy, re-run crossvalidate.py, then --update." % drift)
            rc = 1
        else:
            print("  no drift: the extraction is current")

    return rc


if __name__ == "__main__":
    sys.exit(main())
