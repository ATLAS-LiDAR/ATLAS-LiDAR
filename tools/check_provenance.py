#!/usr/bin/env python3
"""Verify that atlas/ is still the code it claims to be.

WHY THIS EXISTS
While ATLAS lived in a separate repo, "the sandbox agrees with ATLAS on 129 of 129
configurations" was a claim about an external reference. Now that both halves live
here, that comparison is between two implementations in the same repository, and it
would quietly lose its meaning if someone edited atlas/ to make a disagreement go
away.

So the extraction is pinned instead. atlas/PROVENANCE.json records two hashes per
file:

    upstream  what the pipeline repo had at the extraction commit
    local     what we actually ship

They are equal for every file except the ones named in the manifest's `patches`
map, where a local deviation is declared with its justification. That way a
deliberate change is self-documenting and a silent one is caught.

    tools/check_provenance.py              # verify local hashes + declared patches
    tools/check_provenance.py --vs-phm     # also check upstream for drift
    tools/check_provenance.py --update     # re-record local hashes

--vs-phm detects DRIFT: if the pipeline repo's ATLAS changes after this extraction,
the two are no longer the same attack and the copy here needs refreshing - which
means re-running cross-validation, not just re-recording a hash.

Editing atlas/ is not forbidden; it is how upstream fixes and patches land. But it
must be deliberate: change the file, re-run tools/crossvalidate.py, declare the
reason in `patches` if it is a local deviation, then --update.

TWO TIERS, DELIBERATELY
`files` is the attack itself - removal.py, the two *_atlas_common.py modules, the
calibration assets. Those are what tools/crossvalidate.py compares against, so an
undeclared edit there invalidates the 129/129 claim outright. They get the strict
two-hash treatment (upstream vs local, with declared patches).

`mirrors` is the surrounding harness copied over from the pipeline repo: dataset
generation, the detector evaluation matrix, real-world validation against OSF k6nf2,
and the result writeups. A single hash each, because a difference there is a
maintenance problem - the copies have diverged and someone should reconcile them -
not a reason to distrust the fidelity number.
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
                    help="also compare upstream hashes against the pipeline repo")
    ap.add_argument("--update", action="store_true",
                    help="re-record local hashes (only after re-running crossvalidate.py)")
    a = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    files = manifest["files"]
    patches = manifest.get("patches", {})
    rc = 0

    mirrors = manifest.get("mirrors", [])

    if a.update:
        for f, rec in files.items():
            rec["local"] = sha256(os.path.join(ROOT, "atlas", f))
        n = 0
        for group in mirrors:
            for name in group["files"]:
                group["files"][name] = sha256(os.path.join(ROOT, group["local_dir"], name))
                n += 1
        json.dump(manifest, open(MANIFEST, "w"), indent=2, sort_keys=True)
        print("re-recorded %d core + %d mirrored hashes - re-run tools/crossvalidate.py"
              " to justify it" % (len(files), n))
        return 0

    print("checking %d extracted files against atlas/PROVENANCE.json" % len(files))
    clean = patched = 0
    for f, rec in sorted(files.items()):
        p = os.path.join(ROOT, "atlas", f)
        if not os.path.exists(p):
            print("  MISSING   %s" % f)
            rc = 1
            continue
        got = sha256(p)
        if got != rec["local"]:
            print("  EDITED    %s  (undeclared change)" % f)
            print("            recorded local %s" % rec["local"][:16])
            print("            on disk        %s" % got[:16])
            rc = 1
        elif rec["local"] != rec["upstream"]:
            if f not in patches:
                print("  UNDECLARED PATCH  %s differs from upstream with no entry in `patches`" % f)
                rc = 1
            else:
                patched += 1
        else:
            clean += 1

    if rc == 0:
        print("  %d file(s) byte-identical to the %s %s"
              % (clean, manifest["source_repo"], manifest["source_state"]))
        for f in sorted(patches):
            print("  declared patch: %s" % f)
            print("      %s" % patches[f])
        # The extraction came from uncommitted work upstream, so say so every run
        # rather than burying it in a JSON field nobody opens.
        unc = manifest.get("uncommitted_vs_head", {}).get("head_sha256", {})
        if unc:
            print("  WARNING: %d file(s) were extracted from UNCOMMITTED upstream work"
                  % len(unc))
            for f in sorted(unc):
                print("      %s differs from upstream HEAD %s"
                      % (f, manifest["head_commit_at_extraction"][:12]))
            print("      Not reproducible from the pipeline repo's history until"
                  " those changes are committed there.")

    # The harness mirrored out of the pipeline repo: generation, evaluation,
    # real-world validation, writeups. One hash each - see TWO TIERS above.
    if mirrors:
        total = sum(len(g["files"]) for g in mirrors)
        print("\nchecking %d mirrored harness file(s) across %d group(s)"
              % (total, len(mirrors)))
        for group in mirrors:
            edited = missing = 0
            for name, want in sorted(group["files"].items()):
                p = os.path.join(ROOT, group["local_dir"], name)
                if not os.path.exists(p):
                    print("  MISSING   %s/%s" % (group["local_dir"], name))
                    missing += 1
                elif sha256(p) != want:
                    print("  EDITED    %s/%s" % (group["local_dir"], name))
                    edited += 1
            if edited or missing:
                rc = 1
            else:
                print("  %-26s %2d file(s) intact  (from %s/%s)"
                      % (group["local_dir"], len(group["files"]),
                         manifest["source_repo"], group["upstream_dir"].rstrip("/.")))

    if a.vs_phm:
        src = os.path.join(PHM, manifest["source_path"])
        print("\nchecking upstream for drift: %s" % src)
        if not os.path.isdir(src):
            print("  pipeline repo not found - skipping (set PHM_ROOT)")
            return rc
        drift = 0
        for f, rec in sorted(files.items()):
            up = os.path.join(src, f)
            if not os.path.exists(up):
                print("  gone upstream: %s" % f)
                drift += 1
            elif sha256(up) != rec["upstream"]:
                print("  DRIFTED: %s changed upstream since extraction" % f)
                drift += 1
        for group in mirrors:
            for name, want in sorted(group["files"].items()):
                up = os.path.join(PHM, group["upstream_dir"], name)
                if not os.path.exists(up):
                    print("  gone upstream: %s/%s" % (group["upstream_dir"], name))
                    drift += 1
                elif sha256(up) != want:
                    print("  DRIFTED: %s/%s changed upstream" % (group["upstream_dir"], name))
                    drift += 1
        if drift:
            print("\n%d file(s) drifted. Refresh, re-run crossvalidate.py, then --update." % drift)
            rc = 1
        else:
            print("  no drift: the extraction is current")

    return rc


if __name__ == "__main__":
    sys.exit(main())
