#!/usr/bin/env python
"""Add NeurIPS-required Responsible AI metadata to a Croissant file.

The hosting platforms (HuggingFace included) populate only the Croissant CORE
fields. NeurIPS 2026 Evaluations & Datasets additionally requires a minimal set of
RAI fields in the Croissant file, and states that failure to provide them "may
constitute grounds for rejection":

    rai:dataLimitations               rai:dataSocialImpact
    rai:dataBiases                    rai:hasSyntheticData
    rai:personalSensitiveInformation  prov:wasDerivedFrom
    rai:dataUseCases                  prov:wasGeneratedBy

The `rai:` and `prov:` prefixes must also be declared in @context, which the
platform-generated files omit.

Usage:
    python add_croissant_rai.py --dataset kitti --in <core>.json --out <full>.json
"""
import argparse
import json

RAI_CONTEXT = {
    "rai": "http://mlcommons.org/croissant/RAI/",
    "prov": "http://www.w3.org/ns/prov#",
}

# No `citeAs`: the paper is under review, so no title or citation is published
# yet. NeurIPS does not ask for it -- their required set is the 8 RAI fields plus
# the platform-generated core fields, and the word "citeAs" does not appear in
# their hosting guidelines at all. mlcroissant emits a "recommended but does not
# exist" advisory for it, which is expected and not a validation failure. Add it
# once the submission is de-anonymized.

# ---------------------------------------------------------------- shared text
_SYNTH = (
    "Yes. Both attack families are synthetically generated and applied to real "
    "sensor recordings; the underlying point clouds are real LiDAR captures, and "
    "the adversarial perturbation is synthetic. Injection adds a phantom vehicle "
    "assembled from real vehicle point clusters ('traces') harvested from the "
    "TRAINING split, then ground-anchored, raycast against the scene for occlusion "
    "consistency, and given intensities resampled from range-conditioned statistics "
    "measured on real returns of the same sensor. Removal deletes real points inside "
    "an azimuth wedge aimed at a tracked vehicle. No point coordinates are "
    "hallucinated by a generative model."
)

_SENSITIVE = (
    "No personal or sensitive attributes are present or annotated. The data contains "
    "no gender, age, socio-economic, health, political, religious, cultural, or "
    "linguistic attributes. The modality is LiDAR range measurement only: there are "
    "no camera images, no faces, no licence plates, and no audio. Pedestrians and "
    "cyclists appear as unlabelled sparse point returns in some scenes, which are "
    "not personally identifiable at LiDAR resolution and are inherited unchanged "
    "from the source dataset. Geographic scope is inherited from the source dataset "
    "(city-scale driving routes) and is not linked to any individual. Attacks target "
    "only the vehicle/car class."
)

_BIASES = (
    "Biases are inherited from the source dataset and are not corrected here. "
    "(1) Geographic and cultural: the source recordings cover a small number of "
    "cities, so road layouts, vehicle models, signage, and driving conventions are "
    "not globally representative. (2) Class: attacks target only the car/vehicle "
    "class, so robustness conclusions do not transfer to pedestrians, cyclists, or "
    "other classes. (3) Condition: predominantly daytime, fair-weather scenes, so "
    "night, rain, snow, and fog are under-represented. (4) Attack-geometry: phantom "
    "placement is sampled from a bounded ego-relative region (roughly 15-20 m ahead, "
    "+/-3 m lateral), so very close-range and very long-range attacks are "
    "under-represented. (5) Target-availability: removal requires a suitable vehicle "
    "in frame, so scenes without one are absent from removal configs, mildly skewing "
    "those configs toward busier scenes."
)

_IMPACT = (
    "Intended positive impact: enabling systematic, reproducible measurement of how "
    "LiDAR-based 3D object detectors fail under physically-motivated spoofing, so "
    "that defences can be developed and compared before such attacks are encountered "
    "in deployment. Publishing standardised adversarial splits lets robustness claims "
    "be checked rather than asserted. Risk of misuse: the data documents attack "
    "geometry that could inform an attempt to attack a real vehicle. We judge this "
    "risk low relative to the defensive benefit: the underlying attack mechanisms are "
    "already published in the peer-reviewed security literature, this release adds no "
    "new hardware technique, and it contains no hardware designs, firmware, timing "
    "parameters, or laser specifications needed to execute a physical attack. "
    "Mitigation: the release is derived data plus attack metadata only, and the "
    "accompanying paper reports defences alongside attacks."
)

# ------------------------------------------------------------------ per-dataset
KITTI = dict(
    limitations=(
        "(1) Derived from the KITTI 3D object detection VALIDATION split only "
        "(3769 frames); no training or test split is included, so it is an "
        "evaluation resource, not a training corpus. (2) Point clouds are "
        "camera-FOV cropped (roughly |azimuth| <= 40 deg, x > 5 m), matching the "
        "standard KITTI 3D detection evaluation region; they are NOT full 360-degree "
        "sweeps. (3) KITTI 3D object frames are independent, so the dataset carries "
        "no temporal dimension and cannot be used to study attack persistence, "
        "tracking, or temporal filtering. (4) Range quantisation and azimuth-grid "
        "structure of the raw sensor stream are not preserved, because KITTI itself "
        "distributes rectified float32 coordinates; the released clouds match clean "
        "KITTI in this respect but not a raw packet capture. (5) Frame counts differ "
        "per config because frames that could not be attacked were not written "
        "(3769 / 3767 / 3649 for injection, 3384 for removal): attack-success-rate "
        "denominators must be the config's own length, not 3769. (6) Attacks target "
        "the Car class only. (7) Phantom intensity is sampled from the diffuse "
        "range-conditioned distribution rather than saturated, which is appropriate "
        "for measuring geometric robustness but does not reproduce the saturated "
        "returns a real spoofer produces. NOT recommended for: training detectors, "
        "temporal/multi-frame robustness studies, evaluating intensity-based spoofing "
        "defences, or claims about non-vehicle classes."
    ),
    use_cases=(
        "Construct: the dataset is intended to measure how much a 3D object "
        "detector's output can be manipulated by a physically-realisable LiDAR "
        "spoofing attack — specifically the creation of a phantom vehicle "
        "(injection) or the concealment of a real one (removal). Validity evidence: "
        "attacks are occlusion-consistent (phantoms are raycast against the scene "
        "and self-occluded points removed, real points behind a phantom are "
        "deleted); removal probabilities per wedge width are calibrated against "
        "physical measurements reported by Cao et al. (USENIX Security 2023); and the "
        "generated clouds were compared against a real Velodyne spoofing capture on "
        "sensor-lattice invariants (ring-arc spacing matched real hardware to within "
        "3.9%, and single-return-per-ray occupancy was closer to real hardware than "
        "clean KITTI). Established for: benchmarking and comparing detector "
        "robustness to injection and removal attacks; ablating attack geometry "
        "(wedge width, phantom density); measuring attack success rate. NOT "
        "established for: certifying that a detector is safe to deploy; predicting "
        "end-to-end vehicle behaviour or crash risk; fairness auditing (no "
        "demographic attributes exist); evaluating attacks on classes other than "
        "cars; or measuring temporal robustness."
    ),
    derived=["https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d"],
    generated=(
        "Data collection: no new sensor data was collected. All point clouds "
        "originate from the KITTI 3D object detection benchmark (Velodyne HDL-64E, "
        "Karlsruhe, Germany, 2011/2012), used under its CC BY-NC-SA 3.0 licence. "
        "Preprocessing: frames were loaded through the standard OpenPCDet KITTI "
        "pipeline (camera-FOV crop, point-range filter), yielding ~16k points per "
        "cloud. Synthetic generation - injection: a real vehicle point cluster is "
        "sampled from a trace library built from the KITTI TRAINING split only (to "
        "avoid train/validation leakage), placed at a random ego-relative offset "
        "(~15-20 m ahead, +/-3 m lateral) with random yaw, ground-anchored to the "
        "local surface height, given intensities resampled from range-conditioned "
        "statistics measured on real KITTI returns, then raycast against the scene "
        "so that occluded phantom points are dropped and real points behind the "
        "phantom are deleted. Synthetic generation - removal: a car is selected by "
        "range and class, and points inside an azimuth wedge (10-60 deg) aimed at it "
        "are deleted with a per-width probability calibrated to the A-HFR attack "
        "measurements of Cao et al. (USENIX Security 2023). Determinism: every frame "
        "uses a seeded RNG (seed 42), so generation is reproducible. Data "
        "annotation: no human annotation was performed. Clean object labels, "
        "difficulty flags and calibration matrices are inherited verbatim from "
        "KITTI; attack labels (the phantom box, or the box of the vehicle targeted "
        "for removal) are emitted programmatically by the generator, together with "
        "per-frame attack parameters. No crowdworkers, annotation platforms, or "
        "LLM/synthetic agents were involved at any stage. Verification: the released "
        "Parquet was checked field-by-field against the generator's own output for "
        "all 31489 frames with zero mismatches."
    ),
)

NUSCENES = dict(
    limitations=(
        "(1) Derived from the nuScenes trainval VALIDATION split only (150 scenes, "
        "6019 keyframes); no training or test split is included, so it is an "
        "evaluation resource, not a training corpus. (2) Only ATTACKED frames are "
        "shipped. A scene is ~40 keyframes and the attack covers a subset, so "
        "reconstructing a full sequence requires the user's own copy of nuScenes; "
        "the clean frames are deliberately not redistributed. (3) Point clouds are "
        "10-sweep accumulated (~265k points/frame) following the standard nuScenes "
        "detection setting; single-sweep clouds are not provided. (4) Reproducing "
        "the clean frames bit-exactly requires the same detector preprocessing "
        "pipeline (sweep accumulation plus point-range filter); naive sweep "
        "accumulation yields ~1.3x too many points. (5) Attack-success-rate "
        "denominators must be the config's own length, not 6019. (6) Attacks target "
        "the car class only. (7) Removal carries a range-dependent firing gate from "
        "the A-HFR attack model, so not every scheduled frame fires; scheduled and "
        "realised attack frames are recorded separately and must not be conflated. "
        "(8) Phantom intensity is diffuse rather than saturated, appropriate for "
        "geometric robustness but not reproducing a real spoofer's saturated "
        "returns. NOT recommended for: training detectors, evaluating "
        "intensity-based spoofing defences, or claims about non-vehicle classes."
    ),
    use_cases=(
        "Construct: the dataset is intended to measure how much a 3D object "
        "detector's output can be manipulated by a physically-realisable LiDAR "
        "spoofing attack — the creation of a phantom vehicle (injection) or the "
        "concealment of a real one (removal) — and, because nuScenes scenes are "
        "sequences, how that manipulation is distributed in time. Validity "
        "evidence: attacks are occlusion-consistent; removal probabilities per "
        "wedge width are calibrated against physical measurements reported by Cao "
        "et al. (USENIX Security 2023); and generated clouds were compared against a "
        "real Velodyne spoofing capture on sensor-lattice invariants. Established "
        "for: benchmarking detector robustness to injection and removal; ablating "
        "attack geometry and attack duration; measuring attack success rate at "
        "frame and sequence level. NOT established for: certifying deployment "
        "safety; predicting end-to-end vehicle behaviour; fairness auditing (no "
        "demographic attributes exist); non-car classes. Note that the temporal "
        "schedule variants are intended for future temporal/tracking-based "
        "detectors; single-frame detectors are by construction insensitive to when "
        "within a sequence a frame occurs."
    ),
    derived=["https://www.nuscenes.org/nuscenes"],
    generated=(
        "Data collection: no new sensor data was collected. All point clouds "
        "originate from the nuScenes trainval dataset (32-beam spinning LiDAR, "
        "Boston and Singapore, 2018-2019), used under its CC BY-NC-SA 4.0 licence. "
        "Preprocessing: keyframes were loaded through the standard OpenPCDet "
        "nuScenes pipeline with 10-sweep accumulation and point-range filtering, "
        "yielding ~265k points per cloud. Synthetic generation - injection: a real "
        "vehicle point cluster is sampled from a trace library built from the "
        "nuScenes TRAINING split only (to avoid train/validation leakage), placed "
        "either world-fixed ('global') or ego-fixed ('relative') at a random offset "
        "(~15-20 m ahead, +/-3 m lateral) with random yaw, ground-anchored, given "
        "intensities resampled from range-conditioned statistics measured on real "
        "nuScenes returns, then raycast for occlusion consistency. Synthetic "
        "generation - removal: a car is tracked across the attack window with a "
        "constant-velocity associator (dt = 0.5 s, matching the 2 Hz keyframe rate), "
        "and points inside an azimuth wedge (10-60 deg) aimed at it are deleted with "
        "a per-width probability calibrated to the A-HFR measurements of Cao et al. "
        "(USENIX Security 2023), subject to that model's range-dependent firing "
        "gate. Temporal schedules: the default schedule attacks the final 8 frames "
        "of each scene; '_cont' and '_intermit' variants attack one span per scene of "
        "length 2-32 frames, contiguous or with interior gaps, with span length and "
        "placement derived from a per-scene content hash so they are identical "
        "across configs. Determinism: seeded RNGs throughout (seed 42), so "
        "generation is reproducible. Data annotation: no human annotation was "
        "performed. Clean object labels and ego-pose matrices are inherited verbatim "
        "from nuScenes; attack labels and per-frame attack parameters are emitted "
        "programmatically. No crowdworkers, annotation platforms, or LLM/synthetic "
        "agents were involved at any stage."
    ),
)

DATASETS = {"kitti": KITTI, "nuscenes": NUSCENES}


def add_rai(doc, spec):
    ctx = doc.setdefault("@context", {})
    for k, v in RAI_CONTEXT.items():
        ctx.setdefault(k, v)
    # `version` is recommended by mlcroissant and the platform file omits it.
    doc.setdefault("version", "1.0.0")
    doc.pop("citeAs", None)   # withheld during review; see note above
    doc["rai:dataLimitations"] = spec["limitations"]
    doc["rai:dataBiases"] = _BIASES
    doc["rai:personalSensitiveInformation"] = _SENSITIVE
    doc["rai:dataUseCases"] = spec["use_cases"]
    doc["rai:dataSocialImpact"] = _IMPACT
    doc["rai:hasSyntheticData"] = True
    doc["rai:dataCollection"] = spec["generated"]
    doc["prov:wasDerivedFrom"] = spec["derived"]
    doc["prov:wasGeneratedBy"] = spec["generated"]
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    a = ap.parse_args()

    doc = json.load(open(a.src))
    doc = add_rai(doc, DATASETS[a.dataset])
    json.dump(doc, open(a.dst, "w"), indent=2, ensure_ascii=True)

    required = ["rai:dataLimitations", "rai:dataBiases",
                "rai:personalSensitiveInformation", "rai:dataUseCases",
                "rai:dataSocialImpact", "rai:hasSyntheticData",
                "prov:wasDerivedFrom", "prov:wasGeneratedBy"]
    missing = [f for f in required if f not in doc]
    print(f"wrote {a.dst}")
    for f in required:
        v = doc.get(f)
        n = len(v) if isinstance(v, str) else ("bool" if isinstance(v, bool) else len(v))
        print(f"  {f:38s} {'OK':4s} ({n} chars)" if f not in missing else f"  {f:38s} MISSING")
    print("all required RAI fields present" if not missing else f"MISSING: {missing}")


if __name__ == "__main__":
    main()
