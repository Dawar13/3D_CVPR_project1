"""Write the extended-E2 / E3-calibration configs from the frozen E2 ones, so condition names (and therefore every
random corruption) stay equal across models and experiments. Run from MapAnything_Exp1/:

  python scripts/e2x/make_configs.py

Writes configs/e2x_*.json. Nothing here changes a frozen config; each output names its source.
"""

import copy
import json
import os

C = "configs"
CKPT = {"mapanything": "facebook/map-anything", "omnivggt": "Livioni/OmniVGGT", "pi3x": "yyfz233/Pi3X"}
E3 = ["witness", "witness_drop"]
BASELINES = ["filter_rgbcheck", "drop_lowconf"]


def load(name):
    return json.load(open(os.path.join(C, name)))


def save(name, cfg):
    with open(os.path.join(C, name), "w", newline="\n") as f:
        json.dump(cfg, f, indent=1)
        f.write("\n")
    print(f"{name}: {len(cfg['conditions'])} condition entries")


def with_controls(conds, controls, only_pose_families=False):
    out = []
    for c in copy.deepcopy(conds):
        c.pop("controls", None)
        if not c["inputs"]:
            out.append(c)
            continue
        ctr = list(controls)
        if only_pose_families or "pose" not in c["inputs"]:   # the E1/E2 baselines act on pose hints only
            ctr = [x for x in ctr if x not in BASELINES]
        if ctr:
            c["controls"] = ctr
        out.append(c)
    return out


def main():
    # X-A: Pi3X on ETH3D, the E2-D grid verbatim (same names as MapAnything / OmniVGGT) + the E2-D checks
    for src, dst in (("e2d_full.json", "e2x_pi3x_eth3d.json"), ("e2d_checks.json", "e2x_pi3x_checks.json")):
        cfg = load(src)
        cfg["_comment"] = (f"Extended E2 (experiments/E2X.md), third model Pi3X: configs/{src} verbatim except the "
                           f"checkpoint, so every condition and random corruption equals OmniVGGT's (E2-D). Run with "
                           f"--backend pi3x --save_extras. Source comment: " + cfg["_comment"])
        cfg["checkpoint"] = CKPT["pi3x"]
        save(dst, cfg)

    # X-B: real device hints. E2-E clips (Pi3X only; MapAnything and OmniVGGT ran in E2) and room clips (all three)
    ma, ov = load("e2e_mapanything.json"), load("e2e_omnivggt.json")
    real = {"mapanything": ma, "omnivggt": ov, "pi3x": ma}       # Pi3X takes every hint set MapAnything takes
    for model, src in real.items():
        cfg = copy.deepcopy(src)
        cfg["checkpoint"] = CKPT[model]
        cfg["_comment"] = (f"Extended E2 (experiments/E2X.md) X-B, {model}: the E2-E conditions "
                           f"(configs/e2e_{'omnivggt' if model == 'omnivggt' else 'mapanything'}.json), for the room "
                           f"clips (data/clips_e2x_room) and, for Pi3X, the E2-E clips. Baselines R2/R3 on pose hints.")
        cfg["conditions"] = with_controls(src["conditions"], BASELINES)
        save(f"e2x_real_{model}.json", cfg)

    # E3 calibration ('cal' clips and dev clips only; never eval): synthetic families + real hints, with the E3
    # controls and the baselines. Thresholds are chosen from the logged decision statistics (experiments/E2X.md).
    synth = load("e2d_full.json")
    keep = {"rgb_only", "clean_pose", "clean_pose_depth2", "outlier_rot", "outlier_frac", "drift",
            "shared_heading_bias", "anchor_error", "c6_shared_rot", "depth_scale", "depth_one_scale", "noise_rot"}
    for model in CKPT:
        cfg = copy.deepcopy(synth)
        cfg["checkpoint"] = CKPT[model]
        cfg["seeds"] = [0]
        cfg["lowconf_tau"] = 0.4572
        cfg["_comment"] = ("E3 calibration, synthetic hint errors (experiments/E2X.md): the E2-D families that "
                           "stress E3's error model, from configs/e2d_full.json (same names, same corruptions), one "
                           "seed, with the E3 controls (witness = repair, witness_drop = drop ablation), the baselines "
                           "R2/R3 and the ORACLE drop. cal and dev clips only.")
        conds = [c for c in synth["conditions"] if c.get("name", c.get("family")) in keep]
        for c in conds:
            c["seeds"] = 1 if "seeds" in c else c.get("seeds")
        cfg["conditions"] = with_controls(conds, BASELINES + E3)
        for c in cfg["conditions"]:
            if c.get("family") in ("outlier_rot", "outlier_frac"):
                c["controls"] = c.get("controls", []) + ["drop_oracle"]
            if c.get("seeds") is None:
                c.pop("seeds", None)
        save(f"e2x_e3cal_synth_{model}.json", cfg)
        cfg = copy.deepcopy(real[model])
        cfg["checkpoint"] = CKPT[model]
        cfg["_comment"] = ("E3 calibration, real hints (experiments/E2X.md): the E2-E conditions with the E3 "
                           "controls and baselines. cal and dev clips only.")
        cfg["conditions"] = with_controls(real[model]["conditions"], BASELINES + E3)
        save(f"e2x_e3cal_real_{model}.json", cfg)


if __name__ == "__main__":
    main()
