"""Session A2 configs (experiments/E2X.md X-M; DRAFT until tag e2x-a2-prereg). Run from MapAnything_Exp1/:
  python scripts/e2x/make_xm_configs.py
Writes configs/e2x_xm_<model>.json (exp1/run_xm.py) and configs/e2x_m8_<model>.json (exp1/run_exp1.py, M8 grid).
Corruption specs are copied from the frozen E3-cal grid (configs/e2x_e3cal_synth_mapanything.json) so the X-M cases
are the same hint sets the rest of E2X uses.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
C = os.path.join(HERE, "..", "..", "configs")
src = json.load(open(os.path.join(C, "e2x_e3cal_synth_mapanything.json")))
fam = {c["family"]: c for c in src["conditions"] if "family" in c}

CASES = [
    {"name": "one_cam_15", "corruption": {**fam["outlier_rot"]["corruption"], "rot_deg": 15}},
    {"name": "shared_heading_5", "corruption": {**fam["shared_heading_bias"]["corruption"], "rot_deg": 5}},
    {"name": "drift_level2", "corruption": {**fam["drift"]["corruption"], **fam["drift"]["levels"][1]}},
    {"name": "depth_x1.5", "inputs": ["intrinsics", "pose", "depth"],
     "corruption": {**fam["depth_scale"]["corruption"], "factor": 1.5}},
]
MODELS = {"mapanything": "facebook/map-anything", "pi3x": "yyfz233/Pi3X"}
FACTORS = [0.67, 1.0, 1.5]

for model, ckpt in MODELS.items():
    xm = {"checkpoint": ckpt, "backend": model, "inputs": ["intrinsics", "pose"], "depth_views": src["depth_views"],
          "layers": "all", "dial": [round(k / 6, 4) for k in range(7)], "seeds": [0], "cases": CASES}
    json.dump(xm, open(os.path.join(C, f"e2x_xm_{model}.json"), "w"), indent=1)
    conds = [{"name": "rgb_only", "family": "reference", "inputs": []},
             {"name": "clean_pose_depth2", "family": "reference", "inputs": ["intrinsics", "pose", "depth"]}]
    for p in FACTORS:
        for d in FACTORS:
            conds.append({"name": f"m8_pose{p}_depth{d}", "family": "m8", "inputs": ["intrinsics", "pose", "depth"],
                          "corruption": {"target": "pose", "type": "scale", "factor": p,
                                         "also": {"target": "depth", "type": "scale", "factor": d}}})
    m8 = {k: v for k, v in src.items() if k != "conditions"}
    m8.update(checkpoint=ckpt, conditions=conds)
    json.dump(m8, open(os.path.join(C, f"e2x_m8_{model}.json"), "w"), indent=1)
    print(f"{model}: {len(CASES)} X-M cases, {len(conds)} M8 conditions")
