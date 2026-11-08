"""E3 eval configs (experiments/E3.md): the E3-cal grids, unchanged, with the frozen E3 thresholds:
  e3_z = 5.0                 secondary calibration (E2X E3-cal amendment 1: ETH3D dev + TUM fr3, accurate ground truth;
                             the primary calibration on SceneFun3D was NOT CALIBRATED for the pose part)
  e3_tau_depth_shared = 0.20 primary calibration (the amendment keeps the depth part as calibrated by the primary rule)
  e3_tau_depth_view = 0.25   primary calibration
  python scripts/e2x/make_e3_eval_configs.py   (from MapAnything_Exp1/)
"""

import json
import os

C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs")
TH = {"e3_z": 5.0, "e3_tau_depth_shared": 0.20, "e3_tau_depth_view": 0.25}
for kind in ("synth", "real"):
    cfg = json.load(open(os.path.join(C, f"e2x_e3cal_{kind}_mapanything.json")))
    cfg.update(TH)
    cfg["_comment"] = (f"E3 eval ({kind}), experiments/E3.md: the E3-cal grid unchanged, E3 thresholds frozen: "
                       "e3_z from the secondary calibration (accurate ground truth), depth taus from the primary one.")
    json.dump(cfg, open(os.path.join(C, f"e3_eval_{kind}_mapanything.json"), "w"), indent=1)
    print(kind, len(cfg["conditions"]), "conditions", TH)
