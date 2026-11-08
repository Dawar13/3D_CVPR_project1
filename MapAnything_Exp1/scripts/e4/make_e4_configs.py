"""E4-A configs (experiments/E4.md): the frozen E3 eval configs (configs/e3_eval_{synth,real}_{model}.json), with the
same conditions, seeds, depth views and baseline settings. Controls: filter_rgbcheck (R2) and drop_lowconf (R3)
exactly where E3 ran them; vtt (E4-A) on every condition E3 had controls on; vtt_gate (MapAnything only, its fusion
gate) on those with pose hints.
E3's own controls (witness, witness_drop, drop_oracle) are not re-run: their rows come from the E3 eval runs.
VtT constants: exp1/vtt.py DEFAULTS, plus the calibrated ones from results/e4_vtt_cal/chosen.json.
  python scripts/e4/make_e4_configs.py   (from MapAnything_Exp1/)
"""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
C = os.path.join(ROOT, "configs")
CAL = json.load(open(os.path.join(ROOT, "results", "e4_vtt_cal", "chosen.json")))
VTT = {k: CAL[k] for k in ("vtt_kappa_rot", "vtt_sys_rot_deg", "vtt_kappa_dir", "vtt_sys_dir_deg", "vtt_pe_a", "vtt_pe_b",
                             "vtt_depth_sys")}

for model in ("mapanything", "pi3x"):
    for kind in ("synth", "real"):
        cfg = json.load(open(os.path.join(C, f"e3_eval_{kind}_{model}.json")))
        for k in [k for k in cfg if k.startswith("e3_")]:
            del cfg[k]
        n = 0
        for c in cfg["conditions"]:
            old = c.get("controls") or []
            if old:   # R2 / R3 exactly where E3 ran them; VtT wherever E3 had controls; the gate only with pose hints
                c["controls"] = ([x for x in ("filter_rgbcheck", "drop_lowconf") if x in old] + ["vtt"] +
                                 (["vtt_gate"] if model == "mapanything" and "pose" in c["inputs"] else []))
                n += 1
        cfg.update(VTT)
        cfg["_comment"] = (f"E4-A eval ({kind}, {model}), experiments/E4.md: the E3 eval grid unchanged; controls R2, R3, "
                           f"VtT{' and its MapAnything gate variant' if model == 'mapanything' else ''}; VtT constants "
                           "calibrated on dev / cal clips only (results/e4_vtt_cal/chosen.json).")
        out = os.path.join(C, f"e4_eval_{kind}_{model}.json")
        json.dump(cfg, open(out, "w"), indent=1)
        print(out, len(cfg["conditions"]), "condition entries,", n, "with controls", VTT)
