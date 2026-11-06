# Research vision and where the repo stands

Written 7 Nov 2026. It merges the research vision (the handover note "3D Vision Research — Consolidated Context", 31 Oct 2026, kept outside the repo) with the state of this repo after E0–E2 and the E2X preparation. Status lines point to the files they come from. Nothing here changes a preregistration, a config or a decision rule.

## 1. The vision (Project 1, "Self-Healing Geometric Priors")

**Research question.** Can a pretrained 3D foundation model notice that an external geometric hint (camera pose, lens settings, depth) is unreliable, and reduce that hint's influence using the images and cross-view consistency?

The contribution is **not** "add an attention gate". A paper has to show, in this order:

1. clean hints help;
2. corrupted hints cause reproducible harm;
3. simple filtering does not already solve the problem;
4. a reliability-aware method beats strong simple baselines;
5. the method keeps clean-hint performance and generalises.

Working rules from the vision, all adopted in this repo (`context.md` §6):
- Prove the failure before building architecture. Do not build a large model before the central hypothesis survives the diagnostic experiment.
- Fair baselines: the deployable filter must use no ground truth. Test subtle errors, not only easy 15° outliers.
- Separate measured results, hypotheses, thresholds and projections. Never describe a script as a finished experiment.
- Held-out data for final evaluation. Record commands, commits, configs, seeds and outputs.

There is a second, separate project (RS-3D, rolling-shutter-aware 3D) in its own repository. Its code, results and claims are kept out of this repo.

## 2. The five requirements against the evidence

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Clean hints help | **Shown** | E0, E1, E2-E (real iPad poses and LiDAR help both models) |
| 2 | Wrong hints harm, reproducibly | **Shown** | E1: 2,350 runs on 10 held-out ETH3D scenes, Strong GO (`MapAnything_Exp1/reports/e1_results.md`). E2: 13,674 runs, two models, literature-sized errors, real device hints (`MapAnything_Exp1/reports/e2_results/e2_results.pdf`) |
| 3 | Simple filtering does not solve it | **Partly shown** | E2-F: E1's filter fixes single gross outliers but misses errors shared by all cameras and hurts real phone data; "drop all poses" removes most damage but also all benefit (`results/e2def_agg/f_remaining_gap.csv`, `f_real_hints.csv`). The gap is in *keeping good hints*, not in removing damage |
| 4 | A reliability method beats the baselines | **Open** | E3 witness prototyped and tested with fake models only (`exp1/e3.py`, `exp1/witness.py`, `tests/test_e3.py`). Calibration runs in E2X session A |
| 5 | Keeps clean performance, generalises | **Open** | E3 calibration on correct hints (false-action rule, `experiments/E2X.md` §5); third model Pi3X (X-A); whole-room real hints (X-B). All in session A |

Caveats from the vision, and how they were handled:
- *"15° is an easy outlier."* E2 used error sizes from published sources, shared (agreeing) errors and drift (`configs/e2_lit_realistic.json`, `reports/e2_literature.md`).
- *"Verify the filter uses no oracle labels."* `filter_rgbcheck` compares hints with the images-only prediction; ORACLE controls are labelled as such (`context.md` §6 rule 3).
- *"Audit GeoCond and related work."* Done in E2-A (`reports/e2_literature.md` §1). GeoCond (arXiv 2609.18465) is about unreliable images and predictions, not unreliable input hints. It is adjacent work to cite, not prior art for our findings.
- *"Scale fitted over all views before subset scoring."* **Still open for shape errors.** Per-view shape errors (`view_pts_err_scale`) use the one clip-wide scale `s` fitted over all views, the corrupted view included (`MapAnything_Exp1/exp1/metrics.py:195-214`). Rotation errors, the main harm readout, do not depend on this scale, and a per-view Sim3-aligned shape error (`view_shape_err_sim3`) is already saved. To do before the paper: a sensitivity check that refits `s` on the uncorrupted views only, or reports `view_shape_err_sim3` next to the spread results (E1 S2).

## 3. Where the repo moved away from the vision, and why

| Vision (31 Oct) | Repo now (4 Nov) | Why |
|---|---|---|
| Learned reliability gate **inside** feature fusion | Main method E3 is **training-free and outside** the network: photos-only witness, an error model (shared tilt, camera-mount rotation, drift, single bad camera, depth scale), repair or drop only when the photos clearly disagree | E2 showed the damage is inside the network (median-scale probe changed nothing) and that real errors are mostly *agreeing*, where per-view gates and model confidence are weakest. A training-free check is cheaper to make safe on correct hints |
| In-network gate as the method | Kept as the **G track** (`experiments/G.md`): oracle gates (G1) and a token-signal probe (G2), dev clips only, not preregistered | It decides whether a learned gate is worth building. Kill rules are written in advance |
| Framing: "self-healing priors" | Framing: **"They Know, but They Obey"** (model confidence flags the bad hint, AUROC 0.93, yet it copies 77–96% of the error) plus break-even curves per real hint source | That is what the evidence supports; a method claim waits for E3 |
| Target: CVPR 2027 main track (submission ~16 Nov 2026) | Target: **CVPR 2027 workshop** (team preference, 2 Nov 2026) | The method is not validated in time for the main deadline |
| MapAnything, OmniVGGT optional | MapAnything, OmniVGGT (E2-D), Pi3X prepared (E2X) | Reviewers will ask whether "obeying" is common |

## 4. Choosing the primary paper track

On 31 Oct the vision judged the rolling-shutter project to have the stronger evidence. Project 1 has since run E1 and E2 in full. Both projects still lack a *deployable* method that has been shown to work: here E3 (E2X session A), there EXP2 (correction from images alone).

Proposed rule: choose the primary track after both decisive runs are back, using the stop rules already fixed in `experiments/E2X.md` §5 for this project. Do not build both learned architectures in parallel.

## 5. What happens next

The full step-by-step list is in [`NEXT_STEPS.md`](../NEXT_STEPS.md). In short:

1. Freeze E2X (`e2x-prereg` tag) and push the tags.
2. Run E2X session A (`experiments/E2X_instructions.md`): third model, whole-room real hints, E3 calibration, gate tests G0–G2.
3. Decide with the fixed rules: method paper / analysis paper / in-network gate.
4. Session B: frozen E3 eval (`e3-prereg`), then the workshop paper.
