# Gate track: can a gate inside the model stop bad hints? (DRAFT, not preregistered)

Status on 2 Nov 2026: code written, not yet run on real weights. This plan needs the team's approval and a logged amendment in `context.md`. The E2 decision also still reads "proposed" in `context.md`.

The idea comes from "Self-Healing Alternating Attention". Our E1 and E2 results and the model code changed the design:

- Harm stays on the bad camera; it does not spread to all views (`MapAnything_Exp1/reports/e2_audit.md`).
- MapAnything has no separate prior tokens: hints are added to the image features once, before the transformer (`MapAnything_Exp1/third_party/map-anything/mapanything/models/mapanything/model.py:1048-1052, 1168-1173`). So the gate acts where hints enter: `F_E = LN(F_I + sum gamma * F_hint)`.
- OmniVGGT: depth tokens blend with its learned placeholder; pose embeddings are scaled before the zero-initialised adapters.
- Hints are normalised together before the gate in OmniVGGT, and poses in MapAnything, so gamma = 0 on one bad view does not undo its effect on normalisation. That is why input-level drops are run next to the gates.

## Tests in this package

Dev clips only: 3 ETH3D scenes (8 clips) and 3 phone rooms (8 clips). The runner refuses other clips without `--allow_eval`. **These are pilots: six scenes cannot give a verdict.** A positive pilot justifies freezing a `g-prereg` tag; it is not a result for the paper.

| Test | Question | Config |
|---|---|---|
| G0 | How much AI depth error is overall scale, per-view scale or distortion? | no 3D model |
| G1a | Which of those errors actually harms each model? Conditions `ai_fix_*`, `gt_ai_*` use ground truth and are ORACLE | `configs/g1_oracle.json` |
| G1b | How much damage would a perfect gate remove? ORACLE gates and drops | `configs/g1_oracle.json` |
| G2 | Do tokens at global attention layers reveal bad hints, including agreeing errors, in unseen scenes? | `configs/g2_signal.json` |

Not in this package: the one-step Jacobian test (only worth doing if G2 passes) and a trained gate.

## Proposed pass conditions (not frozen; change them only before results are read)

G1b, per scenario, using the best in-network oracle gate:

- Damage left <= 0.25 (shape for depth scenarios, rotation for pose scenarios), where damage left = (x - clean) / (always - clean).
- The gate is better than never using the hint.
- The scenario has real harm: always is worse than clean.
- In `depth_mix_ai`, the known-source rule is the same as the oracle view drop, so a gate cannot win there. The scenarios that matter are `depth_lidar_view_bad` and `depth_lidar_patch_bad`, where knowing the source does not help. Both are synthetic.

G2, per task, on held-out scenes (the layer is picked on the other scenes):

- AUROC >= 0.80, and at least 0.05 above the best baseline (confidence and hint-vs-photos disagreement).
- `pose_drift` and `pose_shared` matter most, because existing filters are weakest there.

Kill rules:

- G1b fails on both bad-LiDAR scenarios: a depth gate cannot beat simple rules, so stop the depth gate.
- G2 token scores do not beat the baselines on any task: the signal is not there, so do not train a gate.

## Limits to keep in mind

- Six dev scenes, so per-scene values are saved and no confidence intervals are claimed.
- The G2 probe has few samples per fold and many features. Treat it as a screen.
- Token scores use a photos-only run as reference, so this design takes two forward passes, not one.
- Exploratory check already done on the E2 results, including eval summaries: photos-only scale is about x1.40 on the rooms, like AI depth at x1.42. Keep all thresholds on dev data and disclose this when the tag is frozen.

## Rules

Results are tracked in git. Do not edit frozen grids after eval results exist. A later eval run must come once from a `g-prereg` tag.
