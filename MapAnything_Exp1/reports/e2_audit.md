# E2-B: post-hoc audit of E1

**Status:** done 2026-10-29. **Plan:** `experiments/E2.md` §2; method choices in Amendment 2.
**Gate result: PASS.** The model's copying of bad hints is real behaviour, not a code shortcut.

This is a **post-hoc audit**, kept separate from the frozen E1 decision (Strong GO), which it does not change.
- It re-reads E1's files: `results/e1_eval`.
- The E1 references and the single-view and fraction pose families were re-run on the new VM with E1's exact corruptions: `configs/e2b_audit.json`, `results/e2b_audit_eval`, 420 runs, 0 failures, 1 h 02 min, peak 8.21 GB ([gpu_log](../results/e2b_audit_eval/gpu_log.txt)). E1 saved no predicted poses, so checks 3 and 4 needed this re-run.
- Script: `exp1/e2_audit.py`. All tables: [`reports/e2_audit/`](e2_audit/).

---

## Check 1. Is "copying" real, or does the code hand the input poses back? → **real**

**Source code (MapAnything `3d10cf7a`).** The output pose comes only from the network:
- The pose head runs on the last transformer features (`mapanything/models/mapanything/model.py:1504-1513`).
- Its output is split into translation and quaternion (`model.py:1762-1764`) and scaled by the predicted metric scale (`model.py:1800-1802`).
- Post-processing builds the 4×4 `camera_poses` from those network outputs (`mapanything/utils/inference.py:363-378`). The input views are used there only for the image (`inference.py:341`).
- Our harness reads exactly that key (`exp1/model_io.py:127`).

No path copies the input pose into the output.

**E1 data** ([copy_check.csv](e2_audit/copy_check.csv)). If the code copied inputs, absorption would be exactly 1 in many runs.
- Rotation absorption: 1 of 4,550 values rounds to 1.0000 at 4 decimals, and 0 are exactly 1.
- Position absorption: 1 of 4,157 rounds to 1.0000, and 0 are exactly 1.
- The values spread continuously, from 0.27 to 2.16 for rotation.

**GPU check** (clean hints; 10 eval clips × 9 views). The output pose differs from the input pose on every view:
- rotation by 0.045–0.73° (median 0.18°);
- camera centre by 0.004–0.17 m (median 0.019 m).
- 0 of 90 views are identical to their input.

---

## Check 2. Which E1 harms survive the stricter E2 standard?

**Rule.** "Harmful" now needs both of these:
- **Statistically clear:** Holm-adjusted p < 0.05, two-sided paired t-test over the 10 scenes. The Holm family is all 132 (setting × metric) tests.
- **Large enough** (E2 §0.2): shape +1 pp, rotation +0.25°, real size +5 pp, scale ratio +5 percentage points off.

The comparison is always against the same scenes without that hint (E1's floor). Source: [practical.csv](e2_audit/practical.csv), [practical_family.csv](e2_audit/practical_family.csv).

**Result.**
- 16 of 132 tests are harmful under the strict rule. Another 17 pass the uncorrected CI and the size threshold but not Holm. E1's S1 rule flagged 40.
- By family: 9 of E1's 19 families are harmful at some setting, 7 at a literature-realistic setting (labels: `reports/e2_literature.md` §3).

**Harmful under the strict rule:**

| Wrong hint | Setting | Literature label | Metric | Without hint | With wrong hint | Δ (95% CI) | Scenes worse | Holm p |
|---|---|---|---|---|---|---|---|---|
| One camera turned | 15° | realistic | rotation | 1.41° | 3.03° | +1.62° (1.19, 2.01) | 10/10 | 0.006 |
| | 30° | edge | rotation | 1.41° | 5.94° | +4.53° | 10/10 | < 0.001 |
| | 90° | beyond | rotation | 1.41° | 17.9° | +16.5° | 10/10 | < 0.001 |
| Every camera turned (random axis) | 5° | realistic (compass) | rotation | 1.41° | 5.62° | +4.20° | 10/10 | < 0.001 |
| Same heading error on every camera | 10° | realistic | rotation | 1.41° | 6.17° | +4.76° (3.22, 6.32) | 10/10 | 0.038 |
| First (reference) camera turned | 15° | realistic | rotation | 1.41° | 3.16° | +1.75° | 10/10 | 0.003 |
| All distances scaled | ×2 | realistic (monocular SLAM) | real size | 25.2% | 78.9% | +53.7 pp | 10/10 | 0.005 |
| | ×2 | | scale ratio | 27.2% off | 75.1% off | +47.9 pp | 9/10 | 0.031 |
| Depth hints scaled (with poses) | ×1.5 | realistic (monocular depth) | real size | 5.8% | 16.1% | +10.3 pp | 10/10 | 0.049 |
| | ×1.5 | | scale ratio | 4.7% off | 15.4% off | +10.7 pp | 10/10 | 0.031 |
| One depth map scaled | ×1.5 | realistic (monocular depth) | shape | 3.2% | 5.8% | +2.5 pp | 10/10 | 0.031 |
| Several cameras wrong (15° + 0.25 L each) | 10% / 20% / 40% | unsourced fraction | rotation | 1.41° | 3.05° / 5.59° / 10.3° | +1.64° / +4.17° / +8.84° | 10/10 | ≤ 0.004 |
| Odometry drift | level 2 / 3 | **beyond** | rotation | 1.41° | 3.41° / 6.88° | +2.00° / +5.46° | 10/10 | ≤ 0.001 |

**Clear without correction but not after Holm** (likely real; not established with 10 scenes and 132 tests):

| Wrong hint | Setting | Metric | Δ (95% CI) | Scenes worse | raw p | Holm p |
|---|---|---|---|---|---|---|
| One camera moved far (GPS jump) | 10 L | shape | +24.7 pp (15.5, 37.0) | 10/10 | 0.0025 | 0.24 |
| Same heading error on every camera | 5° | rotation | +1.53° (0.72, 2.37) | 9/10 | 0.0077 | 0.69 |
| Depth hints scaled (with poses) | ×1.5 | shape | +4.5 pp (2.5, 6.5) | 10/10 | 0.0023 | 0.22 |
| All distances scaled | ×1.5 | real size | +18.8 pp (4.3, 31.2) | 8/10 | 0.030 | 1.0 |
| First camera turned | 15° | shape | +11.6 pp (6.9, 14.9) | 9/10 | 0.0006 | 0.063 |
| Every camera turned | 2° | rotation | +0.73° (0.30, 1.14) | 8/10 | 0.011 | 0.96 |
| Wrong focal, all photos, no poses | ×0.8 / ×1.2 (edge) | rotation | +0.81° / +1.42° | 8/10 | 0.055 / 0.013 | 1.0 |

Full list (17 rows): [practical.csv](e2_audit/practical.csv), rows with `harmful_ci_only` true and `harmful` false.

**Not harmful even without correction.**
- Wrong focal length with poses (×1.1: rotation +0.05°, below the 0.25° bar).
- Principal-point shift, noisy depth, a forgotten metric flag, and camera positions moved by up to 3 L.

**Statistical power.**
- With 10 scenes, an exact sign-flip test cannot go below p = 0.00195. The smallest Holm threshold over 132 tests is 0.00038, so only a t-test can pass, and only for large or very consistent effects.
- Consequence for E2-C: its preregistration names 12 primary tests, not hundreds.

---

## Check 3. Is the harm the bad camera itself, or does it spread? → **the bad camera itself**

Rotation error on camera pairs that include a bad camera vs pairs of two good cameras, against the same pairs with correct hints. Source: [pairs.csv](e2_audit/pairs.csv); 10 scenes, 3 seeds.

| Wrong hint | Bad-camera pairs: correct hints → wrong hint | Good-camera pairs: correct hints → wrong hint (Δ, 95% CI) | Good pairs with photos only |
|---|---|---|---|
| One camera 5° | 0.24° → 4.35° | 0.249° → 0.256° (+0.008°; 0.003, 0.014) | 1.43° |
| One camera 15° | 0.24° → 14.1° | 0.248° → 0.261° (+0.013°; 0.003, 0.024) | 1.42° |
| One camera 30° | 0.24° → 28.7° | 0.247° → 0.251° (+0.005°; −0.002, 0.013) | 1.42° |
| One camera 90° | 0.25° → 88.7° | 0.244° → 0.243° (−0.002°) | 1.40° |
| 10% / 20% / 40% of cameras wrong | 0.23–0.25° → 14.2–15.2° | +0.005° / +0.030° / +0.036° | 1.44–1.47° |
| One camera moved 10 L | 0.26° → 0.82° | +0.032° (−0.019, 0.089) | 1.41° |

- The bad camera's own pairs take on almost the whole injected rotation.
- Pairs of good cameras change by at most 0.036°, far below the 0.25° bar. In 0 of 10 scenes do they become worse than photos only.
- E1's "spread" signal (S2, +20.4% with 20% wrong) measured the good views' rotation against view 0, a different quantity. The absolute size of the spread is small either way.

---

## Check 4. Do the bad view's 3D points move as much as its camera? → **yes for rotations and large shifts; partly for small shifts**

Source: [consistency.csv](e2_audit/consistency.csv), [consistency_views.csv](e2_audit/consistency_views.csv), [consistency_stats.json](e2_audit/consistency_stats.json). Medians over 30 bad views (10 scenes × 3 seeds).

| Wrong hint | Camera absorption (E1) | Camera moved along the shift, vs correct-hint run | Points moved along the shift | Depth-compensation slope |
|---|---|---|---|---|
| One camera moved 0.1 L | 0.88 | 0.87 | 0.54 | 0.48 |
| 0.25 L | 0.91 | 0.90 | 0.67 | 0.50 |
| 0.5 L | 0.99 | 0.98 | 0.70 | 0.46 |
| 1 L | 1.00 | 1.00 | 0.95 | 0.26 |
| 3 L | 0.97 | 0.97 | 0.95 | 0.03 |
| 10 L | 0.97 | 0.97 | 0.97 | −0.01 |

**Rotations.**
- The model rotates the bad camera in the injected direction: the axis cosine between the injected and the output rotation is ≥ 0.9994 at every size.
- The share it takes on is 0.89 at 5°, 0.94 at 15°, 0.96 at 30° and 0.99 at 90° (`rot_along`).

**Positions.**
- The camera takes on 0.87–1.00 of the shift. Its points follow fully for large shifts but only about 0.5–0.7 for small ones.
- The reason is depth re-fitting. The depth-compensation slope compares the bad view's depth change with the change that would keep its points in place: 1 = full compensation, 0 = points move rigidly with the camera.
- For small shifts (≤ 0.5 L) the slope is about 0.5: the model re-fits that view's depth so that about half of an axial shift is absorbed and the points stay closer to where the other photos put them.
- From 3 L on it does not (slope ≈ 0).
- Across views, the lag between camera and points grows with the share of the shift along the viewing direction (Spearman 0.33).
- So for small position errors the model partly uses the images; for large ones it trusts the hint completely. Masking or the metric's median cannot explain the gap: the camera-shift metric and E1's absorption agree within 0.02.

---

## Check 5. Which E1 harms happen at literature-realistic sizes?

Labels from `reports/e2_literature.md` §3 (`configs/e2_lit_realistic.json`). Strict harm at a **realistic** size:
- one camera turned 15° (an indoor compass error);
- every camera turned 5° (compass-derived rotations);
- the same 10° heading error on every camera;
- the first camera turned 15°;
- distances ×2 (monocular SLAM);
- depth hints ×1.5 with poses, and one depth map ×1.5 (monocular-network depth).

Not established at realistic sizes:
- **Drift.** E1's levels are beyond every sourced drift rate. E2-C C6 adds a level near gyro-only drift.
- **Several bad cameras.** The fraction of views affected has no source.
- **GPS jumps (1–10 L).** Not strict-harmful. At 10 L, shape is worse in 10/10 scenes, but that does not survive Holm.
- **Lens errors.** Below the practical bar with poses.

**Pattern.** The realistic harms are heading-type rotation errors at compass level, scale errors at monocular-SLAM level, and depth-scale errors at monocular-network level. Working phone VIO (< 2° orientation error) and active depth sensors (2–3.5% scale bias) are milder than any harmful setting found.

---

## New VM vs E1 (environment parity)

The same 420 runs on driver 550 / CUDA 12.6 vs E1's driver 580 / CUDA 13.0, both torch 2.14.1. Source: [parity.csv](e2_audit/parity.csv).

| Metric | Median relative difference | Largest absolute difference |
|---|---|---|
| shape | 0.05% | 0.0006 (0.06 pp) |
| rotation | 0.08% | 0.0049° |
| real size | 0.19% | 0.0024 |

- 1 of 420 values is bit-identical (scale ratio).
- The largest differences are 16× (shape), 21× (real size) and 51× (rotation) below the practical thresholds, so E1 and E2 numbers are comparable. Each E2 part still reruns its own references.

---

## What this means for the rest of E2
- **Gate passed.** Copying is behaviour, so "the model trusts bad hints" stands.
- **Harm comes from the bad camera itself.**
  - It is mostly a placement error, with almost no spread to good cameras.
  - For small position errors, the bad view's depth is partly re-fitted.
  - E2-C C3 measures whether the bad view's own shape bends.
- **Smaller effects need more power.** Only large effects survive strict correction with 10 scenes. E2-C keeps 12 primary tests and uses 3 clips per scene to reduce noise within each scene.
- **The realistic threats** are rotation/heading errors at compass level, global scale at monocular-SLAM level, and depth scale at monocular-network level. E2-D (OmniVGGT) and E2-E (real sensors) should centre on these.
