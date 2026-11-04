# E2X — Extended E2 (third model, larger real errors) and E3 calibration

Started 2026-11-03 after the user's go-ahead ("do all the laptop work … we just have to make this a robust paper"). Target: CVPR 2027 workshop (user's preference, 2026-11-02). Rules of `experiments/E2.md` §0 apply unchanged: the scene is the unit, paired comparisons, Holm within a part, the E2 practical thresholds, dev and `cal` data never reported as results, and every eval grid frozen (git tag) before its eval run.

## 1. Why

E2 left three gaps a reviewer would hit first:
1. **One model obeys.** MapAnything copies wrong hints; OmniVGGT copies about 60% of rotation errors, and no harm was proven for it. A third hint-taking model is needed to say whether "obeying" is common.
2. **Real hints were too good.** The ARKit hints in the E2-E clips (about 2 m of walking) had a median error of 0.87° and helped both models. AI depth was the only real hint that hurt.
3. **No method yet.** E3 needs a training-free fix that repairs bad hints and leaves good ones alone. Its thresholds must be set on data that is neither dev nor eval.

## 2. What the laptop work found (2026-11-03; no model run)

### 2.1 Real-data candidates
| Source | Device hint | Ground truth | Finding | Decision |
|---|---|---|---|---|
| ADVIO (Zenodo 1476931, seq. 09, CC BY-NC) | iPhone ARKit, Pixel ARCore, Tango | Inertial navigation **from the same iPhone IMU** plus hand-placed fix points (σ = 25 cm; paper §4.1) | ARKit vs GT, 1 s relative rotation: median 0.26°, max 1.0° (`results/e2x_survey/advio_probe.log`). The GT shares the phone's gyroscope, so it cannot judge rotations at the 0.25° level, and its positions are too rough over short clips (`advio_spans.log`). | **Not used** for scoring. |
| Zurich Urban MAV (AGZ, no restriction) | Onboard GPS + Pixhawk attitude | Pix4D photogrammetry, one pose per second | GPS − GT: median 4.69 m, p90 9.71 m, max 42.67 m, while the drone moves 0.71 m per GT frame (median). The autopilot attitude is **not** the camera's: when the camera turns 10–30° in 3 s, the attitude log turns about 2.7° (`zurich_probe.log`), so the camera turns on its own. | Only "GPS position + accurate orientation" hints are possible (semi-real). **Optional**, after the core. The server allows range reads of the 29.8 GB zip. |
| SceneFun3D test split (ARKitScenes Validation), whole recordings | ARKit (VIO) | COLMAP + laser scan | See 2.2. | **Used** (room clips, `cal` visits). |

### 2.2 SceneFun3D: ground truth has failures, ARKit drifts
- **Ground-truth failures** (all 121 landscape test videos with matched frames, `results/e2x_survey/sf_gt_spikes.log`). Frames that jump away from their neighbours (frame vs the interpolation of frames i−1 and i+1):
  - GT: 493 of 73,512 frames > 5°, 80 > 30°, 136 position spikes > 0.3 m;
  - ARKit: 13 frames > 5°, **none** > 30°, 2 position spikes.
  - Some GT segments are turned ~180° for many frames (`sf_alignment.log`: 16% of videos have a frame > 90° from the device after one video-wide alignment).
  - So GT, not ARKit, causes the near-180° "device errors". The E2-E clip `eval__sf422829__c1` (view 1, found in E2) is one such case.
- **Photo check of poses** (`exp1/photo_check.py`, no model): RANSAC-cleaned SIFT matches between a clip's own images, scored by their Sampson distance to the epipolar geometry of the poses under test.

  | Poses tested | Median over views | p99 | Max | File |
  |---|---|---|---|---|
  | ETH3D GT (13 clips) | 0.11 px | 0.31 px | 0.43 px | `results/e2x_photo_check/eth3d_clips.csv` |
  | SceneFun3D GT (E2-E clips) | 0.28 px | 0.56 px | 1.0 px | `e2e_clips.csv` |
  | ARKit poses (same clips) | 0.85 px | 10.7 px | 16.1 px | `e2e_clips.csv` |

  - Limit: views with fewer than 15 clean matches to any partner are not scored (341 of 490). The known bad view of `eval__sf422829__c1` sits in low-texture frames (white cabinets) and is not scored. Hence the two trajectory filters in 3.2.
- **Device error vs clip span** (ARKit vs GT relative to view 0, GT spike frames removed; 150 landscape test videos; `results/e2x_survey/sf_device_survey_clean.csv`, `sf_device_survey.log`):

  | 10 views spread over | Clips | GT extent (median) | Worst view's rotation error: median | p90 | Clips > 5° | Length ratio p10–p90 |
  |---|---|---|---|---|---|---|
  | consecutive keyframes (E2-E) | 1457 | 0.80 m | 1.23° | 2.71° | 4.6% | 0.952–1.031 |
  | 25% of the recording | 1008 | 1.41 m | 1.63° | 4.19° | 8.9% | 0.958–1.026 |
  | 50% | 414 | 2.24 m | 2.05° | 5.36° | 10.6% | 0.965–1.022 |
  | the whole recording | 150 | 3.36 m | 2.88° | 13.46° | 20.7% | 0.953–1.021 |

  - The p99 of every row is ~179°: persistent GT flips, which the room-clip filters remove (3.2).
  - Over whole recordings, ARKit's rotation error is several times the photos-only uncertainty measured in E2-C8 (~0.8° typical). This is the regime where real pose hints may stop helping.
- **Bug found and fixed:** a timed-out download left an empty cached trajectory, which made that video look shortest and changed the "longest video" choice (`exp1/arkit_clips.py` `cached`: fetch before writing). The survey above is after the fix. With a complete cache the first 18 selected videos equal E2-E's exactly (`data/clips_e2x_room/selection.csv` vs `data/clips_e2e/selection.csv`), so E2-E was not affected.

### 2.3 Third model: Pi3X
- Research by sub-agent, source read at `yyfz/Pi3@9fa3ddb`; citations in the header of `exp1/model_io_pi3x.py`. Takes pose, intrinsics and depth hints, each with its own per-view mask. Missing hint = feature × 0, as in MapAnything.
- Hint scale is normalised: depth by one mean over all views, translations by the same factor or, without depth, by their mean length. So the metric scale of hints cannot reach it, as for OmniVGGT, but the pose-to-depth scale ratio can.
- SpatialBench (arXiv 2605.27367) reports that Pi3X "partially overrides" injected poses, while MapAnything adheres. This is a direct contrast to test.
- Licences: code BSD-3, weights CC BY-NC 4.0. Training data per the π3 paper includes ARKitScenes (split not stated) and ScanNet++, with ETH3D for evaluation only. **Caveat for X-B:** whether ARKitScenes Validation (our SceneFun3D videos) was in Pi3X's training is unverified; the Pi3X real-hint results carry this caveat.
- Other candidates (same research): WorldMirror cannot take partial hints, rescales each depth map to [0, 1] and has a licence excluding the EU/UK/South Korea. DA3 takes poses only, all-or-nothing. Not used.
- Connector `exp1/model_io_pi3x.py`, runner `--backend pi3x`, tests `tests/test_pi3x.py` (fake model; the E2-D grid builds completely). Real-weight check on the VM (dev gate).

### 2.4 E3 prototype on dev data
- `exp1/witness.py` fits the error model (shared rotation in the frame of view 0, camera-mount rotation, drift, per-view outliers) between the hints and the photos-only prediction.
- On E2-C/E2-F **dev** extras (3 ETH3D clips; `results/e2x_survey/e3_witness_dev.csv`; first prototype, before the mount term):
  - clean hints: shared 0.6°, drift 0.9°;
  - a 15° outlier: residual 15.0° on the bad view vs ≤ 0.9° on good views;
  - first camera 15°: shared 14.75°;
  - shared heading 5°: shared 4.74°;
  - drift levels: 4.2 / 7.7 / 15.1° total.
- On SceneFun3D dev clips the photos-only witness itself is off by up to 15° on single views, and by 2–4.5° shared, even against GT hints (MapAnything confidence ≈ 1.0 there, its minimum).
- So a fixed threshold would "repair" good hints. E3 therefore measures the witness's own uncertainty per clip.

### 2.5 Code changes for E2X (all tested on CPU with fake models; `tests/test_e2x.py`, `test_e3.py`, `test_pi3x.py`)
- `exp1/e3.py` and the runner control `witness` / `witness_drop` (`run_exp1.run_e3`). The control:
  - logs every decision to `e3_decisions.jsonl` (merged across processes by `exp1/shard.py`);
  - converts the model's depth into the pose hints' units (Pi3X and OmniVGGT normalise hint scale away);
  - skips the depth check when neither pose hints nor a metric output exist;
  - for OmniVGGT, drops intrinsics together with poses.
- R2 (`filter_rgbcheck`) and R3 (`drop_lowconf`) now run on any condition with pose hints, clean ones included, so their false alarms on good hints are measured. Before, they required a pose corruption. `drop_oracle` still requires one.
- Pose **AUC@30** (VGGT/MapAnything convention: per pair the larger of rotation and translation-direction error, accuracy at 1–30°, averaged) is written as an extra per-run column (`exp1.metrics.auc`).
  - For the finished E2 eval runs it is computed from the saved predicted poses (`scripts/e2x/auc_from_extras.py` → `results/<run>/auc30.csv`).
  - The recomputed RRA@5 matches `per_run.csv` to 1e-16 on all 11,938 runs.
- `scripts/e2x/check_clips.py` applies the exclusion rule of 3.2.
- `exp1/e3_calibrate.py` applies the calibration rule of §5 to the logged decisions (tested: it recomputes every logged decision exactly). `scripts/e2x/analyse_e2x.py` runs X-A/X-B with the frozen E2 code plus the descriptive angles.
- Step-by-step instructions for the rest: `experiments/E2X_instructions.md`.

## 3. Data

### 3.1 ETH3D (unchanged)
The E2 clips: 24 eval clips (10 scenes) and 3 dev c0 clips.

### 3.2 SceneFun3D room clips and `cal` visits (`exp1/arkit_clips.py --mode room --n_cal 10`)
- **Same 18 videos as E2-E** (3 dev, 15 eval). Per video, 10 keyframes evenly spread over:
  - the whole usable recording (`r0`);
  - its first half (`r1`);
  - its second half (`r2`).
- **GT frame filters, before keyframing** (room mode only):
  - drop frames that jump away from their neighbours by > 2° or > 0.10 m;
  - drop frames > 30° from the device after one robust video-wide alignment;
  - ARKit never jumps > 30° (2.2), so the second filter only removes GT failures. Real device errors > 30° cannot be studied here.
- **`cal` split:** the next 10 visits of the same seeded order. They get room clips and consecutive clips (`--mode consec --n_cal 10 --only <cal videos>`). For E3 calibration only, never evaluated.
- **Exclusion rule, fixed now, applied before any model run on these clips:**
  - a clip is excluded if it fails the E0 GT check (median_rel ≥ 3% on a pair with > 5% overlap);
  - or if a GT view fails the photo check: its score is > 2.0 px (twice the worst SceneFun3D GT view in the E2-E clips and 4.6× the worst ETH3D GT view) **and** more than 2× the score of the device pose of the same view, scored on the same matches.
  - Amended 2026-11-03, before any model run on these clips: the first version, without the device comparison, also excluded clips where GT and ARKit poses agree with each other within a few degrees but both fit the matches badly. That pattern means unreliable matches on wide-baseline room pairs, not a wrong GT (`results/e2x_clip_check/`).
  - A first run also used looser match settings (15 matches, 4000 SIFT) than the ones the 2 px threshold was set with. It was rerun with the original settings (30 matches, 2000 SIFT), which reproduce the saved E2-E scores exactly.
  - The list is written to `data/clips_e2x_room/excluded.csv` and copied to `results/e2x_clip_check/`.
  - **Result (2026-11-03, before any model run):** 21 of 114 clips excluded.
    - 20 failed the E0 check, several on pairs with 60–100% overlap, so these are wrong GT frames, not occlusion.
    - 1 failed the photo check: `cal__sf422826__c1`, where views 7–9 are flipped 178° in the GT and view 6 scores 385.65 px while the device pose scores 0.58 px.
    - Kept: room clips 36 eval (all 15 scenes), 7 dev (3 scenes), 24 cal; consecutive `cal` clips 26. All 10 `cal` visits keep clips.
    - Device error on the kept dev/cal room clips (dev/cal only; eval unread): worst-view rotation 1.2–8.3°.
- **AI depth hint:** as in E2-E (Depth Anything V2 metric-indoor Large, `exp1/mono_depth.py`), computed on the VM before any run.

## 4. Parts and grids (configs generated by `scripts/e2x/make_configs.py` from the frozen E2 configs)

| Part | Model | Clips | Config | Per clip |
|---|---|---|---|---|
| X-A checks | Pi3X | ETH3D dev c0 | `e2x_pi3x_checks.json` (= `e2d_checks.json`) | E0-style checks + invariances |
| X-A | Pi3X | ETH3D dev → eval | `e2x_pi3x_eth3d.json` (= `e2d_full.json`) | 138 |
| X-B | Pi3X | E2-E clips dev → eval | `e2x_real_pi3x.json` | 11 + baselines |
| X-B | MapAnything, OmniVGGT, Pi3X | room clips dev → eval | `e2x_real_<model>.json` | E2-E conditions + R2/R3 |
| E3-cal synthetic | MapAnything, Pi3X (OmniVGGT opt-in) | ETH3D dev c0 + `cal` clips | `e2x_e3cal_synth_<model>.json` | 27 conditions with R2, R3, `witness`, `witness_drop`, ORACLE |
| E3-cal real | same | SceneFun3D dev + `cal` clips | `e2x_e3cal_real_<model>.json` | E2-E conditions with the same controls |

- Same condition names as E2-D, so every model gets identical corrupted inputs.

## 5. Preregistration draft (to be frozen as tag `e2x-prereg` after the user's approval, before any eval run)

### X-A: Pi3X on ETH3D (third model)
- The E2-D tests verbatim (`experiments/E2.md`, E2-D section):
  - **D-H:** harm per family with `exp1/e2_audit.practical`, shape and rotation. No real-size readouts: Pi3X normalises hint scale.
  - **D-B:** behaviour P1–P5, P10, P11, same definitions and thresholds.
  - Holm within each, per model.
- **Side-by-side classification over three models:** same / MapAnything only / Pi3X only / … / opposite, as in E2-D.
- **"Obeying is common"** is supported if P1 (copying) is a finding for Pi3X. The SpatialBench contrast is reported either way.
- **Dev gate:**
  - 0 failures;
  - `repeat_clean` within 1/10 of the practical thresholds;
  - `clean_pose` rotation error below `rgb_only` on ≥ 2 of 3 dev clips.

### X-B: real hints over whole rooms (three models) and Pi3X on the E2-E clips
- E2-E's E-H tests verbatim: each real-hint condition vs `rgb_only`, scene values, shape and rotation (plus real size for MapAnything metric-hint conditions). Holm over all X-B tests of one clip set, per model.
- **Primary question:** is `dev_pose` (whole-room ARKit poses) harmful, neutral or helpful vs `rgb_only`, per model, and how does it compare with the E2-E short clips (descriptive, paired by visit)?
- **Descriptive:**
  - real error sizes per room clip (rotation, translation, scale vs GT);
  - where they fall on the C1 trust curves;
  - R2/R3 on the real hints, Δ vs no remedy.
- **Dev gate:** 0 failures; `gt_pose` rotation error below `rgb_only` in ≥ 2 of 3 dev visits, per model.

### E3 calibration (no eval data; nothing here is a result)
- **What is chosen:** `e3_z` from {2, 2.5, 3, 4, 5}, `e3_tau_depth_shared` from {0.05, 0.10, 0.15, 0.20} and `e3_tau_depth_view` from {0.10, 0.15, 0.25}. The floors stay at their defaults (0.5°, 1.0°, 0.5°) and `e3_n_orders` = 3.
- **Rule (Neyman–Pearson style), per model:**
  - take the smallest `e3_z` whose false-action rate on **ground-truth hints** in the `cal` and dev clips is ≤ 5% of clips for any group action (shared / mount / drift repair) and ≤ 2% of views for single-view drops;
  - the same for the depth taus on GT depth (`gt_all`, `clean_pose_depth2`);
  - ties go to the larger threshold.
  - False actions are read from `e3_decisions.jsonl`, which logs every fitted value and witness uncertainty, so any threshold can be evaluated without rerunning.
- **Reported, not used to choose:**
  - detection rates on the synthetic families;
  - accuracy after repair vs R2/R3/ORACLE at the default setting.
- The E3 eval grid and its tests are preregistered separately (`e3-prereg`) after calibration and before any E3 eval run.

### Angles that make the paper stronger (added 2026-11-04, after the user chose "strengthen, don't pivot")
- These are **descriptive** analyses, computed by `scripts/e2x/analyse_e2x.py` and labelled DESCRIPTIVE in every output. They add no preregistered test and change no grid.
- **A. Break-even ("fused as if perfect").**
  - Principle: weighing each source by its reliability (Gauss's least squares; inverse-variance weighting; Kalman 1960) is never worse than the better source alone. A model that gives hints near-full weight instead gets worse than images only once the hint's error passes the photos' own error.
  - Measured: per model and C1 family, the error size at which a hint stops helping. MapAnything, ETH3D eval, already finished, post hoc (`results/e2x_angle_preview/`):
    - one wrong camera still helps at 5° (Δ −1.06°), breaks even near 10° (−0.08°) and hurts clearly at 30° (+3.88°, CI 2.06–4.96);
    - depth scale hurts at ×0.5/×0.63 and ×1.587/×2 (CI above 0), not at ×0.794–×1.26;
    - OmniVGGT is never worse than its images-only mode.
  - Pi3X comes from X-A.
  - Weaker photos, already run: the same 15° case on 4- and 20-photo clips (`results/e2c_n4_eval`, `e2c_n20_eval`).
- **B. Reality map.** Every real hint source placed on these curves by its measured or published error size (§2.1–2.2, `reports/e2_literature.md`). Phone VIO and LiDAR fall below break-even, which is why they helped. AI depth (×1.42) falls above, which is why it hurt. Compass headings (indoors 12.45° average) straddle it. Whole-room VIO is measured in X-B. This answers "is it rare?" source by source.
- **C. Knows but obeys, blind to agreement.** E2-C's AUROC 0.93 vs absorption 0.96, with P10/P11, for three models (X-A). Real errors are mostly *agreeing*: one shared AI-depth scale, drift-like ARKit errors (E2-E Spearman 0.52 with view order), compass bias. That is exactly where model confidence and per-view filters are blind, and what E3's shared/mount/drift model targets.
- **D. Safety vs benefit.** For every remedy, two numbers on the same clips:
  - the share of the good-hint benefit kept, (E_rgb − E_remedy) / (E_rgb − E_clean) on correct hints;
  - the share of the bad-hint damage removed.
  
  Existing remedies sit at corners (use all hints: keeps all, removes nothing; drop all poses: removes 80–90% but keeps nothing). The honest target for E3 is the top-right. Preview on calibration runs (X-cal); final in session B.
- **E. Benchmark release.** Error families at literature-sourced sizes (each with its citation), real device hints (whole-room ARKit, LiDAR, AI depth), GT-audited clips, three models and standard metrics (AUC@30, RRA/RTA@5, AbsRel, δ1.25) packaged as a reusable stress test. SpatialBench injects only correct priors.
- **Optional, laptop only: real compass hints from ADVIO.** ADVIO records the raw magnetometer and accelerometer. Its reference orientation comes from gyroscope integration with fix points, and the device trackers agree with it to < 2° on average (paper p. 13), well below typical compass errors of 5–20°. A tilt-compensated compass heading per frame would give *measured* phone-compass rotation errors indoors, and later real compass-hint clips scored on rotation only (ADVIO positions are too rough).

### Continue / stop rules after session A (fixed before it runs)
- **GO for the method paper** if both hold:
  - in X-B, real whole-room pose hints are **harmful** (E-H verdict HURTS on rotation or shape) for at least one model, or E3-cal shows E3 removing more bad-hint damage than the best of R1–R3 at equal or better benefit kept;
  - E3 is **calibrated** for MapAnything (`exp1/e3_calibrate.py` finds a z on the grid; not "NOT CALIBRATED").
- **Analysis paper (no method claim)** if E3 is NOT CALIBRATED for MapAnything, i.e. the photo witness cannot be made safe on correct hints. E3 is then reported as a negative result.
- **Inside-the-network direction (trust dial / learned gate)** only if Shubh's G1 oracle dial removes most of the damage (experiments/G.md thresholds).
- Whatever the outcome, X-A (three models) and angles A–E are written up.

## 6. VM session A (one session; queue ordered by priority so an early stop loses only the tail)
1. Setup in parallel: envs `mapanything`, `pi3x`, `omnivggt`, `aidepth` (`scripts/vm_setup_e2x.sh`). The GPU starts when MapAnything is ready.
2. AI depth for the E2-E clips (the laptop copies have none; checked against E2-E's record `results/e2e_mono/mono_depth.csv`), then the room and `cal` clips.
3. MapAnything: E3-cal synthetic + real; room dev → gate → room eval; then Shubh's gate test (G0–G2, dev clips only, `experiments/G.md`; about 10 minutes; a failure there is logged and the queue goes on).
4. Pi3X: checks; ETH3D dev → gate → eval; E2-E dev → gate → eval; room dev → gate → eval; E3-cal.
5. OmniVGGT: room dev → gate → eval. Its E3-cal is opt-in (`RUN_OV_E3CAL=1`): about 9,000 runs at OmniVGGT's measured fp32 speed would add 3.5–4 hours, and its images-only witness is unstable.

**Size and time** (run counts from the configs and clip folders, `scripts/session_progress_e2x.py`; speeds measured in the E2 session on the same GPU type, `results/*/gpu_log.txt`):
- Recorded runs: MapAnything 10,554; Pi3X 15,061; OmniVGGT 817, plus 9,064 if opt-in. E3 controls add hidden passes: two witness passes per clip and one depth pass per depth hint set.
- Measured E2 speeds: MapAnything about 18,000 runs/h (e2df eval: 4,392 runs in 14.5 min); OmniVGGT about 2,500 runs/h (e2d eval: 3,312 runs in 77 min).
- **Estimate, not measured:** setup about 0.3 h, MapAnything about 0.75 h, Pi3X 1–1.75 h (speed unknown until its dev run), OmniVGGT rooms about 0.35 h. **About 2.5–3.5 GPU-hours in total**, plus 3.5–4 h with the OmniVGGT E3-cal.

## 7. Open items
- **User:** approve §5, then the tag `e2x-prereg` is set.
- **User:** submit the LaMAR access request (only the user can).
- **Not in session A:**
  - more AI depth estimators;
  - Zurich GPS hints;
  - TUM RGB-D;
  - R4 (robust BA, CPU);
  - standard metrics (AUC@30 etc.) are computed on the laptop from saved extras.
