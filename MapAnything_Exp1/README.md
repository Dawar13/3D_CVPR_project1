# Experiment 1: MapAnything geometric-prior corruption audit

Inference only. No Self-Healing Attention, no gates, no fine-tuning, no bundle adjustment.

**Status: none of this code has been run yet.** It compiles; nothing has executed
against the real MapAnything package. Run it on the GPU server, not the Mac.

## Layout

```
configs/pilot.json            preregistered condition grid (edit BEFORE looking at eval results)
exp1/geometry.py              SE(3)/SO(3) utilities, pose validation
exp1/corrupt_geometry.py      deterministic, logged pose corruption
exp1/datasets.py              ETH3D -> fixed clip .npz files; dev/eval split
exp1/model_io.py              the ONLY file that calls the MapAnything API
exp1/metrics.py               depth / point-map / pose metrics
exp1/controls.py              deployable RGB-consistency pose filter
exp1/run_exp1.py              runs every condition x clip x seed (resumable)
exp1/aggregate.py             bootstrap CIs, paired deltas, figures
tests/                        unit tests for the corruption utilities
scripts/env_report.sh         Stage A inspection dump
scripts/setup_server.sh       clone + dedicated conda env
reports/                      Deliverable A and results templates
```

## Commands (on the server, GPUs 4/5/6)

```bash
# 0. setup + inspection
bash scripts/setup_server.sh
bash scripts/env_report.sh > reports/env_dump.txt
conda activate mapanything
python -m pytest tests -q

# 1. minimal official inference: run the README example in third_party/map-anything first
#    and confirm finite outputs before going further (Stage A, step 6).

# 2. data: ETH3D high-res multi-view TRAINING, undistorted DSLR images + GT depth
#    (multi_view_training_dslr_undistorted.7z, multi_view_training_dslr_depth.7z)
#    unpacked into one root. Download on the server only.
python -m exp1.datasets --eth3d_root /data/eth3d --out data/clips --n_views 10 --size 518 336

# 3. smoke test: one dev clip, 3 conditions
CUDA_VISIBLE_DEVICES=4 python -m exp1.run_exp1 --config configs/pilot.json \
    --clips 'data/clips/dev__courtyard__c0.npz' --out results/smoke \
    --only rgb_only clean_pose zero_noise

# 4. dev pilot (3 scenes), then eval (10 scenes) once the protocol is frozen
CUDA_VISIBLE_DEVICES=4 python -m exp1.run_exp1 --config configs/pilot.json \
    --clips 'data/clips/dev__*.npz'  --out results/dev_pilot
CUDA_VISIBLE_DEVICES=4 python -m exp1.run_exp1 --config configs/pilot.json \
    --clips 'data/clips/eval__*.npz' --out results/eval_pilot

# 5. tables + figures
python -m exp1.aggregate --run results/eval_pilot --metric pts_relerr_scale
python -m exp1.aggregate --run results/eval_pilot --metric rre_deg
```

The dev/eval split is fixed by a hash of the scene name (`datasets.dev_eval_split`).
The smoke-test scene name above is a placeholder; use whichever clip landed in `dev__`.

## Design decisions

- **Clips**: 10 consecutive DSLR images per scene, resized+centre-cropped to 518x336
  once, at prep time. GT depth is nearest-resampled with the same transform and
  intrinsics are adjusted. All conditions read the same file, and the runner
  checks the image hash on every run.
- **Poses** are OpenCV cam2world, re-expressed relative to view 0, in metres.
  ETH3D is metric, so `is_metric_scale=True` goes with every pose.
- **RGB-only** means the optional keys are absent, which is the README's interface.
  No zero poses are fabricated.
- **Reference ladder**: `rgb_only` → `rgb_K` (intrinsics) → `clean_pose` (K + pose),
  plus `clean_pose_noK`. Every corrupted condition keeps clean K, so corruption
  effects are measured against `clean_pose` and `rgb_only` with only poses changing.
- **Corruption** (`corrupt_geometry.py`): each severity is an exact magnitude with a
  random direction. Rotation: `R' = R·Exp(θu)`, so the camera turns about its own
  centre. Translation: `c' = c + f·L·v`, where L is the median distance of the
  clip's camera centres from their centroid. View 0 is never corrupted by default.
  Every applied transform goes to `corruption_manifest.json`.
- **Metrics** (`metrics.py`): everything is in the view-0 camera frame with **no
  rigid alignment**, so pose errors stay visible. Each metric is reported as raw
  metric and with a single per-clip median scale. Point-map relative error, depth
  AbsRel/δ1, pairwise RRE/RTE, RRA@5/RTA@5 and camera-centre error are all reported.
  Lower is better except δ1, RRA and RTA. Evaluation uses GT-valid pixels only, and
  model masking is off.
- **Statistics**: seeds are averaged within a clip. 95% CIs come from a percentile
  bootstrap over clips, and deltas are paired. `per_scene.csv` and Figure 3 show
  per-clip spread.
- **Contamination**: for outlier conditions the runner also scores the corrupted
  views and the *uncorrupted* views separately, alongside RGB-only and clean-prior
  predictions on the same views (`subset_deltas.csv`). This shows whether one bad
  pose damages the other views.
- **Controls**: `drop_oracle` (an ORACLE that drops the truly corrupted views'
  poses) and `filter_rgbcheck` (DEPLOYABLE: drops poses that disagree with the
  RGB-only prediction by >10° rotation or >15° centre direction).

## Things to verify against the installed commit before trusting any number

The code follows the README API as I remember it. I did not read it from the
checkout, so check each of these in `third_party/map-anything` and fix
`exp1/model_io.py` if anything differs:

1. `MapAnything.from_pretrained`, `preprocess_inputs`, and the `model.infer` kwargs
   in `model_io.INFER_KW`.
2. Input keys `img` (HxWx3 uint8), `intrinsics` (3x3), `camera_poses` (4x4 OpenCV
   cam2world) and `is_metric_scale`, plus whether torch tensors or numpy are expected.
3. Output keys `pts3d`, `depth_z`, `camera_poses`, `conf`, `mask`, and their shapes.
4. **Whether `preprocess_inputs` resizes 518x336 inputs.** `metrics.evaluate` raises
   if the output resolution differs from GT. Change `--size` to a resolution the
   model keeps.
5. **Whether poses may be given for only some views.** Both controls depend on
   this. If they can't, the controls will show up in `failures.jsonl`.
6. How the model normalises supplied poses (the first-view frame? scene-scale
   normalisation?). Record this in the Deliverable A report before interpreting results.
7. Licence: the default checkpoint `facebook/map-anything` is CC-BY-NC. The
   Apache-2.0 variant is `facebook/map-anything-apache`.

## Not done yet

- Depth corruption (Condition 5). This is deliberately deferred until the pose pilot works.
- Resume skips any key that already has a row, including FAILED rows. To retry a
  failure, delete its row from `per_run.csv`.
