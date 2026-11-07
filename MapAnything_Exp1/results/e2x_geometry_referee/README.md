# Geometry referee (planned baseline R5): first look, DEV and CAL clips only (2026-11-07)

**What it is.** `scripts/e2x/geometry_referee.py`, CPU only, no 3D model.
- Feature matches between a clip's own images are cleaned with RANSAC, without using any pose.
- Each hint set is then scored:
  - **pose check:** per view, the median Sampson distance (px) of its matches under the hinted poses and intrinsics;
  - **depth check:** matches triangulated with the hinted poses give a depth per match; k is the median ratio depth hint / triangulated depth.
- **Data:** 53 clips (3 ETH3D dev c0 clips, 50 SceneFun3D cal clips) for synthetic errors. 65 clips (cal plus SceneFun3D dev) for the real iPad hints.
- **Status:** a development look only, nothing preregistered, eval clips not used.

**Logs.** `results/e2x_geometry_referee_synth.log` and `results/e2x_geometry_referee_real.log` (beside this folder).

**Files.** `synth/` (configs/e2x_e3cal_synth_mapanything.json) and `real/` (configs/e2x_e3cal_real_mapanything.json), each with:
- `views.csv`: per view;
- `runs.csv`: per hint set;
- `summary.csv`: AUROCs.

## Findings (AUROC: 1 = perfect separation, 0.5 = chance)

**Errors shared by all cameras** (hint set vs the correct hint sets of the same clips, median view score):

| Error | AUROC |
|---|---|
| shared heading 2° / 5° / 10° | 0.998 / 1.000 / 1.000 |
| camera-mount error 2° / 5° | 1.000 / 1.000 |
| drift levels 1 / 2 / 3 | 0.990 / 0.995 / 1.000 |
| noise on every camera 0.5° / 1° / 2° / 5° | 0.998 / 0.985 / 1.000 / 1.000 |

MapAnything's own confidence reacts weakly to exactly these agreeing errors (E2-C P10/P11).

**Single wrong camera** (bad view vs good views within a run):
- 5° / 15° / 30° / 90°: 0.911 / 0.919 / 0.900 / 0.933;
- wrong first camera: 0.894;
- 10% / 20% / 40% of cameras wrong: 0.925 / 0.883 / 0.779.

For shared, drift and noise errors this per-view test is not meaningful: every non-reference view is corrupted.

**Depth scale** (pose hints correct):
- all depth ×1.5 → k = 1.498; ×0.67 → k = 0.669; correct depth → k = 0.999. AUROC 1.000 for both.
- One view ×1.5: AUROC 0.845 with the clip-level k. A per-view ratio is the better statistic there.

**Real iPad hints** (cal and dev clips):
- iPad poses fit the matches worse than the GT poses: median view score 1.139 px vs 0.305 px; AUROC 0.990.
- iPad LiDAR is scale-consistent with the iPad poses: k = 1.009, vs 0.999 for GT depth with GT poses.

**Coverage.** Only 40% of views could be scored: they need ≥ 30 RANSAC-clean matches with some partner. Wide-baseline room clips often have too few, e.g. `cal__sf483632__r2` has 0 matched pairs.

## What it means (for the E3 design; not a result)

- **Plain geometry sees the agreeing errors the model is blind to.** This is a candidate "aha" for the paper.
- **Detection is not harm.** The referee also flags real iPad poses and 0.5° noise, which *help* the model (E2-E).
  - A deployable rule must compare the *size* of the measured hint error with what the photos alone would get (the break-even idea), not merely detect a mismatch.
  - Next step: turn Sampson scores into rotation errors in degrees (per-pair relative rotation from the essential matrix) and act only above break-even.
- **Low-texture and wide-baseline views are blind spots,** where the model-based witness (E3) is the fallback.
