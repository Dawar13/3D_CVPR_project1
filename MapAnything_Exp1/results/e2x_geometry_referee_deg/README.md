# Geometry referee in degrees (E2X L2, planned baseline R5): DEV and CAL clips only (2026-11-07)

**What it is.** `scripts/e2x/geometry_referee_deg.py`, CPU only, no 3D model.
- The photos give their own relative pose per pair of views: RANSAC-clean SIFT matches → essential matrix → rotation and translation direction. No pose hint is used for this.
- A hint set is compared with it **in degrees**:
  - per pair, the angle between the photo and the hinted relative rotation, and between the translation directions;
  - per view, the median over its pairs; per hint set, the median over all pairs.
- Earlier version in pixels (Sampson distance): `results/e2x_geometry_referee/README.md`.

**Rule** (fixed in the script before any detection was looked at; same false-action budget as E3):
- drop a camera hint if its rotation disagreement is above τ_view, the smallest grid value that drops ≤ 2% of correct-hint views;
- drop all camera hints if the set disagreement is above τ_set, the smallest grid value that acts on ≤ 5% of correct-hint clips;
- if no grid value meets the budget, that part is NOT CALIBRATED and is never loosened.

**Data:**
- synthetic errors (`synth/`): 3 ETH3D dev clips plus the SceneFun3D `cal` clips;
- real iPad hints (`real/`): SceneFun3D `cal` and dev clips.

Eval clips are refused by the script.

## Result: NOT CALIBRATED (both revisions)

| Revision | Folder | Rule | Outcome (`chosen.json`) |
|---|---|---|---|
| 1 | `../e2x_geometry_referee_deg_rev1/` | every view / set with ≥ 1 usable pair | NOT CALIBRATED (view and set) |
| 2 | this folder | abstain below 3 usable pairs per view, 10 per set | NOT CALIBRATED (view and set); 17% of views and 20% of hint sets scored |

**Why** (from the ground-truth hint rows of `synth/` and `real/`, revision 1):
- ETH3D dev: 24.3 usable pairs per clip on average.
  - Correct hints disagree with the photos by a median of 0.84° per view; the largest single view is 2.81° (`eth3d_dev_descriptive.csv`).
- SceneFun3D whole-room iPad clips: the median clip has **1** usable pair, and many have 0.
  - Correct hints still disagree widely: the 95th percentile is 21.6° (rotation) and 48.4° (direction) per hint set.
  - Some clips with many pairs disagree systematically, e.g. `cal__sf470352__c2`: 36 pairs, median 8.65° rotation, 55.9° direction.
  - So either the dataset's ground truth or the matching is unreliable there. SceneFun3D ground truth is known to have misregistrations (E2X.md §2.2).

## Descriptive: ETH3D dev (laser-registered ground truth), `eth3d_dev_descriptive.csv`

**One wrong camera:** the referee measures the size of the error itself.

| True error | Bad camera reads | Good cameras read (median) |
|---|---|---|
| 5° | 5.17° | 0.82° |
| 15° | 14.97° | 1.01° |
| 30° | 29.86° | 0.95° |
| 90° | 89.96° | 1.01° |

**Errors shared by every camera** leave relative rotations unchanged, so they show mainly in the translation direction:

| Error | Set direction disagreement |
|---|---|
| shared heading 2° / 5° / 10° | 3.94° / 6.90° / 8.20° |
| none (correct hints) | 3.61° |

**Noise and drift** show in the relative rotations: noise 0.5° / 1° / 2° / 5° reads 1.11° / 1.58° / 3.44° / 6.47°; drift levels 1 / 2 / 3 read 1.20° / 2.27° / 4.61° (set medians).

**Depth-scale errors** are invisible to it, as expected (it uses poses only).

## What it means (for E3 and the paper; not a preregistered result)
- Where photos overlap well (ETH3D), plain two-view geometry measures a wrong camera's error in degrees, with no 3D model. That is enough to apply a "drop only above break-even" rule.
- On whole-room iPad captures it cannot be made safe under the fixed false-alarm budget. The photos give too little evidence, and the ground truth itself is doubtful.
- So R5 is not a deployable baseline on the room clips. On ETH3D-like clips it is a strong reference. The model-based witness (E3) has to cover the rest.
