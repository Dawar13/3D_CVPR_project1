# Next steps: start here

Updated 7 Nov 2026. This is the checklist from where the repo stands now to the workshop paper. Each step links to the detailed instructions. The reasons and the vision are in [`docs/VISION.md`](docs/VISION.md); the full context and decision log are in [`context.md`](context.md).

## Where things stand

| Step | Status |
|---|---|
| E0 pipeline check | **PASS** (27 Oct) |
| E1 do wrong hints hurt? | **Strong GO** (28 Oct) |
| E2 how, why, real hints, second model, remedies | **Done** (2 Nov). Decision "GO to E3, narrowed" still marked *proposed* in `context.md` |
| E2X third model, whole-room real hints, E3 calibration | **Prepared on the laptop, not run.** No `e2x-prereg` tag yet |
| G gate tests | Code in the main tree and in the session-A queue; never run on real weights |
| E3 frozen eval | Not prepared (session B) |

## 0. Housekeeping (before anything else)

- [ ] **Push the preregistration tags.** GitHub has no tags at all (`git ls-remote --tags origin` is empty), while the docs cite `e1-prereg`, `e2-prereg-C` and `e2-prereg-DEF`. Whoever has them locally runs:
  ```bash
  git push origin e1-prereg e2-prereg-C e2-prereg-DEF
  ```
  Without them, the "frozen before eval" claim cannot be checked by anyone else.
- [ ] **Clips are not in git.** `data/` is git-ignored. Session A needs `data/clips`, `clips_e2`, `clips_e2e`, `clips_e2x_room` and `clips_e2x_cal` (about 1.1 GB) on the laptop that builds the bundle. A fresh clone cannot run session A until they are copied over or rebuilt (`exp1/datasets.py`, `exp1/arkit_clips.py`). Expected counts are in `experiments/E2X_instructions.md` §2.
- [ ] **Confirm the E2 decision** in `context.md` §10 (it still says "proposed").
- [ ] **Decide if the repo stays public.** It holds unpublished results.
- [ ] **Do not use `3D-CVPR_gate_tests_only/`.** It is an older copy of code already in the main tree; see its `HOW_TO_USE.md`.

## 1. E2X session A (one GPU session, about 2.5–3.5 GPU-hours, estimate)

Follow [`experiments/E2X_instructions.md`](experiments/E2X_instructions.md):

1. [ ] Laptop checks: tests pass, `make_configs.py` changes no config, clip counts match (§2).
2. [ ] Read and approve `experiments/E2X.md` §5 (tests, calibration rule, stop rules) and §3.2 (clip exclusion). Commit, then `git tag e2x-prereg` and push the tag (§3).
3. [ ] Build the bundle: `bash MapAnything_Exp1/scripts/vm_launch_e2x.sh build` (§4).
4. [ ] Rent the VM: ≥ 80 GB GPU, driver ≥ 570, ≥ 16 cores, ≥ 64 GB RAM, ≥ 150 GB disk (§5).
5. [ ] Send and start: `bash MapAnything_Exp1/scripts/vm_launch_e2x.sh send user@HOST PORT` (§6).
6. [ ] Monitor; when something fails, use the table in §8. Never patch a failed dev gate.
7. [ ] Fetch: `bash MapAnything_Exp1/scripts/vm_fetch_e2x.sh user@HOST PORT`. Switch the VM off only after `SESSION_GPU_DONE` and the results are on the laptop (§9).

The queue includes the gate tests G0–G2 (MapAnything, dev clips only). To run them on their own instead, see [`RUN_GATE_TESTS.md`](RUN_GATE_TESTS.md).

## 2. Analyse and decide (laptop, about 1 hour)

`experiments/E2X_instructions.md` §10:

- [ ] Commit the raw results first.
- [ ] E3 calibration: `python -m exp1.e3_calibrate ...` → `chosen.json` or **NOT CALIBRATED**.
- [ ] X-A / X-B / angles: `python scripts/e2x/analyse_e2x.py --out results/e2x_agg`.
- [ ] Gate tests: judge `results/g1_summary/`, `results/g2_summary/` against `experiments/G.md`.
- [ ] Apply the stop rules fixed in `experiments/E2X.md` §5:

| If | Then |
|---|---|
| Real whole-room pose hints HURT for ≥ 1 model, **or** E3 beats the best of R1–R3 at equal or better benefit kept, **and** E3 is calibrated for MapAnything | **Method paper**: write `e3-prereg`, prepare session B |
| E3 NOT CALIBRATED for MapAnything | **Analysis paper**; E3 reported as a negative result |
| G1's oracle gate removes most of the damage | Also prepare the in-network learned gate (the original vision, `docs/VISION.md` §3) |

- [ ] Log the decision, with numbers and file links, in `context.md` §10.

## 3. Choose the primary track

- [ ] Compare with the rolling-shutter project's EXP2 result (separate repository) and pick one primary paper (`docs/VISION.md` §4).

## 4. Session B and the paper

- [ ] Freeze `e3-prereg`: cut-offs from calibration, E3 eval grid, remedies R0–R3 (+ R4 if run), success rules (`experiments/E2X_instructions.md` §11).
- [ ] Run the frozen E3 eval once.
- [ ] Laptop meanwhile: R4 robust bundle adjustment (CPU), optional ADVIO compass study, benchmark write-up (angle E).
- [ ] Laptop meanwhile: scale-fit sensitivity check for the spread results (`docs/VISION.md` §2, last caveat). Descriptive only.
- [ ] Write the workshop paper. Every number links to a results file.
