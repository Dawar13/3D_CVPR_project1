# Project context: Do 3D foundation models trust bad hints?

Last updated: 27 Oct 2026. Update the "Status and decisions log" at the bottom after every experiment.

## 1. One-paragraph summary

Feed-forward 3D models such as **MapAnything** and **OmniVGGT** take images plus optional *hints*: camera poses, intrinsics (lens settings) and depth maps. With clean hints they get much better. Real hints come from GPS, odometry, cheap depth sensors or wrong EXIF data, so they are often wrong.

We want to find out three things:
1. Do wrong hints make these models worse than using images alone, and for which kinds of error?
2. Why does it happen? Is it the average-based scale normalisation, or contamination inside attention?
3. Can bad hints be detected and switched off at inference time, using only the images as the witness, with no training?

The project is modular:
- Proving (1) alone is a workshop-level analysis.
- (1) + (2) + (3) with a clear accuracy gain is a CVPR-level method paper.

Target venues: CVPR 2027 (registration 10 Nov 2026, submission 16 Nov 2026), or a CVPR/ICML 2027 workshop as fallback.

## 2. Hypotheses (each is tested by one experiment)

- **H1 (problem exists):** for some realistic hint errors, MapAnything with hints is worse than MapAnything with images only. Realistic errors include a gross outlier pose (a GPS jump), a wrong metric scale on all poses, accumulated drift, or a wrong focal length on all views. The damage also spreads to views whose hints were correct. → **E1**
  - **E1 result (2026-10-28): CONFIRMED, decision Strong GO** (`MapAnything_Exp1/reports/e1_results.md`).
    - Harm is driven by wrong camera *rotations*: one camera 15° off made rotation error +114% vs images-only (10/10 scenes); drift level 2 +142%; a shared 5° heading +108%.
    - Wrong scale and wrong depth scale are also harmful.
    - Wrong camera *positions* are mostly absorbed safely; only a 10 L GPS jump hurts.
    - Spread to good views is real but modest: +20% rotation error with 20% of views wrong.
    - The model copies bad poses almost fully (absorption 0.96), even though its confidence flags the bad view (AUROC 0.91).
- **H2 (mechanism):** if the damage comes mainly from the mean-based pose/depth scale (a 0% breakdown point), replacing it with a median removes most of it. If it does not, the damage happens inside the network. → **E2**
- **H3 (detectability):** an image-consistency score computed without ground truth can identify which hints are bad, and dropping them recovers most of what an oracle recovers. → **E3**

## 3. Background you need (verified facts only)

**MapAnything** (Keetha et al., Meta + CMU; arXiv 2509.13414, v3 Jan 2026):
- Inputs: N images, plus optional per-view intrinsics (converted to ray directions), camera poses (OpenCV cam2world; split into quaternion + translation) and depth.
- Every input is encoded to features of the same size: F_I (DINOv2 image), F_R (rays), F_D (normalised depth), F_Q (rotation), F_T (translation **divided by ẑp, not a unit direction**; code check, `reports/e0_interface.md` Q8), F_ẑp (pose scale), F_ẑd (depth scale).
- They are fused per view as F_E = LN( LN(F_I) + Σ_m LN(F_m) ) (paper), over the hints that were provided, then passed to a 16-layer alternating-attention transformer. In code: F_E = LN(F_I + Σ_m mask_m · F_m). A missing hint is the same encoder output multiplied by 0, not a learned token (e0_interface Q7).
- **Pose scale ẑp** = the *mean* distance from view 0 of the posed cameras with non-zero translation (view 0 and unposed views are excluded). All translations are divided by it. **Depth scale ẑd** = the per-view mean of valid depth along the ray. The normalisation always happens; only the scale *features* F_ẑp/F_ẑd are gated per view by `is_metric_scale` (default True when absent) (e0_interface Q8-Q9).
- Training gave hints with selection probabilities (overall 0.9; each modality 0.5). The paper describes no noise added to hints.
- Limitations section (verbatim points): (a) it does not explicitly account for noise or uncertainty in geometric inputs; (c) test-time compute and iterative inference remain unexplored, which "ties into effectively handling noise in the inputs"; (d) multi-modal features are fused before the transformer.
- Clean-hint benefit (Table 2, two-view, averaged over ETH3D/SN++v2/TAv2): points rel. error 0.08 (images only) → 0.02 (all hints); inliers 57.5 → 82.0.
- API constraints (repo docs):
  - If any view has `camera_poses`, view 0 must also have them.
  - Do not give both `intrinsics` and `ray_directions`.
  - `depth_z` needs intrinsics or rays.
  - `preprocess_inputs` converts intrinsics → rays, depth_z → depth along ray, and poses → quats + trans.

**OmniVGGT** (Peng et al., CVPR 2026; arXiv 2511.10560): VGGT plus a GeoAdapter.
- Camera hints are injected through zero-initialised convolutions; depth hints are added directly.
- A missing hint is replaced by a learned placeholder token, chosen by a binary present/absent switch.
- Used for the cross-model replication only (E4), after GO.

**Closest related work. We must cite these and differ from them:**
- **VPGGT** (arXiv 2610.05531, Oct 2026, contemporaneous): trains OmniVGGT with simulated *pose* drift and adds a "prior residual connection".
  - Pose noise only; intrinsics and depth stay clean.
  - It is training-based and does not detect *which* hint is bad.
  - Key number: on TUM RGB-D the released OmniVGGT with realistic noisy poses still beat images-only: translation accuracy at 5° (RTA@5) 58.89 vs 38.00, rotation accuracy at 5° 100.00 vs 99.67 (Table 3, "w/C n"; checked in the PDF 2026-10-29). The earlier figure 57.11 is Table 1(a), where ground-truth depth was also supplied on all views. So mild realistic noise is NOT catastrophic. Our case must be gross or correlated errors.
- **WorldMirror** (arXiv 2510.10726): reports graceful degradation under moderate pose and depth noise.
- **TCO** (arXiv 2604.03878): a test-time method that uses hints as constraints on an image-only model.
- **Pow3R** (CVPR 2025): the first feed-forward model with priors; it examined incorrect camera inputs.

## 4. Assets and licences

| Asset | Where | Licence / notes |
|---|---|---|
| MapAnything code | github.com/facebookresearch/map-anything | Open source |
| MapAnything weights | HF `facebook/map-anything` (13 datasets) | CC BY-NC 4.0 (research OK) |
| MapAnything weights (alt) | HF `facebook/map-anything-apache` (6 datasets) | Apache 2.0 |
| OmniVGGT code / weights | github.com/Livioni/OmniVGGT-official, HF `Livioni/OmniVGGT` | Checked 2026-10-27: code MIT (`LICENSE`), HF card MIT; 1.217 B params, 4.87 GB fp32. Trained from VGGT-1B (README:293), whose weights are CC BY-NC 4.0 and whose code is under the VGGT License: treat as research/non-commercial and **acknowledge VGGT in the paper** (VGGT License 1.b.ii) |
| ETH3D high-res multi-view, training split | eth3d.net | CC BY-NC-SA 4.0; per scene `dslr_undistorted` + `dslr_depth` + `dslr_jpg` (the last only for `dslr_calibration_jpg`, needed to undistort GT depth); `scripts/e0_download_eth3d.sh` |

Everything is free to download for research. Nothing requires payment or private data.

## 5. Codebase

- `MapAnything_Exp1/` holds Shubh's Experiment-1 harness, imported as-is. It is inference only. **It has never been run.**
  - `exp1/datasets.py`: ETH3D → fixed 10-view clips (`.npz`: images, K, poses relative to view 0, GT depth). The dev/eval split is by scene-name hash (3 dev / 10 eval).
  - `exp1/corrupt_geometry.py`: exact-magnitude, random-direction pose corruption, logged to a manifest. View 0 is never corrupted.
  - `exp1/model_io.py`: the ONLY file that calls the MapAnything API.
  - `exp1/run_exp1.py`: condition × clip × seed runner. It is resumable, scores contamination on the uncorrupted views, and runs the controls `drop_oracle` (cheating upper bound) and `filter_rgbcheck` (deployable baseline).
  - `exp1/metrics.py`: depth AbsRel/δ1, point error, pairwise pose errors; raw metric plus one per-clip median scale; no alignment.
  - `exp1/aggregate.py`: bootstrap CIs, paired deltas, and `benefit_lost` (>1 means worse than images-only).
  - `scripts/e0_*`: E0 checks (layout, GT self-consistency, official example, smoke check, determinism) and `e0_phase_c.sh`, which runs them all. `tests/test_runner_dryrun.py` runs the whole runner on CPU with a fake model, as a plumbing check.
- `experiments/`: per-experiment plans (`E0.md`) and VM runbooks (`E0_runbook.md`).
- `papers/` holds PDFs and short notes of the papers in §3. Add more here.
- `MapAnything_Exp1/results/` is tracked in git (user decision 2026-10-31): every run folder, including the `.npz` extras. `data/` (clips, rebuilt from ETH3D) and `third_party/` (pinned by commit) stay git-ignored. Copy VM results back into `results/` and commit them.

## 6. Rules (non-negotiable)

1. **Preregister.** Finalise every condition grid and GO/STOP threshold *before* running on the eval split. Never change them after seeing eval results. The dev split is only for debugging.
2. **Images-only and clean-hint references are always run** on the same clips, and every comparison is paired.
3. **No ground truth at test time** for anything called "deployable". GT is used only for scoring and for oracle controls, which must be labelled ORACLE.
4. **Verify the interface first.** Check every assumption flagged in `MapAnything_Exp1/README.md` ("Things to verify") against the installed commit before trusting any number. Record file:line evidence.
5. **One change at a time.** Every new condition or patch gets its own name, a config entry and a manifest entry.
6. **Reproducibility.** Log the MapAnything commit, checkpoint id, torch/CUDA versions, seeds and image hashes (the runner already does).
7. **Honesty.**
   - Never invent or "expect" numbers in reports.
   - Label projections as projections.
   - Report failures and negative results.
   - If something looks too good, check for leakage first.
8. **Do not touch shared environments** on the server. Use the dedicated conda env.

## 7. Experiment plan: the initial set only (E0–E3)

**Phase map (updated 2026-10-29, `experiments/E2.md`):**
- E0: pipeline check (done).
- E1: does the problem exist? (done: Strong GO).
- **E2: the nature of the problem.** Literature, audit, behaviour tests, a second model, real sensor hints, existing remedies. No solution work.
- E3: our method.
- E4: generality of the method.

The E3 text below is the earlier sketch; it will be rewritten from the E2 results.

### E0: Setup and smoke test (target: day 1)
- Run `scripts/setup_server.sh` and `scripts/env_report.sh`, then the official README example, then `pytest`.
- Prepare clips: `python -m exp1.datasets --eth3d_root <ETH3D> --out data/clips --n_views 10 --size 518 336`.
- Run the smoke test on one dev clip with `rgb_only`, `clean_pose` and `zero_noise`.
- Fill in `reports/exp1_environment_and_reproduction.md`. In particular, find out what MapAnything does when a hint is missing (term skipped? zeros? a learned token?). This decides what "gate = 0" means later.
- **PASS:** outputs are finite; output resolution equals 518×336; `zero_noise` ≈ `clean_pose` (bf16 noise only); partial poses are accepted; runtime and memory per 10-view clip are recorded.

### E1: Does the problem exist? (target: days 2–3)
Shubh's grid in `configs/pilot.json` (noise and outlier families with controls), PLUS these new families. Each needs new corruption code with unit tests:

| New family | Definition | Why |
|---|---|---|
| `outlier_trans_large` | 1 non-reference view; translation +{1, 3, 10}×L, random direction | GPS jump; stresses the mean pose scale |
| `scale_all` | all non-reference translations × {0.67, 1.5}, metric flag on | wrong metric scale; errors agree with each other |
| `drift` | accumulated heading-rate bias + velocity bias along view order (2 severities) | realistic odometry drift (cf. VPGGT) |
| `focal_all` | fx, fy × {0.8, 1.2} on all views; poses clean | wrong EXIF/calibration; shared error |
| `focal_only` | same as above but no poses (reference: `rgb_K`) | isolates the intrinsics effect |
| *(E1b, if time)* `depth_scale` | give GT depth on 2 views, then × 1.5 | wrong depth-sensor scale |

- **Readouts:**
  - `benefit_lost` and paired Δ vs `rgb_only` for `pts_relerr_scale`, `rre_deg` and `depth_absrel_scale`;
  - the fraction of eval clips worse than `rgb_only`;
  - contamination: the uncorrupted views' error vs `clean_pose` on the same views;
  - `filter_rgbcheck` and `drop_oracle` recovery.
- **GO (H1 true):** at least one realistic family has `benefit_lost > 1` on the eval split, i.e. it is worse than images-only.
  - The 95% CI of the paired Δ vs `rgb_only` must exclude 0.
  - At least 7 of 10 eval scenes must be worse.
  - `filter_rgbcheck` must recover less than half of what `drop_oracle` recovers (or not apply, as with correlated errors).
- **STOP / downgrade:** everything degrades gracefully and stays better than images-only, or the simple filter already fixes it. Then write it up as a careful analysis note (workshop at most) and pivot.

### E2: The nature of the problem (full plan: `experiments/E2.md`)
Six parts, each preregistered before its eval run (tags `e2-prereg-<part>`), Holm-corrected, with practical thresholds (shape ≥ 1 pp, real size ≥ 5 pp, rotation ≥ 0.25°, scale ≥ 5%):
- **E2-A** literature audit: closest work, classical tools, literature-realistic error sizes, datasets with real sensor hints. Gate: stop if a paper already measures our core findings.
- **E2-B** post-hoc audit of E1: is "copying" real (not a code shortcut), practical significance, bad-camera vs good-camera pairs, absorption consistency, realistic re-labelling. Gate: stop if outputs are copied by code.
- **E2-C** behaviour on MapAnything/ETH3D (3 clips per scene): trust curves (C1), knows-but-obeys (C2), which outputs are damaged (C3), which hint channel, incl. the mean→median scale probe as a diagnostic (C4), photos as evidence (C5), independent vs correlated at equal size (C6), partial hints (C7), what photos alone reveal (C8).
- **E2-D** OmniVGGT (second model), **E2-E** real sensor hints, **E2-F** existing remedies (remaining gap).
- **Decision:** GO to E3 / analysis paper / narrow (table in `experiments/E2.md` §7).

The original E2 (mean vs median as the whole experiment) is now the C4 diagnostic.

### E3: Can bad hints be detected without ground truth? (target: days 4–6)
- For every corrupted run, compute per-view scores:
  - (a) the model's confidence on that view;
  - (b) disagreement with the images-only prediction, as a continuous version of `filter_rgbcheck`;
  - (c) **image consistency with leave-one-group-out**: rerun with view i's hint removed, then measure how much cross-view consistency improves. Consistency means warping DINO features (or pixels) of neighbouring views into view i using the predicted depth and pose, and comparing them.
- For correlated errors (`scale_all`, `drift`, `focal_all`), compute the same score at the **group** level (all hints vs none).
- **Readouts:**
  - AUROC for flagging the truly corrupted views or groups;
  - the accuracy recovered by dropping flagged hints, as a fraction of `drop_oracle`;
  - extra runtime.
- **GO for the method paper:** score (c) reaches AUROC ≥ 0.85 and recovers ≥ 70% of the oracle gain, and clearly beats (b) on the correlated or subtle cases.
- **STOP:** (c) is no better than (b). The deployable baseline is then already enough; publish the E1/E2 analysis only.

### Later (not now): E4 generality of the method (OmniVGGT and others); two-pass refinement; soft gates or light training. Only after E3 = GO. (The OmniVGGT *problem* replication moved into E2-D.)

## 8. Compute budget
**Measured in E0 (2026-10-27, A100-SXM4-80GB, bf16 autocast, `memory_efficient_inference=False`):**
- one 10-view inference at 518×336 takes **0.26 s** after warm-up (first call ≈ 0.55 s), peak GPU memory **8.26 GB** (`MapAnything_Exp1/reports/e0_logs/7_smoke_check.txt`, `8_det_*.txt`);
- output is bit-identical run to run;
- MapAnything has 1.228 B parameters (fp32 checkpoint, 4.91 GB).

**Projection, not measured:** E1's ~2-3k passes come to ~10-15 GPU-minutes of pure inference, plus model loading per process. Any GPU with ≥ 16 GB and bf16 would fit the measured peak. The estimates below were written before E0.

- One GPU with ≥ 24 GB (A100/H100/RTX 4090/A6000). MapAnything is about 1B parameters and runs in bf16.
- E0: < 0.5 GPU-h.
- E1: about 2–3k forward passes ≈ 1–2 GPU-h.
- E2: < 1 GPU-h.
- E3: leave-one-group-out adds about 10 passes per corrupted run, plus feature warping ≈ 2–4 GPU-h.
- **Total for E0–E3: under 10 GPU-hours.** No training. No gradients needed.
- Disk: ETH3D archives (several GB; check sizes on the download page), checkpoints (a few GB), plus results. Keep 50 GB free.
- RAM: 32 GB or more is recommended for clip preparation.

## 9. Open questions to resolve (update as answered)
Answered from the source code @ `3d10cf7a` (2026-10-27); evidence in `MapAnything_Exp1/reports/e0_interface.md`. "Runtime" items are confirmed by E0 Phase C.
- [x] What does MapAnything feed when a hint is missing? The encoder runs on a placeholder (zeros / identity pose) and its feature is multiplied by 0 before the fusion LayerNorm. No learned token. "Gate = 0" = omit the key. (Q7)
- [x] Does `preprocess_inputs` resize 518×336? No: 518×336 is the 3:2 mapping and is kept exactly (runtime check pending). (Q5)
- [x] Are partial per-view poses accepted, with view 0 posed? Yes (runtime check pending). (Q6)
- [x] Where are ẑp and ẑd computed? `geometry.py:1558-1595` / `1523-1555`, called at `model.py:1122-1126` / `987-991`. E2 can swap in a median from `model_io` without editing `third_party`. (Q8, E2/E3 section)
- [x] Is the `metric` flag per view or global? Per view; it gates only the scale features. (Q9)
- [x] **DECIDED 2026-10-27: turn them upright** (rotate image, K and pose together in `datasets.py`, with a test; rebuild the 2 clips before any E1 eval run). Original question: eval clips `relief` and `relief_2` are made entirely of photos ETH3D stores rotated by 90° (sideways). Their GT is consistent, but MapAnything's own converter turns them upright (`data_processing/wai_processing/scripts/conversion/eth3d.py:210-249`). Option 1: rotate them upright in `datasets.py` (image, K, pose) to match MapAnything's benchmark. Option 2: keep them sideways. Pairing keeps comparisons fair either way.
- [x] **DECIDED 2026-10-27: add real-size readouts.** For `scale_all` and `depth_scale`, `pts_relerr_metric`, `depth_absrel_metric` and the model's `metric_scaling_factor` are primary readouts next to the planned ones; other families are unchanged. `model_io` must record `metric_scaling_factor`. Original question: `scale_all` and `depth_scale` reach the model only through the scale features, while the planned E1 readouts are scale-normalised. Should `*_metric` metrics / `metric_scaling_factor` be primary readouts for these families? (e0_interface, "Implication for the preregistration")

## 10. Status and decisions log
- 2026-10-27: Project set up. Shubh's Exp-1 harness imported (never run). Plan E0–E3 agreed. Next: E0.
- 2026-10-27: E0 Phase A (laptop, no GPU) done. **E0 is NOT yet PASS**; no model has been run.
  - Interface checked against the source @ `3d10cf7a`: `model_io.py` needed no functional change.
  - **Bug fixed in `datasets.py`:** ETH3D GT depth is on the distorted grid. It is now undistorted via `dslr_calibration_jpg`, which needs a third archive. The thin-prism model matches pycolmap to 1e-12 px.
  - E0 scripts and tests added, plus a visual check (Step 7b: demo reconstructions of the smoke clip as PLY files and an overview PNG). 30 tests pass on CPU, including a fake-model dry run of the whole runner.
  - `setup_server.sh` pins the commit.
  - Open: the E1 readout question in §9.
  - Next: Phases B and C on the VM (`experiments/E0_runbook.md`).
- 2026-10-27: **E0 PASS on the dev split; 3 eval scenes pending.** VM: A100-SXM4-80GB; torch 2.14.1+cu130; MapAnything `3d10cf7a`; HF snapshot `a1d87e90`. Evidence: `MapAnything_Exp1/reports/exp1_environment_and_reproduction.md`, `reports/e0_logs/`.
  - Smoke clip `dev__delivery_area__c0`:
    - 0 failures; no resize; `zero_noise` ≡ `clean_pose`;
    - clean hints vs images-only: `pts_relerr_scale` −58%, `rre_deg` −75%;
    - both controls ran and dropped exactly the corrupted view;
    - visual check OK.
  - Speed and determinism: 0.26 s per 10-view inference, 8.26 GB peak; bit-identical runs, so the noise floor is 0.
  - GT check PASS on all 10 available clips (3 dev + 7 eval).
  - Bugs found and fixed:
    1. ETH3D GT depth is on the distorted grid (6048×4032, while the undistorted images are ~6200×4130), so the harness as delivered crashed on the first image;
    2. the dev/eval split was computed over whichever scene subset was passed in, which mislabelled 3 eval scenes as dev; no model ran on them, and they were rebuilt.
  - Not done: `relief`, `relief_2` and `terrains` (eval) could not be downloaded because eth3d.net refused TLS from ~14:07 UTC, from both the VM and the laptop. **Download, build clips and GT-check them before any E1 eval run.**
  - Single-clip observations, no interpretation:
    - `rgb_K` worsens the metric-scale errors while improving the scale-normalised ones;
    - `clean_pose_noK` is slightly worse than `rgb_only` on `pts_relerr_scale`.
  - The 10 clips (45 MB) are on the laptop in `data/clips/`, so the next VM can skip the 16 GB download for them.
  - Next: decide the §9 E1 readout question, then E1 code (new corruption families + tests).
- 2026-10-27 (later): **E0 fully PASS.**
  - eth3d.net came back at ~15:15 UTC; `relief`, `relief_2` and `terrains` were fetched and built, and all **13/13 clips pass the GT check**.
  - Results package: `MapAnything_Exp1/reports/e0_results/`. It holds an offline HTML page with a 3D viewer, figures, PLY point clouds, `numbers.csv` and `gt_check.csv`, all regenerated by `scripts/e0_make_results.py`. Online copy: https://claude.ai/artifact/4rESTcPRbXYFACZmyCoSWf (private).
  - New open question for E1, see §9: the `relief` and `relief_2` clips are all sideways photos.
- 2026-10-27: **E1 preregistration decisions (made before any E1 code or eval run):**
  1. turn the sideways `relief` and `relief_2` photos upright;
  2. real-size errors and `metric_scaling_factor` are primary readouts for the wrong-scale families.

  E1 coding has not started; waiting for the go-ahead. GPU: the A100 VM is shared with the rolling-shutter project (about 35 GB in use), and E1 needs about 8.3 GB per run, so it can share the card.
- 2026-10-27: **E1 dev stage done; NOT frozen, no eval run** (on hold for discussion at the user's request).
  - Code, grid (`configs/e1_full.json`, 235 inferences per clip), analysis (`exp1/e1_analysis.py`) and tests (57 pass) are written.
  - relief and relief_2 are upright, and their GT check passes.
  - Dev run: 705/705 runs, 0 failures, ~11 min; repeat runs are bit-identical on the shared GPU.
  - Dev checks pass after amendments 1-13 in `experiments/E1.md`. The open point is amendment 13: turning the metric flag off changes `clean_pose` results by up to 148%, so the plan's sanity expectation for `scale_all_nonmetric_x1.5` does not hold, while that run equals `metric_flag_off` exactly.
- 2026-10-27: **E1 preregistered at `71582c1` (tag `e1-prereg`).** The user approved amendments 13 (metric-flag sanity check) and 14 (5% minimum effect for S1). The grid, code and decision rules are frozen; the eval split is run once from this commit.
- 2026-10-28: **E1 DONE. Decision: Strong GO.**
  - Run: eval split run once from `71582c1` on 10 scenes; 2350/2350 runs, 0 failures, ~1 h 55 min (the shared GPU was slowed by the rolling-shutter project).
  - Signals (`results/e1_eval/agg/signals.json`): S1, S2, S3, S5, S6 and S7 YES; S4 NO (the model's confidence does flag bad views, AUROC 0.91).
  - Key numbers are in `reports/e1_results.md` and the plain summary in `reports/e1_results/e1_results.html` / `.pdf`.
  - Eval-run diagnostics: all pass except `gauge_rigid` (electro rotation differs by 0.026°, 9.4% relative, better not worse; 6/10 scenes bit-identical). This is numerical precision and does not enter the decision.
  - Caveats:
    - S2 is marginal (+20.4% vs the 20% bar).
    - benefit_lost is unstable for the lens and depth families, so the report leads with "% vs without the hint" figures (presentation only; decision unchanged).
    - Realistic ranges follow the preregistered rule, not yet literature sources.
  - **Next: E2** (mean vs median pose scale). Also note for E3: the simple filter fails on correlated errors (drift, shared heading, scale) and on 5° outliers, while the model's own confidence already flags single bad views.
- 2026-10-29: **E2 started** (plan: `experiments/E2.md`, saved verbatim; amendments 1-7).
  - **New VM** `gpucloud`: A100-SXM4-80GB, driver 550 / CUDA 12.4, torch 2.14.1+cu126 (E1 used cu130). It is shared with the rolling-shutter project (tmux `rs3d`, not ours). Every job is capped at 25% of GPU memory and starts only with ≥ 20 GB free.
  - **E2-A literature: gate PASS** (`MapAnything_Exp1/reports/e2_literature.md`).
    - No paper measures our core findings on these models.
    - Partial overlaps to cite: Pow3R (2-view; it *refuses* gross wrong hints, and its confidence tracks agreement), VPGGT (drift-style pose noise on OmniVGGT, poses only, no gate), WorldMirror (wrong focal and depth noise, sometimes worse than no prior), TCO.
    - Literature-realistic sizes replace E1's rule (`configs/e2_lit_realistic.json`). E1's drift levels are **beyond** every sourced drift rate.
    - Datasets for E2-E: ARKitScenes validation split (open download), TUM RGB-D (open), ScanNet++ v2 (signed form from the user and the PI; MapAnything trained on it, only its 30 test scenes are held out).
    - ETH3D is not in MapAnything's or OmniVGGT's training data.
    - Correction: the VPGGT key number is Table 3, 58.89 vs 38.00 (§3 updated).
  - **E2-B audit of E1: gate PASS** (`MapAnything_Exp1/reports/e2_audit.md`).
    - Copying is real behaviour: the output pose comes from the network's pose head (`model.py:1504-1513`, `inference.py:363-378`), with 0 exact-1 absorptions.
    - Under the strict standard (Holm over 132 tests, plus E2 practical thresholds), 16 tests and 9 families remain harmful; 7 families are harmful at literature-realistic sizes.
    - Realistic harms: one camera 15°, compass-level all-camera rotation, shared 10° heading, first camera 15°, scale ×2, depth ×1.5.
    - Harm is the bad camera itself: good-camera pairs change by at most 0.036°.
    - For small position errors the model re-fits about half of the bad view's depth; for large ones it moves the points rigidly.
    - The new VM reproduces E1 within 0.005° rotation and 0.06 pp shape.
  - **E2-C prepared, not frozen.**
    - Code and tests: 79 pass on the laptop (1 skipped: the MapAnything median-swap test, which passes on the VM).
    - Grid: `configs/e2c_full.json` (192 inferences per clip) and `configs/e2c_views.json`.
    - Clips: 24 eval + 8 dev ten-view clips, plus the 4- and 20-view sets. All pass the GT check, and every c0 is byte-identical to E1's clip.
    - The dev run on the 3 dev c0 clips is running. The preregistration draft (12 primary tests) is in `experiments/E2.md`; it is frozen only after the dev check and the user's approval.
- 2026-10-29 (later): **E2-C preregistered at `b232c09` (tag `e2-prereg-C`); eval run started.**
  - The user chose the trimmed grid: seeds 0-1, one-camera rotations up to 30° (Amendment 9). 3,232 eval inferences, about 5.5 h on the shared GPU.
  - The VM code was verified by sha256 against the tag.
  - **E2-D prepared.** OmniVGGT has its own conda env (`omnivggt`, torch 2.7.0+cu126), weights at HF snapshot `29c67625`, and a wrapper `exp1/model_io_omnivggt.py` (CPU smoke test OK).
    - Its checks (`configs/e2d_checks.json`) are queued to start after the E2-C run. They run from a separate copy (`~/e2d`), so the frozen E2-C code is never touched.
    - By its code, OmniVGGT removes the scale of pose and depth hints and ignores the principal point, so some E1 families cannot affect it; the checks test this.
  - **E2-E download postponed by the user's choice:** keep the raw ETH3D data; the VM disk has 7 GB free. Still open: the ScanNet++ access form needs the user's and the PI's signatures.
- 2026-10-30: **E2-C eval run lost with the VM; restarts from zero (Amendment 10 in `experiments/E2.md`).**
  - The user terminated the gpucloud VM during the run (at least 1,227 of 3,232 inferences done, 0 failures). The partial results were not copied back, and no eval metric had been read.
  - Safe on the laptop: all code, all clips (`data/clips*`, image hashes checked against the earlier runs) and every earlier result. Gone with the VM: the raw ETH3D data (only needed to rebuild clips), the conda envs and the weight caches.
  - Next VM: RTX PRO 6000 (Blackwell, needs CUDA ≥ 12.8 wheels). Order: setup → parity gate vs E1 → the E2-C eval run from `e2-prereg-C` → the E2-D OmniVGGT checks. `scripts/setup_omnivggt.sh` added.
- 2026-10-30 (later): **Paid-VM plan (user: use every paid hour).** Amendment 11 in `experiments/E2.md`.
  - Two short sessions with no idle GPU. Session 1: parity gate → E2-C eval → E2-D dev runs (OmniVGGT grid `configs/e2d_full.json`, MapAnything companion `configs/e2d_mapanything.json`). Session 2: the D/E/F eval runs after their preregistrations.
  - Speed-ups: setup installs both models in parallel and the GPU starts as soon as MapAnything is ready (`scripts/vm_setup_e2.sh`, launched from the laptop by `scripts/vm_launch_e2.sh`). Each job is split over up to 4 processes on the one GPU (`exp1/shard.py`; outputs byte-identical to one process in `tests/test_shard.py`, and re-checked on the GPU).
  - E2-F: R5 (VPGGT) cannot run, since no code or weights are public (project-page code link returns 404, checked 2026-10-30). R6 (TCO, github.com/cvlab-stonybrook/TCO) has public code but no licence file; the paper reports 161.5 s per 10-view ETH3D scene at 40 steps (Table 9, TITAN RTX), and it needs a prior on every view. Whether to run it is the user's call.
