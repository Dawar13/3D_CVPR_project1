# Exp 1: Environment and reproduction (Deliverable A)

> E0, 2026-10-27. Static reading of the source, then the GPU run (`scripts/e0_phase_c.sh`).
> Every number below comes from the linked file. Smoke numbers are from one dev clip and are a
> pipeline check, **not research results**.

## Hardware / software
- Host, GPU(s), driver, CUDA, torch:
  - rented VM `gpucloud`, Ubuntu 22.04, 14 CPUs, 98 GB RAM;
  - 1× NVIDIA A100-SXM4-80GB, driver 580.126.09 (driver CUDA 13.0);
  - Python 3.12.16, torch 2.14.1+cu130 (CUDA 13.0).

  Sources: `reports/env_dump.txt`, `reports/e0_logs/provenance.txt`.
- Free disk / GPU memory at start: 558 GB free after setup; GPU 0 MiB used (`reports/env_dump.txt`).

## MapAnything
- Repo commit: `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`, pinned by `scripts/setup_server.sh` (`reports/mapanything_commit.txt`).
- Checkpoint: `facebook/map-anything`, CC BY-NC 4.0, HF snapshot `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa` (`reports/e0_logs/provenance.txt`). 1.228 B params, fp32 weights.
- Install:
  - Miniforge in `~/miniforge3`, then `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash scripts/setup_server.sh`.
  - Deviation: pip then resolved torch 2.14.1+**cu130** from PyPI. It works with driver 580 (CUDA 13.0).
  - Tests pass on the VM: 30 passed.

## Verified interface (see README "Things to verify")
Full evidence: `reports/e0_interface.md`.

| Item | Verified? | Where in code (file:line) | Notes |
|---|---|---|---|
| RGB-only = omit keys | yes (static + runtime) | `mapanything/models/mapanything/model.py:725-754, 844-848, 999-1003, 1289-1298` | missing modality = feature × 0 before one LayerNorm; `rgb_only` ran in every smoke run |
| Pose convention (OpenCV cam2world) | yes (static + runtime) | `README.md:249, 385`; `mapanything/utils/inference.py:245-263`; `model.py:716-789` | clean poses cut `rre_deg` by 74.8% vs `rgb_only` on the smoke clip (`7_smoke_check.txt`); a convention error would make poses hurt |
| Partial per-view poses allowed | yes (static + runtime) | `inference.py:178-188`; `model.py:725-754` | both controls ran for 3 seeds and dropped exactly the corrupted view |
| Pose/scale normalisation inside model | yes (static) | `mapanything/utils/geometry.py:1558-1595`, `1523-1555`; `model.py:987-991, 1116-1165` | ẑp = mean ‖t‖ over posed views with t ≠ 0; ẑd = per-view mean depth along the ray; scale features gated per view by `is_metric_scale` |
| Output resolution == 518x336 input | yes (static + runtime) | `mapanything/utils/image.py:44, 459-464`; `mapanything/utils/cropping.py:231-245, 429-447` | no size error in any run |
| Confidence output | yes (runtime) | `inference.py:322-330` | `conf` (1,336,518) float32, finite |

## Minimal official example
- `python scripts/e0_official_example.py --eth3d_root ~/eth3d`, using the first 5 undistorted DSLR images of `delivery_area` (DSC_0675-0679).
- Result: 15 output keys, all floating-point outputs finite; `pts3d` (1,336,518,3), `depth_z` (1,336,518,1), `camera_poses` (1,4,4), `conf` (1,336,518), `metric_scaling_factor` (1,1).
- 1.038 s for 5 views including first-call warm-up; peak 6.44 GB. **PASS** (`reports/e0_official_example.txt`).

## Data (ETH3D, E0 Steps 4-5)
- Raw data: 13 scenes × 3 archives = 16 GB. `dslr_jpg` is needed for the depth undistortion.
- **Download:** all 13 scenes are in. eth3d.net refused TLS from about 14:07 to 15:15 UTC, so `relief`, `relief_2` and `terrains` (eval) were fetched after it came back, on the same VM (`reports/e0_eval3_console.txt`).
- Clips: 10 views at 518×336.
  - dev = `delivery_area`, `facade`, `terrace`;
  - eval = the other 10;
  - logs: `reports/e0_logs/4_clips.txt`, `4_clips_eval.txt`, `4_clips_eval_relabel.txt`, `4_clips_eval3.txt`;
  - copies on the laptop in `data/clips/` (60 MB), git-ignored.
- **GT self-consistency, all 13 clips: PASS** (`reports/e0_logs/5_clip_check_all.txt`, `reports/e0_results/gt_check.csv`):
  - worst pair median relative depth difference 0.02%-0.21%;
  - flat radial profiles;
  - 2-9 of 9 pairs per clip above 5% overlap (sparse GT).
- **Sideways photos:** the `relief` and `relief_2` clips consist entirely of images ETH3D stores rotated by 90° (`reports/e0_logs/4_layout_all.txt`). Their geometry is consistent (both PASS). MapAnything's own converter turns them upright, though (`eth3d.py:210-249`). **Decide before any E1 eval run.**
- **Split bug found and fixed:** `prepare_eth3d` used to compute the dev/eval split over the scenes passed in. A 7-scene run therefore labelled the eval scenes `electro`, `kicker` and `pipes` as `dev` (`reports/e0_eval_console_mislabelled.txt`).
  - Now `datasets.split_of` always splits over the full 13-scene list; test `test_split_is_fixed_over_full_eth3d_list`.
  - The 3 clips were deleted and rebuilt as `eval__`. **No model ever ran on them.**

## Smoke run (`results/smoke`, `results/smoke_ctrl`; clip `dev__delivery_area__c0`)
- **Status:** all rows `ok`, `failures.jsonl` empty, all outputs finite. `zero_noise` vs `clean_pose`: max |Δpts3d| = **0 m** (L = 1.72 m) (`reports/e0_logs/7_smoke_check.txt`).
- **Hints help:** `clean_pose` vs `rgb_only`, `pts_relerr_scale` 0.0174 vs 0.0415 (−58.0%), `rre_deg` 0.184 vs 0.732 (−74.8%).
- **Intrinsics only:** `rgb_K` vs `rgb_only`, `pts_relerr_scale` −59.6%, `rre_deg` +3.9% (roughly equal). Note: `rgb_K` is worse on the metric-scale errors on this clip (`pts_relerr_metric` 0.150 vs 0.127).
- **Visual check** (`results/e0_demo/dev__delivery_area__c0__overview.png`): with clean hints the points and cameras lie on GT; nothing is mirrored, flipped or mis-scaled; `rgb_only` underestimates the trajectory scale (s = 1.116). **PASS.**
- **Runtime and memory:** 0.26 s per 10-view inference after warm-up (median 0.263 s over smoke runs; 0.258 s in determinism runs), peak **8.26 GB** (`7_smoke_check.txt`, `8_det_*.txt`).
- **Determinism:** bit-identical within and across processes; noise floor 0 (`8_det_b.txt`).

## Status
- [x] Reproduced: interface, data and smoke runs behave as expected on the dev split.
- [x] Blocking issue found and fixed: ETH3D GT depth is on the distorted grid; the harness as delivered crashed on it (`reports/e0_interface.md`, "Other findings" 1).
- [x] Split bug found and fixed (see Data above).
- [x] All 13 clips built and GT-checked.
- [ ] Open for E1 (not E0): sideways `relief` / `relief_2` photos (see Data above).
- Figures, 3D point clouds, numbers and an offline results page: `reports/e0_results/` (`README.md` there).
