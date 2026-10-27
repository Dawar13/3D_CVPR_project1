# E1 results: do wrong hints make MapAnything worse than no hints?

Eval split, run once from the frozen commit `71582c1` (tag `e1-prereg`). Plan and amendments: `experiments/E1.md`. Analysis: `python -m exp1.e1_analysis --run results/e1_eval`. All numbers below come from `results/e1_eval/agg/` (copied to `reports/e1_results/`).

## Decision: **Strong GO**

| Signal | Name | Plain meaning | Eval result |
|---|---|---|---|
| S1 | Harm | Wrong hints make it worse than no hints | YES |
| S2 | Spread | One bad hint damages the good views | YES |
| S3 | Blind trust | The model copies bad hints | YES |
| S4 | Silent | The model's confidence gives no warning | NO |
| S5 | Simple fix fails | The easy filter is not enough | YES |
| S6 | Agrees-but-wrong | Correlated / lens / depth errors hurt, or scale goes wrong unnoticed | YES |
| S7 | Realistic | The harm happens at realistic error sizes | YES |

Decision rule (frozen): STOP if not S1; Strong GO if S1, S5, S7 and (S2 or S6); otherwise Weak GO.

## Setup

- 10 eval clips × 10 views (518×336; relief / relief_2 turned upright, 336×518), 2350 runs ok, 0 failed (`results/e1_eval/per_run.csv`).

- MapAnything `3d10cf7a30`, checkpoint `facebook/map-anything`, torch 2.14.1+cu130, GPU NVIDIA A100-SXM4-80GB.

- Scene size L per clip (metres): courtyard 1.60, electro 2.41, kicker 1.70, meadow 6.76, office 0.25, pipes 0.90, playground 0.71, relief_2 1.24, relief 1.19, terrains 0.81.

- Statistics: seeds averaged within a clip; 95% CIs = bootstrap over clips; paired by clip. benefit_lost = (x − ceiling)/(floor − ceiling), ratio of means; 1 = as bad as the floor.

## 1. Per-family summary (worst realistic setting)

| Group | Damage | S1 harm (realistic) | Setting | Floor | Shape BL | Shape vs floor | Rotation BL | Rotation vs floor | Rotation Δ | Scenes worse | vs photos+lens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P: independent pose errors | All cameras slightly turned | YES | 2° | rgb_only | 0.18 | -44.7% | 1.63 | +51.9% | +0.734° | 80% | +134.9% |
| P: independent pose errors | All cameras slightly moved | NO | 0.02×L | rgb_only | 0.01 | -53.7% | 0.01 | -81.9% | -1.158° | 10% | +15.1% |
| P: independent pose errors | One camera turned wrong | YES | 30° | rgb_only | 0.94 | -3.1% | 4.88 | +320.5% | +4.531° | 100% | +550.2% |
| P: independent pose errors | One camera moved a little | NO | 0.25×L | rgb_only | 0.13 | -47.3% | 0.02 | -80.6% | -1.139° | 20% | +31.0% |
| P: independent pose errors | One camera moved far | only extreme | 3×L | rgb_only | 2.13 | +61.6% | 0.11 | -73.6% | -1.041° | 80% | +301.6% |
| P: independent pose errors | Several cameras wrong (15° + 0.25 L each) | YES | 20% | rgb_only | 0.93 | -3.7% | 4.58 | +295.2% | +4.173° | 100% | +511.0% |
| C: correlated pose errors | All distances scaled wrong | YES | ×0.67 | rgb_only | 1.54 | +29.5% | 0.16 | -69.7% | -0.986° | 70% | +221.8% |
| C: correlated pose errors | Odometry drift along the path | YES | level 2.0 | rgb_only | 0.80 | -11.1% | 2.71 | +141.5% | +2.000° | 100% | +273.4% |
| C: correlated pose errors | Same heading error on every camera | YES | 5° | rgb_only | 0.90 | -5.6% | 2.31 | +107.9% | +1.525° | 90% | +221.4% |
| C: correlated pose errors | First (reference) camera wrong | YES | level 1.0 | rgb_only | 4.00 | +163.6% | 2.50 | +123.5% | +1.746° | 100% | +555.1% |
| K: lens errors | Wrong focal length, all photos (with poses) | YES | ×1.1 | clean_pose_noK | 0.96 | -2.2% | 4.10 | +17.6% | +0.046° | 90% | -66.4% |
| K: lens errors | Wrong focal length, all photos (no poses) | YES | ×1.1 | rgb_only | 0.98 | -1.4% | 1.89 | +31.6% | +0.447° | 80% | – |
| K: lens errors | Wrong focal length, one photo | NO | ×0.7 | clean_pose_noK | 0.13 | -44.0% | 2.37 | +7.8% | +0.020° | 70% | -69.2% |
| K: lens errors | Image centre shifted | YES | 0.05×width | clean_pose_noK | 0.25 | -38.2% | 3.38 | +13.5% | +0.035° | 70% | -67.6% |
| D: depth-hint errors | Depth hints scaled wrong (with poses) | YES | ×1.5 | clean_pose | 11.69 | +139.0% | 11.72 | +38.5% | +0.095° | 100% | +170.5% |
| D: depth-hint errors | Depth hints scaled wrong (no poses) | YES | ×0.67 | rgb_K | – | +43.0% | – | -1.2% | -0.011° | 90% | – |
| D: depth-hint errors | Noisy depth hints | NO | σ0.05 | clean_pose | 0.09 | -11.8% | 0.05 | -3.4% | -0.008° | 50% | -74.0% |
| D: depth-hint errors | One depth hint scaled wrong | YES | ×1.5 | clean_pose | 6.99 | +77.8% | 6.67 | +20.4% | +0.050° | 100% | +101.3% |
| M: metric flag | Metric flag forgotten | NO | 1.0 | rgb_only | -0.24 | -67.4% | -0.04 | -85.8% | -1.213° | 0% | -18.9% |

`Shape` = `pts_relerr_scale`, `Rotation` = `rre_deg`. "vs photos+lens" = the same run against `rgb_K` (all poses dropped); negative = the damaged hints are still better than dropping them.

## 2. All S1 settings

| Condition | Metric | Realistic | benefit_lost | vs floor | Δ vs floor | 95% CI | Scenes worse |
|---|---|---|---|---|---|---|---|
| noise_rot_rot_deg=2 | rre_deg | yes | 1.63 | +51.9% | 0.7343 | [0.2988, 1.1417] | 80% |
| noise_rot_rot_deg=5 | rre_deg | no | 4.60 | +297.2% | 4.2025 | [3.7663, 4.6060] | 100% |
| outlier_rot_rot_deg=15 | rre_deg | yes | 2.38 | +114.3% | 1.6154 | [1.1907, 2.0101] | 100% |
| outlier_rot_rot_deg=30 | rre_deg | yes | 4.88 | +320.5% | 4.5311 | [4.1122, 4.9290] | 100% |
| outlier_rot_rot_deg=90 | pts_relerr_scale | no | 2.66 | +90.7% | 0.0644 | [0.0050, 0.1142] | 90% |
| outlier_rot_rot_deg=90 | rre_deg | no | 15.16 | +1169.1% | 16.5294 | [16.0943, 16.9364] | 100% |
| outlier_trans_large_trans_frac=10 | pts_relerr_scale | no | 7.39 | +347.8% | 0.2470 | [0.1549, 0.3702] | 100% |
| outlier_frac_frac=0.1 | rre_deg | yes | 2.40 | +115.9% | 1.6386 | [1.2377, 2.0152] | 100% |
| outlier_frac_frac=0.2 | rre_deg | yes | 4.58 | +295.2% | 4.1734 | [3.7462, 4.5684] | 100% |
| outlier_frac_frac=0.4 | rre_deg | no | 8.57 | +625.3% | 8.8399 | [8.3985, 9.2643] | 100% |
| scale_all_factor=0.5 | pts_relerr_scale | no | 3.34 | +127.4% | 0.0905 | [0.0380, 0.1551] | 80% |
| scale_all_factor=1.5 | pts_relerr_metric | yes | 1.97 | +74.6% | 0.1879 | [0.0432, 0.3121] | 80% |
| scale_all_factor=1.5 | scale_err | yes | 1.86 | +68.2% | 0.1450 | [0.0109, 0.2578] | 80% |
| scale_all_factor=2 | pts_relerr_metric | no | 3.77 | +213.3% | 0.5373 | [0.3996, 0.6704] | 100% |
| scale_all_factor=2 | scale_err | no | 3.04 | +161.0% | 0.3422 | [0.2274, 0.4462] | 90% |
| drift_level=2 | rre_deg | yes | 2.71 | +141.5% | 2.0002 | [1.5720, 2.3947] | 100% |
| drift_level=3 | rre_deg | no | 5.68 | +386.3% | 5.4615 | [4.9969, 5.8824] | 100% |
| shared_heading_bias_rot_deg=5 | rre_deg | yes | 2.31 | +107.9% | 1.5255 | [0.7203, 2.3717] | 90% |
| shared_heading_bias_rot_deg=10 | rre_deg | no | 5.07 | +336.5% | 4.7568 | [3.2215, 6.3150] | 100% |
| anchor_error_level=1 | pts_relerr_scale | yes | 4.00 | +163.6% | 0.1162 | [0.0688, 0.1486] | 90% |
| anchor_error_level=1 | rre_deg | yes | 2.50 | +123.5% | 1.7457 | [1.3241, 2.1337] | 100% |
| focal_all_factor=0.8 | rre_deg | no | 7.42 | +36.5% | 0.0955 | [0.0467, 0.1524] | 90% |
| focal_all_factor=1.1 | rre_deg | yes | 4.10 | +17.6% | 0.0461 | [0.0213, 0.0718] | 80% |
| focal_all_factor=1.2 | rre_deg | no | 8.38 | +42.0% | 0.1097 | [0.0643, 0.1529] | 90% |
| focal_all_noposes_factor=0.8 | pts_relerr_scale | no | 2.67 | +99.8% | 0.0709 | [0.0179, 0.1118] | 90% |
| focal_all_noposes_factor=0.8 | rre_deg | no | 2.63 | +57.5% | 0.8125 | [0.1855, 1.5609] | 80% |
| focal_all_noposes_factor=1.1 | rre_deg | yes | 1.89 | +31.6% | 0.4469 | [0.0574, 0.8955] | 70% |
| focal_all_noposes_factor=1.2 | pts_relerr_scale | no | 2.06 | +63.3% | 0.0449 | [0.0017, 0.0742] | 90% |
| focal_all_noposes_factor=1.2 | rre_deg | no | 3.85 | +100.7% | 1.4237 | [0.5956, 2.2995] | 80% |
| principal_shift_shift=0.05 | rre_deg | yes | 3.38 | +13.5% | 0.0354 | [0.0029, 0.0693] | 70% |
| principal_shift_shift=0.1 | rre_deg | no | 6.94 | +33.8% | 0.0883 | [0.0453, 0.1330] | 90% |
| depth_scale_factor=0.67 | pts_relerr_scale | yes | 7.58 | +85.5% | 0.0277 | [0.0111, 0.0442] | 90% |
| depth_scale_factor=1.5 | pts_relerr_scale | yes | 11.69 | +139.0% | 0.0449 | [0.0254, 0.0648] | 100% |
| depth_scale_factor=1.5 | rre_deg | yes | 11.72 | +38.5% | 0.0950 | [0.0517, 0.1443] | 100% |
| depth_scale_factor=1.5 | pts_relerr_metric | yes | 8.78 | +179.4% | 0.1033 | [0.0688, 0.1392] | 100% |
| depth_scale_factor=1.5 | scale_err | yes | 8.01 | +215.3% | 0.0960 | [0.0679, 0.1264] | 100% |
| depth_scale_noposes_factor=1.5 | pts_relerr_metric | yes | 2.52 | +76.8% | 0.1663 | [0.0762, 0.2384] | 90% |
| depth_scale_noposes_factor=1.5 | scale_err | yes | 2.62 | +77.9% | 0.1400 | [0.0714, 0.1985] | 90% |
| depth_one_scale_factor=1.5 | pts_relerr_scale | yes | 6.99 | +77.8% | 0.0252 | [0.0170, 0.0329] | 100% |
| depth_one_scale_factor=1.5 | rre_deg | yes | 6.67 | +20.4% | 0.0502 | [0.0110, 0.0933] | 90% |

## 3. Spread to good views (S2), realistic settings

| Condition | Per-view metric | Rise on uncorrupted views | 95% CI of Δ |
|---|---|---|---|
| outlier_rot_rot_deg=5 | view_pts_err_scale | +3.0% | [-0.0004, 0.0024] |
| outlier_rot_rot_deg=5 | pred_rot_err | +2.7% | [0.0016, 0.0112] |
| outlier_rot_rot_deg=15 | view_pts_err_scale | +3.9% | [-0.0003, 0.0029] |
| outlier_rot_rot_deg=15 | pred_rot_err | +7.8% | [0.0077, 0.0285] |
| outlier_rot_rot_deg=30 | view_pts_err_scale | +2.7% | [-0.0004, 0.0019] |
| outlier_rot_rot_deg=30 | pred_rot_err | +5.7% | [-0.0018, 0.0356] |
| outlier_trans_small_trans_frac=0.1 | view_pts_err_scale | +0.7% | [-0.0001, 0.0005] |
| outlier_trans_small_trans_frac=0.1 | pred_rot_err | +0.0% | [-0.0026, 0.0027] |
| outlier_trans_small_trans_frac=0.25 | view_pts_err_scale | +3.6% | [0.0006, 0.0016] |
| outlier_trans_small_trans_frac=0.25 | pred_rot_err | +2.3% | [-0.0018, 0.0113] |
| outlier_trans_large_trans_frac=1 | view_pts_err_scale | +8.6% | [0.0010, 0.0048] |
| outlier_trans_large_trans_frac=1 | pred_rot_err | +3.8% | [-0.0024, 0.0187] |
| outlier_trans_large_trans_frac=3 | view_pts_err_scale | +13.0% | [0.0017, 0.0070] |
| outlier_trans_large_trans_frac=3 | pred_rot_err | +7.1% | [0.0015, 0.0292] |
| outlier_frac_frac=0.1 | view_pts_err_scale | +5.1% | [-0.0003, 0.0033] |
| outlier_frac_frac=0.1 | pred_rot_err | +3.9% | [-0.0007, 0.0197] |
| outlier_frac_frac=0.2 | view_pts_err_scale | +4.9% | [-0.0023, 0.0046] |
| outlier_frac_frac=0.2 | pred_rot_err | +20.4% | [0.0206, 0.0741] |

## 4. Blind trust (S3) and confidence (S4)

| Condition | Median absorption | IQR | Corrupted views |
|---|---|---|---|
| outlier_rot_rot_deg=5 | 0.86 | [0.85, 0.90] | 30 |
| outlier_rot_rot_deg=15 | 0.95 | [0.93, 0.96] | 30 |
| outlier_rot_rot_deg=30 | 0.96 | [0.94, 0.97] | 30 |
| outlier_rot_rot_deg=90 | 0.99 | [0.98, 0.99] | 30 |
| outlier_trans_small_trans_frac=0.1 | 0.88 | [0.78, 0.97] | 30 |
| outlier_trans_small_trans_frac=0.25 | 0.91 | [0.86, 0.95] | 30 |
| outlier_trans_small_trans_frac=0.5 | 0.99 | [0.96, 1.01] | 30 |
| outlier_trans_large_trans_frac=1 | 1.00 | [0.97, 1.03] | 30 |
| outlier_trans_large_trans_frac=3 | 0.97 | [0.93, 0.99] | 30 |
| outlier_trans_large_trans_frac=10 | 0.97 | [0.93, 0.99] | 30 |
| outlier_frac_frac=0.1 | 0.94 | [0.93, 0.97] | 30 |
| outlier_frac_frac=0.2 | 0.95 | [0.93, 0.97] | 60 |
| outlier_frac_frac=0.4 | 0.95 | [0.93, 0.96] | 120 |
| focal_all_factor=0.8 | 0.86 | [0.74, 0.88] | 100 |
| focal_all_factor=0.9 | 0.90 | [0.79, 0.94] | 100 |
| focal_all_factor=1.1 | 0.86 | [0.78, 0.94] | 100 |
| focal_all_factor=1.2 | 0.85 | [0.80, 0.92] | 100 |
| focal_all_noposes_factor=0.8 | 0.87 | [0.83, 0.89] | 100 |
| focal_all_noposes_factor=0.9 | 0.89 | [0.85, 0.93] | 100 |
| focal_all_noposes_factor=1.1 | 0.88 | [0.80, 0.93] | 100 |
| focal_all_noposes_factor=1.2 | 0.85 | [0.82, 0.94] | 100 |
| focal_one_factor=0.7 | 0.55 | [0.46, 0.68] | 30 |
| focal_one_factor=1.3 | 0.67 | [0.56, 0.75] | 30 |
| depth_scale_factor=0.67 | 0.47 | [0.33, 0.57] | 20 |
| depth_scale_factor=1.5 | 0.50 | [0.29, 0.60] | 20 |
| depth_scale_noposes_factor=0.67 | 0.51 | [0.44, 0.55] | 20 |
| depth_scale_noposes_factor=1.5 | 0.75 | [0.69, 0.93] | 20 |
| depth_one_scale_factor=1.5 | 0.48 | [0.31, 0.57] | 10 |

Absorption = model error ÷ hint error on the corrupted view (1 = copies the bad hint, 0 = ignores it).

| Family | AUROC confidence | AUROC disagree (rot) | AUROC disagree (dir) | Runs |
|---|---|---|---|---|
| outlier_rot | 0.94 | 0.99 | 0.52 | 120 |
| outlier_trans_small | 0.80 | 0.42 | 0.79 | 90 |
| outlier_trans_large | 0.97 | 0.49 | 0.98 | 90 |
| outlier_frac | 0.90 | 1.00 | 0.81 | 90 |
| focal_one | 0.94 | 0.51 | 0.52 | 60 |
| POOLED (S4) | 0.91 | 0.71 | 0.73 | 450 |

Within-run AUROC: chance a clean view scores "cleaner" than the corrupted view of the same run (confidence: higher; disagreement with images-only: lower). S4 is YES when the pooled confidence AUROC ≤ 0.65.

## 5. Scale (C1, D1, M1)

| Condition | Injected factor | Metric flag | Predicted size ÷ clean-hint size | 95% CI | Filter: poses dropped |
|---|---|---|---|---|---|
| zero_noise | 1 | on | 1.000 | [1.000, 1.000] | 0.00 |
| gauge_rigid | 1 | on | 1.000 | [1.000, 1.000] | 0.00 |
| scale_all_nonmetric_x1.5 | 1.5 | off | 1.157 | [1.058, 1.277] | 0.00 |
| scale_all_x1.0 | 1 | on | 1.000 | [1.000, 1.000] | 0.00 |
| scale_all_factor=0.5 | 0.5 | on | 0.653 | [0.544, 0.778] | 0.70 |
| scale_all_factor=0.67 | 0.67 | on | 0.761 | [0.693, 0.837] | 0.70 |
| scale_all_factor=0.8 | 0.8 | on | 0.852 | [0.816, 0.891] | 0.70 |
| scale_all_factor=1.25 | 1.25 | on | 1.184 | [1.149, 1.215] | 0.70 |
| scale_all_factor=1.5 | 1.5 | on | 1.378 | [1.315, 1.433] | 0.70 |
| scale_all_factor=2 | 2 | on | 1.685 | [1.567, 1.798] | 0.70 |
| depth_scale_factor=0.67 | 0.67 | on | 0.921 | [0.885, 0.956] | 0.00 |
| depth_scale_factor=1.5 | 1.5 | on | 1.108 | [1.078, 1.143] | 0.00 |
| depth_scale_noposes_factor=0.67 | 0.67 | on | 0.879 | [0.847, 0.917] | 0.00 |
| depth_scale_noposes_factor=1.5 | 1.5 | on | 1.326 | [1.277, 1.372] | 0.00 |
| metric_flag_off | 1 | off | 1.157 | [1.058, 1.277] | 0.00 |

## 6. Fixes: oracle vs simple filter (S5)

| Condition | Metric | Oracle | Damaged | Oracle | Filter | Filter recovery | Poses dropped |
|---|---|---|---|---|---|---|---|
| outlier_rot_rot_deg=5 | pts_relerr_scale | drop_oracle | 0.0371 | 0.0336 | 0.0373 | -0.05 | 0.70 |
| outlier_rot_rot_deg=5 | rre_deg | drop_oracle | 1.0750 | 0.3656 | 1.0483 | 0.04 | 0.70 |
| outlier_rot_rot_deg=15 | pts_relerr_scale | drop_oracle | 0.0479 | 0.0338 | 0.0342 | 0.97 | 1.63 |
| outlier_rot_rot_deg=15 | rre_deg | drop_oracle | 3.0292 | 0.3647 | 0.4222 | 0.98 | 1.63 |
| outlier_rot_rot_deg=30 | pts_relerr_scale | drop_oracle | 0.0689 | 0.0333 | 0.0338 | 0.99 | 1.63 |
| outlier_rot_rot_deg=30 | rre_deg | drop_oracle | 5.9449 | 0.3786 | 0.4271 | 0.99 | 1.63 |
| outlier_rot_rot_deg=90 | pts_relerr_scale | drop_oracle | 0.1354 | 0.0334 | 0.0341 | 0.99 | 1.53 |
| outlier_rot_rot_deg=90 | rre_deg | drop_oracle | 17.9432 | 0.3770 | 0.4209 | 1.00 | 1.53 |
| outlier_trans_small_trans_frac=0.1 | pts_relerr_scale | drop_oracle | 0.0330 | 0.0335 | 0.0334 | – | 0.63 |
| outlier_trans_small_trans_frac=0.1 | rre_deg | drop_oracle | 0.2545 | 0.3613 | 0.3026 | – | 0.63 |
| outlier_trans_small_trans_frac=0.25 | pts_relerr_scale | drop_oracle | 0.0374 | 0.0335 | 0.0360 | 0.37 | 0.93 |
| outlier_trans_small_trans_frac=0.25 | rre_deg | drop_oracle | 0.2747 | 0.3463 | 0.3342 | – | 0.93 |
| outlier_trans_small_trans_frac=0.5 | pts_relerr_scale | drop_oracle | 0.0453 | 0.0340 | 0.0395 | 0.51 | 1.07 |
| outlier_trans_small_trans_frac=0.5 | rre_deg | drop_oracle | 0.2790 | 0.3566 | 0.3358 | – | 1.07 |
| outlier_trans_large_trans_frac=1 | pts_relerr_scale | drop_oracle | 0.0640 | 0.0337 | 0.0371 | 0.89 | 1.50 |
| outlier_trans_large_trans_frac=1 | rre_deg | drop_oracle | 0.3082 | 0.3817 | 0.4192 | – | 1.50 |
| outlier_trans_large_trans_frac=3 | pts_relerr_scale | drop_oracle | 0.1148 | 0.0336 | 0.0340 | 1.00 | 1.57 |
| outlier_trans_large_trans_frac=3 | rre_deg | drop_oracle | 0.3726 | 0.3256 | 0.3635 | 0.19 | 1.57 |
| outlier_trans_large_trans_frac=10 | pts_relerr_scale | drop_oracle | 0.3180 | 0.0337 | 0.0344 | 1.00 | 1.67 |
| outlier_trans_large_trans_frac=10 | rre_deg | drop_oracle | 0.3827 | 0.3895 | 0.4320 | – | 1.67 |
| outlier_frac_frac=0.1 | pts_relerr_scale | drop_oracle | 0.0517 | 0.0338 | 0.0345 | 0.96 | 1.67 |
| outlier_frac_frac=0.1 | rre_deg | drop_oracle | 3.0524 | 0.3517 | 0.3964 | 0.98 | 1.67 |
| outlier_frac_frac=0.2 | pts_relerr_scale | drop_oracle | 0.0684 | 0.0354 | 0.0358 | 0.99 | 2.53 |
| outlier_frac_frac=0.2 | rre_deg | drop_oracle | 5.5872 | 0.4593 | 0.5101 | 0.99 | 2.53 |
| outlier_frac_frac=0.4 | pts_relerr_scale | drop_oracle | 0.1007 | 0.0367 | 0.0386 | 0.97 | 4.27 |
| outlier_frac_frac=0.4 | rre_deg | drop_oracle | 10.2537 | 0.6514 | 0.8395 | 0.98 | 4.27 |
| scale_all_factor=0.5 | pts_relerr_scale | rgb_K (drop all poses) | 0.1615 | 0.0286 | 0.1681 | -0.05 | 0.70 |
| scale_all_factor=0.5 | rre_deg | rgb_K (drop all poses) | 0.5318 | 0.9144 | 0.6042 | – | 0.70 |
| scale_all_factor=0.5 | pts_relerr_metric | rgb_K (drop all poses) | 0.4337 | 0.2165 | 0.4340 | -0.00 | 0.70 |
| scale_all_factor=0.67 | pts_relerr_scale | rgb_K (drop all poses) | 0.0920 | 0.0286 | 0.0954 | -0.05 | 0.70 |
| scale_all_factor=0.67 | rre_deg | rgb_K (drop all poses) | 0.4280 | 0.9144 | 0.4914 | – | 0.70 |
| scale_all_factor=0.67 | pts_relerr_metric | rgb_K (drop all poses) | 0.2815 | 0.2165 | 0.2832 | -0.03 | 0.70 |
| scale_all_factor=0.8 | pts_relerr_scale | rgb_K (drop all poses) | 0.0593 | 0.0286 | 0.0610 | -0.06 | 0.70 |
| scale_all_factor=0.8 | rre_deg | rgb_K (drop all poses) | 0.3401 | 0.9144 | 0.4056 | – | 0.70 |
| scale_all_factor=0.8 | pts_relerr_metric | rgb_K (drop all poses) | 0.1700 | 0.2165 | 0.1725 | – | 0.70 |
| scale_all_factor=1.25 | pts_relerr_scale | rgb_K (drop all poses) | 0.0276 | 0.0286 | 0.0279 | – | 0.70 |
| scale_all_factor=1.25 | rre_deg | rgb_K (drop all poses) | 0.1985 | 0.9144 | 0.2599 | – | 0.70 |
| scale_all_factor=1.25 | pts_relerr_metric | rgb_K (drop all poses) | 0.2379 | 0.2165 | 0.2393 | -0.06 | 0.70 |
| scale_all_factor=1.5 | pts_relerr_scale | rgb_K (drop all poses) | 0.0372 | 0.0286 | 0.0383 | -0.13 | 0.70 |
| scale_all_factor=1.5 | rre_deg | rgb_K (drop all poses) | 0.2306 | 0.9144 | 0.3180 | – | 0.70 |
| scale_all_factor=1.5 | pts_relerr_metric | rgb_K (drop all poses) | 0.4398 | 0.2165 | 0.4406 | -0.00 | 0.70 |
| scale_all_factor=2 | pts_relerr_scale | rgb_K (drop all poses) | 0.0851 | 0.0286 | 0.0928 | -0.14 | 0.70 |
| scale_all_factor=2 | rre_deg | rgb_K (drop all poses) | 0.4210 | 0.9144 | 0.5091 | – | 0.70 |
| scale_all_factor=2 | pts_relerr_metric | rgb_K (drop all poses) | 0.7891 | 0.2165 | 0.7801 | 0.02 | 0.70 |
| drift_level=1 | pts_relerr_scale | rgb_K (drop all poses) | 0.0421 | 0.0286 | 0.0412 | 0.06 | 0.77 |
| drift_level=1 | rre_deg | rgb_K (drop all poses) | 1.6867 | 0.9144 | 1.6548 | 0.04 | 0.77 |
| drift_level=2 | pts_relerr_scale | rgb_K (drop all poses) | 0.0631 | 0.0286 | 0.0595 | 0.10 | 0.97 |
| drift_level=2 | rre_deg | rgb_K (drop all poses) | 3.4140 | 0.9144 | 3.1901 | 0.09 | 0.97 |
| drift_level=3 | pts_relerr_scale | rgb_K (drop all poses) | 0.1145 | 0.0286 | 0.0609 | 0.62 | 4.87 |
| drift_level=3 | rre_deg | rgb_K (drop all poses) | 6.8753 | 0.9144 | 2.9339 | 0.66 | 4.87 |
| shared_heading_bias_rot_deg=2 | pts_relerr_scale | rgb_K (drop all poses) | 0.0404 | 0.0286 | 0.0405 | -0.01 | 0.70 |
| shared_heading_bias_rot_deg=2 | rre_deg | rgb_K (drop all poses) | 1.1218 | 0.9144 | 1.1387 | -0.08 | 0.70 |
| shared_heading_bias_rot_deg=5 | pts_relerr_scale | rgb_K (drop all poses) | 0.0671 | 0.0286 | 0.0658 | 0.03 | 0.80 |
| shared_heading_bias_rot_deg=5 | rre_deg | rgb_K (drop all poses) | 2.9393 | 0.9144 | 2.9187 | 0.01 | 0.80 |
| shared_heading_bias_rot_deg=10 | pts_relerr_scale | rgb_K (drop all poses) | 0.1237 | 0.0286 | 0.0826 | 0.43 | 5.20 |
| shared_heading_bias_rot_deg=10 | rre_deg | rgb_K (drop all poses) | 6.1706 | 0.9144 | 4.2437 | 0.37 | 5.20 |

## 7. Signal evidence
```json
{
 "S1": {
  "value": true,
  "evidence": [
   {
    "condition": "noise_rot_rot_deg=2",
    "metric": "rre_deg",
    "benefit_lost": 1.6289864474112492,
    "frac_worse_than_floor": 0.8,
    "delta_ci_lo": 0.2988056830072171,
    "realistic": true
   },
   {
    "condition": "noise_rot_rot_deg=5",
    "metric": "rre_deg",
    "benefit_lost": 4.599962981014102,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 3.7663437645514755,
    "realistic": false
   },
   {
    "condition": "outlier_rot_rot_deg=15",
    "metric": "rre_deg",
    "benefit_lost": 2.3838382494609083,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 1.1906881674864989,
    "realistic": true
   },
   {
    "condition": "outlier_rot_rot_deg=30",
    "metric": "rre_deg",
    "benefit_lost": 4.881497445580003,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 4.112187708476324,
    "realistic": true
   },
   {
    "condition": "outlier_rot_rot_deg=90",
    "metric": "pts_relerr_scale",
    "benefit_lost": 2.6646042285781055,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.004962619810593878,
    "realistic": false
   },
   {
    "condition": "outlier_rot_rot_deg=90",
    "metric": "rre_deg",
    "benefit_lost": 15.159591446933742,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 16.094326310416537,
    "realistic": false
   },
   {
    "condition": "outlier_trans_large_trans_frac=10",
    "metric": "pts_relerr_scale",
    "benefit_lost": 7.3850444956591685,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 0.15493843086068476,
    "realistic": false
   },
   {
    "condition": "outlier_frac_frac=0.1",
    "metric": "rre_deg",
    "benefit_lost": 2.4036532894852884,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 1.2376555165503609,
    "realistic": true
   },
   {
    "condition": "outlier_frac_frac=0.2",
    "metric": "rre_deg",
    "benefit_lost": 4.575050135021321,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 3.7461923101053505,
    "realistic": true
   },
   {
    "condition": "outlier_frac_frac=0.4",
    "metric": "rre_deg",
    "benefit_lost": 8.57252507658048,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 8.398532139082342,
    "realistic": false
   },
   {
    "condition": "scale_all_factor=0.5",
    "metric": "pts_relerr_scale",
    "benefit_lost": 3.33864843672644,
    "frac_worse_than_floor": 0.8,
    "delta_ci_lo": 0.03796843353019064,
    "realistic": false
   },
   {
    "condition": "scale_all_factor=1.5",
    "metric": "pts_relerr_metric",
    "benefit_lost": 1.9670632920219524,
    "frac_worse_than_floor": 0.8,
    "delta_ci_lo": 0.04322344946339605,
    "realistic": true
   },
   {
    "condition": "scale_all_factor=1.5",
    "metric": "scale_err",
    "benefit_lost": 1.8635560806825227,
    "frac_worse_than_floor": 0.8,
    "delta_ci_lo": 0.010895249693438438,
    "realistic": true
   },
   {
    "condition": "scale_all_factor=2",
    "metric": "pts_relerr_metric",
    "benefit_lost": 3.7651324798662364,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 0.3995925828848276,
    "realistic": false
   },
   {
    "condition": "scale_all_factor=2",
    "metric": "scale_err",
    "benefit_lost": 3.037995880753176,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.22741673549748814,
    "realistic": false
   },
   {
    "condition": "drift_level=2",
    "metric": "rre_deg",
    "benefit_lost": 2.7134371918501468,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 1.5720067165549776,
    "realistic": true
   },
   {
    "condition": "drift_level=3",
    "metric": "rre_deg",
    "benefit_lost": 5.678467427947417,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 4.9969034797566305,
    "realistic": false
   },
   {
    "condition": "shared_heading_bias_rot_deg=5",
    "metric": "rre_deg",
    "benefit_lost": 2.306778999273643,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.7202694235423188,
    "realistic": true
   },
   {
    "condition": "shared_heading_bias_rot_deg=10",
    "metric": "rre_deg",
    "benefit_lost": 5.074840927902033,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 3.221480650345937,
    "realistic": false
   },
   {
    "condition": "anchor_error_level=1",
    "metric": "pts_relerr_scale",
    "benefit_lost": 4.002733182155257,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.06877406729368729,
    "realistic": true
   },
   {
    "condition": "anchor_error_level=1",
    "metric": "rre_deg",
    "benefit_lost": 2.495409950592027,
    "frac_worse_than_floor": 1.0,
    "delta_ci_lo": 1.3241333042498027,
    "realistic": true
   },
   {
    "condition": "focal_all_factor=0.8",
    "metric": "rre_deg",
    "benefit_lost": 7.420983135403699,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.04668029679136065,
    "realistic": false
   },
   {
    "condition": "focal_all_factor=1.1",
    "metric": "rre_deg",
    "benefit_lost": 4.099348907660609,
    "frac_worse_than_floor": 0.8,
    "delta_ci_lo": 0.021309939786919592,
    "realistic": true
   },
   {
    "condition": "focal_all_factor=1.2",
    "metric": "rre_deg",
    "benefit_lost": 8.375809087833087,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.0643297384314797,
    "realistic": false
   },
   {
    "condition": "focal_all_noposes_factor=0.8",
    "metric": "pts_relerr_scale",
    "benefit_lost": 2.6700448015445977,
    "frac_worse_than_floor": 0.9,
    "delta_ci_lo": 0.017888105319477834,
    "realistic": false
   },
   {
    "condition": "focal_all_noposes_factor=0.8",
    "metric": "rre_deg",
    "benefit_lost": 2.6269370584385654,
    "frac_worse_than_floor": 0.8,
    "delta_ci_lo": 0.18545385557556046,
    "realistic": false
   },
   {
    "condition": "focal_all_noposes_factor=1.1",
    "metric": "rre_deg",
    "benefit_lost": 1.894842124801192,
    "frac_worse_than_floor": 0.7,
    "delta_ci_lo": 0.05738213564029238,
    "realistic": true
   },
   {
    "condition": "focal_all_noposes_factor=1.2",
    "metric": "pts_relerr_scale",
    "benefit_lost"
```

## Checks on the eval run
| Check | Result | Detail |
|---|---|---|
| no failures, every condition x clip x seed x control present | PASS | failures file: False; missing runs: 0 of 2350 |
| gauge_rigid ~ clean_pose (max rel. diff <= 5%) | FAIL | max rel. diff 0.0943 |
| scale_all_nonmetric_x1.5 == metric_flag_off (exact) | PASS | max rel. diff 0.00e+00 |
| INFO: scale_all_nonmetric_x1.5 vs clean_pose (metric-flag effect, not gating) | INFO | max rel. diff 0.6903 |
| scale_all_x1.0 and zero_noise equal clean_pose | PASS | max |d pts3d| per clip: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| severity sweeps roughly monotone (most severe >= 95% of mildest, per damage direction) | PASS | ok (checked: depth_noise[0.05->0.1], drift[1->3], focal_all[0.9->0.8], focal_all[1.1->1.2], focal_all_noposes[0.9->0.8], focal_all_noposes[1.1->1.2], noise_rot[0.5->5], noise_trans[0.005->0.05], outlier_frac[0.1->0.4], outlier_rot[5->90], outlier_trans_large[1->10], outlier_trans_small[0.1->0.5], principal_shift[0.05->0.1], scale_all[0.8->0.5], scale_all[1.25->2], shared_heading_bias[2->10]) |
| drop_oracle ~ clean_pose quality on the remaining views (median ratio <= 1.25) | PASS | median ratio 1.024 over 390 runs |
| per-view logs complete; absorption and metric_scale_ratio filled and finite | PASS | per_view rows 23500 (expected 23500); absorption filled on 100% of single-outlier corrupted views; scale ratio finite: True |

## Figures
`reports/e1_results/`: fig_dose_response.png (headline), fig_scene_heatmap.png, fig_contamination.png, fig_absorption.png, fig_scale.png; demo_*.png = dev-clip visual examples (qualitative).

## Limitations
- One clip of 10 views per scene, 10 eval scenes; CIs are over scenes.
- Realistic ranges follow the preregistered rule (amendment 2), not yet literature-sourced (E1.md §6).
- ETH3D GT depth is sparse (≈ 6-35% of pixels per clip); depth hints are that sparse GT.
- Single model (MapAnything); OmniVGGT replication is E4.

## Reproduce
```
git checkout e1-prereg
python -m exp1.run_exp1 --config configs/e1_full.json --clips 'data/clips/eval__*.npz' --out results/e1_eval
python -m exp1.e1_analysis --run results/e1_eval --checks
python scripts/e1_make_report.py
```
