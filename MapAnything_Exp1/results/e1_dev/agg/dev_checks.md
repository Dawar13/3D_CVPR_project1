| Check | Result | Detail |
|---|---|---|
| no failures, every condition x clip x seed x control present | PASS | failures file: False; missing runs: 0 of 705 |
| gauge_rigid ~ clean_pose (max rel. diff <= 5%) | PASS | max rel. diff 0.0268 |
| scale_all_nonmetric_x1.5 == metric_flag_off (exact) | PASS | max rel. diff 0.00e+00 |
| INFO: scale_all_nonmetric_x1.5 vs clean_pose (metric-flag effect, not gating) | INFO | max rel. diff 1.4809 |
| scale_all_x1.0 and zero_noise equal clean_pose | PASS | max |d pts3d| per clip: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| severity sweeps roughly monotone (most severe >= 95% of mildest, per damage direction) | PASS | ok (checked: depth_noise[0.05->0.1], drift[1->3], focal_all[0.9->0.8], focal_all[1.1->1.2], focal_all_noposes[0.9->0.8], focal_all_noposes[1.1->1.2], noise_rot[0.5->5], noise_trans[0.005->0.05], outlier_frac[0.1->0.4], outlier_rot[5->90], outlier_trans_large[1->10], outlier_trans_small[0.1->0.5], principal_shift[0.05->0.1], scale_all[0.8->0.5], scale_all[1.25->2], shared_heading_bias[2->10]) |
| drop_oracle ~ clean_pose quality on the remaining views (median ratio <= 1.25) | PASS | median ratio 1.003 over 117 runs |
| per-view logs complete; absorption and metric_scale_ratio filled and finite | PASS | per_view rows 7050 (expected 7050); absorption filled on 100% of single-outlier corrupted views; scale ratio finite: True |
