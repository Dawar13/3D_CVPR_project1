# Scale-fit caveat (docs/VISION.md section 2): does it touch E1's spread result S2?

Checked 2026-11-07 on the laptop from saved files; no model run.

- Caveat: per-view shape errors (`view_pts_err_scale`) use one clip-wide scale fitted over all views, the corrupted
  view included (`exp1/metrics.py`, `per_view`). Rotation errors do not use that scale.
- E1's S2 (spread to uncorrupted views) is decided by `exp1/e1_analysis.py` lines 284-286: realistic settings only,
  rel_rise >= 0.2 and CI above 0, over both metrics in `results/e1_eval/agg/contamination.csv`.
- The only row that passed (`results/e1_eval/agg/signals.json`, S2.evidence): outlier_frac_frac=0.2,
  metric `pred_rot_err` (rotation of the uncorrupted views), rel_rise 0.2041,
  CI low 0.0206.
- Shape rows (`view_pts_err_scale`) passing the rule without the realistic filter: 1 of 13, the 10x GPS jump
  (`outlier_trans_large_trans_frac=10`, rel_rise 0.389, CI low 0.005), which is not realistic, not part of S2 and not a
  reported claim. It is also the case where a clip-wide scale fit is most exposed to the corrupted view, so it should
  not be quoted without a scale refitted on the uncorrupted views.

**Conclusion:** S2 rests on a rotation readout, which the scale fit does not affect. The caveat does not change S2.
Shape-based spread claims (none were made) would need the scale refitted on uncorrupted views, or
`view_shape_err_sim3` (saved in the E2 runs, not in E1's).
