# Break-even preview (DESCRIPTIVE, post hoc)

`angle_break_even.csv` was computed on 2026-11-04 by `scripts/e2x/analyse_e2x.py --split eval` from runs that E2 had
already finished and analysed (`results/e2c_eval` for MapAnything, `results/e2d_eval` for OmniVGGT). It is a post-hoc
descriptive view for the "break-even" angle of experiments/E2X.md, not a preregistered test.
Per model and C1 family: scene-mean difference vs the floor (images only for pose, lens and scale hints; clean poses for
depth hints), bootstrap 95% CI over the 10 eval scenes, and whether the CI lies entirely above 0 (hint worse than floor).
