# Exp 1: Environment and reproduction (Deliverable A)

> Fill this in from `scripts/env_report.sh` output and the smoke run. Nothing below is verified yet.

## Hardware / software
- Host, GPU(s) used, driver, CUDA, torch:
- Free disk / GPU memory at start:

## MapAnything
- Repo commit (`reports/mapanything_commit.txt`):
- Checkpoint id + licence:
- Install commands and any deviations from README:

## Verified interface (see README "Things to verify")
| Item | Verified? | Where in code (file:line) | Notes |
|---|---|---|---|
| RGB-only = omit keys | | | |
| Pose convention (OpenCV cam2world) | | | |
| Partial per-view poses allowed | | | |
| Pose/scale normalisation inside model | | | |
| Output resolution == 518x336 input | | | |
| Confidence output | | | |

## Minimal official example
- Command, sample, output shapes, finiteness:

## Smoke run (`results/smoke`)
- rgb_only / clean_pose / zero_noise: finite? zero_noise max |Δpts3d| vs clean_pose:
- Runtime, peak memory per 10-view clip:

## Status
- [ ] Reproduced  - [ ] Blocking issue:
