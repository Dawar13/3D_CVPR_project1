# For Dawar: running the gate tests

This zip is the repo at `610f30f` plus new, **uncommitted** code for the gate tests. Plan and pass conditions are in `experiments/G.md`. Please read that first; nothing is preregistered yet.

**Dev clips only.** Please do not use `--allow_eval`.

**Testing so far (laptop CPU):** all 18 tests in `tests/test_gate.py` pass. The OmniVGGT hooks ran on its real aggregator class with tiny random weights and `evo` stubbed out. The MapAnything hooks have only been tested against fake classes with the same call shape. The runner and analyses ran end to end with fake models. **Nothing has run on real weights or a GPU**, and `third_party` was not installed the way it is on the server, so the smoke step may find a bug.

## What is new

| File (under `MapAnything_Exp1/` unless noted) | What it does |
|---|---|
| `experiments/G.md` (repo root) | Plan, proposed pass conditions, kill rules, limits |
| `scripts/g_run_dev.sh` | The only command you need: `smoke`, `full`, `pack` |
| `exp1/run_gate.py` | Runner: applies hints and gates, runs the model, writes results. Refuses non-dev clips by default |
| `exp1/depth_hints.py` | Builds depth hints with one error isolated, and partly bad hints. No model |
| `exp1/g_analysis.py` | Summaries: `decompose`, `verify_mono`, `oracle`, `signal`. No model |
| `exp1/model_io.py`, `exp1/model_io_omnivggt.py` | **Added at the end of each file only:** `set_hint_gate`, `clear_hint_gate`, `TokenCapture`. Existing functions are unchanged, and with no gate set the model is untouched |
| `configs/g1_oracle.json`, `configs/g2_signal.json` | The test grids |
| `tests/test_gate.py` | Unit tests |

The E2 runner `exp1/run_exp1.py` and the frozen tags are not changed. `mono_depth.py` adds `mono_depth` to the dev room clips in place, as it did in E2-E.

## 1. Before you start

- The 11 commits in this zip are not on GitHub, because `Shubhf` has no push access. Please push `main` and the tags `e1-prereg`, `e2-prereg-C`, `e2-prereg-DEF`, or add `Shubhf` as a collaborator. The repo is public, so decide first whether it should stay public.
- `git status` will show `papers/` as deleted because those files are not in the zip's working folder. Run `git checkout -- papers` if you want them back. Do not commit the deletion.
- Commit the new code before the runs, so `run_info.json` records a useful commit.

## 2. Environments

Conda envs, as in `MapAnything_Exp1/scripts/vm_setup_e2.sh`:

| Env | Used for | Setup |
|---|---|---|
| `mapanything` | MapAnything | `scripts/setup_server.sh` |
| `omnivggt` | OmniVGGT | `scripts/setup_omnivggt.sh` |
| `aidepth` | Depth Anything V2 | python 3.11, torch 2.7.0, `transformers>=4.45,<5`, pillow, numpy |

Other names: set `MA_ENV`, `OV_ENV`, `AI_ENV`. All three models are downloaded from Hugging Face on first use.

## 3. Run

```bash
bash MapAnything_Exp1/scripts/g_run_dev.sh smoke
```

This generates AI depth for the dev rooms, runs the unit tests in both envs, then runs one room clip and one ETH3D clip on both models. It stops on the first failure.

**Check the printed `[check]` lines:**

- `gamma1_vs_no_gate` must be about 0. The run stops by itself if it is above 0.001.
- `gamma0_vs_absent` should be much smaller than `hint_vs_absent`. For depth on MapAnything it should be about 0. For OmniVGGT, or pose hints, some difference can come from normalisation. If it is close to `hint_vs_absent`, the gate is not working: stop and report.

Also, `verify_mono` reports how many clips match the E2-E AI depth. Different bits with the same model revision are acceptable, but report the count. A different revision stops the run.

If smoke passes:

```bash
bash MapAnything_Exp1/scripts/g_run_dev.sh full     # resumes if interrupted
bash MapAnything_Exp1/scripts/g_run_dev.sh pack
```

**Size (estimate):** about 850 inferences per model. At E2 speeds, roughly 10 minutes for MapAnything and 1 to 2 hours for OmniVGGT on one process.

## 4. Send back

Send `MapAnything_Exp1/results/g_dev_results.tgz`. The summaries are in `g1_summary/` and `g2_summary/`, plus `g0_dev/depth_decompose.csv`.

If something fails, send `results/g_logs/` and any `failures.jsonl`. Please do not patch the configs to get a run through without telling us, so that the plan stays accurate.
