# E2X instructions: running the rest of extended E2, step by step

Written 2026-11-04. The plan, the preregistration and the reasons are in [`E2X.md`](E2X.md). This file only says **what to do, in what order, what you should see, and what to do when something goes wrong**. A plain-language overview of the whole project is in `MapAnything_Exp1/reports/e2x_plan/e2x_plan.pdf`.

Commands are for **Git Bash on the laptop**, run from the **repo root** (`3D-CVPR_Exp/`), unless a step says "on the VM".

---

## 0. Where things stand

**Done on the laptop (no GPU):**
- A third model, Pi3X:
  - connector `exp1/model_io_pi3x.py`, runner flag `--backend pi3x`;
  - tests `tests/test_pi3x.py`.
- New clips, all checked before any model saw them (21 of 114 excluded, reasons in `results/e2x_clip_check/`):
  - whole-room iPad clips `data/clips_e2x_room/`: 36 eval, 7 dev, 24 cal;
  - consecutive cal clips `data/clips_e2x_cal/`: 26.
- The E3 fix:
  - `exp1/witness.py`, `exp1/e3.py`;
  - runner controls `witness` and `witness_drop`;
  - calibration `exp1/e3_calibrate.py`;
  - tests `tests/test_e3.py`, `tests/test_e2x.py`.
- Configs `configs/e2x_*.json`, made from the frozen E2 configs by `scripts/e2x/make_configs.py`.
- VM scripts:
  - `scripts/vm_launch_e2x.sh` (laptop: build and send);
  - `scripts/vm_setup_e2x.sh` (VM: installs);
  - `scripts/session_queue_e2x.sh` (VM: the GPU jobs);
  - `scripts/session_progress_e2x.py` (VM: progress bars);
  - `scripts/vm_fetch_e2x.sh` (laptop: results back).
- Analysis `scripts/e2x/analyse_e2x.py`.
- Pose AUC@30 for every finished E2 run (`results/*/auc30.csv`).

**Still to do (this file):**
1. Laptop checks.
2. Freeze the plan.
3. Build the bundle.
4. Rent the VM.
5. Send and start.
6. Monitor.
7. Fix problems if any.
8. Fetch the results.
9. Analyse and decide.
10. Prepare session B.

---

## 1. Rules (same as E0–E2)

- **Never edit a config, clip or exclusion list after step 2.** If something must change, write an amendment in `E2X.md` *before* running, and say why.
- **Eval clips are read only by the frozen queue and by the analysis.** Debug on `dev__*` and `cal__*` clips only.
- **`cal` clips are for choosing E3's cut-offs only.** Their numbers are never reported as results.
- **A failed dev gate is not a problem to "fix".** The queue skips the eval run it guards. Report it; do not patch the gate.
- **Every number in a report must come from a results file.** Link the file.
- **Shubh's gate package:** dev clips only, no `--allow_eval`, and no config patches without telling him.

---

## 2. Step 1: laptop checks (about 15 minutes)

```bash
cd MapAnything_Exp1
git status --short | head -50            # expect the new E2X files; papers/ shows as deleted (do not commit that)
python -m pytest -q tests --ignore=tests/test_gate.py -p no:cacheprovider   # ~12 min on the laptop CPU
```
- **Expected:** all pass, with 1 skipped (the MapAnything median-swap test, which needs MapAnything installed and runs on the VM).

```bash
python scripts/e2x/make_configs.py && git diff --stat configs/   # must print no changed config
for d in data/clips_e2x_room data/clips_e2x_cal data/clips_e2e data/clips_e2 data/clips; do
  echo "$d: $(ls $d/*.npz | wc -l) clips"; done
```
- **Expected clip counts:** `clips_e2x_room` 67 (24 cal + 7 dev + 36 eval); `clips_e2x_cal` 26; `clips_e2e` 49; `clips_e2` 32; `clips` 13.
- Excluded clips sit in `excluded/` subfolders, outside every glob.
- If a count differs, stop and find out why before going on.

---

## 3. Step 2: approve and freeze the plan (5 minutes)

1. Read `E2X.md`, in particular:
   - **section 5**: the X-A/X-B tests, the E3 calibration rule, the angles, and the continue/stop rules;
   - **section 3.2**: the clip exclusion rule.
2. If you agree, commit everything and set the tag. The queue runs exactly this commit.

```bash
cd ..   # repo root
git add experiments/E2X.md experiments/E2X_instructions.md \
  MapAnything_Exp1/exp1 MapAnything_Exp1/configs MapAnything_Exp1/scripts MapAnything_Exp1/tests \
  MapAnything_Exp1/results/e2x_survey MapAnything_Exp1/results/e2x_photo_check MapAnything_Exp1/results/e2x_clip_check \
  MapAnything_Exp1/results/e2x_angle_preview MapAnything_Exp1/results/*/auc30.csv MapAnything_Exp1/reports/e2x_plan
git status --short | grep -v '^??' | head -60      # check: no papers/ deletion staged, no data/ files
git commit -m "E2X preregistration: Pi3X connector, room/cal clips (checked), E3 witness controls, configs, VM queue

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag e2x-prereg
git rev-parse --short e2x-prereg
```
- Do **not** push unless you have decided about the repo being public. Pushing is a separate decision.
- After this point **nothing in configs, clips or exclusion lists may change**. See the rules.

---

## 4. Step 3: build the bundle (2 minutes)

```bash
bash MapAnything_Exp1/scripts/vm_launch_e2x.sh build
```
- **Expected:** a folder `../3D-CVPR_Exp_bundle_e2x/` with `e2x_tree.tgz` (code at the tag, plus the E2-E AI-depth record) and `vm_setup_e2x.sh`. The last line says `bundle ready ... (e2x-prereg <commit>)`.
- **"tag e2x-prereg missing":** do step 2 first.
- **"excluded.csv missing":** the clip checks were not run. Do not continue; ask.

---

## 5. Step 4: rent the VM

| Need | Why |
|---|---|
| 1 GPU with **≥ 80 GB**: RTX PRO 6000 (96 GB, used in E2) or A100/H100 80 GB | Up to 4 model processes share the GPU |
| NVIDIA driver **≥ 570** (≥ 580 is best) | Blackwell needs CUDA ≥ 12.8 wheels. The setup picks cu130 / cu128 / cu126 from the driver |
| **≥ 16 CPU cores, ≥ 64 GB RAM** | Setup installs four environments in parallel; the queue runs one process per 16 GB of free GPU memory and per 4 CPUs, at most 4 |
| **≥ 150 GB disk** | Weights (MapAnything ~5 GB, Pi3X ~5.4 GB, OmniVGGT ~5 GB, Depth Anything ~1.3 GB), four conda envs (~20 GB), clips (1.1 GB) and results |
| Ubuntu 22.04 or newer, SSH access | The scripts use `bash`, `tmux`, `wget` and `conda` (installed by the setup) |

- The laptop key `~/.ssh/id_ed25519` is the one rented VMs accept.
- You need the **user, host and port** for the next step.

---

## 6. Step 5: send and start (about 5–15 minutes of laptop time; the rest runs by itself)

```bash
bash MapAnything_Exp1/scripts/vm_launch_e2x.sh send user@HOST PORT
```
What happens, in order:
1. The code is uploaded (~1 MB) and `vm_setup_e2x.sh` starts **at once** in tmux. It creates four conda envs (`mapanything`, `pi3x`, `omnivggt`, `aidepth`) and installs them in parallel.
2. The clips are **streamed** (1.1 GB, no local archive) while the envs install.
3. The tmux session `gpu` waits for MapAnything and the clips, then starts `scripts/session_queue_e2x.sh`.
4. `setup_ai` first regenerates the AI depth on the E2-E clips. The laptop copies have none, because it was made on the E2 VM. It checks the new depth against E2-E's record (`~/logs/mono_verify.txt`), then does the room and cal clips. No job touches the SceneFun3D clips before this is finished.

The command ends by printing the progress command for step 6.

---

## 7. Step 6: monitor

**Progress bars** (from the laptop):
```bash
ssh -p PORT user@HOST 'cd ~/x/MapAnything_Exp1 && ~/miniconda3/envs/mapanything/bin/python scripts/session_progress_e2x.py; tail -8 ~/logs/session.txt'
```

**What else to look at** (on the VM):

| File / command | Meaning |
|---|---|
| `tmux ls` | Sessions `setup_ma`, `setup_px`, `setup_ov`, `setup_ai`, `gpu` |
| `~/logs/ready_*` / `~/logs/FAILED_*` | An env finished installing / failed (see the matching `~/logs/setup_*.txt`) |
| `~/logs/mono_verify.txt` | `N/49 bit-identical to E2-E; same model revision: True`. Different bits are acceptable (another torch/GPU) but note the number. A different revision stops the AI-depth step |
| `~/logs/session.txt` | One line per job, the dev-gate verdicts (`GATE ... PASS/FAIL`), and the markers `ROOM_MA_EVAL_DONE`, `GATE_MA_DONE`, `PX_ETH3D_EVAL_DONE`, `PX_E2E_EVAL_DONE`, `ROOM_PX_EVAL_DONE`, `ROOM_OV_EVAL_DONE`, `SESSION_GPU_DONE` |
| `~/logs/gpu.txt` | Everything the queue printed (errors included) |

**Order of the jobs** (most important first, so stopping early loses only the tail):

| # | Model | Job | Runs | Notes |
|---|---|---|---|---|
| 1 | MapAnything | E3-cal synthetic, ETH3D dev | 414 | |
| 2 | MapAnything | E3-cal synthetic + real, cal clips; real, room dev | 9,237 | waits for the AI depth |
| 3 | MapAnything | room dev → gate → room eval | 147 + 756 | |
| 4 | MapAnything | Shubh's gate tests G0–G2, dev only | ~850 (estimate in G.md) | failures are logged and the queue goes on |
| 5 | Pi3X | checks, ETH3D dev → gate → eval | 39 + 414 + 3,312 | |
| 6 | Pi3X | E2-E dev → gate → eval; room dev → gate → eval | 168 + 861 + 147 + 756 | |
| 7 | Pi3X | E3-cal synthetic + real | 9,364 | |
| 8 | OmniVGGT | room dev → gate → eval | 133 + 684 | |
| — | OmniVGGT | E3-cal (opt-in only) | 9,064 | adds 3.5–4 h; switch on with `RUN_OV_E3CAL=1` |

**Time:** about 2.5–3.5 GPU-hours in total. This is an estimate: the speeds come from the E2 session on the same GPU type, and Pi3X's speed is unknown until its dev run (`E2X.md` §6). The final marker is `SESSION_GPU_DONE`.

---

## 8. Step 7: when something goes wrong

| Symptom | What it means | What to do |
|---|---|---|
| `~/logs/FAILED_mapanything` | MapAnything env or weights failed (the `gpu` session exits) | Read `~/logs/setup_ma.txt`. Fix the install (often a driver/wheel mismatch: see `cu_build` in `vm_setup_e2x.sh`), then `bash ~/vm_setup_e2x.sh ma` and `tmux new -d -s gpu "bash ~/vm_setup_e2x.sh gpu 2>&1 \| tee -a ~/logs/gpu.txt"` |
| `~/logs/FAILED_pi3x` or `FAILED_omnivggt` | That model's jobs are skipped; the rest run | Read `~/logs/setup_px.txt` / `setup_ov.txt`, fix, rerun `bash ~/vm_setup_e2x.sh px` (or `ov`), then rerun the queue (below). Finished jobs are skipped automatically |
| `~/logs/FAILED_mono` or `mono_verify.txt` says "different revision" | AI depth could not be made, or differs from E2-E | The queue still runs, but every `ai_depth` condition fails. Stop and report. Do not change the Depth Anything model |
| `GATE ... FAIL` in `session.txt` | A dev run failed its automatic check, so its eval run was skipped **by design** | Do not patch. Copy the gate lines and the dev run's `failures.jsonl` (if any) and report |
| `failures.jsonl` in a results folder | Some runs raised errors | `head -c 3000 results/<run>/failures.jsonl` and report. Do not rerun with changed settings |
| Out of GPU memory, or the GPU is shared with someone else | Too many processes | Rerun the queue with fewer processes: `SHARDS=1 bash ~/x/MapAnything_Exp1/scripts/session_queue_e2x.sh` (in tmux) |
| SSH connection dropped | Nothing; jobs run in tmux | Reconnect and check progress |
| The queue stopped half way (VM restart, crash) | | Rerun it in tmux: `tmux new -d -s gpu2 "bash ~/x/MapAnything_Exp1/scripts/session_queue_e2x.sh 2>&1 \| tee -a ~/logs/gpu.txt"`. Every job resumes from its shard folders; finished runs are not repeated |
| `GATE_MA had errors` | Shubh's gate code failed on real weights (it was never run on them before) | Expected to be possible; the rest is unaffected. Send `~/logs/gpu.txt` and `results/g_*` to Shubh |
| You want the OmniVGGT E3-cal after all | | After `SESSION_GPU_DONE`: `RUN_OV_E3CAL=1 bash ~/x/MapAnything_Exp1/scripts/session_queue_e2x.sh` (everything else is skipped as done) |

---

## 9. Step 8: fetch the results, then switch the VM off

```bash
bash MapAnything_Exp1/scripts/vm_fetch_e2x.sh user@HOST PORT
grep -E "SESSION_GPU_DONE|GATE|DONE|FAIL" MapAnything_Exp1/results/vm_logs_e2x/session.txt
```
- **Expected folders:** `results/e2x_*` (one per job), `results/e2x_mono/`, `results/g0_dev`, `results/g1_*`, `results/g2_*` and `results/vm_logs_e2x/`.
- Only after `SESSION_GPU_DONE` is in the log **and** the folders are on the laptop, switch the VM off.

---

## 10. Step 9: analyse and decide (laptop, about 1 hour)

**a. Commit the raw results first**, so the analysis is traceable:
```bash
git add MapAnything_Exp1/results/e2x_* MapAnything_Exp1/results/g0_dev MapAnything_Exp1/results/g1_* \
  MapAnything_Exp1/results/g2_* MapAnything_Exp1/results/vm_logs_e2x
git commit -m "E2X session A results (raw)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

**b. E3 calibration** (cal and dev runs only; the script refuses eval runs):
```bash
cd MapAnything_Exp1
python -m exp1.e3_calibrate --runs 'results/e2x_e3cal_*_ma_*' --out results/e2x_e3cal_agg/mapanything
python -m exp1.e3_calibrate --runs 'results/e2x_e3cal_*_px_*' --out results/e2x_e3cal_agg/pi3x
```
- `chosen.json` gives `e3_z`, `e3_tau_depth_shared` and `e3_tau_depth_view`, or **NOT CALIBRATED**.
- `false_action_rates.csv` shows how often E3 acted on correct hints at every setting.
- `detection_rates.csv` shows how often it acts on wrong ones. It is descriptive.

**c. X-A, X-B and the angles:**
```bash
python scripts/e2x/analyse_e2x.py --out results/e2x_agg            # eval runs; --split dev for a dry run
```

| Output | Contents |
|---|---|
| `xa_harm_pi3x.csv`, `xa_behaviour_pi3x.csv`, `xa_*_vs_pi3x.csv` | Preregistered X-A (Pi3X harm and behaviour, side by side with MapAnything and OmniVGGT) |
| `xb_room_tests_<model>.csv`, `xb_e2e_tests_pi3x.csv` | Preregistered X-B (HURTS / HELPS verdicts) |
| `xb_room_costs_*`, `xb_room_remedies_*`, `xb_room_real_errors.csv` | Descriptive |
| `angle_break_even.csv`, `angle_safety_preview.csv` | Descriptive angles A and D |

**d. Shubh's gate tests:** `results/g1_summary/`, `results/g2_summary/`, judged against the thresholds in `experiments/G.md`. Send them to Shubh.

**e. Decide with the rules fixed in `E2X.md` §5, "Continue / stop rules":**

| If | Then |
|---|---|
| Real whole-room pose hints HURT for ≥ 1 model, **or** E3 beats the best of R1–R3 on the cal safety-vs-benefit preview, **and** E3 is calibrated for MapAnything | **GO** for the method paper: write `e3-prereg` and prepare session B (step 10) |
| E3 is NOT CALIBRATED for MapAnything | Analysis paper; E3 becomes a reported negative result |
| G1's oracle dial removes most of the damage | Also prepare the inside-the-network trust gate (learned module, about 1 GPU-day, estimate) |

**f. Record it:** add a dated entry to `context.md` → "Status and decisions log". Include the numbers and links to the files.

---

## 11. Step 10: what comes after (session B, not prepared yet)

1. **Freeze `e3-prereg`:**
   - the cut-offs from step 9b;
   - the E3 eval grid:
     - ETH3D eval families;
     - whole-room iPad eval clips with real hints;
     - E2-E clips with AI depth;
   - remedies to beat: R0–R3, plus R4 if it has run;
   - the success rules: damage removed, harm on good hints, and the safety-vs-benefit plot (`E2X.md` §5 and §7 of the PDF).
2. **Session B:** the frozen E3 eval run. Optionally:
   - the learned trust gate;
   - more AI-depth estimators;
   - the OmniVGGT E3-cal.
3. **Laptop work meanwhile:**
   - R4 (robust bundle adjustment, CPU);
   - optional ADVIO compass study (`E2X.md` §5, angles);
   - the benchmark write-up (angle E).

---

## 12. File map

| What | Where |
|---|---|
| Plan, preregistration, angles, stop rules | `experiments/E2X.md` |
| This runbook | `experiments/E2X_instructions.md` |
| Plain-language overview (PDF) | `MapAnything_Exp1/reports/e2x_plan/e2x_plan.pdf` |
| Third model connector | `MapAnything_Exp1/exp1/model_io_pi3x.py` |
| E3 method / calibration | `exp1/witness.py`, `exp1/e3.py`, `exp1/run_exp1.py` (`run_e3`), `exp1/e3_calibrate.py` |
| Photo check, clip builder, exclusion | `exp1/photo_check.py`, `exp1/arkit_clips.py` (`--mode room`, `--n_cal`), `scripts/e2x/check_clips.py` |
| Configs | `configs/e2x_*.json` (made by `scripts/e2x/make_configs.py`) |
| VM scripts | `scripts/vm_launch_e2x.sh`, `vm_setup_e2x.sh`, `session_queue_e2x.sh`, `session_progress_e2x.py`, `vm_fetch_e2x.sh`, `setup_pi3x.sh` |
| Analysis | `scripts/e2x/analyse_e2x.py`, `exp1/e2def_analysis.py` (frozen E2 code, reused) |
| Survey and checks (inputs to the plan) | `results/e2x_survey/`, `results/e2x_photo_check/`, `results/e2x_clip_check/`, `results/e2x_angle_preview/` |
| Shubh's gate package | `experiments/G.md`, `RUN_GATE_TESTS.md`, `exp1/run_gate.py`, `exp1/g_analysis.py` |
