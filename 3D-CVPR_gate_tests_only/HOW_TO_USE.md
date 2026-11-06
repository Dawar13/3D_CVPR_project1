# Gate tests only (SUPERSEDED, do not copy)

> **7 Nov 2026: do not use this folder.** Every file here is already in the main tree (commit `dfe9f1a`), and the gate tests run as step 4 of the E2X session-A queue (`MapAnything_Exp1/scripts/session_queue_e2x.sh`). To run them on their own, follow `RUN_GATE_TESTS.md` at the repo root.
>
> **Copying this folder over the repo breaks E3 on OmniVGGT.** `MapAnything_Exp1/exp1/model_io_omnivggt.py` here is older: it lacks `INTRINSICS_ALONE = False`, so `run_exp1.py:540` would fall back to `True` and E3 would drop poses without intrinsics on OmniVGGT. All other files are byte-identical to the main tree. The folder is kept only as a record of what was delivered.

Original note:

These files are not a complete project. They need the full repo at commit 610f30f, which holds the data clips, results and model code.

Three files replace existing ones: MapAnything_Exp1/exp1/model_io.py, MapAnything_Exp1/exp1/model_io_omnivggt.py (both only gain code at the end) and MapAnything_Exp1/README.md (gains a note at the top). Everything else is new.
