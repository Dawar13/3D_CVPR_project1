# Gate tests only

These files are not a complete project. They need the full repo at commit 610f30f, which holds the data clips, results and model code.

Three files replace existing ones: MapAnything_Exp1/exp1/model_io.py, MapAnything_Exp1/exp1/model_io_omnivggt.py (both only gain code at the end) and MapAnything_Exp1/README.md (gains a note at the top). Everything else is new.

From this folder, with the repo at that commit:

    cp -R MapAnything_Exp1 experiments RUN_GATE_TESTS.md /path/to/3D-CVPR_Exp/

Then follow RUN_GATE_TESTS.md. Where it says "this zip", read "the repo plus these files".
