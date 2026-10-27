# E0 results: vanilla MapAnything on ETH3D

Everything here is regenerated from the result files by `python scripts/e0_make_results.py`
(CPU only, no model). It shows one dev clip (`dev__delivery_area__c0`) and checks the pipeline; it is
**not a research result**.

| File | What it is |
|---|---|
| `e0_results.html` | The full results page. It works **offline**: the data, images and 3D viewer are all inside, so double-click to open. Drag to rotate the 3D view. |
| `fig_errors.png` | Error bars per run: 3D shape error, 3D error at real size, camera angle error. Lower is better. |
| `fig_3d_views.png` | Perspective views of the 3 reconstructions over the laser truth, with the true cameras. |
| `overview.png` | Bird's-eye and side views with true (black) and predicted (red) cameras. |
| `photos.jpg` | The 10 input photos. |
| `gtcheck_dev__delivery_area__c0.png` | Ground-truth self-check for the smoke clip; all clips are in `../e0_clip_check/`. |
| `3d/*.ply` | Point clouds for the laser truth and the 3 runs (open in MeshLab or CloudCompare). All four share the same pixels, so they overlay point for point. |
| `numbers.csv` | Every number on the page, with its source file. |
| `gt_check.csv` | Ground-truth self-check summary per clip, including the radial profile. |

Sources: `results/smoke/`, `results/smoke_ctrl/`, `results/e0_demo/`, `results/det_b/`,
`reports/e0_clip_check/clip_check.csv`, `data/clips/dev__delivery_area__c0.npz`.
