# E2-E data plan: real device hints (SceneFun3D on ARKitScenes Validation)

Research notes from 2026-10-30. Sources were opened by a research agent; the small files it fetched (READMEs, scripts, split CSVs, sample `.traj` files, the size survey) are kept outside the repo. Items marked *unverified* must be checked before the E2-E preregistration. Nothing here is a result.

## Decision
Use the **SceneFun3D test split**, not raw ARKitScenes.
- **Raw ARKitScenes has no ground-truth pose file.** Each video has one trajectory, `lowres_wide.traj`, which holds the on-device ARKit poses (ARKitScenes `download_data.py:13-15,43-45`; `DATA.md:61-66`; paper §1, §3.1). GT exists only implicitly, in `highres_depth`.
  - *Unverified:* a search snippet of ARKitScenes issue #15 says the released poses are laser-registered. Issues are now disabled, so this could not be checked.
- **SceneFun3D** (github.com/SceneFun3D/scenefun3d; file structure at scenefun3d.github.io/documentation/dataset/file-structure) adds per video:
  - `hires_poses.traj`: SuperGlue + COLMAP poses in the laser-scan frame;
  - `hires_wide` (RGB, 10 FPS) and `hires_depth` (rendered from the Faro laser scan);
  - `<vid>_transform.npy`: laser-scan frame → ARKit frame;
  - `lowres_poses.traj`: the same ARKit trajectory as ARKitScenes' `lowres_wide.traj` (SceneFun3D `download_data.py:121-124`).
- **Held out:** SceneFun3D `test_set.csv` has 220 videos in 85 visits, all from ARKitScenes **Validation** (the downloader maps Validation→test, `download_data.py:179,205`). OmniVGGT trained on ARKitScenes (E2-A, `reports/e2_literature.md`).
  - *Still to confirm:* that OmniVGGT used only the Training split.
- **Licence:** SceneFun3D data CC-BY-4.0 plus the ARKitScenes terms, which are non-commercial (ARKitScenes `LICENSE:7`). Code MIT. Research use is fine.

## Hints and ground truth per frame
| | Device hint | Ground truth |
|---|---|---|
| Pose | `lowres_poses.traj` (ARKit), at the `hires_wide` timestamps (an exact subset of the `.traj` timestamps) | `hires_poses.traj` (COLMAP, laser frame) |
| Intrinsics | ARKit `lowres_wide_intrinsics` × 7.5 (256→1920 px); `hires_wide_intrinsics` may be COLMAP-refined (*unverified*) | `hires_wide_intrinsics` |
| Depth | `lowres_depth` (iPad LiDAR, 256×192, uint16 mm) plus `confidence` (0–2) | `hires_depth` (laser render) |

**Formats** (ARKitScenes `DATA.md`; official loader `tenFpsDataLoader.py`):
- `.traj` rows: timestamp, angle-axis rotation (rad), translation (m). Each row is world→camera; invert it for cam2world (`tenFpsDataLoader.py:13-43`).
- OpenCV camera axes, gravity z-up world, about 10 Hz.
- `.pincam`: `w h fx fy cx cy`.
- Frames are matched by the timestamp in the file name: poses rounded to 3 decimals with ±5 ms, intrinsics ±1 ms (`tenFpsDataLoader.py:247,331-336,257-263`).

**Caveat:** in 2 of 4 checked videos, the hires camera frame is rotated 180° about the optical axis relative to the `.traj`. Image, intrinsics and pose must be turned together, as was done for the sideways ETH3D clips. The clip builder needs a test for this.

**Early look (4 videos, not a result):** device vs COLMAP relative poses differ by a median of 0.2–0.45° and 0.4–1.4 cm over about 1 s, and 0.5–0.8° and 0.7–2.5 cm over about 3 s. Real ARKit pose hints may therefore be mild compared with E1's harmful sizes. The preregistration should allow for "real hints help".

## Download
- 15 videos, one per visit. The selection rule must be fixed in the preregistration (e.g. seeded random visits).
- Assets:
  - SceneFun3D: `hires_wide hires_wide_intrinsics hires_depth hires_poses transform`;
  - ARKitScenes: `lowres_poses lowres_depth lowres_wide_intrinsics confidence`.
- Command (SceneFun3D repo root, needs `moviepy`): `python -m data_downloader.data_asset_download --split custom --video_id_csv my15.csv --download_dir DIR --dataset_assets <assets>`.
- **Size:** 11.6 GB for 15 videos, measured from the servers' reported file sizes; the laser scans would add 1.1 GB. The laptop has 9.3 GB free and the last VM had 7 GB.
  - So either download on a VM with enough disk (on its CPU, alongside GPU jobs), or fetch only the needed frames with HTTP range requests. Both servers accept range requests. This would be about 0.4 GB for 30 frames per video (an estimate, untested).
- Clips: 3 × 10 frames per video. Clips are a few MB each, so only clips go to the GPU VM.
