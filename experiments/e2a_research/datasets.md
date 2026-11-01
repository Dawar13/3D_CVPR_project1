# E2-E draft: datasets with REAL sensor hints + better GT (literature/web check)

Date read for every source: **2026-10-29**. Literature/web only: no code, no repo edits, no data downloads.

**How the quotes were obtained.**
- *(page)*: the quote was taken from the web page through the WebFetch tool. The tool relays text through an extraction model, so a few words could differ from the page; re-check before it goes into a paper.
- *(pdf-txt)*: the quote is exact text extracted locally with PyMuPDF from the PDF that WebFetch saved. The extracted texts are in `scratchpad/e2_lit/ds_txt/` (`arkitscenes_paper.txt`, `scannetpp_terms.txt`, `tum_sturm12.txt`, `lamar.txt`).
- Anything I could not open or confirm is marked **UNVERIFIED** and collected in §6.

---

## 0. Bottom line

- **Start now, no approval needed: ARKitScenes, raw assets, *Validation* split.** It has iPad ARKit poses, LiDAR depth and per-frame intrinsics, plus GT poses and GT depth from a Faro laser scan. Access:
  - it is an open per-video download script;
  - it is **not** in either MapAnything checkpoint's training table;
  - for OmniVGGT, its *training* split is in the training table, so use Validation.

  Caveat: I could not confirm which raw `.traj` file holds the ARKit poses and which the GT poses. **SceneFun3D**, which is built on ARKitScenes, documents this split explicitly (`lowres_poses.traj` = ARKit; `hires_poses.traj` = COLMAP in the laser-scan frame). It is also open download (CC-BY-4.0 plus the ARKitScenes terms), so it is the fallback or companion.
- **Also start now, as a small side set: TUM RGB-D.** It is CC BY 4.0, open, and 0.3–2 GB per sequence. It has real Kinect depth and real **intrinsics error**: the factory/ROS default K differs from the calibrated K. GT is motion capture.
  - It has **no device poses and no GT depth**, so it only tests depth and intrinsics hints.
  - VPGGT used TUM, which makes the results comparable.
- **Request access today: ScanNet++ v2.** It is the best fit:
  - raw ARKit poses + iPhone LiDAR, against COLMAP poses aligned to a sub-mm laser scan and rendered depth;
  - DSLR images as well.

  The process is slow: a terms-of-use PDF with **handwritten signatures of the requester and the PI**, approval, then a token. **It is in MapAnything's training data (926 scenes)**; only the 30-scene MapAnything test list is held out.
- **Optional request: LaMAR.** It needs a Google form. It covers large indoor/outdoor scenes with ARKit, iPad LiDAR and HoloLens.
  - Caveat: its GT pipeline *uses the on-device relative poses as input*, so the hint and the GT are not independent.

---

## 1. Comparison table

| Dataset | Licence & access | Real hints | GT & stated accuracy | Size | In MapAnything training? | In OmniVGGT training? | Held-out split to use | Fit |
|---|---|---|---|---|---|---|---|---|
| **ScanNet++ v2** (iPhone stream) | "ScanNet++ Terms of Use", non-commercial. Register, then submit the signed PDF (handwritten signatures of requester + PI), approval, then a personal token. Wait time **not stated** | ARKit poses ("raw ARKit output") + ARKit intrinsics per frame; iPhone LiDAR depth 256×192; RGB 1920×1440, 60 FPS | Faro Focus Premium laser scan, "sub-millimeter resolution"; iPhone frames registered with COLMAP and filtered by depth agreement with the scan; depth rendered from the mesh. **No numeric pose accuracy given** | Default bundle (DSLR 2MP + iPhone + meshes) ≈ 1.5 TB; iPhone-only size not stated; per-scene download **UNVERIFIED** (script only after approval) | **YES** (both checkpoints): 926 scenes in training; 30 "unique held-out scenes" for benchmarking. MapAnything converts **DSLR only** | **YES** ("ScanNet++" in the 19-dataset list; version not stated; via CUT3R/DUSt3R preprocessing, which uses DSLR **and iPhone** frames: inference) | MapAnything: only the 30 scenes in `scannetppv2_scene_list_test.npy` (file exists on HF, contents not read). OmniVGGT: unknown | Best hint/GT pairing; strong leakage risk; slow access |
| **ARKitScenes** (raw) | Apple custom licence ("non-commercial purposes only"). Open download via `download_data.py`, per video and per asset; no form mentioned | ARKit world-tracking poses (stored and released); iPad LiDAR depth 256×192 (60 FPS) + confidence; per-frame intrinsics (`*_intrinsics`); RGB 256×192 / 640×480 / 1920×1440 | Faro Focus S70 laser scans; GT poses by render-and-match PnP + dense photometric refinement against the scan; `highres_depth` 1920×1440 at 10 FPS projected from the Faro mesh. **No numeric accuracy given**; frames dropped if LR–HR depth RMSE > 7 cm | Raw total not stated (3DOD subset 623.4 GB / 5047 scans); per-video and per-asset download confirmed | **NO** (not in Table 1, either checkpoint) | **YES** (Table 8: "ARKitScene", 1.2M frames, 2.07%). The same paper's App. A.4 calls ARKitScenes "unseen": internal inconsistency | Official **Validation** split (venue-disjoint 10%); test 10% never released | **Start here.** Confirm the `.traj` provenance first |
| **SceneFun3D** (on ARKitScenes) | Data "under the CC-BY-4.0 license" + you "agree with the license & terms" of ARKitScenes. Open toolkit download per visit/video; no form mentioned | `lowres_poses.traj` = "ARKit camera pose trajectory"; `lowres_depth` iPad LiDAR 256×192 at 60 FPS; intrinsics files | `hires_poses.traj` = poses "estimated with COLMAP in the Faro laser scan coordinate system" (SuperGlue + COLMAP); `hires_depth` projected from the Faro mesh; 5 mm laser scan asset. No numeric pose accuracy; notes ARKit-to-frame interpolation errors of "approximately 1-2 cm" | Not stated; per-visit download; `--download_only_one_video_sequence` option | Not listed (but it **is ARKitScenes captures**) | Not listed by name; **overlaps ARKitScenes** (scene split mapping UNVERIFIED) | `train_val_set` / `test_set`; mapping to ARKitScenes Training/Validation UNVERIFIED | Cleanest documented ARKit-vs-GT separation; OmniVGGT overlap risk |
| **CA-1M** (Apple, on ARKitScenes) | "CC-by-NC-ND"; open per-video tars (curl lists) | iPad LiDAR depth 256×192; gravity; **ARKit poses not mentioned** | GT pose `RT.json` "registered to the laser scanner"; GT depth 512×384 rendered from Faro | Not stated | Not listed | Shares ARKitScenes captures | `val` | Depth hints only; ND licence |
| **TUM RGB-D** | "CC BY 4.0" on the site (the 2012 paper says CC-BY 3.0); open per-sequence .tgz | Kinect depth 640×480 (pre-registered to RGB); factory/ROS default K (525/525/319.5/239.5) vs calibrated K → **real intrinsics error**. **No device poses** | Mocap, 8 cameras at 100 Hz; paper: frame-to-frame error "< 1 mm and 0.5 deg", absolute "< 10 mm and 0.5 deg". **No GT depth** | 0.32–2.01 GB per sequence (examples) | **NO** | **NO** (not in the 19; not in VGGT's list) | Any sequence with public GT (`*_validation` have no GT) | Good small side test for depth and K hints; colour/depth unsynchronised (~20 ms); rolling shutter |
| **LaMAR** | Google form ("request access to the dataset via this form"); data licence **not stated** (repo docs CC BY 4.0, code MIT) | iPad/iPhone ARKit poses + per-frame intrinsics, iPad LiDAR depth 10 Hz, 1080p rolling-shutter RGB; HoloLens 2 tracker poses + ToF depth | NavVis laser-scan reference; GT kept if "correct within 10cm with a confidence of 99.7%" (σt = 3.33 cm). **GT uses the on-device relative poses as input** | Not stated | **NO** | **NO** | `query_val_*` (public GT); test GT private | Real but large-scale AR; GT and hint not independent; GT depth for phone frames not documented |
| **ADVIO** | CC BY-NC 4.0; open (Zenodo) | ARKit (60 Hz), ARCore (30 Hz), Tango poses; Tango point clouds; **no depth maps**; iPhone 1280×720 | IMU + "known fixation points"; no accuracy number | 71–255 MB per set, ≈5.1 GB total | NO | NO | n/a | Weak: no GT depth, weak GT |
| **MobileBrick** | Repo "MIT license" (data licence not separately stated); one zip | ARKit LiDAR depth ("ARKit Depth"), RGB 1440×1920; whether **raw ARKit poses** are released is **UNVERIFIED** | Poses: PnP + manual + BA; translation RMSE 2.060 mm vs ARKit 4.454 mm; rotation 0.522° vs 0.581° | Not stated | NO | NO | 18-sequence eval set | Object-centric; ARKit already near-GT (mild hints) |
| **OpenLORIS-Scene** | Form ("Fill this form"); licence not stated on the page | RealSense D435i depth 848×480; **T265 poses not mentioned** | OptiTrack (office scene only) / offline LiDAR SLAM; no number | Not stated | NO | NO | — | Poor: no device poses, planar robot motion |
| **Aria Digital Twin** | Licence agreement + email | No depth sensor; MPS/on-device poses not mentioned on the page | Digital-twin GT depth, 6DoF GT | Not stated | NO (ASE, the synthetic one, is in the NC model) | **YES via VGGT init** ("Aria Digital Twin" in VGGT's training list) | — | Reject |

---

## 2. Training-set facts (Task A item 5 and Task B)

### 2.1 MapAnything (arXiv 2509.13414, v3 dated 23 Jan 2026), https://arxiv.org/html/2509.13414
- Table 1 caption *(page)*: "Datasets used for training and testing MapAnything."
- Table 1 rows *(page, as relayed)*:
  - BlendedMVS 493; Mapillary Planet-Scale Depth 71,428; **ScanNet++ v2 926**; Spring 37; TartanAirV2-WB 49; UnrealStereo4K 9.
  - Then "*Additionally used for our CC BY-NC model:*" Aria Synthetic Environments 103,890; DL3DV-10K 10,109; Dynamic Replica 523; MegaDepth 269; MVS-Synth 120; ParallelDomain-4D 1,528; SAIL-VOS 3D 171.
  - Then "*Unique held-out scenes for dense up-to-N-view benchmarking:*" **ETH3D 13 (CC BY-NC-SA 4.0)**; **ScanNet++ v2 30**; TartanAirV2-WB 5.
- *(page)* "For ScanNet++ v2 and TartanAirV2-WB, we split the scenes into a training, validation, and a held-out test set"
- *(page)* the Apache model is "trained on six datasets", i.e. the first six rows (all include ScanNet++ v2).
- *(page)* RMVD benchmark: "we don't use ETH3D due to the distortion issue"
- ARKitScenes, TUM, LaMAR, ADVIO, SceneFun3D, CA-1M, MobileBrick, OpenLORIS and ADT do **not** appear in Table 1.
- Model cards: https://huggingface.co/facebook/map-anything ("License: cc-by-nc-4.0") and https://huggingface.co/facebook/map-anything-apache ("License: apache-2.0"). **Neither card lists datasets.** The GitHub README (https://github.com/facebookresearch/map-anything) says *(page)* "The only difference is the training data composition and resulting license terms." and "Please see our paper for detailed information about the specific datasets used in each model variant."
- Split files on HF (https://huggingface.co/datasets/facebook/map-anything/tree/main/mapanything_dataset_metadata):
  - `train/` holds 13 `*_scene_list_train.npy` files, one per training dataset, including `scannetppv2_scene_list_train.npy` (11.9 kB), and **no eth3d file**;
  - `val/` holds 13 files, again no eth3d;
  - `test/` holds `eth3d_scene_list_test.npy` (413 B), `scannetppv2_scene_list_test.npy` (667 B) and `tav2_wb_scene_list_test.npy` (351 B).
  - I could not read the .npy contents: the CDN redirect could not be followed. **UNVERIFIED which 30 ScanNet++ scenes are held out.** These are tiny files; read them on the VM before choosing scenes.
- ScanNet++ in MapAnything uses **DSLR only**. https://raw.githubusercontent.com/facebookresearch/map-anything/main/data_processing/wai_processing/scripts/conversion/scannetppv2.py *(page)*: `if modality != "dslr":` / `raise NotImplementedError("Only DSLR support is implemented")`. Poses come from `dslr/nerfstudio/transforms.json`; depth is rendered from `mesh_aligned_0.05.ply`. So iPhone images were never input, **but the scenes and their geometry were**.
- Inference, not verified: 926 + 30 = 956 = 856 (nvs_sem_train) + 50 (nvs_sem_val) + 50 (sem_test), the splits that have scan data. This is consistent with MapAnything re-splitting all scan-bearing scenes, which would put **official val/sem_test scenes in MapAnything training**. Do not use the official ScanNet++ val split as "held out" for MapAnything. Use only their test list.

### 2.2 OmniVGGT (arXiv 2511.10560v2), https://arxiv.org/html/2511.10560 ; README https://github.com/Livioni/OmniVGGT-official
- *(page)* "We train our model using images from 19 public datasets, including: ARKitScenes [6], BlendedMVS [71], DL3DV [33] …". The relayed list is: ARKitScenes, BlendedMVS, DL3DV, Dynamic Replica, HyperSim, Kubric, MapFree, MegaDepth, Matterport 3D, MVS-Synth, ScanNet, ScanNet++, Spring, TartanAir, UASOL, Unreal 4K, Virtual KITTI, Waymo, WildRGBD.
- Appendix Table 8 *(page)*: caption "Training datasets statistic." / "Each training epoch is composed of 20 datasets…". It lists ARKitScene (1.2M frames, 2.07%) and ScanNet++ (7.8M frames, 13.88%).
- **Inconsistency** *(page)*, App. A.4:
  - "Tables 9 and 10 report the results on ARKitScene and OmniWorld (Game), respectively." and "The methods have not been trained on these datasets."
  - Table 9 caption: "Impact of Auxiliary Information Injection on ARKitScenes (unseen)".
  - Most likely the evaluation used a split OmniVGGT did not train on, but **the paper does not name the split (UNVERIFIED)**.
- README *(page)*: "Follow CUT3R to download and preprocess the datasets."
  - CUT3R (https://github.com/CUT3R/CUT3R/blob/main/docs/preprocess.md) processes ARKitScenes with `--splits Training Test` and ScanNet++ from "pre-computed pairs provided by DUSt3R".
  - DUSt3R's ARKitScenes script (https://raw.githubusercontent.com/naver/dust3r/main/datasets_preprocess/preprocess_arkitscenes.py) reads `subdirs = ['Test', 'Training']`, `lowres_depth`, `vga_wide` and `lowres_wide.traj`.
  - DUSt3R's ScanNet++ script (https://raw.githubusercontent.com/naver/dust3r/main/datasets_preprocess/preprocess_scannetpp.py) uses both `dir_dslr` and `dir_iphone`, iPhone poses from `iphone/colmap`, and depth rendered from `mesh_aligned_0.05.ply`.
  - Inference: OmniVGGT's ScanNet++ data likely included iPhone frames of its training scenes. The scene list is DUSt3R's `scene_list.json`, which I did not read.
- OmniVGGT is initialised from VGGT-1B (context.md §4, README:293). VGGT's training list (https://arxiv.org/html/2503.11651) *(page)*: "The model was trained using a large and diverse collection of datasets, including:" Co3Dv2, BlendMVS, DL3DV, MegaDepth, Kubric, WildRGB, ScanNet, HyperSim, Mapillary, Habitat, Replica, MVS-Synth, PointOdyssey, Virtual KITTI, Aria Synthetic Environments, **Aria Digital Twin**, and an Objaverse-like synthetic set. ScanNet++, ARKitScenes and TUM are not mentioned there.
- TUM, LaMAR and ADVIO are in neither the OmniVGGT nor the VGGT list.

### 2.3 Task B: ETH3D
- **MapAnything: benchmark only, not training.**
  - Table 1 lists ETH3D (13 scenes) only under "*Unique held-out scenes for dense up-to-N-view benchmarking:*".
  - HF metadata has `test/eth3d_scene_list_test.npy` and no ETH3D file in `train/` or `val/`.
  - It is excluded from RMVD ("we don't use ETH3D due to the distortion issue").
  - Note: the ETH3D high-res training split also has 13 scenes, so MapAnything's benchmark very likely uses **the same 13 scenes as our E0/E1**. That is inference; the npy is unread. So our ETH3D scenes are not trained on, but the authors evaluated on them (possible model-selection exposure, not training leakage).
- **OmniVGGT: evaluation only.**
  - *(page)* "Following RobustMVD, we evaluate the multi-view depth performance"; ETH3D appears in Table 11 and Fig. 4. It is not in the 19-dataset training list or Table 8.
  - VGGT, its initialisation, also uses ETH3D for evaluation only *(page)*: "We also compare the accuracy of our predicted point cloud to DUSt3R and MASt3R on the ETH3D".
  - CUT3R's preprocessing doc has no ETH3D entry.

---

## 3. Per-dataset evidence (quotes and URLs)

### 3.1 ScanNet++ v2
- Site: https://scannetpp.mlsg.cit.tum.de/scannetpp/ (the old kaldir.vc.in.tum.de URL redirects here).
  - *(page)* "1000+ 3D indoor scenes containing sub-millimeter resolution laser scans, registered 33-megapixel DSLR images" and "commodity RGB-D streams from iPhone."
  - Access *(page)*: "Once your application is approved, you will receive a personalized token to download the data."
- Terms PDF: https://scannetpp.mlsg.cit.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf *(pdf-txt)*:
  - "Researcher shall use the Database only for non-commercial research and educational purposes. Commercial use is strictly prohibited."
  - "Signatures must be handwritten. Digital signatures cannot be accepted."
  - "Only established researchers with a publicly verifiable research profile (postdocs, junior professors, senior researchers in companies) can apply independently." Students need PI details and a PI signature.
- Changelog: https://scannetpp.mlsg.cit.tum.de/scannetpp/changelog. v2 (20 Dec 2024) *(page)*:
  - "1000+ scenes released with 600+ new scenes."
  - "Improved DSLR and iPhone poses and anonymization for all scenes (including updates to v1 scenes)"
  - "New scenes are added to the train set. The other splits remain the same"
- Documentation: https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation *(page)*:
  - iPhone:
    - "pose_intrinsic_imu.json … contains ARKit poses and IMU information from the iPhone"; poses are "a 4x4 camera-to-world extrinsic matrix from raw ARKit output"; intrinsics are "a 3x3 intrinsic matrix of the RGB image".
    - "aligned_pose … contains ARKit poses that are scaled and transformed to our mesh space".
    - Depth: "Depth images as 16 bit png in millimeters in a single binary file from iPhone Lidar sensor"; "much lower resolution: 256 x 192"; RGB "The resolution is 1920 x 1440."
    - iPhone COLMAP: "Images here have been filtered based on agreement of depth between iPhone Lidar and the laser scanner."
  - DSLR COLMAP: "aligned with the 3D scans, which implies the poses are in metric scale."
  - Splits: "nvs_sem_train … 856 scenes", "nvs_sem_val … 50 scenes", "nvs_test … 50 scenes (no scan data)", "sem_test … 50 scenes"; "the scenes in nvs_test split do not contain 3D information like meshes and iphone depth maps."
  - Sizes: "The default download (low-res DSLR images, iPhone data, 3D meshes and semantics) occupies about 1.5 TB on disk"; DSLR 2 MP 371 GB; meshes + semantics 132 GB.
- Paper (ICCV 2023, arXiv 2308.11417v1): https://arxiv.org/html/2308.11417 *(page)*:
  - "We acquire point clouds of the scenes using the Faro Focus Premium laser scanner."
  - iPhone RGB and LiDAR are "both recorded at 60 FPS synchronously".
  - Poses are refined "with dense photometric error guided by the geometry of the laser scan".
  - "Depth images from iPhone Lidar are noisy and low resolution."
  - **No numeric GT pose accuracy given.**
- Toolbox: https://github.com/scannetpp/scannetpp *(page)*: `python -m iphone.prepare_iphone_data …` (extracts frames and depth); `python -m common.render …` renders depth ("uint16 png, where the unit is mm and 0 means invalid depth").
- Fit: excellent hint/GT pairing on the same frames. Hints are raw ARKit pose + intrinsics + LiDAR; GT is COLMAP pose aligned to the laser scan + rendered mesh depth.
  - Pitfall: the GT iPhone subset is filtered by LiDAR–scan depth agreement, which biases it toward frames where the depth hint is good.
  - It is unknown whether the ARKit poses were used to initialise the iPhone COLMAP (independence UNVERIFIED).

### 3.2 ARKitScenes
- Repo: https://github.com/apple/ARKitScenes *(page)*: "ARKitScenes is the largest indoor 3D dataset consisting of 5,047 captures of 1,661 unique scenes."
- Licence: https://github.com/apple/ARKitScenes/blob/main/LICENSE *(page)*: a custom Apple licence granting a "personal, non-commercial, non-exclusive license"; the base grant is for "non-commercial purposes only".
- Download: https://github.com/apple/ARKitScenes/blob/main/DATA.md *(page)*:
  - `python3 download_data.py raw --split Training --video_id 47333462 --download_dir /tmp/ARKitScenes/`
  - `--raw_dataset_assets` selects assets.
  - Splits come from `raw/raw_train_val_splits.csv`; data lands in `Training` and `Validation`.
  - "The total Size is 623.4 GB for 5047 threedod scans" is the only size given.
  - No form or token is mentioned.
- Raw assets: https://github.com/apple/ARKitScenes/blob/main/raw/README.md *(page)*:
  - lowres_wide "RGB images of the wide camera (256x192)" at 60 FPS;
  - lowres_depth "the depth image acquired by AppleDepth Lidar (256x192)";
  - highres_depth "the ground-truth depth image projected from the mesh generated by Faro's laser scanners (1920x1440)", Filtered 10 FPS;
  - wide 1920×1440 (Filtered 10 FPS);
  - vga_wide 640×480 (30 FPS).
- Paper (arXiv 2111.08897 v3): https://arxiv.org/pdf/2111.08897 *(pdf-txt)*:
  - "The on-device ARKit world tracking poses as well as the scene reconstruction are stored and provided with the dataset"
  - GT pose: "We use RANSAC [42] and PnP to estimate initial camera poses using these sparse correspondences. To improve over these initial estimates we jointly solve for the camera poses of all keyframes that minimize a dense photometric error metric."
  - "This dense ground truth depth map, along with the per-keyframe ground truth pose are provided as part of the dataset."
  - "we utilized a Faro Focus S70 stationary laser scanner on a tripod"
  - "We split the venues of ARKitScenes into 80% for training, 10% for validation and the remaining 10% are a held-out test set that we do not release."
  - Also (ar5iv page, https://ar5iv.labs.arxiv.org/html/2111.08897): frames are ignored if the "Root Mean Square Error (RMSE) between the LR and a downscaled HR depth map is more than 7cm".
  - **No numeric GT accuracy given.**
- **UNVERIFIED:** whether `lowres_wide.traj` holds the ARKit poses or the refined GT poses, and where the per-keyframe GT poses are stored. The threedod and raw READMEs do not say. DUSt3R uses `lowres_wide.traj` as training poses.
- Fit: good, and open. Pitfalls:
  - GT depth exists only on the filtered 10 FPS `wide` frames, while LiDAR depth is 60 FPS. Frames must be matched by timestamp; ARKit-to-frame interpolation errors of "approximately 1-2 cm" are reported by SceneFun3D (§3.3).
  - Images can be portrait (CA-1M notes images are rotated upright).
  - The iPad camera is rolling shutter (LaMAR paper, §3.6).
  - The 4:3 aspect differs from our 3:2 ETH3D clips. What MapAnything's `preprocess_inputs` does with 4:3 is not checked here.

### 3.3 SceneFun3D (CVPR 2024; built on ARKitScenes)
- File structure: https://scenefun3d.github.io/documentation/dataset/file-structure *(page)*:
  - `lowres_poses.traj` "contains the ARKit camera pose trajectory in the ARKit coordinate system";
  - `hires_poses.traj` "contains the camera poses estimated with COLMAP in the Faro laser scan coordinate system"; "we employ SuperGlue and COLMAP to estimate accurate camera poses in the Faro laser scan coordinate system";
  - `<video_id>_transform.npy` is a "4x4 transformation matrix that registers the Faro laser scan to the ARKit coordinate system";
  - frames are "not temporally synchronized with these ARKit poses", giving "misalignment errors of approximately 1-2 cm in some cases";
  - `hires_depth` is "the ground-truth depth image projected from the mesh generated by Faro's laser scanners";
  - `lowres_depth` is "depth images acquired with the iPad Lidar sensor (256x192) - 60 FPS".
- Licence: https://scenefun3d.github.io/documentation/license/ *(page)*: "The SceneFun3D data assets are under the CC-BY-4.0 license." and "you also agree with the license & terms and conditions of the ARKitScenes dataset".
- Downloader: https://scenefun3d.github.io/documentation/toolkit/data-downloader/ *(page)*:
  - `python -m data_downloader.data_asset_download` with `--split` (train_val_set / test_set / custom) and `--visit_id/--video_id`;
  - assets include `laser_scan_5mm`, `lowres_wide`, `lowres_depth`, `hires_wide`, `hires_depth`, `lowres_poses`, `hires_poses`, `transform`, `arkit_mesh`.
  - No form is mentioned. Sizes are not stated.
- Main page (https://scenefun3d.github.io/): "710 high-resolution real-world 3D indoor scenes".
- **UNVERIFIED:** which ARKitScenes split each SceneFun3D visit comes from (needed for OmniVGGT leakage). The CVPR PDF returned HTTP 403.

### 3.4 CA-1M (Apple "Cubify Anything")
- https://github.com/apple/ml-cubifyanything *(page)*:
  - "We generally share the same underlying captures" (as ARKitScenes);
  - "CA-1M will only include captures which were successfully registered to the laser scanner";
  - per frame: 1024×768 RGB, 256×192 LiDAR depth, GT `RT.json` pose, GT depth 512×384 rendered from Faro;
  - "All data is released under the CC-by-NC-ND".
  - **ARKit poses are not mentioned.** Splits are train/val.

### 3.5 TUM RGB-D
- Site: https://cvg.cit.tum.de/data/datasets/rgbd-dataset *(page)*:
  - "contains the color and depth images of a Microsoft Kinect sensor along the ground-truth trajectory";
  - "obtained from a high-accuracy motion-capture system with eight high-speed tracking cameras (100 Hz)";
  - licence "Creative Commons 4.0 Attribution License (CC BY 4.0)".
- Download page: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download *(page)*:
  - fr1_xyz 0.47 GB, fr1_desk 0.32 GB, fr2_desk 2.01 GB, fr3_long_office_household 1.58 GB;
  - "The `*_validation` sequences do not contain ground truth."
- File formats: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats *(page)*:
  - "The color and depth images are already pre-registered using the OpenNI driver from PrimeSense";
  - depth is 640×480 16-bit, scale 5000;
  - "We recommend to use the ROS default parameter set (i.e., without undistortion)" (fx = fy = 525.0, cx = 319.5, cy = 239.5) versus calibrated Freiburg 1 fx = 517.3, fy = 516.5, cx = 318.6, cy = 255.3 with distortion.
- Tools: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/tools *(page)*: "The Kinect provides the color and depth images in an un-synchronized way."
- Paper (Sturm et al., IROS 2012): https://cvg.cit.tum.de/_media/spezial/bib/sturm12iros.pdf *(pdf-txt)*:
  - "the relative error on a frame-to-frame basis in the ground truth data is lower than 1 mm and 0.5 deg … the absolute error over the whole motion capture area is lower than 10 mm and 0.5 deg";
  - "the depth images arrive on average around 20 ms later than the color images";
  - "the Kinect uses a rolling shutter for the color camera";
  - the paper says "(CC-BY 3.0)"; the current site says CC BY 4.0.
- Fit: depth and intrinsics hints only. There is no device pose and no GT depth, so depth metrics would need GT-pose-based multi-view checks, or would be limited to pose and point metrics.

### 3.6 LaMAR (ECCV 2022)
- Repo: https://github.com/microsoft/lamar-benchmark *(page)*:
  - validation queries have "known ground-truth poses"; "We keep private the ground-truth poses of the test queries";
  - docs licence "Creative Commons Attribution 4.0 International Public License", code MIT;
  - **the data licence is not stated.**
- RAW-DATA.md: https://github.com/microsoft/lamar-benchmark/blob/main/RAW-DATA.md *(page)*: phone `depths.txt` lists LiDAR depth maps with confidence; HoloLens `trajectories.txt` odometry "only consistent within a sub-map".
- Access: https://lamar.ethz.ch/lamar/ *(page)*: "You can request access to the dataset via this form."
- Paper: https://arxiv.org/pdf/2210.10770 *(pdf-txt)*:
  - "the iPad has a single, higher-resolution camera with rolling shutter";
  - "We also recorded outputs of the real-time AR tracking algorithms available on each device, which includes relative camera poses and sensor calibration";
  - "we use ARKit for iPhone/iPad and the on-device tracker for HoloLens. The tracker also outputs per-frame camera intrinsics";
  - GT: "then register each AR sequence individually to the reference map using local feature matching and relative poses from the on-device tracker";
  - "We retain images whose poses are correct within 10cm with a confidence of 99.7%. … σt=3.33cm".
- Fit: real ARKit hints, but the GT is coupled to them (it uses tracker relative poses) and is only cm-level. GT depth for phone frames is not documented, and the scenes are large.

### 3.7 ADVIO
- https://github.com/AaltoVision/ADVIO *(page)*:
  - "CC BY-NC 4.0";
  - "The Apple ARKit poses … captured at 60 Hz"; ARCore 30 Hz; Tango 60 Hz; "Tango point cloud data";
  - GT: "calculated based on the raw IMU data and a set of known fixation points";
  - 23 sets, "71 Mb to 255 Mb (packed)", "around 5.1 Gb".
- No depth maps and no GT depth, so it is rejected for our purpose.

### 3.8 MobileBrick (CVPR 2023)
- Repo: https://github.com/ActiveVisionLab/MobileBrick *(page)*:
  - "135 sequences … for training", "18 sequences … for evaluation";
  - "arkit_depth (the confidence and depth maps provided by ARKit)"; "gt_depth (The high-resolution depth maps projected from the aligned GT shape)"; "pose (4x4 transformation matrix from camera to world …)", not stated whether this is ARKit or GT.
- Paper: https://ar5iv.labs.arxiv.org/html/2303.01932 *(page)*: refined poses have translation RMSE 2.060 mm (ARKit 4.454 mm) and rotation 0.522° (ARKit 0.581°). ARKit hints are therefore nearly clean on small objects.

### 3.9 Rejected quickly
- **OpenLORIS-Scene**, https://lifelong-robotic-vision.github.io/dataset/scene *(page)*: D435i depth 848×480. GT is "obtained by an OptiTrack motion capture system for the office scene", other scenes use "offline LiDAR SLAM". Access is "Fill this form"; T265 pose output is not mentioned.
- **Aria Digital Twin**, https://www.projectaria.com/datasets/adt/ *(page)*: "200 sequences (~400 mins)"; depth "by leveraging the complete scene reconstruction". It has no depth sensor, and it is in VGGT's training list.

---

## 4. Practical fit: building 10-frame clips with matched hints and GT
- **ARKitScenes / SceneFun3D:**
  - Pick 10 `wide`/`hires_wide` frames (10 FPS) spread over a sequence for baseline.
  - Hint pose: ARKit pose interpolated to each frame timestamp (1–2 cm interpolation error per SceneFun3D).
  - Hint depth: the nearest LiDAR 256×192 frame by timestamp.
  - GT: the refined/COLMAP pose and the Faro-projected depth.
  - Express ARKit and GT in one frame by aligning each clip at view 0, as in our ETH3D clips.
- **ScanNet++:**
  - Take iPhone frames that have COLMAP poses (the filtered subset).
  - Hint = raw ARKit pose + ARKit K + LiDAR depth.
  - GT = COLMAP pose + depth rendered from the laser mesh (toolbox `common.render`).
  - DSLR frames could add a high-res image condition, but their device "hints" are absent.
- **TUM:**
  - 10 frames per clip, associated by timestamp (`associate.py`, ~20 ms depth lag).
  - Hint = Kinect depth + ROS-default K.
  - GT = mocap poses + calibrated K. No GT depth.
- All the phone/iPad sets have rolling-shutter RGB (LaMAR states this for the iPad; TUM states it for the Kinect). This may add residual error between hint and GT that is unrelated to the hint.

## 5. Recommendation
1. **Start immediately (no approval wait): ARKitScenes raw, Validation split**, downloaded per video with assets `wide`, `wide_intrinsics`, `highres_depth`, `lowres_depth`, `confidence`, `lowres_wide`, `lowres_wide_intrinsics` and `lowres_wide.traj`.
   - It is unseen by both MapAnything checkpoints, and the Validation split should be unseen by OmniVGGT.
   - First task on the VM: establish whether `lowres_wide.traj` is ARKit or GT. Do this by comparing it with SceneFun3D's `lowres_poses.traj` / `hires_poses.traj` for an overlapping video, or by reading the ARKitScenes issues.
   - If ambiguous, switch to **SceneFun3D** (explicit ARKit vs COLMAP-in-laser-frame poses, CC-BY-4.0 plus ARKitScenes terms, open download), after checking its visit→ARKitScenes split mapping.
2. **In parallel, start now: TUM RGB-D**, 3–5 sequences with public GT (a few GB). It tests real depth and real intrinsics errors only.
3. **Request access now: ScanNet++ v2.** The signed PDF needs handwritten signatures of the requester and the PI. The wait time is unknown, so file it today.
   - When granted, use **only the 30 scenes in MapAnything's `scannetppv2_scene_list_test.npy`** and check them against DUSt3R's ScanNet++ `scene_list.json` for OmniVGGT.
4. **Optional request: LaMAR** (Google form), as an outdoor/large-scale check. Report that its GT is coupled to the on-device tracker.

## 6. Not verified / could not open
- Which 30 ScanNet++ scenes (and which 13 ETH3D scenes) MapAnything holds out. The `.npy` files exist on HF but the download redirect could not be fetched.
- Whether ScanNet++ official val/sem_test scenes are inside MapAnything's 926 training scenes. The arithmetic 926 + 30 = 856 + 50 + 50 is suggestive only.
- ScanNet++ per-scene/per-asset download options and the iPhone-only size. The download script is only given after approval.
- ScanNet++ approval wait time, and whether the iPhone COLMAP was initialised from ARKit (hint/GT independence).
- ARKitScenes: which raw `.traj` holds the ARKit poses versus the GT poses; total raw size.
- Which ARKitScenes split OmniVGGT trained on: the CUT3R/DUSt3R scripts read `Training` and `Test` folders, while the official download produces `Training` and `Validation`. Also the split behind OmniVGGT's "ARKitScenes (unseen)" Table 9.
- Which ScanNet++ scenes and version OmniVGGT trained on (DUSt3R `scene_list.json` not read).
- SceneFun3D's mapping to ARKitScenes splits; its sizes; the CVPR paper PDF (HTTP 403).
- LaMAR data licence and sizes; whether phone sessions ship GT depth.
- MobileBrick: whether raw ARKit poses are released.
- Approval wait times for ScanNet++ and LaMAR: neither page states one.
- Any `facebook/map-anything` model-card dataset list: the cards list none, so Table 1 of the paper is the only source.
