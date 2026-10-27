# E0 Step 3: MapAnything interface verification

**Commit read:** `facebookresearch/map-anything` @ `3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9`
(2026-08-07, "Secure Gradio prediction loading and state (#165)"). `scripts/setup_server.sh`
now checks out this exact commit, so the server runs the code read here.
**Method:** static reading of the source on 2026-10-27, then runtime confirmation on the same day
(A100-SXM4-80GB, torch 2.14.1+cu130; logs in `reports/e0_logs/`). All paths below are relative to
`third_party/map-anything/`. Items marked **runtime** are now confirmed unless stated otherwise;
see "Runtime confirmation" at the end.

**Verdict for `exp1/model_io.py`:** every call, input key and output index in it matches the
code. No functional change was needed. A docstring note was added, plus `official_example()`
for Step 6, so `model_io.py` stays the only file that calls MapAnything.

## The 10 questions

**1. Signatures and `infer` kwargs.**
- `MapAnything(nn.Module, PyTorchModelHubMixin)` (`mapanything/models/mapanything/model.py:92`).
  `from_pretrained` comes from `huggingface_hub`. The override `_from_pretrained` (`model.py:95-104`)
  only disables fetching DINOv2 hub weights, because the checkpoint already contains them.
- `preprocess_inputs(input_views, resize_mode="fixed_mapping", size=None, norm_type="dinov2", patch_size=14, resolution_set=518, verbose=False)`
  (`mapanything/utils/image.py:348-356`).
- `infer(views, memory_efficient_inference=True, minibatch_size=None, use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=True, edge_normal_threshold=5.0, edge_depth_threshold=0.03, apply_confidence_mask=False, confidence_percentile=10, ignore_calibration_inputs=False, ignore_depth_inputs=False, ignore_pose_inputs=False, ignore_depth_scale_inputs=False, ignore_pose_scale_inputs=False, use_multiview_confidence=False, ...)`
  is decorated `@torch.inference_mode()` (`model.py:2027-2049`).
- All six harness kwargs exist. Two of them differ from the README defaults:
  - `memory_efficient_inference=False` only controls minibatching of the dense heads (`model.py:2071`).
  - `apply_mask=False` means no `mask` key is returned and nothing is zeroed (`mapanything/utils/inference.py:411-505`). `mask_edges` only acts inside that branch (`inference.py:447`).
- The `ignore_*` flags work by setting input probabilities to 0/1 (`model.py:1975-2018`). They are useful later (see E2).

**2. `img` format and `data_norm_type`.**
- `preprocess_inputs` accepts an (H,W,3) torch tensor, an (H,W,3) numpy array, or a PIL image (`image.py:424-444, 506-532`).
- A tensor whose max is ≤ 1 is scaled ×255; otherwise it is clamped to uint8 (`image.py:514-518`).
- It returns `img` normalised and batched, shape (1,3,H,W), and **sets `data_norm_type=["dinov2"]` itself** (`image.py:619-624`). `data_norm_type` is required by `infer` (`inference.py:42`), so the harness must not set it.
- The harness passes uint8 (H,W,3) torch tensors (`exp1/model_io.py:56`). That is correct.
- The encoder's expected normalisation is `dinov2` for the DINOv2 configs (`configs/model/encoder/dinov2_large.yaml:6`). The checkpoint's own config is on HF and was not read. **runtime:** finite outputs in Step 6 and "hints help" in Step 7.

**3. `intrinsics`, `camera_poses`, `is_metric_scale`.**
- `intrinsics`: must be (3,3) (`image.py:584-585`). Inside `infer` it is converted to unit ray directions (`inference.py:218-229`) on an integer pixel grid, i.e. OpenCV convention (`mapanything/utils/geometry.py:212-214`).
- `camera_poses`: (4,4) tensor or ndarray, or a `(quats, trans)` tuple (`image.py:637-667`). The batched (B,4,4) form is split into quaternion + translation (`inference.py:245-263`).
  - Convention: OpenCV cam2world (`README.md:249, 385`).
  - The world frame can be anything: every pose is re-expressed in the frame of view 0 inside the model (`model.py:716-789`, `geometry.py:814`).
- `is_metric_scale`: passed through unchanged (`image.py:669-678`). Its shape is (B,) bool, one per view.
- **Defaults when absent:** a missing optional key means that modality is absent. A missing `is_metric_scale` defaults to **True** (`inference.py:265-272`).
- Allowed keys are fixed (`inference.py:29-40`); any other key raises an error (`inference.py:145-150`).

**4. Outputs.** One dict per view (`inference.py:322-330`, `README.md:170-194`):
- `pts3d` (B,H,W,3), world frame;
- `pts3d_cam`;
- `depth_z` (B,H,W,1) = `pts3d_cam[...,2:3]` (`inference.py:351-353`);
- `camera_poses` (B,4,4), built from `cam_quats` and `cam_trans` (`inference.py:362-378`);
- `conf` (B,H,W);
- `intrinsics` (B,3,3);
- `metric_scaling_factor` (B,);
- `non_ambiguous_mask` (B,H,W);
- `mask` (B,H,W,1), **only when `apply_mask=True`**.

Geometry is predicted up to scale and then multiplied by `metric_scaling_factor`, which comes from a learned scale token (`model.py:225-226, 1584-1596, 1668-1681, 1789-1805`).

`model_io.run` indexes `[0]` for B, squeezes the last dimension of `depth_z`/`mask`, and falls back to all-ones when `mask` is absent (`exp1/model_io.py:84-92`). That is correct.

**5. Resizing.** `preprocess_inputs` always maps the batch's mean aspect ratio to a fixed resolution (`image.py:459-464`). For 518, the 3:2 entry is **(518, 336)** (`image.py:44`); our input aspect is 518/336 = 1.5417, which maps to key 1.542.
- `crop_resize_if_necessary` (`mapanything/utils/cropping.py:385-467`) rescales by `max(target/size) + 1e-8` (`cropping.py:231`). It floors the result back to exactly 518×336 (`cropping.py:240`) and resizes to the same size.
  - Pillow returns an exact copy for a same-size resize: checked locally, PIL 11.1.0, for both BICUBIC and LANCZOS.
  - The crop box is (0,0,518,336) (`cropping.py:429-447`).
  - K changes by about 1e-8 relative.
- **518×336 is kept exactly.** All mapped sizes are multiples of 14.
- **runtime:** `metrics.evaluate` raises if the prediction size differs from GT (`exp1/metrics.py:73-75`).

**6. Partial poses.** Accepted. The only rule is "if any view is posed, view 0 must be posed" (`inference.py:178-188`). Inside the model, views without a pose get their camera mask set to False (`model.py:725-754`).
- Both controls never drop view 0, because corruption never touches it.
- **runtime:** smoke-run control rows.

**7. Missing hints are zero-masked, not learned tokens.**
- The encoders still run on placeholder input: zero rays, zero depth, identity pose.
- Their output features are then multiplied by a per-view 0/1 mask and summed into the image features. One LayerNorm follows the sum:

```python
# rays   model.py:861-867
ray_dirs_features_across_views = ray_dirs_features_across_views * per_sample_ray_dirs_input_mask[..., None, None, None]
all_encoder_features_across_views = all_encoder_features_across_views + ray_dirs_features_across_views
# depth  model.py:1020-1023, 1033-1036, 1043-1052 (depth, depth-scale x depth mask x metric mask)
# pose   model.py:1086-1088, 1144-1146, 1156-1165, 1168-1173 (rot, trans, trans-scale x cam mask [x metric mask])
# fusion model.py:1289-1298 -> self.fusion_norm_layer = LayerNorm(eps=1e-6)  (model.py:113-115, 220)
```

So the code computes **F_E = LN(F_I + Σ_m mask_m · F_m)**.
- The paper's inner LNs, if they exist, are inside the encoders (not checked).
- **"gate = 0" for a modality on a view is exactly "omit that key".** The no-key images-only run is feature-identical to the `ignore_*` images-only mode (`model.py:1991-2004`).
- Caveat for E3: omitting a view's pose also changes ẑp for the other views (Q8). So a per-view gate is not purely local.

**8. Scale normalisation (E2 patch points).**
- **Pose scale ẑp** (`normalize_pose_translations`, `geometry.py:1558-1595`; called at `model.py:1116-1126`):
  - formula: ẑp = Σ_v ‖t_v‖ / #{v : ‖t_v‖ > 0}, over translations already in the view-0 frame;
  - view 0 (t = 0) and unposed views (placeholder t = 0, `model.py:760-762`) are excluded from both the sum and the count;
  - **a mean** (0% breakdown point), one scalar per scene.
  - F_T encodes **t_v/ẑp, not a unit direction** (`model.py:1140-1142`). F_ẑp encodes log ẑp, the same value for every posed view (`model.py:1149-1154`).
- **Depth scale ẑd** (`normalize_depth_using_non_zero_pixels`, `geometry.py:1523-1555`; called at `model.py:987-991`):
  - formula: the per-view **mean of valid depth along the ray**; `depth_z` is converted to depth along the ray first (`inference.py:235-243`);
  - F_D encodes depth/ẑd after the log1p transform (`geometry.py:1666-1679`, `model.py:1011-1018`);
  - F_ẑd encodes log ẑd (`model.py:1026-1031`).

**9. `is_metric_scale`.**
- Per view, per batch element (`model.py:934-954` depth, `1095-1105` pose).
- It **only gates the scale features** F_ẑd and F_ẑp (`model.py:1043-1045, 1162-1165`). Normalised depth and t_v/ẑp are fed regardless.
- Default when absent: True (`inference.py:265-272`).
- `ignore_{pose,depth}_scale_inputs=True` switches the scale features off for all views (`model.py:2015-2016` with `945-950`, `1107-1113`).
- The harness sets True on posed views (`exp1/model_io.py:59-61`); unposed views default to True, which is harmless.

**10. Checkpoint and licence.**
- `configs/pilot.json:3` uses `facebook/map-anything`, per `context.md` §4 CC BY-NC 4.0 (research use).
- The Apache-2.0 alternative is `facebook/map-anything-apache` (`README.md:148`). The code itself is Apache-2.0 (file headers, e.g. `cropping.py:1-4`).
- `run_info.json` records the checkpoint id but not the HF revision. `scripts/e0_phase_c.sh` records the HF snapshot hash in `reports/e0_logs/provenance.txt`.

## Model internals relevant to E2/E3

- **Fusion (E3 gates):** additive and masked, followed by one LayerNorm (`model.py:1177-1305`; sum at `865-867`, `1048-1052`, `1168-1173`; LN at `1289-1298`). These run in fp32 with autocast off (`model.py:1574-1581`). A soft gate g∈[0,1] would scale `mask_m` at those three sum sites.
- **E2 mean→median patch:** `model.py:25-26` imports both normalisers into the model module's namespace by name. So a **median version can be swapped in from `exp1/model_io.py` behind a flag**, without editing `third_party`, by setting `mapanything.models.mapanything.model.normalize_pose_translations = median_fn` (and the same for `normalize_depth_using_non_zero_pixels`). `ablations.py:917, 1052` also calls them, but the `MapAnything` class does not use it.
  - **The median must be taken over the non-zero entries only.** Otherwise view 0 and unposed views (t = 0) dominate.
- **Two routes by which one bad translation spreads to the other views, and how E2 can separate them:**
  - **Route (i):** ẑp scales every other view's F_T input t_v/ẑp. This route is always on.
  - **Route (ii):** the shared log ẑp feature. This route is on only when the view is flagged metric.
  - `infer(..., ignore_pose_scale_inputs=True)` removes (ii) only, so it is a built-in "scale feature off" isolation run. The depth branch works the same way: route (i) is per view only, and route (ii) is `ignore_depth_scale_inputs`.
- **Implication for the preregistration (logic, not a measured result):**
  - Multiplying all non-reference translations by s (`scale_all`) leaves t_v/ẑp exactly unchanged. Its only input channel is log ẑp, which needs the metric flag on, and its main effect should be on `metric_scaling_factor`.
  - The E1 readouts in `context.md` §7 (`pts_relerr_scale`, `rre_deg`, `depth_absrel_scale`) remove or ignore global scale. They may therefore show *no* effect for `scale_all` even if the metric output is badly wrong.
  - The same holds for E1b `depth_scale` (depth/ẑd is unchanged by a global ×1.5).
  - **Decide before any E1 eval run** whether `*_metric` metrics (and `metric_scaling_factor`) are primary readouts for these families.

## Other findings (beyond the 10 questions)

1. **GT depth/image mismatch in `exp1/datasets.py` (fixed).**
   - ETH3D `ground_truth_depth/dslr_images/*` belongs to the **distorted** images, which use a THIN_PRISM_FISHEYE camera. MapAnything's own ETH3D converter re-projects it onto the undistorted grid before use (`data_processing/wai_processing/scripts/conversion/eth3d.py:253-414`; layout `:443-469`). The harness paired it directly with undistorted images and PINHOLE K.
   - Fix: `datasets.py` now re-projects every target pixel through the distorted calibration. This needs `<scene>/dslr_calibration_jpg/` from the `dslr_jpg` archive, a third download; without it the code fails loudly. `--no_depth_undistort` keeps the old behaviour for A/B diagnostics.
   - Checks:
     - the thin-prism projection matches `pycolmap` 4.2.1 to 9e-13 px over 10k random points and 5 parameter sets;
     - end-to-end synthetic tests are in `tests/test_e0_clip_pipeline.py`.
   - On the synthetic slanted-plane scene, the old lookup **still passed the Step-5 "median < 3%" rule** (median 0.61%). Its error grew from 0.17% at the image centre to 1.57% in the corners. The fixed clip was flat, at about 0.12-0.15%. A median-only criterion would therefore miss this class of bug, so `scripts/e0_check_clips.py` also reports a radial error profile.
   - **Real-data evidence (runtime):** on ETH3D the undistorted images are about 6200×4130 (6193-6208 × 4127-4136 per camera), while every depth map is 6048×4032, the distorted camera's size. The harness as delivered therefore **crashed on the first image** with "24385536 floats, expected 6208x4135" (`reports/e0_phase_c_console_attempt1.txt`), and the planned A/B with the old lookup cannot run at all. With the fix, the dev clips pass Step 5 with flat radial profiles (`reports/e0_logs/5_clip_check.txt`).
   - ETH3D GT depth is **sparse**: 8-22% of full-resolution pixels hold a laser point. Nearest sampling keeps that density in the clips, which is the same as MapAnything's own loader (`cropping.py:247-253`, `INTER_NEAREST`). As a result, only 2-5 of the 9 (0, j) pairs per dev clip reach the 5% overlap that Step 5 needs.
2. **Half-pixel convention (not changed).** ETH3D K uses the COLMAP convention, where pixel centres sit at +0.5. `adjust_K` treats it as OpenCV. At clip resolution the offset is 0.5·(518/6048) ≈ 0.04 px, which is negligible.
3. **Rotated images.** Some ETH3D images are stored rotated 90° (`eth3d.py:210-249`; now `datasets.ETH3D_ROTATED`). Their geometry is consistent, so nothing is altered. The layout and clip checks report any that land in a clip, because sideways images may be out of distribution for the model.
4. **Smoke clip name in the harness README.** `dev__courtyard__c0` cannot exist, because courtyard is an eval scene. The fixed split is dev = `delivery_area, facade, terrace`; eval = the other 10 (`datasets.dev_eval_split`). E0 Step 7 already uses `ls dev__* | head -1`.
5. **`exp1/aggregate.py:130`** passes one-element Series to `errorbar` when a family has a single severity. This currently gives a pandas/matplotlib FutureWarning, which may become a TypeError in newer versions. It does not affect E0, which runs no aggregate, nor E1, whose families have 3 severities. Not changed.
6. **Harness dry run.** `tests/test_runner_dryrun.py` runs `run_exp1`, the smoke checker and `aggregate` end to end on CPU with a **fake** model. That covers all 6 smoke conditions, both controls over 3 seeds, contamination subsets, the manifest and resume. It passes, so the plumbing works. It says nothing about MapAnything's numbers.

## Runtime confirmation (E0 Phase C, 2026-10-27)

All items below come from logs in `reports/e0_logs/` and `reports/e0_official_example.txt`.
- **Q1/Q2/Q4:** the README example and every harness `infer` call ran without error. The output keys, shapes and dtypes match Q4 (`reports/e0_official_example.txt`). One correction: `metric_scaling_factor` arrives as shape **(1, 1)**, not (B,). The harness does not use it. The torch hub log line "Loading pretrained dinov2_vitg14" confirms the encoder is DINOv2 ViT-g/14.
- **Q5:** no prediction-size error in any run; outputs are 336×518.
- **Q6:** partial poses are accepted. `drop_oracle` and `filter_rgbcheck` ran for all 3 seeds and each dropped exactly the corrupted view (`7_smoke_check.txt`).
- **Q7 (indirect):** `zero_noise` is identical to `clean_pose`, and repeated runs are bit-identical (below).
- **Determinism:** max |Δ pts3d| = 0 within a process (3 repeats) and across two processes, as is |Δ| of depth, pose and conf (`8_det_a.txt`, `8_det_b.txt`). The noise floor for "no change" is exactly 0 on this GPU and software stack.
- **Q10:** checkpoint `facebook/map-anything`, HF snapshot `a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa` (`provenance.txt`).
- **Split bug (harness, fixed):** `prepare_eth3d` used to split dev/eval over the scenes it was given. A run over 7 eval scenes labelled `electro`, `kicker` and `pipes` as dev. Now `datasets.split_of` always splits over the full ETH3D list, and a test covers it. The mislabelled clips only had the GT check and were rebuilt as `eval__`.
- **Software stack on the VM** (`reports/env_dump.txt`): pandas 3.1.0 and matplotlib 3.11.2. Finding 5 (`aggregate.py:130`) may now raise rather than warn for any single-severity family. Keep every E1 family at ≥ 2 severities, or patch that line before running `aggregate` on such a family.
