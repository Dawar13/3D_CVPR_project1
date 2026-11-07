# M7: were the models ever shown wrong hints during training? (E2X, laptop audit, 2026-11-07)

Source: a code and paper audit by a research sub-agent.
- **Code:** read in shallow clones at the commits listed below.
- **Papers:** read through the arXiv HTML summariser. **Every paper quote must be checked against the PDF before it goes into the paper.**
- No model was run. This is the input to hypothesis M7 in `experiments/E2X.md` (section X-M).

## How each model got its hints during training

| Model | Hints | Training hint source | Noise on hints | Dropout / selection | Where it says so |
|---|---|---|---|---|---|
| MapAnything | rays (K), pose (quat + trans + scale), depth, metric flag | **Exact ground truth**; the same batch goes to the model and to the loss (`preds = model(batch)`, then `criterion(batch, preds)`) | **None** (grep for noise/perturb/jitter finds nothing on hints; only ColorJitter on images). Also: 90% random depth sparsification (p 0.5); metric-scale features withheld (p 0.05) | 0.9 overall per sample; each view kept with 0.95; rays, depth, pose each 0.5 per sample, shared by all views | Paper §3.3; `configs/model/task/aug_training.yaml:5-23` (used by `bash_scripts/train/main/mapa_curri_24v_13d_36ipg_64g.sh:46`); `model.py:897, 944-951, 1107-1110, 1200-1243`; `utils/inference.py:108-110`; `datasets/base/base_dataset.py:115-124`; commit `3d10cf7a` (our pin) |
| OmniVGGT | pose + K (one token), depth | Exact ground truth (clones of the batch extrinsics and depth) | **None** (App. A.2 lists only ColorJitter) | Camera: none with p 0.1, else k ~ U{0..S}, given to the *first* k views. Depth: none with p 0.3 (config), else k ~ U{0..S} random views | §4.3, App. A.2; `omnivggt/models/omnivggt_aggregator.py:135-161, 190-191`; `configs/train.py:32-33`; `train_omnivggt.py:235-252`; commit `c8cbfc6` |
| Pi3X | rays (from K), relative pose, depth | Ground truth, then perturbed | **Poses only:** about 50% of samples get noise on views 1..N-1. Magnitude unknown: `add_randomized_smooth_pose_noise_torch` is not defined in the release; a related released function uses rotation σ ~ U(0, 2°) per axis and translation σ ~ U(0, 0.05). Separately, a ×U(0.8, 1.2) scale jitter on the normalisation (not a wrong hint) | Per view: depth 0.3, rays 0.5, pose 0.2 | No paper. Training branch `97ae7a43`, `pi3/models/pi3x_dirty_training/pi3x_training.py:308-346, 350, 369, 378-384`; `datasets/base/utils.py:465-556`. **Unverified** whether the released weights were trained with this code |
| WorldMirror | pose, K, depth | Ground truth from the dataset views (`src/models/models/worldmirror.py:229-269`) | **None in the training code.** App. D noise (rotation 20°/40°, translation σ 0.05/0.1, K ×0.8/0.6, depth ×N(1, 0.05/0.1)) is evaluation only | Each modality on with p 0.5, independently, per batch (all views together; App. C.2) | §3.1, App. C.2, App. D; `training/wrapper.py:52, 296-314`; commit `c61300e` |
| DA3 | pose + K (camera token) | Not stated; no training code released | Not mentioned | Pose conditioning active with p 0.2 | §3.4; repo `3d835ec` is inference only |
| VPGGT | OmniVGGT's hints | Ground truth plus simulated corruption, **poses only** | **Yes:** drift model (heading-rate bias, heading increments, roll/pitch, velocity bias, translation noise; Table 6), scaled by g ~ LogU(0.6, 1.8). Clean vs corrupted proportion not given | OmniVGGT's | Sec. 5.2, App. A.1, Table 6; no code |
| Pow3R | K1, K2, sparse D1/D2, relative pose | Ground truth (depth with a random mask; translation normalised) | Not described; none in the released helper | Uniform number and choice of modalities | §3.2, §3.4; `pow3r/datasets/utils/modalities.py:19-41, 57-67`; commit `25d7bb6` |

## What this says about M7

- **MapAnything: copying is a zero-loss shortcut in training.** The pose hint is exactly the training target and was never wrong, so copying it is always optimal during training. That fits our measurements: absorption 0.93 at 15° and 0.96 at 30° (E2-C), no override at any size.
- **"Clean training → copying" is not the whole story.** OmniVGGT (absorption 0.56 at 15°, E2-D) and WorldMirror also trained on exact hints yet partly override them (SpatialBench). Pi3X is the only one of the five SpatialBench models with training-time hint noise, and it partly overrides, which fits the hypothesis but is one model.
- **The design differences are the stronger candidate explanation (not tested yet; M6 tests it):**
  - MapAnything adds pose features to every patch token before the transformer.
  - OmniVGGT feeds poses through a zero-initialised adapter, and in training only the first k views get poses.
  - Pi3X injects relative poses inside attention at 5 depths.
- **Controlled test, not planned yet:** the same model fine-tuned with and without hint noise, with copying measured. VPGGT trains with noise but measures the opposite failure (output less accurate than the hint), not how much of a wrong hint is copied.

## Related ideas outside 3D (to cite after checking)

- The "copycat problem" in imitation learning (Wen et al., NeurIPS 2020, arXiv 2010.14876).
- Causal confusion (de Haan et al., NeurIPS 2019).
- Exposure bias / scheduled sampling (Bengio et al., 2015).
- Shortcut learning (Geirhos et al., 2020).
- Depth completion evaluated with GT sparse depth and degrading under noise (arXiv 2412.14103).

The last three were not re-opened by the audit.

## Could not verify

- Pi3X noise magnitude, and whether the released weights used the "dirty" training code.
- DA3 and Pow3R training hint sources (no training code).
- WorldMirror's paper never states outright that training priors are ground truth; that comes from the code.
- VPGGT's clean vs corrupted proportions.
- All paper quotes came from the summariser.
