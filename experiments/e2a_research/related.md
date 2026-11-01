# E2-A literature audit (draft): wrong geometric hints in feed-forward 3D models

Date read: 2026-10-29 (all sources below opened on this date).
Scope: learned/feed-forward 3D models that take geometric hints (poses, intrinsics, depth, scale), and test-time methods that check or reject hints.

**How the sources were read.**
- Every cited paper was opened: the arXiv abstract page, then the HTML full text (`arxiv.org/html/<id>`) and, where it mattered, the PDF.
- The web-fetch tool caps verbatim quotes at about 125 characters and summarises long pages with a small model. To guard against that, the load-bearing numbers were re-checked against the PDF text (`pdftotext`):
  - WorldMirror Table 12 and App. D;
  - TCO Sec. 9 and Table 8;
  - VPGGT Tables 1 and 3, Sec. 5.2 and App. A.1/A.7;
  - Pow3R Sec. 4.1, Fig. 6 and App. D;
  - GeoFF3D Sec. 4 and Table 5.
- Quotes and numbers that were seen **only** through the HTML fetch are marked *(HTML)*. Re-check them before they go into the paper.

---

## 0. Bottom line for the E2-A gate

**No paper fully overlaps** with our core findings. No paper measures, on MapAnything / OmniVGGT / VGGT-family / Pow3R / WorldMirror:
- copying of wrong rotation or scale hints;
- correlated versus independent errors;
- lens and depth-scale errors;
- confidence-flags-but-model-obeys.

Four papers **partly** overlap and must be cited and differentiated:

| Paper | What it already showed |
|---|---|
| **Pow3R** (2503.17316, CVPR 2025) | Wrong focal / relative rotation on 2 views: the model follows small errors and "refuses" large ones; confidence correlates with hint–GT agreement. Strong partial overlap with our (i) and (iv). |
| **VPGGT** (2610.05531, Oct 2026) | OmniVGGT family; drift-style correlated pose noise (heading-rate and velocity bias) in training and eval; the "prior dilution" metric; explicitly no reliability gate. Partial overlap with our (ii); the closest contemporaneous work. |
| **WorldMirror** (2510.10726, ICML 2026) | Intrinsics ×0.8/×0.6, per-pixel depth noise, 20°/40° rotation and translation noise; on DTU some noisy priors are worse than no prior. Strong partial overlap with our (iii). |
| **TCO** (2604.03878, CVPR 2026) | Test-time method on VGGT. Small rotation/translation/focal perturbations; the combined 5°/10%/10% setting is worse than image-only VGGT on accuracy. Weak partial overlap with our (iii). |

**Correction needed in `context.md` §3 (VPGGT key number).** The quoted "57.11 vs 38.00" is from VPGGT **Table 1(a)**, and that row also supplies **ground-truth depth on 100% of views**: "GT/N specifies pose quality only; supplied depth is ground truth" (PDF, Table 1 note). The clean, pose-only comparison is **Table 3**: released OmniVGGT, TUM RGB-D, RTA@5 is **58.89 with noisy poses (Cn) vs 38.00 with RGB only**, and RRA@5 is 100.00 vs 99.67.

---

## 1. Closest-work table

Legend for "wrong hints tested":
- **IndP** = independent per-view pose noise;
- **CorrP** = correlated pose error (drift, shared heading, global scale);
- **K** = intrinsics error;
- **D** = depth error;
- **S** = scale error.

| # | Paper | ID / venue / URL | Hints accepted | Wrong hints tested | Train / test-time | Key numbers on wrong or noisy hints (quoted) | Code / weights / licence |
|---|---|---|---|---|---|---|---|
| 1 | MapAnything (Keetha et al.) | arXiv 2509.13414 (v3, 23 Jan 2026); https://arxiv.org/abs/2509.13414 | intrinsics (rays), poses (quat + trans), depth, metric-scale flag | **None.** The limitations state "MapAnything does not explicitly account for noise or uncertainty in geometric inputs" *(HTML)*. The only input degradation is depth sparsification: "equal probability of providing dense depth or 90% randomly sparsified depth" *(HTML)* | Training-based (input-selection probabilities; no hint noise described) | none | Code: "licensed under an open-source Apache 2.0 license" (GitHub README). Weights: `facebook/map-anything` CC-BY-NC 4.0; `facebook/map-anything-apache` Apache 2.0 |
| 2 | OmniVGGT (Peng et al.) | arXiv 2511.10560, CVPR 2026; https://arxiv.org/abs/2511.10560 | intrinsics, extrinsics, depth (+ mask) | **None.** Only missing hints: "ensuring the model is robust to various partial-information scenarios during inference" *(HTML)*. The Tables 9-13 percentages are coverage, not corruption *(HTML)* | Training-based (stochastic modality subsets) | none | README: "This project is licensed under the MIT License"; HF `Livioni/OmniVGGT` |
| 3 | VPGGT, "Deep Prior Learning for Embodied Perception" (Wu, Guo, Liu, Li) | arXiv 2610.05531 (4 Oct 2026); https://arxiv.org/abs/2610.05531 | camera (K + pose) and depth priors, OmniVGGT-based | **IndP + CorrP (poses only).** TUM/TartanAir use accumulated drift: "A sequence-constant heading-rate bias contributes" (PDF, App. A.1); translation uses "a sequence-constant velocity bias and white increments" *(HTML)*. C3VD: "The tracker-style profile independently perturbs each absolute pose" *(HTML)*. **No K / D:** "Only poses are corrupted; intrinsics and supplied depth D remain ground truth." (PDF, Sec. 5.2). No separate rotation/translation/scale sweep; "The main evaluation uses a profile multiplier of 1" *(HTML)* | Training-based (noise in training) plus a parameter-free "prior residual connection" | **Table 3, TUM, released OmniVGGT** (R/T/A = RRA@5/RTA@5/AUC@30): RGB only 99.67/38.00/61.94; with exact poses 100.00/70.78/78.95; with noisy poses 100.00/58.89/72.21 (PDF + HTML). **Table 1(a)** (all views GT depth + cameras): GT poses 97.22/67.78/76.66; noisy 97.22/57.11/71.30. **Partial coverage:** VPGGT-PRC with 30% noisy cameras gives RTA 49.78 vs 50.33 RGB-only, i.e. slightly worse (PDF). On TartanAir "on Ocean with Cn, VPGGT obtains 85.81 AUC, below the input prior's 91.66" (PDF) | No code URL in the paper *(HTML)* |
| 4 | WorldMirror (Liu et al.) | arXiv 2510.10726 (v2, 2 Aug 2026), ICML 2026; https://arxiv.org/abs/2510.10726 | poses, intrinsics, depth ("any-prior prompting") | **IndP (rotation 20°/40°, translation σ 0.05/0.1), K (scale ×0.8/×0.6), D (per-pixel multiplicative N(1, σ), σ 0.05/0.1).** Rotation noise follows Pow3R: "rotating the rotation matrix clockwise by 0° (clean), 20°, and 40°" (PDF, App. D). The paper does not say whether the noise is per view or on all views | Training-based (prior dropout p = 0.5; no noise in training) *(HTML)* | **Table 12** (PDF): None 7S-Acc 0.044, DTU-Acc 0.982. Rot 40°: 0.025 / 1.017. Intrinsics 0.6×: 0.051 / **2.691**. Intrinsics 0.8×: DTU-Acc 1.047. Depth σ 0.1: 0.064 / 1.387. Quote: "Camera pose priors show strong robustness up to 20° rotation error, maintaining better performance than the baseline." "Depth priors are more sensitive to noise due to direct geometric guidance" | GitHub `Tencent-Hunyuan/HunyuanWorld-Mirror`; HF `tencent/HunyuanWorld-Mirror`. Licence "TENCENT HUNYUANWORLD-MIRROR COMMUNITY LICENSE AGREEMENT", which states "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA." |
| 5 | TCO, "Learning 3D Reconstruction with Priors in Test Time" (Zhou, Wu, Dave, Samaras) | arXiv 2604.03878, CVPR 2026; https://arxiv.org/abs/2604.03878 | poses, intrinsics (point-map task); depth (pose task), as penalty terms on an image-only VGGT | **IndP-like + K**: "We perturb the camera pose and intrinsic parameters by adding a small random perturbation to the ground truth values." (PDF, Sec. 9). Rotation 1/3/5°, translation 1/5/10%, focal 1/5/10%. Depth priors never perturbed; per-view versus all views is not stated | **Test-time** optimisation (LoRA on VGGT); no per-view weighting or rejection *(HTML)* | **Table 8, ETH3D** (PDF), mean Acc / Comp / N.C.: VGGT base 0.280/0.305/0.853; clean priors 0.1765/0.1816/0.8822; rot 5° 0.2344/0.2143/0.8569; trans 10% 0.2816/0.2381/0.8489; focal 10% 0.2549/0.2654/0.8527; 5°/10%/10% **0.2901**/0.2708/**0.8413**. Quote: "our method still outperforms the base image-only model VGGT in most cases." Runtime (Table 9) *(HTML)*: 1.301 s with 0 steps vs 161.547 s with 40 steps (10 views, TITAN RTX) | GitHub `cvlab-stonybrook/TCO` is public; no LICENSE file seen |
| 6 | G-CUT3R (Khafizov et al.) | arXiv 2508.11379 (v2, 29 Sep 2025); https://arxiv.org/abs/2508.11379 | depth (+ mask), intrinsics, poses, fused into CUT3R by zero-conv | **None.** It aims to handle "noisy depth maps and both intrinsic and extrinsic camera parameters" but runs no corruption experiment *(HTML)* | Training-based (modality dropout) | none | No repository found (paper says source code is provided; no URL) |
| 7 | Pow3R (Jang, Weinzaepfel, Leroy, Agapito, Revaud) | arXiv **2503.17316**, CVPR 2025; https://arxiv.org/abs/2503.17316 | intrinsics K1/K2, relative pose P12, dense/sparse depth (2-view) | **Wrong focal on one camera (f1/f_gt down to 0.1), wrong relative rotation ±40° about y.** A "controllability" study on "10K Habitat test pairs" (PDF). No translation, depth or correlated errors | Training-based (random modality subsets; no noise) | Fig. 6 is plots only, no table. Quotes (PDF): "the model accepts to follow the guidance whenever it is close to the ground truth value ... but then refuses to follow it when it gets too different". "the output confidence clearly correlates with the agreement between the guidance and the ground truth." "The network starts to recover in this case when f1 ≳ 0.5 f1_gt." "This suggests that Pow3R would be robust to noise in the auxiliary information." | GitHub `naver/pow3r`: inference code and checkpoint released ("TODO: release training code"). LICENSE: "Non-Commercial License" |
| 8 | VGGT (Wang et al.) | arXiv 2503.11651, CVPR 2025; https://arxiv.org/abs/2503.11651 | **images only** (no hint inputs) | n/a | n/a | none | README: code "under a commercial-use-friendly license"; "the original checkpoint remains non-commercial"; separate VGGT-1B-Commercial |
| 9 | VGGT-Ω (Wang et al.) | arXiv 2605.15195 (v2, 25 Oct 2026), CVPR 2026 Oral; https://arxiv.org/abs/2605.15195 | images; optional auxiliary inputs (temporal order, camera, depth, scale) discussed only qualitatively, provided only at fine-tuning | **None.** Priors in pretraining are "often detrimental"; "comprehensive exploration is beyond the scope of this paper" *(HTML)* | n/a | none | Project page links `facebookresearch/vggt-omega` and HF `facebook/VGGT-Omega` (gated request). Licence not stated on the pages read |
| 10 | SpatialBench (Peng et al.) | arXiv 2605.27367 (v2, 29 May 2026); https://arxiv.org/abs/2605.27367 | benchmark; injects **GT** depth and/or camera priors into DA3, MapAnything, OmniVGGT, π3-X, WorldMirror | **None (GT priors only):** "We inject ground-truth depth and/or camera (pose + intrinsic) priors" *(HTML)* | Evaluation only | AUC@30, Sparse: MapAnything 0.579 → 0.934 with GT camera *(HTML)*. Quotes: "DA3-Giant and MapAnything exhibit strong prior adherence"; others are "partially overriding them with their own predictions"; "MapAnything fails on object-centric scenes when GT camera priors are injected" *(HTML)* | not checked |
| 11 | UAVFF3D (Yang et al.) | arXiv 2605.17942; https://arxiv.org/abs/2605.17942 | MapAnything, Pi3X, DA3 with intrinsics (C), poses (P), or both (CP) | **None declared** (prior source not stated; no injected noise) *(HTML)* | Evaluation plus UAV fine-tuning | Zero-shot MapAnything (Table 7/11): AbsRel 0.044 (RGB) → 0.105 (P) / 0.099 (CP); Chamfer 2.90 → 5.66 / 5.15. Quote: "the AbsRel and Chamfer-L1 of MapAnything degrade markedly under P and CP inputs"; priors "may not fully match the pretrained assumptions about scale, focal length, pose layout, and scene geometry" *(HTML)* | not checked |
| 12 | GeoFF3D (Yang et al.) | arXiv 2608.28288 (v2, 31 Aug 2026); https://arxiv.org/abs/2608.28288 | georeferenced translations (primary), optional rotations, intrinsics, depth | **IndP (GNSS-like)**: "Gaussian noise is added to horizontal translation, vertical translation, and yaw with standard deviations of 0.5 m, 0.8 m, and 1.0" (PDF; the unit symbol for yaw is garbled in the PDF text) | Training-based: "randomly dropping and perturbing the geometric priors to improve robustness to incomplete and noisy georeferencing inputs" (PDF) | Table 5 (PDF), CD / ATE: default noise 4.50/5.52; 2× noise 5.12/6.34; translation-only 6.56/10.67. No images-only run of the same model under noise | "The code is available at https://github.com/yanxian-ll/GeoFF3D" (PDF); licence not checked |
| 13 | DAVIO (dense mono-inertial SLAM) | arXiv 2609.27702; https://arxiv.org/abs/2609.27702 | DA3 conditioned on **real VIO poses** + intrinsics | Real odometry poses versus GT poses (system level) | Test-time system (no training of DA3) | "Since DA3's default interface returns the input poses, this fit serves as a consistency check rather than as independent evidence of depth accuracy." The system "degrades far less when GT poses are replaced by real odometry"; the error rise is "increases DAVIO's global error by 26 %" vs "ScaRF's by 110 % (0.040 to 0.084 m)" *(HTML)* | Project page https://be2rlab.github.io/DAVIO/ |
| 14 | VIDAR | arXiv 2607.17171; https://arxiv.org/abs/2607.17171 | DA3-Streaming conditioned on real SVO+IMU poses | Real noisy poses (no controlled noise) | Test-time system | EuRoC: pose injection cuts scale error from 0.458 to 0.009 ("VIO injection reduces it to 0.009"); F@0.10 0.463 vs 0.504 with GT poses *(HTML)* | not checked |
| 15 | Depth Anything 3 | arXiv 2511.10647; https://arxiv.org/abs/2511.10647 | optional camera token (pose + FOV) | **None** (GT poses only) *(HTML)* | Training (pose conditioning "probability 0.2") | none | not checked |
| 16 | UniScale | arXiv 2602.23224; https://arxiv.org/abs/2602.23224 | intrinsics, poses | **None.** Only "encouraging robustness to missing inputs" *(HTML)* | Training-based | none | "We will release our implementation upon acceptance." |
| 17 | Any4D | arXiv 2512.10935; https://arxiv.org/abs/2512.10935 | depth, intrinsics, poses (incl. IMU), Doppler | **None.** It says it does "not account for sensor noise" *(HTML)* | Training-based (modality dropout 0.5) | none | not checked |

Adjacent work: **E3-relevant, but about bad images or predictions, not bad hints.**

| Paper | ID | Relevance | Quote |
|---|---|---|---|
| GeoCond | 2609.18465 (16 Sep 2026) | Trained reliability adapter on a frozen VGGT; pose-level uncertainty and a refinement gate. No input hints | failures are "poorly captured by native aleatoric confidence" |
| Emergent Outlier View Rejection | 2512.04012 (3 Dec 2025) | Training-free rejection of **outlier images** from VGGT internals | "without any additional fine-tuning or supervision" |
| ReVar3R | 2610.07883 | Training-free perturbation uncertainty for VGGT/π3/MASt3R; beats built-in confidence | "built-in confidence is outperformed in most evaluated conditions" |
| Can These Views Be One Scene? | 2605.18754 (18 May 2026) | 3D foundation models produce geometry even for inconsistent image sets | flawed inputs "still receive high 3D consistency scores" |
| DEFECt3R (Geometric Inconsistency Localization) | 2609.31247 | Learned pixel-level inconsistency detector between images | "a lightweight learning-based classifier" |
| On Geometric Understanding and Learned Priors | 2512.11508 | Image-level perturbations on DUSt3R/VGGT/DA3; no hint inputs | "challenging input-level perturbations, such as occlusions, scene ambiguities, and varying camera configurations" |
| NoDrift3R | 2607.07168 | Drift of the model's *own* poses in pose-free recon; no hint inputs | "a synergistic pose-free framework" |

---

## 2. Searches run

All are WebSearch queries run on 2026-10-29. "Hits" = result links returned and screened by title/snippet; relevant ones were then opened.

| # | Query | Hits | Relevant found (opened) |
|---|---|---|---|
| 1 | noisy pose priors feed-forward 3D reconstruction robustness | 10 | WorldMirror; GeoCond, 2512.04012, 2512.11508, NoDrift3R (adjacent) |
| 2 | MapAnything noisy inputs robustness corrupted camera poses intrinsics | 9 | UAVFF3D |
| 3 | VGGT camera priors robustness noisy intrinsics extrinsics conditioning | 10 | WorldMirror, TCO, UniScale, 2512.11508 |
| 4 | corrupted camera inputs 3D foundation model prior reliability wrong priors pointmap | 9 | none |
| 5 | uncertainty-aware prior injection feed-forward multi-view reconstruction noisy depth camera priors | 10 | Pow3R, WorldMirror, UniScale |
| 6 | robustness to inaccurate camera intrinsics feed-forward 3D reconstruction wrong focal length prior | 9 | UAVFF3D; Multi-View LRM 2406.07648 (2024; full text not opened, excluded) |
| 7 | outlier pose priors transformer reconstruction prior verification test-time reject bad hints | 9 | 2512.04012 (adjacent) |
| 8 | conditioning on noisy depth feed-forward multi-view geometry transformer depth prior noise sensor | 9 | VPGGT |
| 9 | GeoCond conditioning-aware reliability adapter feed-forward 3D reconstruction | 9 | GeoCond (adjacent) |
| 10 | test-time prior verification 3D reconstruction feed-forward detect unreliable camera priors | 10 | TCO, Pow3R, UAVFF3D |
| 11 | "prior dilution" OR "prior-aware" feed-forward reconstruction noisy GPS poses odometry drift VGGT MapAnything 2026 | 9 | SpatialBench, UAVFF3D, GeoCond |
| 12 | Pi3X π³-X prior injection camera poses intrinsics depth conditioning noisy | 10 | SpatialBench (π3-X has no paper with noisy-prior tests) |
| 13 | feed-forward reconstruction geometric priors ablation noise rotation degrees translation noise focal error "worse than" image-only 2026 arXiv | 9 | UAVFF3D |
| 14 | model trusts erroneous geometric prior over images confidence detects inconsistent prior multi-view transformer | 9 | 2604.09366 (no), 2605.18754, 2609.31247 (adjacent) |
| 15 | OmniVGGT noisy camera priors evaluation robustness wrong depth prior scale | 9 | VPGGT, SpatialBench, WorldMirror |
| 16 | noisy GPS prior feed-forward 3D reconstruction georeferenced poses MapAnything VGGT UAV robustness to pose prior noise | 9 | GeoFF3D |
| 17 | conflicting prior and image evidence feed-forward 3D model follows wrong pose hint analysis attention contamination | 9 | none new |
| 18 | G-CUT3R guided 3D reconstruction camera depth prior github code (lookup) | 18 | no code repo found |

Total: 18 searches, about 176 result links screened (with duplicates).

**Citing-paper check (Semantic Scholar Graph API, 2026-10-29).**
- `arXiv:2509.13414/citations`, MapAnything: **399** citing papers (384 with abstracts), in one page (limit 500).
- `arXiv:2511.10560/citations`, OmniVGGT: **25** citing papers (23 with abstracts).
- Screening:
  1. A keyword regex over titles and abstracts gave 54 strong-keyword hits ("noisy pose", "camera prior", "pose prior", "drift", ...).
  2. 26 hits had noise/corruption phrases near prior/pose/depth words.
  3. 64 hits had "prior/condition" + "pose/intrinsic/depth/GPS/IMU" + "robust/noise/error" co-occurring.
  4. 25 abstracts were read in full, and 9 papers were opened (VPGGT, WorldMirror, TCO, UAVFF3D, DAVIO, VIDAR, GeoFF3D, UniScale, Any4D).
- **No citing paper beyond those in the table tests wrong hints on MapAnything or OmniVGGT.**
- Other citing papers screened and judged not relevant: Any3R, VI3, FoundationVIO, UniMetric, IntriAdapt, SurroundNEXO, DrivingDepth, Foundation4D, Glob3R, Hybrid Cross-Device Localization (MapAnything conditioned on inputs, no corruption), Geometry OR Tracker (fixes bad calibration outside the model), ViGeo, LDCM.
- SpatialBench did not appear in the Semantic Scholar list; it was found by search.

---

## 3. Overlap verdicts

Our core findings:
- (i) copying / absorption of wrong rotation or scale hints;
- (ii) correlated (drift, shared heading, global scale) versus independent errors;
- (iii) lens and depth-hint errors;
- (iv) the model's own confidence flagging bad hints, or test-time detection of bad hints from images.

Rule used: a paper that only adds noise in training, or only reports aggregate accuracy under noise, is at most "partly".

### Pow3R (2503.17316, CVPR 2025): **partly (strong on i and iv)**
- **(i) partly.** It measures output versus input for wrong focal and wrong relative rotation, which is a direct copying test, but only 2-view, ±40° about y, on Habitat. Its conclusion is the **opposite** of ours: "the model accepts to follow the guidance whenever it is close to the ground truth value ... but then refuses to follow it when it gets too different" (PDF Sec. 4.1). App. D: "The model adheres to the guidance until it reaches a breaking point" (PDF).
- **(ii) no.** 2-view only; no drift, shared or scale errors.
- **(iii) partly.** Wrong focal on K1 only: "The model fails to generate accurate pointmaps for blatantly false input focals" (PDF, Fig. D). No depth-hint errors.
- **(iv) partly (strong).** "In all cases, the output confidence clearly correlates with the agreement between the guidance and the ground truth." (PDF). Fig. 6 caption: "the model seems to not take the input information at face value and instead makes an informed decision whether to use it or not, in which latter case the confidence becomes low." But it gives no per-view detection AUROC and no deployable rejection method, and it claims without testing that "Pow3R would be robust to noise in the auxiliary information."
- **Consequence for us.** "Confidence flags the bad hint" is **not new as a concept**; we must cite Pow3R. What is new is "confidence flags it **and** the model still copies it (absorption 0.96)" in multi-view MapAnything, which contradicts Pow3R's "refuses to follow".

### VPGGT (2610.05531, Oct 2026, contemporaneous): **partly (ii); closest on OmniVGGT**
- **(i) partly.** It measures how much of the prior's accuracy survives: "prior dilution, where predictions are less accurate than their supplied pose priors" (abstract, *(HTML)*), with closure κ = (M_out − M_0)/(M_prior − M_0). That is the **under-use** direction, not copying, and it gives no per-error absorption. Its conclusion: "prior availability alone does not ensure prior preservation, particularly at partial coverage" *(HTML)*.
- **(ii) partly.** It uses correlated drift ("A sequence-constant heading-rate bias contributes", PDF A.1) on TUM/TartanAir and independent noise on C3VD. But these are dataset-specific profiles at fixed size ("profile multiplier of 1"), never compared at equal magnitude, and it has no global-scale or shared-heading condition. Its released-OmniVGGT result **favours hints**: RTA@5 58.89 (noisy) vs 38.00 (RGB only), Table 3, PDF.
- **(iii) no.** "Only poses are corrupted; intrinsics and supplied depth D remain ground truth." (PDF Sec. 5.2).
- **(iv) no.** "The present model does not include an explicit reliability gate" (PDF App. A.7); detecting and rejecting unreliable priors is called "an important direction" *(HTML)*.
- **Also relevant to us.** It cites MapAnything ("identifies noise and uncertainty in geometric inputs as a remaining limitation") and Pow3R ("examines responses to incorrect camera parameters") *(HTML)*. So VPGGT's authors frame the same gap but do not fill it.

### WorldMirror (2510.10726, ICML 2026): **partly (strong on iii)**
- **(i) no.** No input-versus-output analysis.
- **(ii) no.** Rotation "clockwise by ... 20°, and 40°" and isotropic translation noise; whether the error is shared across views is not specified (PDF App. D).
- **(iii) partly (strong).** Intrinsics "scale ... by factors of 1.0× (clean), 0.8×, and 0.6×" and depth "pixel-wise multiplicative noise" (PDF). Table 12: DTU-Acc 2.691 at 0.6× and 1.047 at 0.8×, vs 0.982 with no prior; depth σ 0.1 gives DTU-Acc 1.387, i.e. **worse than no prior**. Its depth error is per-pixel noise, **not** a depth-scale error. The paper's own reading is "graceful performance degradation", although its table shows harm on DTU.
- **(iv) no.**

### TCO (2604.03878, CVPR 2026): **partly (weak on iii); different paradigm**
- **(i) no.**
- **(ii) no.** "small random perturbation"; unclear whether all or some views are perturbed (PDF Sec. 9).
- **(iii) partly.** Focal perturbation of 1/5/10%. Table 8: 5°/10%/10% gives Acc 0.2901 vs VGGT base 0.280 (worse), and trans 10% gives 0.2816 (worse) (PDF).
- **(iv) no.** Priors are uniform penalty terms, with no per-view rejection *(HTML)*.
- It is a test-time method, but it **uses** hints rather than **verifying** them.

### SpatialBench (2605.27367): **no (adjacent on i)**
GT priors only. It reports that "DA3-Giant and MapAnything exhibit strong prior adherence" while OmniVGGT, π3-X and WorldMirror are "partially overriding them with their own predictions" *(HTML)*. This is consistent with our absorption finding for MapAnything, but it tests no wrong hints.

### UAVFF3D (2605.17942): **no (adjacent)**
The prior source is unstated and no deliberate corruption is applied. It shows that MapAnything zero-shot dense geometry gets **worse than RGB-only** with poses in the UAV domain: AbsRel 0.044 → 0.105 *(HTML)*. This is useful as "hints can hurt even when not corrupted", but it is a domain-mismatch effect and does not overlap our findings.

### GeoFF3D (2608.28288): **no**
- Independent GNSS-like translation and yaw noise, used in training and test.
- No images-only comparison of the same model, no correlated errors, no copying or confidence analysis.
- It states that "Translation priors serve as coordinate-conditioning signals rather than being directly copied to the output." (PDF)

### DAVIO (2609.27702) / VIDAR (2607.17171): **no**
- Both use real VIO poses as hints for DA3, and both report gains, not harm. VIDAR: scale error 0.458 → 0.009 *(HTML)*.
- DAVIO's remark that "DA3's default interface returns the input poses" is relevant to our E2-B check ("copying by code"), but for DA3, not MapAnything.
- Both are useful as real-sensor-hint references for E2-E.

### No overlap on (i)-(iv)
MapAnything, OmniVGGT, G-CUT3R, VGGT, VGGT-Ω, DA3, UniScale and Any4D: **no**. None tests wrong hints; quotes are in §1.

### Adjacent E3 work
GeoCond, Emergent Outlier View Rejection, ReVar3R, "Can These Views Be One Scene?", DEFECt3R: **no** for hints. They detect bad *images* or unreliable *predictions*.
- GeoCond and ReVar3R both report that native confidence is weak for their failure types. That is worth citing next to our AUROC 0.91.
- Emergent Outlier View Rejection is the closest **training-free** detector design, but it works on images, not hints.

**Fully overlapping papers: none.**

---

## 4. What is already known, what is still open

**Known (2025–2026).**
- Prior-conditioned feed-forward models degrade but usually stay better than images-only under **moderate independent** pose noise:
  - WorldMirror up to 20° rotation;
  - TCO up to 3-5° or 5-10%;
  - VPGGT's released OmniVGGT under realistic drift-style noise beats RGB-only on TUM translation accuracy (58.89 vs 38.00).
- **Wrong intrinsics and strong depth noise can be worse than no prior**: WorldMirror on DTU at 0.6×/0.8× focal and σ = 0.1 depth; TCO-VGGT at 5°/10%/10%. The authors frame this as "graceful degradation" rather than harm.
- In two-view Pow3R, a model follows small wrong focal/rotation hints, rejects gross ones, and its confidence drops when the hint disagrees with the truth.
- With **clean** hints, MapAnything and DA3 adhere strongly to injected poses (SpatialBench). Even un-corrupted hints can hurt MapAnything zero-shot in an unfamiliar domain (UAVFF3D).
- No published model has a reliability gate. VPGGT says so explicitly, and MapAnything lists noisy inputs as a limitation.

**Still open (no paper measures these on MapAnything/OmniVGGT/VGGT-family/WorldMirror).**
- Paired, per-error-type harm versus images-only with scene-level statistics in **multi-view** MapAnything.
- **Absorption**: how much of a wrong rotation or metric scale ends up in the output (our 0.96; ×1.5 scale → ×1.38). This sits against Pow3R's "refuses to follow" and VPGGT's "dilution".
- **Correlated versus independent** errors at matched magnitude: shared heading, drift, global scale.
- **Depth-scale** errors, as opposed to per-pixel depth noise.
- **Contamination** of views whose hints were correct.
- The "**knows but obeys**" pattern: confidence flags the bad view but the output still copies it.
- **Training-free, image-witnessed detection and switching-off** of bad hints, including group-level detection for correlated errors.

The closest threats are Pow3R for the confidence idea (iv) and VPGGT for drift-style noise on OmniVGGT (ii). We must cite both and state the differences:
- multi-view MapAnything versus 2-view Pow3R;
- copying versus refusal;
- matched correlated-versus-independent design;
- lens/depth-scale coverage;
- test-time detection without training.

---

## 5. Not opened or not used
- **Multi-View LRM** (2406.07648, 2024): the abstract was opened, but the noise-robustness claim came only from a search snippet. Excluded.
- **Pi3X README / GitHub issue #161** ("Pi3X does not benefit from extra camera intrinsics and poses"): seen in search results only. Not opened, not cited.
- **GeoFF3D**: the HTML (arxiv.org/html/2608.28288 and /v2) returned 404, so it was read from the PDF instead.
- **WorldMirror App. D**: the full paragraph could not be quoted verbatim through the fetch tool. It was read in full from the PDF text; the quotes above are from the PDF.
- **DAVIO**: the absolute error numbers seen through the HTML fetch (0.069 → 0.080 m) do not match the quoted "+26 %". Only the percentages are quoted; re-check against the PDF before citing.
- **Licences not checked**: SpatialBench, UAVFF3D, VIDAR, DA3, Any4D, GeoFF3D. VGGT-Ω's licence is not stated on its project page or repo landing page.
