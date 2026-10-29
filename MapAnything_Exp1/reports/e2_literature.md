# E2-A: literature audit

**Date read:** 2026-10-29 for every source. **Plan:** `experiments/E2.md` §1.
**Gate result: PASS.** No paper already measures our core findings on these models. Four papers overlap partly and must be cited and differentiated (§1.3).

**How it was done.**
- Three literature agents searched and opened the sources. Their full drafts, with every quote and URL, are kept as working notes; this report takes the tables from them.
- I re-checked the load-bearing numbers myself against the PDF text:
  - Pow3R p. 7;
  - VPGGT Table 3;
  - ADVIO p. 13;
  - Photo Tourism p. 12;
  - Strasdat et al. p. 7;
  - Feigl et al. Table 3;
  - Abadi et al. p. 1;
  - Metric3D v2 Table 3.
- Items marked *(HTML)* were read only through a web-page summariser. Re-check them against the PDF before they go into the paper.

---

## 1. Closest work

### 1.1 Table

Wrong hints tested: **IndP** independent per-view pose noise; **CorrP** correlated pose error (drift, shared heading, global scale); **K** intrinsics; **D** depth.

| Paper | ID / venue | Hints accepted | Wrong hints tested | Training / test-time | Key numbers on wrong hints | Code, weights, licence |
|---|---|---|---|---|---|---|
| MapAnything | 2509.13414 v3 | rays (K), pose, depth, metric flag | none; limitation: "does not explicitly account for noise or uncertainty in geometric inputs" *(HTML)* | training (input dropout) | none | code Apache 2.0; weights CC BY-NC 4.0 or Apache 2.0 |
| OmniVGGT | 2511.10560, CVPR 2026 | K, pose, depth | none (missing hints only) | training | none | MIT |
| VPGGT ("Deep Prior Learning for Embodied Perception") | 2610.05531, Oct 2026 | K, pose, depth (OmniVGGT-based) | IndP + CorrP drift ("A sequence-constant heading-rate bias", App. A.1); **poses only**: "Only poses are corrupted; intrinsics and supplied depth D remain ground truth" (Sec. 5.2) | training + "prior residual connection" | Table 3, TUM, released OmniVGGT: RTA@5 **58.89** noisy poses vs **38.00** images only; RRA@5 100.00 vs 99.67 (verified in PDF). No reliability gate (App. A.7) | no code URL |
| WorldMirror | 2510.10726, ICML 2026 | pose, K, depth | IndP rotation 20°/40°, translation noise; K ×0.8/×0.6; per-pixel depth noise σ 0.05/0.1 (App. D) | training (prior dropout) | Table 12, DTU accuracy: no prior 0.982; K ×0.6 **2.691**; depth σ 0.1 **1.387** (worse than no prior); authors call it "graceful performance degradation" | Tencent community licence, not valid in EU/UK/South Korea |
| TCO | 2604.03878, CVPR 2026 | pose, K, depth as test-time penalties on VGGT | small IndP + K: rotation 1/3/5°, translation 1/5/10%, focal 1/5/10% (Sec. 9) | **test-time** (uses hints, never rejects them) | Table 8, ETH3D accuracy: VGGT 0.280; 5°/10%/10% **0.2901** (worse) | public code, no LICENSE file |
| G-CUT3R | 2508.11379 | depth, K, pose | none | training | none | no repository found |
| Pow3R | 2503.17316, CVPR 2025 | K, relative pose, depth (2 views) | wrong focal on one camera; relative rotation up to ±40° (10K Habitat pairs) | training | qualitative (Fig. 6): "accepts to follow the guidance whenever it is close to the ground truth value ... but then refuses to follow it when it gets too different"; "the output confidence clearly correlates with the agreement between the guidance and the ground truth" (p. 7, verified) | inference code + checkpoint, Non-Commercial licence |
| VGGT | 2503.11651, CVPR 2025 | images only | n/a | n/a | n/a | code permissive; original checkpoint non-commercial |
| VGGT-Ω | 2605.15195 | images; optional priors only at fine-tuning | none ("beyond the scope") *(HTML)* | n/a | none | gated HF; licence not stated |
| SpatialBench | 2605.27367 | injects **GT** priors into DA3, MapAnything, OmniVGGT, π3-X, WorldMirror | none (GT only) | evaluation | "DA3-Giant and MapAnything exhibit strong prior adherence"; others "partially overriding them" *(HTML)* | not checked |
| UAVFF3D | 2605.17942 | MapAnything, Pi3X, DA3 with K and/or poses | none injected | evaluation + fine-tuning | zero-shot MapAnything depth AbsRel 0.044 (images) → 0.105 (with poses) in the drone domain *(HTML)* | not checked |
| GeoFF3D | 2608.28288 | GNSS translations (+ rotation, K, depth) | IndP GNSS-like noise in training and test | training | no images-only comparison under noise | code on GitHub |
| DAVIO / VIDAR | 2609.27702 / 2607.17171 | DA3 with **real** VIO poses | real odometry vs GT poses | test-time systems | both report gains from real VIO poses *(HTML)* | — |
| Depth Anything 3, UniScale, Any4D | 2511.10647, 2602.23224, 2512.10935 | pose / K / depth | none | training | none | — |

Adjacent to E3, about bad **images** or unreliable **predictions**, not bad hints: GeoCond (2609.18465), Emergent Outlier View Rejection (2512.04012), ReVar3R (2610.07883), "Can These Views Be One Scene?" (2605.18754), DEFECt3R (2609.31247). GeoCond and ReVar3R both report that the models' built-in confidence is weak for their failure types.

### 1.2 Searches
- 18 web searches with different wordings, about 176 results screened.
- Citing papers via the Semantic Scholar API: 399 for MapAnything, 25 for OmniVGGT. These were keyword-screened, about 25 abstracts were read in full, and 9 papers were opened.
- Query list: see the working notes.

### 1.3 Overlap verdicts against our findings
Our findings:
- (i) the model copies wrong rotation and scale hints;
- (ii) correlated vs independent errors;
- (iii) lens and depth-scale errors;
- (iv) its own confidence flags the bad hint, or bad hints are detected at test time.

| Paper | (i) copying | (ii) correlated | (iii) lens / depth | (iv) confidence / detection | Verdict |
|---|---|---|---|---|---|
| Pow3R | partly: 2-view, finds the **opposite** ("refuses to follow" gross errors) | no | partly (focal on one camera) | partly: confidence correlates with hint–truth agreement; no detector, no AUROC | **partly (strongest threat for iv)** |
| VPGGT | partly, opposite direction: "prior dilution" (output less accurate than the hint) | partly: drift profile at one fixed size; never matched against independent noise | no (poses only) | no ("does not include an explicit reliability gate") | **partly** |
| WorldMirror | no | no | partly: focal ×0.6/0.8, per-pixel depth noise; not depth *scale* | no | **partly** |
| TCO | no | no | weak | no (uses hints, never checks them) | **partly (weak)** |
| all others | no | no | no | no | no |

**What we must say in the paper:**
- **Copying vs refusing.** Multi-view MapAnything copies a bad pose (median absorption 0.96) where Pow3R reports refusing gross errors. E2-C C1 tests up to 180°, which checks directly whether MapAnything has an "override point".
- **Knows but obeys.** "Confidence flags the bad hint" is Pow3R's observation; "flags it and copies it anyway" is ours.
- **What VPGGT does not cover.** Correlated vs independent errors at matched size, lens and depth-scale errors, and contamination of good views.
- **Detection.** No published model or method detects and switches off bad hints at test time from the images.

---

## 2. Classical tools for wrong measurements (context for E3; no implementation in E2)

| Tool | Source | One-line relevance |
|---|---|---|
| Robust M-estimation | Huber 1964, Ann. Math. Statist. 35(1) | Middle ground between MapAnything's mean pose scale and the C4 median probe |
| Least median of squares; breakdown point | Rousseeuw 1984, JASA 79 | The mean (least squares) has breakdown point 0 ("ε* = 0"); a median tolerates nearly 50% contamination |
| Pairwise consistency maximisation (PCM) | Mangelson et al., ICRA 2018 | Keep the largest mutually consistent set of measurements, with no trusted backbone. Cannot see errors shared by all hints |
| Graduated non-convexity (GNC) | Yang et al., RA-L 2020 (1909.08605) | Robust to "70-80% of outliers" without an initial guess |
| Rotation averaging | Hartley et al., IJCV 2013 | Classical way to check per-view rotation hints against image-derived relative rotations. Our harm is rotation-driven |
| Robust bundle adjustment | Triggs et al. 2000, LNCS 1883 | Robust losses fix isolated bad observations, but "If all of the observations had been affected about equally ... we might still know that something was wrong, but not be able to fix it by simple data cleaning" |
| Covariance intersection | Julier & Uhlmann, ACC 1997 | Fuses estimates whose error correlation is unknown, e.g. hints computed from the same images |

---

## 3. Realistic error sizes (replaces the E1 rule "realistic = all but the most severe setting")

Machine-readable labels: `configs/e2_lit_realistic.json`.
- **Unit conversions** use L = 0.25–6.76 m (median 1.2 m) and a median clip path of 5.6 m.
- **Labels:**
  - "realistic" = inside the documented range of at least one named real hint source;
  - "edge" = at the limit of a source;
  - "beyond" = larger than every source found;
  - "unsourced" = plausible, but no source gives a size.

| E1 family | Settings | Label | Main evidence (source) |
|---|---|---|---|
| noise_rot (all views) | 0.5, 1, 2° | realistic | phone VIO orientation "less than < 2° error ... on average" (ADVIO, Cortés et al. ECCV 2018, 1807.09828, p. 13) |
| | 5° | realistic (compass, not VIO) | compass mean deviations 5.5–6.9° (Tartachynska & Pylypyak 2026) |
| noise_trans | 0.005–0.05 L | realistic | VIO 5.94–14.55 cm/m (Feigl et al., GRAPP 2020, Table 3) |
| outlier_rot (one view) | 5, 15° | realistic | indoor magnetometer heading "average heading error ... 12.45 degrees" (Abadi et al., IPIN 2013 per file name, p. 1); isolated deviations 13–25° |
| | 30° | edge | ">30°" spread in some device tests (Jones 2021, conference abstract) |
| | 90° | beyond | no source; a gross blunder only |
| outlier_trans_small | 0.1–0.5 L | unsourced | VIO "wild jumps" exist (ADVIO), no size given |
| outlier_trans_large (GPS jump) | 1, 3, 10 L | realistic | phone GPS 4.9 m open sky (GPS.gov); 7–13 m urban, max 99.7 m (Merry & Bettinger, PLOS ONE 2019). 10 L is inside for L ≤ 1.3 m |
| outlier_frac | 10–40% | unsourced | the per-view error is realistic (indoor compass); no source gives the fraction of views affected |
| scale_all | ×0.5–×2 | realistic (monocular SLAM) | "A relative scale change of 1 : 0.37 was detected" over one loop (Strasdat et al., RSS 2010, p. 7). ×0.8/×1.25 are at the edge for phone VIO |
| drift | levels 1–3 | **beyond** | Our heading rate is 0.5/1/2° per photo = 0.8/1.6/3.2° per metre on the median clip. Phone VIO stays under 2° over sequences averaging 194 m (< 0.01°/m); gyro-only pedestrian heading is about 0.12°/m (24.3° over 294 steps; Yang et al. 2020, read via OpenAlex). The scale part of levels 1–2 (5.1%/10.6% of the path) is inside VIO drift (Feigl et al.); level 3 is beyond |
| shared_heading_bias | 2, 5, 10° | realistic | compass bias 1–3° after calibration, 2–3× worse later (Tartachynska & Pylypyak 2026); 12.45° indoors (Abadi et al.) |
| anchor_error | 15° / 0.25 L | realistic | indoor compass; VIO-level offset |
| focal_all (± poses) | ×0.9, ×1.1 | realistic | EXIF focal "usually good, but can occasionally be off by more than a factor of two"; SfM accepts 0.7–1.4 (Snavely et al., TOG 2006, p. 12); a vendor flags > 5% (Pix4D) |
| | ×0.8, ×1.2 | edge | inside the 0.7–1.4 acceptance window; no error distribution was found |
| focal_one | ×0.7, ×1.3 | edge | the limit of that window |
| principal_shift | 0.05, 0.1 W | beyond | default vs calibrated principal point ≤ 2.5% of width (TUM RGB-D); crop bugs unsourced |
| depth_scale, depth_one_scale (± poses) | ×0.67, ×1.5 | realistic for **monocular depth networks**, beyond for active sensors | ETH3D AbsRel 0.342–0.416 (Metric3D v2) and 0.545 (ZoeDepth) (Metric3D v2, TPAMI 2024, Table 3). Active sensors: 2–3.5% bias (TUM Kinect depth correction 1.031–1.035; Kinect v1/v2, Wasenmüller & Stricker 2016; iPhone LiDAR ±1 cm, Luetzenburg et al. 2021) |
| depth_noise | σ 5%, 10% | realistic for monocular depth and depth edges | Metric3D v2 zero-shot AbsRel 0.05–0.07 on NYU/KITTI |
| metric_flag_off | — | categorical | a user error, no size to source |

**Pattern.**
- The E1 harms that survive (E2-B) sit at compass/magnetometer-level heading errors, monocular-SLAM-level scale errors and monocular-network depth errors.
- These are realistic for phones without working VIO, for SLAM without IMU, and for depth from a network. Working ARKit/ARCore VIO is milder.
- E1's drift family is beyond every sourced drift rate. E2-C should add literature-sized drift levels (about 0.01–0.1° per metre) before claiming drift harm.

**Gaps:**
- no source for phone tilt (gravity) error, so E1's random-axis rotations include tilt that may be unrealistic;
- no distribution of EXIF focal errors;
- no size for VIO jumps or for crop-induced principal-point errors;
- ETH3D gives no photo timestamps, so per-second rates cannot be converted per photo.

---

## 4. Datasets with real sensor hints (for E2-E)

| Dataset | Licence and access | Real hints | Ground truth | In MapAnything training? | In OmniVGGT training? | Use |
|---|---|---|---|---|---|---|
| **ARKitScenes** (raw) | Apple licence, non-commercial; open per-video download script | ARKit poses, iPad LiDAR 256×192, per-frame K | Faro laser scan; GT poses by PnP + dense photometric refinement; GT depth projected from the scan | **no** | **yes** (Table 8) → use the **Validation** split | **start now.** First check which `.traj` file holds ARKit vs GT poses |
| SceneFun3D (built on ARKitScenes) | CC BY 4.0 + ARKitScenes terms; open | `lowres_poses.traj` = ARKit; iPad LiDAR | `hires_poses.traj` = COLMAP in the laser-scan frame | no | shares ARKitScenes captures (split mapping unverified) | fallback if the ARKitScenes pose files are ambiguous |
| **TUM RGB-D** | CC BY 4.0; open; 0.3–2 GB per sequence | Kinect depth; default K (525/525) vs calibrated K (517.3/516.5 for Freiburg 1) = a real lens error | motion capture, absolute error < 10 mm and 0.5° (Sturm et al. 2012); **no GT depth** | no | no | small side set for real depth and real K errors; VPGGT also uses it |
| **ScanNet++ v2** | non-commercial terms; **signed PDF with handwritten signatures of the requester and the PI**, then approval and a token | raw ARKit poses + K, iPhone LiDAR | COLMAP poses aligned to a sub-mm Faro scan; rendered depth | **yes, 926 scenes**; only MapAnything's 30-scene test list is held out | yes | best pairing, but it needs **your signature request now** and only the 30 held-out scenes are usable |
| LaMAR | Google form; data licence not stated | ARKit poses, K, iPad LiDAR | built *using* the device's relative poses (not independent), within 10 cm at 99.7% | no | no | optional |

**Training-set check for our ETH3D scenes:**
- MapAnything lists ETH3D (13 scenes) only under held-out benchmarking; Hugging Face has an ETH3D scene list only under `test/`.
- OmniVGGT and VGGT use ETH3D for evaluation only.
- So E0/E1 have no training contamination. MapAnything's authors very likely evaluated on the same 13 scenes; that is inferred from the count, and the list file was not read.

---

## 5. What is known, and what is still open

**Known (2025–2026).**
- Hint-conditioned feed-forward models stay better than images-only under **mild, independent** pose noise: WorldMirror up to 20°, TCO up to a few degrees, VPGGT's realistic drift on TUM.
- Wrong intrinsics and strong depth noise can be worse than no hint (WorldMirror DTU, TCO), but authors call this "graceful degradation".
- In two-view Pow3R the model follows small wrong hints, refuses gross ones, and its confidence drops.
- With **correct** hints, MapAnything adheres strongly to injected poses (SpatialBench).
- No published model has a reliability gate.

**Still open (our contribution space).**
- Paired, per-error-type harm vs images-only in multi-view MapAnything with scene-level statistics.
- How much of a wrong rotation or scale the model **copies**, against Pow3R's "refuses" and VPGGT's "dilution".
- **Correlated vs independent** errors at matched size.
- **Depth-scale** errors.
- **Spread** to good views.
- **"Knows but obeys"**: confidence flags the bad view, yet the output copies it.
- **Training-free, image-witnessed detection** of bad hints, including group-level detection for correlated errors.

**Correction made to `context.md` §3.** The VPGGT number "57.11 vs 38.00" came from VPGGT Table 1(a), where ground-truth depth was also given. The pose-only comparison is Table 3: 58.89 vs 38.00.
