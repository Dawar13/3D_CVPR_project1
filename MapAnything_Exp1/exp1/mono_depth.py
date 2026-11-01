"""E2-E AI depth hint (user-approved 2026-11-01): add a monocular metric depth estimate to every E2-E clip.

  python -m exp1.mono_depth --clips 'data/clips_e2e/*.npz' --record results/e2e_mono/mono_depth.csv

Each clip gains `mono_depth` (metres, same grid as `depth`) predicted from the clip's own images, and `mono_model`
(model id @ revision). The run_exp1 device condition feeds it as the depth hint with
{"target": "device", "use": ["mono_depth", ...]}. Runs on the VM in its own env (scripts/vm_setup_e2.sh, `aidepth`),
while MapAnything uses the GPU; the laptop did not have the memory.

Model, fixed before any run: Depth Anything V2, metric, indoor (Hypersim fine-tune), Large, through Hugging Face
transformers, fp32, the processor's default resizing, bicubic upsampling back to the clip size.
The error vs ground truth is printed only for dev clips; eval clips stay unread until the analysis.
"""

import argparse
import csv
import glob
import hashlib
import os
import time

import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def load(model_id=MODEL, device="cpu"):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(device)
    return proc, model


def predict(proc, model, image, device="cpu"):
    """image (H, W, 3) uint8 -> metric depth (H, W) float32 at the image's own size."""
    import torch
    from PIL import Image
    inputs = {k: v.to(device) for k, v in proc(images=Image.fromarray(image), return_tensors="pt").items()}
    with torch.no_grad():
        d = model(**inputs).predicted_depth                      # (1, h, w)
    d = torch.nn.functional.interpolate(d[:, None], size=image.shape[:2], mode="bicubic", align_corners=False)
    return d[0, 0].clamp(min=0).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--record", help="csv: clip, model@revision, sha256 of mono_depth, seconds, dev-only ratio")
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load(a.model, device)
    rev = getattr(model.config, "_commit_hash", None) or "unknown"
    rows = []
    paths = sorted(glob.glob(a.clips), key=lambda p: (not os.path.basename(p).startswith("dev__"), p))
    for path in paths:
        z = dict(np.load(path, allow_pickle=False))
        t = time.time()
        if "mono_depth" not in z:
            z["mono_depth"] = np.stack([predict(proc, model, im, device) for im in z["images"]])
            z["mono_model"] = np.array(f"{a.model}@{rev}")
            np.savez_compressed(path, **z)
        dev = os.path.basename(path).startswith("dev__")
        ratio = ""
        if dev:
            g, m = z["depth"], z["mono_depth"]
            ratio = f"{np.median(m[g > 0] / g[g > 0]):.4f}"
        rows.append((os.path.basename(path), str(z["mono_model"]),
                     hashlib.sha256(np.ascontiguousarray(z["mono_depth"]).tobytes()).hexdigest()[:16],
                     f"{time.time() - t:.1f}", ratio))
        print(f"[ok] {path}  {rows[-1][3]} s" + (f"  AI/GT median depth ratio {ratio}" if dev else ""), flush=True)
    if a.record:
        os.makedirs(os.path.dirname(a.record) or ".", exist_ok=True)
        with open(a.record, "w", newline="") as f:
            csv.writer(f).writerows([("clip", "model", "mono_depth_sha256", "seconds", "dev_ratio_ai_over_gt"), *rows])


if __name__ == "__main__":
    main()
