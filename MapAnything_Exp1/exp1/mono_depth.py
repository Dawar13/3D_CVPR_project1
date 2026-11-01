"""E2-E (optional, needs the user's approval): add an AI depth estimate as a real "sensor" hint to the E2-E clips.

  python -m exp1.mono_depth --clips 'data/clips_e2e/*.npz'

Each clip gains `mono_depth` (metres, same grid as `depth`) from a monocular metric depth network run on the clip's
own images, plus `mono_model` (model id and revision). The run_exp1 device condition uses it with
{"target": "device", "use": ["mono_depth", ...]}. CPU is enough (54 clips x 10 images).

Model: Depth Anything V2, metric, indoor (Hypersim fine-tune), Large, through Hugging Face transformers.
"""

import argparse
import glob

import numpy as np

MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"


def load(model_id=MODEL, revision=None):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, revision=revision).eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return proc, model


def predict(proc, model, image):
    """image (H, W, 3) uint8 -> metric depth (H, W) float32 at the image's own size."""
    import torch
    from PIL import Image
    inputs = proc(images=Image.fromarray(image), return_tensors="pt")
    with torch.no_grad():
        d = model(**inputs).predicted_depth                      # (1, h, w)
    d = torch.nn.functional.interpolate(d[:, None], size=image.shape[:2], mode="bicubic", align_corners=False)
    return d[0, 0].clamp(min=0).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    proc, model = load(a.model)
    rev = getattr(model.config, "_commit_hash", None) or "unknown"
    for path in sorted(glob.glob(a.clips)):
        z = dict(np.load(path, allow_pickle=False))
        if "mono_depth" in z:
            continue
        z["mono_depth"] = np.stack([predict(proc, model, im) for im in z["images"]])
        z["mono_model"] = np.array(f"{a.model}@{rev}")
        np.savez_compressed(path, **z)
        g, m = z["depth"], z["mono_depth"]
        v = g > 0
        r = np.median(m[v] / g[v])
        print(f"[ok] {path}  mono/GT median ratio {r:.3f}", flush=True)


if __name__ == "__main__":
    main()
