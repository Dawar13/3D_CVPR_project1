"""E0 Step 6: MapAnything README image-only example on 5 raw images of one dev scene.

  python scripts/e0_official_example.py --eth3d_root /data/eth3d \
      | tee reports/e0_official_example.txt

Prints output keys, shapes, dtypes, finiteness per key (over all views), runtime and
peak GPU memory. PASS: it runs and every floating-point output is finite.
"""

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from exp1 import model_io  # noqa: E402
from exp1.datasets import ETH3D_TRAIN_SCENES, dev_eval_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth3d_root", required=True)
    ap.add_argument("--scene", help="default: first dev scene")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--checkpoint", default="facebook/map-anything")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    scene = a.scene or dev_eval_split(ETH3D_TRAIN_SCENES)[0][0]
    paths = sorted(glob.glob(os.path.join(a.eth3d_root, scene, "images",
                                          "dslr_images_undistorted", "*.JPG")))[:a.n]
    if len(paths) < a.n:
        raise SystemExit(f"found {len(paths)} images for {scene}, need {a.n}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    print(f"checkpoint {a.checkpoint}  scene {scene}  images {[os.path.basename(p) for p in paths]}")

    model = model_io.load_model(a.checkpoint, a.device)
    preds, runtime, peak = model_io.official_example(model, paths, a.device)

    print(f"\n{len(preds)} views; keys/shapes/dtypes of view 0, finiteness over all views:")
    finite_all = True
    for k in sorted(preds[0]):
        v = preds[0][k]
        if not torch.is_tensor(v):
            print(f"  {k:28s} {type(v).__name__}")
            continue
        fin = all(bool(torch.isfinite(p[k].float()).all()) for p in preds
                  if torch.is_tensor(p[k]) and p[k].is_floating_point())
        finite_all &= fin
        print(f"  {k:28s} {str(tuple(v.shape)):20s} {str(v.dtype):16s} finite={fin}")
    print(f"\nruntime {runtime:.3f} s for {len(preds)} views (includes first-call warm-up); "
          f"peak GPU memory {peak:.2f} GB")
    print("PASS" if finite_all else "FAIL: non-finite outputs")


if __name__ == "__main__":
    main()
