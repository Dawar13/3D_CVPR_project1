import sys, numpy as np, pycolmap
out = sys.argv[1]
rng = np.random.default_rng(0)
res = {"version": pycolmap.__version__}
P, X, Y, U, V = [], [], [], [], []
for t in range(5):
    params = np.array([3400, 3410, 3020, 2010] + list(rng.normal(0, 0.03, 8)))
    cam = pycolmap.Camera(model="THIN_PRISM_FISHEYE", width=6048, height=4032, params=params)
    xy = rng.uniform(-0.8, 0.8, (2000, 2))
    xy[0] = 0.0  # r = 0 branch
    try:
        uv = cam.img_from_cam(xy)
    except Exception:
        uv = cam.img_from_cam(np.c_[xy, np.ones(len(xy))])
    P.append(params); X.append(xy); U.append(np.asarray(uv)[:, :2])
np.savez(out, params=np.array(P), xy=np.array(X), uv=np.array(U), version=pycolmap.__version__)
print("pycolmap", pycolmap.__version__, "saved", np.array(U).shape)
