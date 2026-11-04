"""Analysis for the gate track (experiments/G.md). No model and no torch.

  python -m exp1.g_analysis decompose --clips 'data/clips_e2e/dev__*.npz' --out results/g0_dev/depth_decompose.csv
  python -m exp1.g_analysis oracle --runs results/g1_*_dev --out results/g1_summary_dev
  python -m exp1.g_analysis signal --runs results/g2_*_dev --out results/g2_summary_dev

Dev runs are a PILOT: three scenes per dataset. Scene values are printed so nobody reads a verdict into them that they
cannot carry. Thresholds are the proposed ones in experiments/G.md and are not frozen until the g-prereg tag.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .depth_hints import decompose

METRICS = {"shape": "pts_relerr_scale", "rotation": "rre_deg", "real_size": "pts_relerr_metric"}
G1_MAX_GAP, G2_MIN_AUROC, G2_MIN_LIFT = 0.25, 0.80, 0.05


# ------------------------------------------------------------------------------------------------- G0: depth error parts
def run_decompose(clips, out):
    rows = []
    for p in sorted(glob.glob(clips)):
        z = np.load(p, allow_pickle=False)
        name = os.path.basename(p)[:-4]
        for src, key in (("ai", "mono_depth"), ("lidar", "dev_depth")):
            if key in z.files:
                rows.append({"clip": name, "source": src, **decompose(np.asarray(z[key], np.float32),
                                                                      np.asarray(z["depth"], np.float32))})
    if not rows:
        raise SystemExit(f"no depth hints found in {clips}; run exp1/mono_depth.py first")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("source").median(numeric_only=True).round(3).T.to_string())
    return df


def verify_mono(record, reference):
    """The regenerated AI depth should match E2-E (results/e2e_mono/mono_depth.csv). A mismatch is reported, not fatal:
    another torch or GPU can change the last bits, and then E2-E numbers and these runs differ slightly."""
    a = pd.read_csv(record).set_index("clip")
    b = pd.read_csv(reference).set_index("clip")
    same = a.mono_depth_sha256 == b.mono_depth_sha256.reindex(a.index)
    models = a.model == b.model.reindex(a.index)
    print(f"AI depth: {int(same.sum())}/{len(a)} clips bit-identical to E2-E; same model revision on {int(models.sum())}/{len(a)}")
    if not models.all():
        raise SystemExit("different Depth Anything revision from E2-E: stop and report")


# ------------------------------------------------------------------------------------------------------ G1: oracle gates
def load_runs(patterns):
    frames = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            f = os.path.join(d, "per_run.csv")
            if os.path.exists(f):
                frames.append(pd.read_csv(f, low_memory=False))
    if not frames:
        raise SystemExit(f"no per_run.csv under {patterns}")
    df = pd.concat(frames, ignore_index=True)
    failed = df[df.status != "ok"]
    if len(failed):
        print(f"WARNING: {len(failed)} failed runs, left out:\n{failed[['run_key', 'status']].head(10).to_string()}")
    return df[df.status == "ok"]


def scene_means(df, col):
    """Scene is the unit: average seeds in a clip, then clips in a scene (context.md section 6)."""
    return df.groupby(["clip_id", "scene"])[col].mean().groupby("scene").mean()


def oracle_summary(df):
    """Damage left = (x - clean) / (always - clean), per scene and pooled, for each scenario, role and metric."""
    rows = []
    keys = ["backend", "dataset", "scenario"]
    for (backend, ds, scen), g in df[df.scenario.fillna("") != ""].groupby(keys):
        roles = {r: x for r, x in g.groupby("role")}
        if not {"always", "clean"} <= set(roles):
            continue
        for mname, col in METRICS.items():
            if col not in g or g[col].isna().all() or (backend == "omnivggt" and mname == "real_size"):
                continue
            clean, always = scene_means(roles["clean"], col), scene_means(roles["always"], col)
            for role, x in roles.items():
                s = scene_means(x, col).reindex(always.index)
                den = always - clean.reindex(always.index)
                rows.append({"backend": backend, "dataset": ds, "scenario": scen, "metric": mname, "role": role,
                             "value": s.mean(), "always": always.mean(), "clean": clean.mean(),
                             "gap": (s.mean() - clean.mean()) / (always.mean() - clean.mean())
                             if abs(always.mean() - clean.mean()) > 1e-9 else np.nan,
                             "harm_vs_clean": always.mean() - clean.mean(), "n_scenes": int(s.notna().sum()),
                             "scene_gaps": json.dumps({k: round(float(v), 3) for k, v in ((s - clean) / den).items()})})
    return pd.DataFrame(rows)


def oracle_verdicts(tab):
    """Proposed pass, per scenario (experiments/G.md): shape damage left by the best in-network ORACLE gate <= 0.25, and
    that gate beats never using the hint (so the good hints still earn something)."""
    rows = []
    primary = {"depth": "shape", "pose": "rotation"}
    for (backend, ds, scen), g in tab.groupby(["backend", "dataset", "scenario"]):
        kind = "pose" if scen.startswith("pose") else "depth"
        s = g[g.metric == primary[kind]]
        gates = s[s.role.str.startswith("oracle_gate")]
        never = s[s.role == "never"]
        if gates.empty or never.empty:
            continue
        best = gates.loc[gates.value.idxmin()]
        rows.append({"backend": backend, "dataset": ds, "scenario": scen, "metric": primary[kind], "best_gate": best.role,
                     "gate_gap": best.gap, "never_gap": never.gap.iloc[0], "harm_vs_clean": best.harm_vs_clean,
                     "beats_never": bool(best.value < never.value.iloc[0]), "pass_proposed":
                     bool(best.gap <= G1_MAX_GAP and best.value < never.value.iloc[0] and best.harm_vs_clean > 0)})
    return pd.DataFrame(rows)


def decomposition_table(df):
    """G-exp: which part of AI depth does the harm? Every value relative to photos-only on the same clips."""
    rows = []
    want = {"ai": "AI depth as is", "ai_fix_global": "ORACLE overall scale fixed", "ai_fix_view": "ORACLE distortion only",
            "gt_ai_global": "ORACLE overall scale error only", "gt_ai_view": "ORACLE per-view scale error only",
            "lidar": "LiDAR", "gt_depth": "true depth"}
    r = df[df.dataset == "rooms"]
    for backend, g in r.groupby("backend"):
        ref_name = "rgb_K"
        ref = g[g.condition == ref_name]
        if ref.empty:
            continue
        for mname, col in METRICS.items():
            if backend == "omnivggt" and mname == "real_size":
                continue
            base = scene_means(ref, col)
            for cname, label in want.items():
                x = g[g.condition == cname]
                if len(x):
                    d = scene_means(x, col) - base
                    rows.append({"backend": backend, "metric": mname, "condition": cname, "label": label,
                                 "delta_vs_photos": d.mean(), "n_scenes": int(d.notna().sum()),
                                 "scene_deltas": json.dumps({k: round(float(v), 4) for k, v in d.items()})})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------ G2: token signal
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = pd.Series(np.r_[pos, neg]).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def load_tokens(patterns):
    """One row per unit: a view for view tasks (view 0 is excluded: it is the reference and is never made bad),
    a clip + seed for group tasks. Returns the rows and, per row, the probe feature per layer."""
    rows, feats, refs = [], [], {}
    files = [f for pat in patterns for d in sorted(glob.glob(pat)) for f in sorted(glob.glob(os.path.join(d, "tokens", "*.npz")))]
    if not files:
        raise SystemExit(f"no token files under {patterns}")
    data = []
    for f in files:
        backend = json.load(open(os.path.join(os.path.dirname(os.path.dirname(f)), "run_info.json")))["backend"]
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if "signal" not in meta:
            refs[(backend, meta["clip"])] = np.asarray(z["mean_token"], np.float32)
        else:
            data.append((backend, meta, {k: z[k] for k in z.files if k != "meta"}))
    for backend, meta, z in data:
        ref = refs.get((backend, meta["clip"]))
        if ref is None:
            raise SystemExit(f"no photos-only reference for {backend} {meta['clip']}")
        V = len(z["conf"])
        bad = set(int(i) for i in z["bad_views"])
        diff = np.asarray(z["mean_token"], np.float32) - ref
        conf = z["conf"] / np.median(z["conf"])
        sig = meta["signal"]
        tasks = [(sig["task"], sig["kind"], 1)] if "task" in sig else [(t, k, 0) for t, k in sig["neg_for"]]
        for task, kind, label in tasks:
            base = {"backend": backend, "scene": meta["scene"], "dataset": meta["dataset"], "clip": meta["clip"],
                    "condition": meta["condition"], "seed": meta["seed"], "task": task, "kind": kind}
            common = {f"token_mean_{k}": None for k in z["layers"]}
            if kind == "view":
                for i in range(1, V):
                    if label and i not in bad:
                        continue   # good views of a damaged run are neither positives nor clean negatives
                    r = dict(base, label=label, conf=-float(conf[i]), dis_rot=float(z["dis_rot"][i]),
                             dis_dir=float(z["dis_dir"][i]), dis_depth=float(z["dis_depth"][i]))
                    r.update({**common, **{f"token_mean_{k}": float(z["d_mean"][i, j]) for j, k in enumerate(z["layers"])},
                              **{f"token_p90_{k}": float(z["d_p90"][i, j]) for j, k in enumerate(z["layers"])}})
                    rows.append(r)
                    feats.append(diff[i])
            else:
                views = np.arange(1, V)
                r = dict(base, label=label, conf=-float(z["conf"][views].mean()), dis_rot=float(np.nanmean(z["dis_rot"][views])) if np.isfinite(z["dis_rot"][views]).any() else np.nan,
                         dis_dir=float(np.nanmean(z["dis_dir"][views])) if np.isfinite(z["dis_dir"][views]).any() else np.nan,
                         dis_depth=float(np.nanmean(z["dis_depth"][views])) if np.isfinite(z["dis_depth"][views]).any() else np.nan)
                r.update({**common, **{f"token_mean_{k}": float(z["d_mean"][views, j].mean()) for j, k in enumerate(z["layers"])},
                          **{f"token_p90_{k}": float(z["d_p90"][views, j].mean()) for j, k in enumerate(z["layers"])}})
                rows.append(r)
                feats.append(diff[views].mean(0))
    return pd.DataFrame(rows), feats


def ridge_scores(Xtr, ytr, Xte, lam=1e3):
    """Dual ridge on standardised features: the features outnumber the samples."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    alpha = np.linalg.solve(A @ A.T + lam * np.eye(len(A)), ytr - ytr.mean())
    return B @ (A.T @ alpha)


def signal_summary(df, feats):
    """Leave one scene out. Token layer and probe layer are picked on the other scenes only, so held-out numbers are not
    flattered by picking the best layer afterwards."""
    token_cols = [c for c in df.columns if c.startswith("token_")]
    fixed = ["conf", "dis_rot", "dis_dir", "dis_depth"]
    rows = []
    for (backend, task), g in df.groupby(["backend", "task"]):
        scenes = sorted(g.scene.unique())
        folds = []
        idx_all = g.index.to_numpy()
        n_layers = feats[idx_all[0]].shape[0]
        for sc in scenes:
            te, tr = g[g.scene == sc], g[g.scene != sc]
            if te.label.nunique() < 2 or tr.label.nunique() < 2:
                continue
            res = {"scene": sc}
            for c in fixed:
                res[c] = auroc(te[te.label == 1][c], te[te.label == 0][c])
            cols = [c for c in token_cols if g[c].notna().any()]
            best = max(cols, key=lambda c: np.nan_to_num(auroc(tr[tr.label == 1][c], tr[tr.label == 0][c])))
            res["token_score"], res["token_layer"] = auroc(te[te.label == 1][best], te[te.label == 0][best]), best
            ytr = tr.label.to_numpy(float)
            best_l, best_a = 0, -1.0
            for L in range(n_layers):   # inner choice: split the training scenes in two by scene
                tr_scenes = sorted(tr.scene.unique())
                if len(tr_scenes) < 2:
                    break
                a_mask = tr.scene.isin(tr_scenes[::2]).to_numpy()
                X = np.stack([feats[i][L] for i in tr.index])
                if len(set(ytr[a_mask])) < 2 or len(set(ytr[~a_mask])) < 2:
                    continue
                s = ridge_scores(X[a_mask], ytr[a_mask], X[~a_mask])
                au = auroc(s[ytr[~a_mask] == 1], s[ytr[~a_mask] == 0])
                if np.nan_to_num(au) > best_a:
                    best_l, best_a = L, au
            Xtr = np.stack([feats[i][best_l] for i in tr.index])
            Xte = np.stack([feats[i][best_l] for i in te.index])
            s = ridge_scores(Xtr, ytr, Xte)
            yte = te.label.to_numpy()
            res["token_probe"], res["probe_layer"] = auroc(s[yte == 1], s[yte == 0]), best_l
            folds.append(res)
        if not folds:
            rows.append({"backend": backend, "task": task, "kind": g.kind.iloc[0], "n_folds": 0, "n_pos": int(g.label.sum()),
                         "n_neg": int((1 - g.label).sum()), "note": "no scene has both classes"})
            continue
        f = pd.DataFrame(folds)
        row = {"backend": backend, "task": task, "kind": g.kind.iloc[0], "n_folds": len(f), "n_pos": int(g.label.sum()),
               "n_neg": int((1 - g.label).sum())}
        for c in fixed + ["token_score", "token_probe"]:
            row[c] = f[c].mean()
        base = [row[c] for c in fixed if np.isfinite(row[c])]
        tok = max(row["token_score"], row["token_probe"])
        row["best_baseline"] = max(base) if base else np.nan
        row["pass_proposed"] = bool(tok >= G2_MIN_AUROC and (not base or tok >= max(base) + G2_MIN_LIFT))
        row["folds"] = json.dumps(folds, default=float)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("decompose")
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify_mono")
    p.add_argument("--record", required=True)
    p.add_argument("--reference", required=True)
    for name in ("oracle", "signal"):
        p = sub.add_parser(name)
        p.add_argument("--runs", nargs="+", required=True)
        p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "decompose":
        run_decompose(a.clips, a.out)
        return
    if a.cmd == "verify_mono":
        verify_mono(a.record, a.reference)
        return
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "oracle":
        df = load_runs(a.runs)
        tab, ver, dec = oracle_summary(df), None, decomposition_table(df)
        ver = oracle_verdicts(tab)
        tab.to_csv(os.path.join(a.out, "damage_left.csv"), index=False)
        ver.to_csv(os.path.join(a.out, "verdicts_proposed.csv"), index=False)
        dec.to_csv(os.path.join(a.out, "ai_depth_parts.csv"), index=False)
        print("PILOT, dev scenes only.\n\nWhich part of AI depth does the harm:\n" +
              dec.drop(columns="scene_deltas").round(4).to_string())
        print("\nOracle gate, proposed check:\n" + ver.round(3).to_string())
    else:
        df, feats = load_tokens(a.runs)
        tab = signal_summary(df, feats)
        tab.to_csv(os.path.join(a.out, "signal_auroc.csv"), index=False)
        show = [c for c in tab.columns if c != "folds"]
        print("PILOT, dev scenes only.\n\n" + tab[show].round(3).to_string())


if __name__ == "__main__":
    main()
