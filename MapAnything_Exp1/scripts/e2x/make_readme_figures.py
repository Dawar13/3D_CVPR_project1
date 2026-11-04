"""Figures for the repo README (descriptive). Run from MapAnything_Exp1/:  python scripts/e2x/make_readme_figures.py
Writes ../docs/figures/fig_break_even.png from results/e2x_angle_preview/angle_break_even.csv (post-hoc view of the
finished E2-C / E2-D eval runs). Real-source markers use measured or published sizes, each named in the figure:
results/e2x_survey/sf_device_survey.log (phone tracking), reports/e2_literature.md (indoor compass, Abadi et al.),
reports/e2_results/e2_results.pdf (AI depth scale)."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

MA, OV = "#2a78d6", "#eb6834"   # same slots as the E2 report (dataviz reference palette)
INK2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
OUT = os.path.join("..", "docs", "figures")


def style(ax, ylabel, xlabel):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=3, color=AXIS)
    ax.grid(True, axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)


def main():
    d = pd.read_csv(os.path.join("results", "e2x_angle_preview", "angle_break_even.csv"))
    os.makedirs(OUT, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
    panels = [("c1_rot", "rotation error of the one wrong camera (°, log scale)",
               "change in rotation error vs photos only (°)\n↑ hint hurts    ↓ hint helps", 1.0),
              ("c1_depth", "depth hint scale (× true scale, log scale)",
               "change in shape error vs no depth hint (pp)\n↑ hint hurts    ↓ hint helps", 100.0)]
    for ax, (fam, xl, yl, k) in zip(axs, panels):
        for model, col, name in (("mapanything", MA, "MapAnything"), ("omnivggt", OV, "OmniVGGT")):
            s = d[(d.model == model) & (d.family == fam)].sort_values("size")
            ax.fill_between(s["size"], k * s.ci_lo, k * s.ci_hi, color=col, alpha=0.14, lw=0)
            ax.plot(s["size"], k * s.delta_vs_floor, color=col, lw=2, marker="o", ms=5, label=name,
                    markeredgecolor="white", markeredgewidth=1.2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xscale("log")
        style(ax, yl, xl)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
    ax = axs[0]
    ax.set_xticks([1, 2, 5, 10, 15, 30])
    ax.set_xticklabels(["1", "2", "5", "10", "15", "30"])
    for x, lab in ((1.23, "phone tracking,\nshort walk (typ.)"), (12.45, "indoor compass\n(published avg.)")):
        ax.axvline(x, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(x, 0.99, " " + lab, transform=ax.get_xaxis_transform(), fontsize=7.8, color=INK2, va="top")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax = axs[1]
    ax.set_xticks([0.5, 0.63, 0.794, 1, 1.26, 1.587, 2])
    ax.set_xticklabels(["0.5", "0.63", "0.79", "1", "1.26", "1.59", "2"])
    ax.axvline(1.42, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(1.42, 0.99, "AI depth \n(measured typ.) ", transform=ax.get_xaxis_transform(), fontsize=7.8, color=INK2,
            va="top", ha="right")
    fig.tight_layout(w_pad=3)
    path = os.path.join(OUT, "fig_break_even.png")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    main()
