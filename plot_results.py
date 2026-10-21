"""Params-vs-EM plot for the QLoRA ablation (light + dark PNGs for the README).

Reads only results/*.json and adapters/*/train_info.json (real runs).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

GROUPS = [  # (adapter prefix, label, marker); colors are categorical slots 1-3 (validated)
    ("qv", "q, v", "o"),
    ("qkvo", "q, k, v, o", "s"),
    ("all", "attention + MLP", "^"),
]
RANKS = [8, 16, 32, 64]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series=["#3987e5", "#d95926", "#199e70"]),
}


def em(name):
    return json.loads(Path(f"results/{name}__abstain__full.json").read_text())["parse"]["exact"]


def params(name):
    return json.loads(Path(f"adapters/{name}/train_info.json").read_text())["trainable_params"]


def plot(theme_name):
    t = THEMES[theme_name]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])

    for (prefix, label, marker), color in zip(GROUPS, t["series"]):
        names = [f"{prefix}-r{r}" for r in RANKS]
        xs, ys = [params(n) for n in names], [em(n) for n in names]
        size = 10 if marker == "^" else 8  # triangles read smaller than circles at equal size
        ax.plot(xs, ys, color=color, linewidth=2, marker=marker, markersize=size,
                markeredgecolor=t["surface"], markeredgewidth=1.5, zorder=3)
        # Seed replicates (seeds 1-2) as hollow markers, nudged left/right so equal values
        # (e.g. all-r64: 84.50 and 84.49) don't hide each other.
        for n in names:
            reps = [f"{n}-s{s}" for s in (1, 2)]
            if all(Path(f"results/{r}__abstain__full.json").exists() for r in reps):
                ax.plot([params(n) * 0.93, params(n) * 1.07], [em(r) for r in reps], linestyle="none",
                        marker=marker, markersize=size - 1, markerfacecolor="none",
                        markeredgecolor=color, markeredgewidth=1.5, zorder=2)
        # Direct label at the line's end (text in ink, not series color).
        # qkvo's end point sits under the attention+MLP line, so its label goes below the point.
        offset, ha = ((0, -13), "center") if prefix == "qkvo" else ((10, 0), "left")
        ax.annotate(label, (xs[-1], ys[-1]), xytext=offset, textcoords="offset points",
                    va="center", ha=ha, color=t["ink2"], fontsize=9)

    ax.set_xscale("log")
    ticks = [3e6, 1e7, 3e7, 1e8, 3e8]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_xlim(2.5e6, 4.5e8)
    ax.set_ylim(84.0, 87.0)
    ax.set_xlabel("Trainable parameters (log scale)", color=t["ink2"])
    ax.set_ylabel("Dev exact match (%)", color=t["ink2"])
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0)

    legend = [Line2D([], [], color=c, linewidth=2, marker=m, markersize=7,
                     markeredgecolor=t["surface"], label=l)
              for (_, l, m), c in zip(GROUPS, t["series"])]
    legend.append(Line2D([], [], linestyle="none", marker="o", markersize=7, markerfacecolor="none",
                         markeredgecolor=t["muted"], markeredgewidth=1.5, label="seed 1 & 2 replicates"))
    ax.legend(handles=legend, loc="lower left", frameon=False, labelcolor=t["ink2"], fontsize=9)

    fig.text(0.07, 0.95, "QLoRA ablation: dev EM vs. trainable parameters",
             color=t["ink"], fontsize=13, fontweight="semibold")
    fig.text(0.07, 0.905, "LLaMA-3-8B-Instruct on SQuAD 2.0 dev (n = 11,873). Points left to right: "
             "rank 8, 16, 32, 64. Best prompt-only baseline: 64.47 EM.",
             color=t["ink2"], fontsize=8.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.84, bottom=0.12)
    Path("figures").mkdir(exist_ok=True)
    out = f"figures/params_vs_em_{theme_name}.png"
    fig.savefig(out, facecolor=t["surface"])
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    for name in THEMES:
        plot(name)
