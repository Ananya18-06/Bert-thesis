
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# TUM corporate colors
TUM_BLUE = "#0065BD"
TUM_DARK_BLUE = "#005293"
TUM_LIGHT_BLUE = "#64A0C8"
TUM_PALE_BLUE = "#98C6EA"
TUM_GREEN = "#A2AD00"
TUM_ORANGE = "#E37222"
TUM_GRAY = "#DAD7CB"
TUM_BLACK = "#333333"

 
OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
 
METHOD_LABELS = {
    "input_x_gradient": "Gradient x Input",
    "integrated_gradients": "Integrated Gradients",
    "smooth_grad": "SmoothGrad",
    "random_attribution": "Random (floor)",
    "uniform_attribution": "Uniform (ceiling)"
}
METHODS = list(METHOD_LABELS.keys())

TUM_PALETTE = [TUM_BLUE, TUM_ORANGE, TUM_GREEN, TUM_GRAY, TUM_LIGHT_BLUE]
METHOD_COLORS = {m: TUM_PALETTE[i % len(TUM_PALETTE)] for i, m in enumerate(METHODS)}
COLORS = {
    "input_x_gradient": "#4C72B0", "integrated_gradients": "#55A868", "smooth_grad": "#C44E52",
    "random_attribution": "#999999", "uniform_attribution": "#333333", "attention_attribution": "#DDAA33",
}
 
df = pd.read_csv("outputs/stability_faithfulness_per_sentence.csv")
df = df[df["method"].isin(METHODS)].copy()
df["method_label"] = df["method"].map(METHOD_LABELS)
 
per_type_df = pd.read_csv("outputs/stability_per_perturbation_type.csv")
per_type_df = per_type_df[per_type_df["method"].isin(METHODS)].copy()
NOISE_ORDER = ["n1", "n2", "n3", "bt"]
NOISE_LABELS = {"n1": "char noise\n(rho=0.05)", "n2": "char noise\n(rho=0.10)",
                "n3": "char noise\n(rho=0.20)", "bt": "back-\ntranslation"}
 
 
def savefig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close(fig)

 
 

# Distribution of stability scores per method (boxplot)
def plot_stability_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
 
    for ax, metric, title in zip(axes, ["S_cos", "S_rank"], ["Cosine similarity", "Spearman's rho"]):
        data = [df[df["method"] == m][metric].dropna().values for m in METHODS]
        bp = ax.boxplot(data, patch_artist=True)
        ax.set_xticks(range(1, len(METHODS) + 1))
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.6)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", rotation=15)
 
    fig.suptitle("Distribution of per-sentence stability scores, by method")
    fig.tight_layout()
    savefig(fig, "stability_distribution_by_method.png")
 



# Stability vs. faithfulness scatter, per method, both metrics

def plot_stability_vs_faithfulness():
    n_methods = len(METHODS)
    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 9), sharex="row")
 
    for col, m in enumerate(METHODS):
        sub = df[df["method"] == m].dropna(subset=["S_cos", "S_rank", "F_final"])
 
        for row, metric, label in zip([0, 1], ["S_cos", "S_rank"], ["Cosine similarity", "Spearman's rho"]):
            ax = axes[row, col]
            ax.scatter(sub[metric], sub["F_final"], alpha=0.5, color=COLORS[m], s=20)
 
            if len(sub) > 2:
                r, p = pearsonr(sub[metric], sub["F_final"])
                z = np.polyfit(sub[metric], sub["F_final"], 1)
                xs = np.linspace(sub[metric].min(), sub[metric].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color="black", linewidth=1, linestyle="--")
                ax.annotate(f"r = {r:.2f}\np = {p:.3f}", xy=(0.05, 0.85), xycoords="axes fraction",
                            fontsize=9, bbox=dict(boxstyle="round", fc="white", alpha=0.7))
 
            if row == 0:
                ax.set_title(METHOD_LABELS[m])
            if row == 1:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(f"{label}\nF_final")
 
    fig.suptitle("Stability vs. final faithfulness, per method")
    fig.tight_layout()
    savefig(fig, "stability_vs_faithfulness_scatter.png")
 
 
    
def plot_stability_by_perturbation_type():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": TUM_BLACK,
        "axes.labelcolor": TUM_BLACK,
        "text.color": TUM_BLACK,
        "xtick.color": TUM_BLACK,
        "ytick.color": TUM_BLACK,
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)    
    axes[0].set_ylabel("Stability", fontsize=11)
    legend_handles = [
        mpatches.Patch(
            facecolor=METHOD_COLORS[m],
            edgecolor=TUM_BLACK,
            label=METHOD_LABELS[m]
        )
        for m in METHODS
    ]
    
    # Separate legend below both plots
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(METHODS),
        frameon=False
    )
    

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    n_methods = len(METHODS)
    box_width = 0.7 / n_methods
    group_gap = 1.0

    for ax, metric, title in zip(axes, ["S_cos", "S_rank"], ["Cosine similarity", "Spearman's rho"]):
        for mi, m in enumerate(METHODS):
            sub = per_type_df[per_type_df["method"] == m]
            positions = []
            data = []
            for i, ptype in enumerate(NOISE_ORDER):          # <-- was METHOD_LABELS
                vals = sub.loc[sub["perturbation_type"] == ptype, metric].dropna().values
                data.append(vals)
                pos = i * group_gap + (mi - (n_methods - 1) / 2) * box_width
                positions.append(pos)

            ax.boxplot(
                data,
                positions=positions,
                widths=box_width * 0.9,
                patch_artist=True,
                showfliers=True,
                flierprops=dict(marker="o", markersize=2.5, alpha=0.4,
                                 markerfacecolor=METHOD_COLORS[m], markeredgecolor="none"),
                medianprops=dict(color=TUM_BLACK, linewidth=1.4),
                whiskerprops=dict(color=METHOD_COLORS[m], linewidth=1.1),
                capprops=dict(color=METHOD_COLORS[m], linewidth=1.1),
                boxprops=dict(facecolor=METHOD_COLORS[m], edgecolor=TUM_BLACK,
                               linewidth=0.8, alpha=0.85),
            )

        ax.set_xticks([i * group_gap for i in range(len(NOISE_ORDER))])
        ax.set_xticklabels([NOISE_LABELS[k] for k in NOISE_ORDER])  # <-- was METHOD_LABELS
        ax.set_title(title, fontsize=12, fontweight="bold", color=TUM_DARK_BLUE)
        ax.axhline(0, color=TUM_BLACK, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color=TUM_GRAY, linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Stability", fontsize=11)

    legend_handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], edgecolor=TUM_BLACK,
                                      label=METHOD_LABELS[m]) for m in METHODS]

    fig.tight_layout()
    savefig(fig, "stability_by_perturbation_type_boxplot.png")
    
 
 
if __name__ == "__main__":
    plot_stability_distribution()
    plot_stability_vs_faithfulness()
    plot_stability_by_perturbation_type()
    print("\nAll figures saved to outputs/figures/")
 