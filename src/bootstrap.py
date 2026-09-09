"""
Bootstrap 95% CIs for the stability-faithfulness Pearson correlations (RQ2).
 
Assumes you already have `df` in memory (as in your plotting script) with
columns: "method", "S_cos", "S_rank", "F_final".
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, wilcoxon
import matplotlib.pyplot as plt

 
RNG = np.random.default_rng(42)  # fixed seed for reproducibility
 
 
def bootstrap_pearson_ci(x, y, n_boot=10_000, ci=95, rng=RNG):
    """
    Bootstrap CI for Pearson correlation between x and y.
 
    Resamples (x_i, y_i) PAIRS with replacement -- not x and y independently --
    since we need to preserve the pairing between a sentence's stability and
    its faithfulness score in each resample.
 
    Returns: (point_estimate, ci_low, ci_high, boot_distribution)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
 
    if n < 3:
        return np.nan, np.nan, np.nan, np.array([])
 
    point_estimate, _ = pearsonr(x, y)
 
    boot_corrs = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        xb, yb = x[sample_idx], y[sample_idx]
        # guard against degenerate resamples (zero variance -> undefined correlation)
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot_corrs[b] = np.nan
            continue
        boot_corrs[b], _ = pearsonr(xb, yb)
 
    boot_corrs = boot_corrs[~np.isnan(boot_corrs)]
    alpha = (100 - ci) / 2
    ci_low, ci_high = np.percentile(boot_corrs, [alpha, 100 - alpha])
 
    return point_estimate, ci_low, ci_high, boot_corrs
 
 
def run_all_methods(df, methods, n_boot=10_000):
    rows = []
    for m in methods:
        sub = df[df["method"] == m].dropna(subset=["S_cos", "S_rank", "F_final"])
 
        r_cos, lo_cos, hi_cos, _ = bootstrap_pearson_ci(
            sub["S_cos"], sub["F_final"], n_boot=n_boot
        )
        r_rank, lo_rank, hi_rank, _ = bootstrap_pearson_ci(
            sub["S_rank"], sub["F_final"], n_boot=n_boot
        )
 
        rows.append({
            "method": m,
            "n": len(sub),
            "rho_cos": r_cos,
            "rho_cos_ci_low": lo_cos,
            "rho_cos_ci_high": hi_cos,
            "rho_rank": r_rank,
            "rho_rank_ci_low": lo_rank,
            "rho_rank_ci_high": hi_rank,
        })
 
    return pd.DataFrame(rows)


def plot_forest_ci(results_df, metric_prefix, title, output_path):
    plt.figure(figsize=(8, 4.5), dpi=300)
    
    y_pos = np.arange(len(results_df))
    estimates = results_df[metric_prefix]
    low_err = estimates - results_df[f"{metric_prefix}_ci_low"]
    high_err = results_df[f"{metric_prefix}_ci_high"] - estimates
    
    clean_labels = [m.replace("_", " ").title() for m in results_df["method"]]

    plt.errorbar(
        estimates, y_pos, 
        xerr=[low_err, high_err], 
        fmt='o', color='#1f77b4', ecolor='#1f77b4', elinewidth=2, capsize=4, markersize=6
    )
    
    plt.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    plt.yticks(y_pos, clean_labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Pearson Correlation ($r$) with 95% CI")
    plt.title(title, fontsize=12, pad=10)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved figure to {output_path}")
    


# Average S_rank across the 4 perturbation types (bt, n1, n2, n3) per sentence & method,
# matching the S_rank,m definition in the methodology (Step 10)
agg = df.groupby(["sentence_id", "method"])["S_rank"].mean().reset_index()

# Pivot so each method is a column, aligned by sentence_id (paired samples)
piv = agg.pivot(index="sentence_id", columns="method", values="S_rank")

ig = piv["integrated_gradients"]
sg = piv["smooth_grad"]

# Keep only sentences with a valid S_rank for both methods (paired comparison)
paired = pd.concat([ig, sg], axis=1).dropna()
paired.columns = ["ig", "sg"]
n = len(paired)
print(f"n paired sentences: {n}")

# --- Wilcoxon signed-rank test ---
# Tests whether the paired differences (SmoothGrad - IG) are symmetric around zero,
# without assuming normality (unlike a paired t-test).
stat, p_value = wilcoxon(paired["sg"], paired["ig"])
print(f"Wilcoxon signed-rank statistic: {stat}")
print(f"p-value: {p_value:.3e}")

# --- Bootstrap 95% CI on the mean difference (SmoothGrad - IG) ---
diff = (paired["sg"] - paired["ig"]).values
print(f"Mean difference (SmoothGrad - IG): {diff.mean():.4f}")

rng = np.random.default_rng(42)  # seeded for reproducibility
n_boot = 10_000
boot_means = np.empty(n_boot)
for i in range(n_boot):
    sample = rng.choice(diff, size=n, replace=True)
    boot_means[i] = sample.mean()

ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
print(f"Bootstrap 95% CI on mean difference: [{ci_lo:.3f}, {ci_hi:.3f}]")


if __name__ == "__main__":
    # Ensure outputs/figures directory exists
    os.makedirs("outputs/figures", exist_ok=True)

    df = pd.read_csv("outputs/stability_faithfulness_per_sentence.csv")
    methods = [
        "input_x_gradient", "integrated_gradients", "smooth_grad", "random_attribution", "uniform_attribution"
    ]
    
    results = run_all_methods(df, methods, n_boot=10_000)
    results.to_csv("outputs/rq2_bootstrap_ci.csv", index=False)
    print(results.round(3))
    
    # Plot 1: Cosine Stability vs Faithfulness CI
    plot_forest_ci(
        results, 
        metric_prefix="rho_cos", 
        title="RQ2: Stability ($S_{cos}$) vs Faithfulness ($F_{final}$) 95% CIs", 
        output_path="outputs/figures/rq2_ci_cosine_stability.png"
    )

    # Plot 2: Rank Stability vs Faithfulness CI
    plot_forest_ci(
        results, 
        metric_prefix="rho_rank", 
        title="RQ2: Stability ($S_{rank}$) vs Faithfulness ($F_{final}$) 95% CIs", 
        output_path="outputs/figures/rq2_ci_rank_stability.png"
    )
    
    # Average S_rank across the 4 perturbation types (bt, n1, n2, n3) per sentence & method,
    # matching the S_rank,m definition in the methodology (Step 10)
    agg = df.groupby(["sentence_id", "method"])["S_rank"].mean().reset_index()

    # Pivot so each method is a column, aligned by sentence_id (paired samples)
    piv = agg.pivot(index="sentence_id", columns="method", values="S_rank")

    ig = piv["integrated_gradients"]
    sg = piv["smooth_grad"]

    # Keep only sentences with a valid S_rank for both methods (paired comparison)
    paired = pd.concat([ig, sg], axis=1).dropna()
    paired.columns = ["ig", "sg"]
    n = len(paired)
    print(f"n paired sentences: {n}")

    # --- Wilcoxon signed-rank test ---
    # Tests whether the paired differences (SmoothGrad - IG) are symmetric around zero,
    # without assuming normality (unlike a paired t-test).
    stat, p_value = wilcoxon(paired["sg"], paired["ig"])
    print(f"Wilcoxon signed-rank statistic: {stat}")
    print(f"p-value: {p_value:.3e}")

    # --- Bootstrap 95% CI on the mean difference (SmoothGrad - IG) ---
    diff = (paired["sg"] - paired["ig"]).values
    print(f"Mean difference (SmoothGrad - IG): {diff.mean():.4f}")

    rng = np.random.default_rng(42)  # seeded for reproducibility
    n_boot = 10_000
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(diff, size=n, replace=True)
        boot_means[i] = sample.mean()

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"Bootstrap 95% CI on mean difference: [{ci_lo:.3f}, {ci_hi:.3f}]")
    
    