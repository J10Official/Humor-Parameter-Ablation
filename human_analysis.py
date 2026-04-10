"""
human_analysis.py — Analyse human joke-ranking annotations.

Reads:
  results/annotations.csv          (cleaned rankings)
  results/viewer_label_mapping.json (label→model mapping, seed 42)
  prompts.txt                       (theme list)

Outputs:
  Terminal report with detailed statistics
  results/plots/human_*.png         (figures)

Analyses:
  1. Map labels → models, compute Borda scores and win rates
  2. Per-model rank distribution (how often 1st, 2nd, 3rd)
  3. Inter-annotator agreement (Krippendorff's alpha, Kendall's W)
  4. Per-theme breakdown — which model won each theme
  5. Group comparison — do Group 1 and Group 2 annotators agree?
  6. Pairwise model comparison — head-to-head win counts
  7. Confidence intervals via bootstrap
"""

import csv
import json
import math
import itertools
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR  = Path("results")
CSV_PATH     = RESULTS_DIR / "annotations.csv"
MAPPING_PATH = RESULTS_DIR / "viewer_label_mapping.json"
PROMPTS_PATH = Path("prompts.txt")
PLOT_DIR     = RESULTS_DIR / "plots"

MODELS = ["32B", "30A3", "235A22"]
PARAM_COUNT = {"32B": 32, "30A3": 30, "235A22": 235}
# Borda points: rank 1 → 2, rank 2 → 1, rank 3 → 0
BORDA = {1: 2, 2: 1, 3: 0}

SEP  = "═" * 70
THIN = "─" * 70


# ── Load data ────────────────────────────────────────────────────────────────
def load_prompts() -> list[str]:
    prompts = []
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ". " in line:
                line = line.split(". ", 1)[1]
            prompts.append(line)
    return prompts


def load_mapping() -> dict[str, list[str]]:
    with open(MAPPING_PATH) as f:
        return json.load(f)


def load_annotations() -> list[dict]:
    """Load cleaned CSV. Returns list of dicts with keys:
    annotator, group, rankings (list of 20 3-char strings)."""
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rankings = []
            for i in range(1, 21):
                rankings.append(row[f"Set_{i}"])
            rows.append({
                "annotator": row["Annotator"],
                "group": int(row["Group"]),
                "rankings": rankings,
            })
    return rows


# ── Map labels to models ─────────────────────────────────────────────────────
def resolve_rankings(annotations, all_themes, mapping):
    """
    For each annotator × joke set, resolve label ranking to model ranking.

    Group 1: Joke Sets 1-20 map to themes 11-30 (indices 10-29 in all_themes)
    Group 2: Joke Sets 1-20 map to themes 31-50 (indices 30-49 in all_themes)

    Returns list of dicts: {annotator, group, theme, model_ranking: [model1st, model2nd, model3rd]}
    """
    resolved = []
    label_idx = {"A": 0, "B": 1, "C": 2}

    for ann in annotations:
        if ann["group"] == 1:
            theme_offset = 10  # themes 11-30 → indices 10-29
        else:
            theme_offset = 30  # themes 31-50 → indices 30-49

        for set_idx, ranking_str in enumerate(ann["rankings"]):
            if not ranking_str:
                continue
            theme_idx = theme_offset + set_idx
            if theme_idx >= len(all_themes):
                continue
            theme = all_themes[theme_idx]

            if theme not in mapping:
                continue

            label_to_model = mapping[theme]  # [model_at_A, model_at_B, model_at_C]

            # ranking_str e.g. "BAC" → B is 1st, A is 2nd, C is 3rd
            model_ranking = []
            for ch in ranking_str:
                idx = label_idx[ch]
                model_ranking.append(label_to_model[idx])

            resolved.append({
                "annotator": ann["annotator"],
                "group": ann["group"],
                "theme": theme,
                "model_ranking": model_ranking,  # [1st, 2nd, 3rd]
            })

    return resolved


# ── Analysis functions ────────────────────────────────────────────────────────
def compute_borda(resolved):
    """Compute Borda scores per model."""
    scores = Counter()
    counts = Counter()
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            scores[model] += BORDA[rank_pos + 1]
            counts[model] += 1
    # Normalize by number of ratings
    avg = {}
    for m in MODELS:
        avg[m] = scores[m] / counts[m] if counts[m] > 0 else 0
    return scores, counts, avg


def compute_rank_distribution(resolved):
    """How often each model is ranked 1st, 2nd, 3rd."""
    dist = {m: Counter() for m in MODELS}
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            dist[model][rank_pos + 1] += 1
    return dist


def compute_pairwise(resolved):
    """Head-to-head: for each pair (A,B), how often A ranked above B."""
    wins = defaultdict(lambda: defaultdict(int))
    for r in resolved:
        ranking = r["model_ranking"]
        for i in range(len(ranking)):
            for j in range(i + 1, len(ranking)):
                winner = ranking[i]
                loser = ranking[j]
                wins[winner][loser] += 1
    return wins


def compute_mean_rank(resolved):
    """Average rank per model (lower is better)."""
    ranks = defaultdict(list)
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            ranks[model].append(rank_pos + 1)
    return {m: np.mean(ranks[m]) for m in MODELS}


def per_theme_winners(resolved):
    """For each theme, compute average rank per model → determine winner."""
    theme_data = defaultdict(lambda: defaultdict(list))
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            theme_data[r["theme"]][model].append(rank_pos + 1)

    winners = {}
    details = {}
    for theme in theme_data:
        avg_ranks = {m: np.mean(theme_data[theme].get(m, [2])) for m in MODELS}
        winner = min(avg_ranks, key=avg_ranks.get)
        winners[theme] = winner
        details[theme] = avg_ranks
    return winners, details


# ── Krippendorff's alpha (ordinal) ───────────────────────────────────────────
def krippendorff_alpha_ordinal(resolved):
    """
    Compute Krippendorff's alpha for ordinal data.
    Each "unit" is a (theme, model) pair.
    Each "coder" is an annotator.
    The value is the rank assigned to that model (1, 2, or 3).
    """
    # Build reliability data: rows = units (theme×model), columns = coders
    # First, organize by theme
    theme_annotators = defaultdict(dict)
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            key = (r["theme"], model)
            theme_annotators[key][r["annotator"]] = rank_pos + 1

    # Get all unique annotators and units
    all_annotators = sorted(set(r["annotator"] for r in resolved))
    all_units = sorted(theme_annotators.keys())

    if len(all_units) < 2:
        return float("nan")

    # Build coincidence matrix for ordinal data
    # Values are 1, 2, 3 (ranks)
    n_values = 3
    # Collect all value pairs within each unit
    # Observed disagreement
    D_o = 0.0
    n_total_pairs = 0
    value_counts = Counter()  # marginal counts

    for unit in all_units:
        coders_vals = list(theme_annotators[unit].values())
        n_c = len(coders_vals)
        if n_c < 2:
            continue
        for v in coders_vals:
            value_counts[v] += 1
        # Pairs within this unit
        for i in range(len(coders_vals)):
            for j in range(i + 1, len(coders_vals)):
                v_i, v_j = coders_vals[i], coders_vals[j]
                # ordinal distance squared
                d = ordinal_distance_sq(v_i, v_j, n_values)
                D_o += d / (n_c - 1)
                n_total_pairs += 1 / (n_c - 1)

    # Expected disagreement
    n_total = sum(value_counts.values())
    if n_total < 2:
        return float("nan")
    D_e = 0.0
    values = sorted(value_counts.keys())
    for c in values:
        for k in values:
            if c == k:
                continue
            d = ordinal_distance_sq(c, k, n_values)
            D_e += value_counts[c] * value_counts[k] * d
    D_e /= (n_total * (n_total - 1))

    if D_e == 0:
        return 1.0
    return 1.0 - D_o / (n_total_pairs * D_e) if n_total_pairs > 0 else float("nan")


def ordinal_distance_sq(v1, v2, n_cat):
    """Squared ordinal distance metric for Krippendorff's alpha."""
    if v1 == v2:
        return 0.0
    lo, hi = min(v1, v2), max(v1, v2)
    return (hi - lo) ** 2


def krippendorff_alpha_nominal(resolved):
    """
    Simpler Krippendorff's alpha treating ranks as nominal categories.
    """
    theme_annotators = defaultdict(dict)
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            key = (r["theme"], model)
            theme_annotators[key][r["annotator"]] = rank_pos + 1

    all_units = sorted(theme_annotators.keys())

    D_o = 0.0
    n_pairable = 0
    value_counts = Counter()

    for unit in all_units:
        coders_vals = list(theme_annotators[unit].values())
        n_c = len(coders_vals)
        if n_c < 2:
            continue
        for v in coders_vals:
            value_counts[v] += 1
        for i in range(len(coders_vals)):
            for j in range(i + 1, len(coders_vals)):
                d = 0.0 if coders_vals[i] == coders_vals[j] else 1.0
                D_o += d / (n_c - 1)
                n_pairable += 1 / (n_c - 1)

    n_total = sum(value_counts.values())
    if n_total < 2 or n_pairable == 0:
        return float("nan")
    D_e = 1.0 - sum(n * (n - 1) for n in value_counts.values()) / (n_total * (n_total - 1))

    if D_e == 0:
        return 1.0
    return 1.0 - (D_o / n_pairable) / D_e


def kendall_w(resolved):
    """
    Kendall's W (coefficient of concordance).
    Computed per-group since annotators in different groups rate different themes.
    For each group: items = themes, judges = annotators, values = Borda scores per model.
    We treat each (theme, model) as an item being ranked.
    """
    results = {}
    for grp in [1, 2]:
        grp_data = [r for r in resolved if r["group"] == grp]
        if not grp_data:
            continue

        # Get annotators in this group
        annotators = sorted(set(r["annotator"] for r in grp_data))
        themes = sorted(set(r["theme"] for r in grp_data))
        k = len(annotators)  # number of judges
        n = len(themes) * len(MODELS)  # number of items

        if k < 2:
            continue

        # Build rank matrix: for each annotator, rank all (theme, model) items
        # by the rank given (1=best, 3=worst)
        item_ranks = defaultdict(dict)
        for r in grp_data:
            for rank_pos, model in enumerate(r["model_ranking"]):
                item_ranks[(r["theme"], model)][r["annotator"]] = rank_pos + 1

        items = [(t, m) for t in themes for m in MODELS]
        # Build matrix: items × judges
        R = np.zeros((len(items), k))
        for i, item in enumerate(items):
            for j, ann in enumerate(annotators):
                R[i, j] = item_ranks[item].get(ann, 2)  # default middle rank

        # Kendall's W = 12 * S / (k^2 * (n^3 - n))
        # where S = sum of squared deviations of row sums from mean row sum
        row_sums = R.sum(axis=1)
        mean_sum = row_sums.mean()
        S = np.sum((row_sums - mean_sum) ** 2)
        W = 12 * S / (k ** 2 * (n ** 3 - n)) if (n ** 3 - n) > 0 else 0
        results[grp] = W

    return results


def bootstrap_mean_rank(resolved, n_boot=10000):
    """Bootstrap 95% CI for mean rank per model."""
    model_ratings = defaultdict(list)
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            model_ratings[model].append(rank_pos + 1)

    cis = {}
    rng = np.random.RandomState(42)
    for m in MODELS:
        vals = np.array(model_ratings[m])
        boot_means = np.array([
            rng.choice(vals, size=len(vals), replace=True).mean()
            for _ in range(n_boot)
        ])
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        cis[m] = (vals.mean(), lo, hi)
    return cis


# ── High-Agreement Subset Analysis ────────────────────────────────────────────
def high_agreement_subset_analysis(resolved, agreement_threshold=0.4):
    """
    Identify themes where annotators show strong agreement on the winner,
    then re-run Friedman/Wilcoxon on that subset.

    Agreement metric: fraction of annotator pairs within a theme that agree
    on which model is ranked 1st.

    Saves results to results/high_agreement_subset.json
    """
    # Compute per-theme winner agreement
    theme_data = defaultdict(list)
    for r in resolved:
        theme_data[r["theme"]].append(r["model_ranking"])

    theme_agreement = {}
    for theme, rankings in theme_data.items():
        winners = [r[0] for r in rankings]
        n = len(winners)
        if n < 2:
            continue
        agree = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                if winners[i] == winners[j]:
                    agree += 1
                total += 1
        theme_agreement[theme] = agree / total

    sorted_themes = sorted(theme_agreement.items(), key=lambda x: -x[1])

    print(f"\n  Per-theme winner agreement (fraction of annotator pairs agreeing):")
    print(f"  {'Theme':<55s}  {'Agree':>6s}")
    print(f"  {THIN}")
    for t, a in sorted_themes:
        marker = " ✓" if a >= agreement_threshold else ""
        print(f"  {t[:55]:<55s}  {a:.3f}{marker}")

    # Filter to high-agreement themes
    high_themes = [t for t, a in theme_agreement.items() if a >= agreement_threshold]
    low_themes = [t for t, a in theme_agreement.items() if a < agreement_threshold]

    print(f"\n  Threshold: agreement ≥ {agreement_threshold}")
    print(f"  High-agreement themes: {len(high_themes)} / {len(theme_agreement)}")
    print(f"  Low-agreement themes:  {len(low_themes)} / {len(theme_agreement)}")

    # Also compute per-theme rank spread (max avg rank - min avg rank)
    theme_ranks = defaultdict(lambda: defaultdict(list))
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            theme_ranks[r["theme"]][model].append(rank_pos + 1)

    results = {}

    for label, subset_theme_list in [
        ("high_agreement", high_themes),
        ("low_agreement", low_themes),
        ("all", list(theme_agreement.keys())),
    ]:
        subset = [r for r in resolved if r["theme"] in subset_theme_list]
        if not subset:
            print(f"\n  {label}: no themes")
            continue

        # Mean rank
        ranks = defaultdict(list)
        for r in subset:
            for rank_pos, model in enumerate(r["model_ranking"]):
                ranks[model].append(rank_pos + 1)
        mean_ranks = {m: float(np.mean(ranks[m])) for m in MODELS}

        # Friedman test
        block_ranks = defaultdict(dict)
        for r in subset:
            for rank_pos, model in enumerate(r["model_ranking"]):
                block_ranks[(r["annotator"], r["theme"])][model] = rank_pos + 1

        friedman_data = {m: [] for m in MODELS}
        for key in sorted(block_ranks.keys()):
            block = block_ranks[key]
            if len(block) == len(MODELS):
                for m in MODELS:
                    friedman_data[m].append(block[m])

        n_blocks = len(friedman_data[MODELS[0]])

        subset_result = {
            "n_themes": len(subset_theme_list),
            "n_ratings": len(subset),
            "n_blocks": n_blocks,
            "mean_ranks": mean_ranks,
            "themes": sorted(subset_theme_list),
        }

        print(f"\n  ── {label.upper()} ({len(subset_theme_list)} themes, "
              f"{len(subset)} ratings, {n_blocks} blocks) ──")
        for m in sorted(MODELS, key=lambda x: mean_ranks[x]):
            print(f"    {m:>8s}: mean rank = {mean_ranks[m]:.3f}")

        if n_blocks >= 3:
            arrays = [np.array(friedman_data[m]) for m in MODELS]
            stat, p_val = stats.friedmanchisquare(*arrays)
            subset_result["friedman_chi2"] = float(stat)
            subset_result["friedman_p"] = float(p_val)
            sig_str = "SIGNIFICANT" if p_val < 0.05 else "not significant"
            print(f"    Friedman χ² = {stat:.3f}, p = {p_val:.6f}  ({sig_str})")

            # Pairwise Wilcoxon
            wilcoxon_results = {}
            for m1, m2 in itertools.combinations(MODELS, 2):
                r1 = np.array(friedman_data[m1])
                r2 = np.array(friedman_data[m2])
                diff = r1 - r2
                if np.any(diff != 0):
                    stat_w, p_w = stats.wilcoxon(r1, r2)
                    sig_mark = "*" if p_w < 0.05 else ""
                    pair_key = f"{m1}_vs_{m2}"
                    wilcoxon_results[pair_key] = {
                        "W": float(stat_w),
                        "p": float(p_w),
                        "significant": p_w < 0.05,
                    }
                    print(f"      Wilcoxon {m1} vs {m2}: W={stat_w:.0f}, "
                          f"p={p_w:.4f} {sig_mark}")
            subset_result["wilcoxon"] = wilcoxon_results

        results[label] = subset_result

    # Add per-theme agreement scores to results
    results["per_theme_agreement"] = {
        t: float(a) for t, a in sorted_themes
    }
    results["agreement_threshold"] = agreement_threshold

    # Save to JSON (convert numpy types for serialization)
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return super().default(obj)

    out_path = RESULTS_DIR / "high_agreement_subset.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Saved subset analysis → {out_path}")

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def make_plots(resolved, borda_avg, rank_dist, pairwise, mean_ranks, theme_details, bootstrap_ci):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    model_labels = {
        "32B": "Qwen3-32B",
        "30A3": "Qwen3-30B-A3B\n(MoE)",
        "235A22": "Qwen3-235B-A22B\n(MoE)",
    }
    colors = {"32B": "#3b82f6", "30A3": "#f59e0b", "235A22": "#10b981"}

    # 1. Mean rank with bootstrap CI
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(MODELS))
    means = [bootstrap_ci[m][0] for m in MODELS]
    lo = [bootstrap_ci[m][0] - bootstrap_ci[m][1] for m in MODELS]
    hi = [bootstrap_ci[m][2] - bootstrap_ci[m][0] for m in MODELS]
    bars = ax.bar(x_pos, means, color=[colors[m] for m in MODELS],
                  yerr=[lo, hi], capsize=8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in MODELS])
    ax.set_ylabel("Mean Rank (1=best, 3=worst)")
    ax.set_title("Mean Human Rank per Model (95% Bootstrap CI)")
    ax.set_ylim(0.8, 3.2)
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="Chance level")
    ax.legend()
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{mean:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "human_mean_rank.png", dpi=150)
    plt.close()

    # 2. Rank distribution stacked bar
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(MODELS))
    width = 0.55
    rank_colors = ["#22c55e", "#f59e0b", "#ef4444"]
    rank_labels_plot = ["1st (Best)", "2nd", "3rd (Worst)"]
    bottoms = np.zeros(len(MODELS))
    for rank in [1, 2, 3]:
        vals = [rank_dist[m][rank] for m in MODELS]
        total = [sum(rank_dist[m].values()) for m in MODELS]
        pcts = [v / t * 100 if t > 0 else 0 for v, t in zip(vals, total)]
        ax.bar(x_pos, pcts, width, bottom=bottoms, label=rank_labels_plot[rank - 1],
               color=rank_colors[rank - 1], edgecolor="white", linewidth=0.5)
        for i, (p, b) in enumerate(zip(pcts, bottoms)):
            if p > 5:
                ax.text(x_pos[i], b + p / 2, f"{p:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += pcts
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in MODELS])
    ax.set_ylabel("Percentage of Rankings")
    ax.set_title("Rank Distribution per Model (Human Annotations)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "human_rank_distribution.png", dpi=150)
    plt.close()

    # 3. Per-theme heatmap (average rank per model per theme)
    themes = sorted(theme_details.keys())
    if len(themes) > 0:
        short_labels = ["32B", "30A3", "235A22"]
        fig, ax = plt.subplots(figsize=(14, max(6, len(themes) * 0.35)))
        matrix = np.zeros((len(themes), len(MODELS)))
        for i, t in enumerate(themes):
            for j, m in enumerate(MODELS):
                matrix[i][j] = theme_details[t].get(m, 2.0)
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=3, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(short_labels)
        # Truncate theme names
        theme_short = [t[:40] + "..." if len(t) > 40 else t for t in themes]
        ax.set_yticks(range(len(themes)))
        ax.set_yticklabels(theme_short, fontsize=7)
        ax.set_title("Average Rank per Theme per Model (1=best, 3=worst)")
        for i in range(len(themes)):
            for j in range(len(MODELS)):
                ax.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center",
                        fontsize=7, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Avg Rank")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "human_theme_heatmap.png", dpi=150)
        plt.close()

    # 4. Log-parameter count vs mean rank
    import math as _math
    fig, ax = plt.subplots(figsize=(7, 5))
    params = [PARAM_COUNT[m] for m in MODELS]
    log_params = [_math.log(p) for p in params]
    mr = [mean_ranks[m] for m in MODELS]
    for m, lp, r in zip(MODELS, log_params, mr):
        ax.scatter(lp, r, s=200, c=colors[m], edgecolors="black", zorder=5)
        ax.annotate(m, (lp, r), textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight="bold")
    # trend line
    z = np.polyfit(log_params, mr, 1)
    xs = np.linspace(min(log_params) - 0.2, max(log_params) + 0.2, 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="#2563eb", alpha=0.5, linewidth=1)
    ax.set_xlabel("log(Parameter Count)")
    ax.set_ylabel("Mean Human Rank (lower=better)")
    ax.set_title("log(Params) vs Mean Human Rank")
    ax.set_ylim(1.5, 2.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    if len(MODELS) >= 3:
        rho, p_val = stats.spearmanr(params, mr)
        ax.text(0.05, 0.05, f"Spearman ρ = {rho:.2f} (p={p_val:.3f})",
                transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "human_logparam_vs_rank.png", dpi=150)
    plt.close()

    print(f"  Saved 4 plots → {PLOT_DIR}/human_*.png")


def make_subset_plots(resolved, subset_results):
    """Generate the same 4 plots but restricted to the 22 high-agreement themes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    high_themes = subset_results["high_agreement"]["themes"]
    subset = [r for r in resolved if r["theme"] in high_themes]

    model_labels = {
        "32B": "Qwen3-32B",
        "30A3": "Qwen3-30B-A3B\n(MoE)",
        "235A22": "Qwen3-235B-A22B\n(MoE)",
    }
    colors = {"32B": "#3b82f6", "30A3": "#f59e0b", "235A22": "#10b981"}

    # Compute stats for subset
    sub_mean_ranks = compute_mean_rank(subset)
    sub_rank_dist = compute_rank_distribution(subset)
    sub_bootstrap = bootstrap_mean_rank(subset)
    _, sub_theme_details = per_theme_winners(subset)

    # 1. Mean rank with bootstrap CI (subset)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(MODELS))
    means = [sub_bootstrap[m][0] for m in MODELS]
    lo = [sub_bootstrap[m][0] - sub_bootstrap[m][1] for m in MODELS]
    hi = [sub_bootstrap[m][2] - sub_bootstrap[m][0] for m in MODELS]
    bars = ax.bar(x_pos, means, color=[colors[m] for m in MODELS],
                  yerr=[lo, hi], capsize=8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in MODELS])
    ax.set_ylabel("Mean Rank (1=best, 3=worst)")
    ax.set_title("Mean Human Rank — High-Agreement Subset (22 themes, 95% CI)")
    ax.set_ylim(0.8, 3.2)
    ax.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, label="Chance level")
    ax.legend()
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{mean:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "human_subset_mean_rank.png", dpi=150)
    plt.close()

    # 2. Rank distribution stacked bar (subset)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(MODELS))
    width = 0.55
    rank_colors = ["#22c55e", "#f59e0b", "#ef4444"]
    rank_labels_plot = ["1st (Best)", "2nd", "3rd (Worst)"]
    bottoms = np.zeros(len(MODELS))
    for rank in [1, 2, 3]:
        vals = [sub_rank_dist[m][rank] for m in MODELS]
        total = [sum(sub_rank_dist[m].values()) for m in MODELS]
        pcts = [v / t * 100 if t > 0 else 0 for v, t in zip(vals, total)]
        ax.bar(x_pos, pcts, width, bottom=bottoms, label=rank_labels_plot[rank - 1],
               color=rank_colors[rank - 1], edgecolor="white", linewidth=0.5)
        for i, (p, b) in enumerate(zip(pcts, bottoms)):
            if p > 5:
                ax.text(x_pos[i], b + p / 2, f"{p:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += pcts
    ax.set_xticks(x_pos)
    ax.set_xticklabels([model_labels[m] for m in MODELS])
    ax.set_ylabel("Percentage of Rankings")
    ax.set_title("Rank Distribution — High-Agreement Subset (22 themes)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "human_subset_rank_distribution.png", dpi=150)
    plt.close()

    # 3. Per-theme heatmap (subset)
    themes = sorted(sub_theme_details.keys())
    if len(themes) > 0:
        short_labels = ["32B", "30A3", "235A22"]
        fig, ax = plt.subplots(figsize=(14, max(6, len(themes) * 0.35)))
        matrix = np.zeros((len(themes), len(MODELS)))
        for i, t in enumerate(themes):
            for j, m in enumerate(MODELS):
                matrix[i][j] = sub_theme_details[t].get(m, 2.0)
        im = ax.imshow(matrix, cmap="RdYlGn_r", vmin=1, vmax=3, aspect="auto")
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(short_labels)
        theme_short = [t[:40] + "..." if len(t) > 40 else t for t in themes]
        ax.set_yticks(range(len(themes)))
        ax.set_yticklabels(theme_short, fontsize=7)
        ax.set_title("Average Rank per Theme — High-Agreement Subset (1=best, 3=worst)")
        for i in range(len(themes)):
            for j in range(len(MODELS)):
                ax.text(j, i, f"{matrix[i][j]:.1f}", ha="center", va="center",
                        fontsize=7, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Avg Rank")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "human_subset_theme_heatmap.png", dpi=150)
        plt.close()

    # 4. Log-parameter count vs mean rank (subset)
    import math as _math
    fig, ax = plt.subplots(figsize=(7, 5))
    params = [PARAM_COUNT[m] for m in MODELS]
    log_params = [_math.log(p) for p in params]
    mr = [sub_mean_ranks[m] for m in MODELS]
    for m, lp, r in zip(MODELS, log_params, mr):
        ax.scatter(lp, r, s=200, c=colors[m], edgecolors="black", zorder=5)
        ax.annotate(m, (lp, r), textcoords="offset points", xytext=(10, 10),
                    fontsize=10, fontweight="bold")
    z = np.polyfit(log_params, mr, 1)
    xs = np.linspace(min(log_params) - 0.2, max(log_params) + 0.2, 50)
    ax.plot(xs, np.polyval(z, xs), "--", color="#2563eb", alpha=0.5, linewidth=1)
    ax.set_xlabel("log(Parameter Count)")
    ax.set_ylabel("Mean Human Rank (lower=better)")
    ax.set_title("log(Params) vs Mean Human Rank — High-Agreement Subset")
    ax.set_ylim(1.5, 2.5)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    if len(MODELS) >= 3:
        rho, p_val = stats.spearmanr(params, mr)
        ax.text(0.05, 0.05, f"Spearman ρ = {rho:.2f} (p={p_val:.3f})",
                transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "human_subset_logparam_vs_rank.png", dpi=150)
    plt.close()

    print(f"  Saved 4 subset plots → {PLOT_DIR}/human_subset_*.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    all_themes = load_prompts()
    mapping = load_mapping()
    annotations = load_annotations()

    print(f"Loaded {len(annotations)} annotators, {len(mapping)} theme mappings\n")

    resolved = resolve_rankings(annotations, all_themes, mapping)
    print(f"Resolved {len(resolved)} individual theme ratings")
    print(f"  Group 1: {sum(1 for r in resolved if r['group'] == 1)} ratings "
          f"({len(set(r['annotator'] for r in resolved if r['group'] == 1))} annotators)")
    print(f"  Group 2: {sum(1 for r in resolved if r['group'] == 2)} ratings "
          f"({len(set(r['annotator'] for r in resolved if r['group'] == 2))} annotators)")

    # ── 1. Borda Scores ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  1. BORDA SCORES & WIN RATES")
    print(SEP)
    scores, counts, borda_avg = compute_borda(resolved)
    for m in sorted(MODELS, key=lambda x: borda_avg[x], reverse=True):
        print(f"  {m:>8s}:  avg Borda = {borda_avg[m]:.3f}  "
              f"(total = {scores[m]:>4d}, from {counts[m]} ratings)")

    # ── 2. Mean Rank ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  2. MEAN RANK (1=best, 3=worst)")
    print(SEP)
    mean_ranks = compute_mean_rank(resolved)
    for m in sorted(MODELS, key=lambda x: mean_ranks[x]):
        print(f"  {m:>8s}:  mean rank = {mean_ranks[m]:.3f}")

    # ── 3. Rank Distribution ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  3. RANK DISTRIBUTION")
    print(SEP)
    rank_dist = compute_rank_distribution(resolved)
    print(f"  {'Model':>8s}  {'1st':>6s}  {'2nd':>6s}  {'3rd':>6s}  {'Total':>6s}")
    print(f"  {THIN}")
    for m in MODELS:
        total = sum(rank_dist[m].values())
        r1 = rank_dist[m][1]
        r2 = rank_dist[m][2]
        r3 = rank_dist[m][3]
        print(f"  {m:>8s}  {r1:>4d} ({r1/total*100:4.1f}%)  "
              f"{r2:>4d} ({r2/total*100:4.1f}%)  "
              f"{r3:>4d} ({r3/total*100:4.1f}%)  {total:>4d}")

    # ── 4. Pairwise Head-to-Head ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  4. PAIRWISE HEAD-TO-HEAD")
    print(SEP)
    pairwise = compute_pairwise(resolved)
    for m1, m2 in itertools.combinations(MODELS, 2):
        w1 = pairwise[m1][m2]
        w2 = pairwise[m2][m1]
        total = w1 + w2
        print(f"  {m1:>8s} vs {m2:<8s}:  {w1:>3d}–{w2:<3d}  "
              f"({w1/total*100:.1f}%–{w2/total*100:.1f}%)")

    # ── 5. Inter-Annotator Agreement ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  5. INTER-ANNOTATOR AGREEMENT")
    print(SEP)

    alpha_nom = krippendorff_alpha_nominal(resolved)
    alpha_ord = krippendorff_alpha_ordinal(resolved)
    print(f"  Krippendorff's α (nominal):  {alpha_nom:.4f}")
    print(f"  Krippendorff's α (ordinal):  {alpha_ord:.4f}")

    print()
    print("  Interpretation:")
    if alpha_nom < 0:
        print("    α < 0: Less agreement than chance (systematic disagreement)")
    elif alpha_nom < 0.2:
        print("    α < 0.2: Slight agreement")
    elif alpha_nom < 0.4:
        print("    0.2 ≤ α < 0.4: Fair agreement")
    elif alpha_nom < 0.6:
        print("    0.4 ≤ α < 0.6: Moderate agreement")
    elif alpha_nom < 0.8:
        print("    0.6 ≤ α < 0.8: Substantial agreement")
    else:
        print("    α ≥ 0.8: Almost perfect agreement")

    print("    (Low agreement is expected for subjective humor rating)")

    w_results = kendall_w(resolved)
    print()
    for grp, w in sorted(w_results.items()):
        print(f"  Kendall's W (Group {grp}):  {w:.4f}")
    if w_results:
        print("    W interpretation: 0=no agreement, 1=perfect agreement")
        print("    W > 0.5 is generally considered moderate concordance")

    # ── 6. Bootstrap CI ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  6. BOOTSTRAP 95% CI FOR MEAN RANK")
    print(SEP)
    bootstrap_ci = bootstrap_mean_rank(resolved)
    for m in sorted(MODELS, key=lambda x: bootstrap_ci[x][0]):
        mean, lo, hi = bootstrap_ci[m]
        print(f"  {m:>8s}:  {mean:.3f}  [{lo:.3f}, {hi:.3f}]")

    # ── 7. Per-Theme Winners ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  7. PER-THEME WINNERS")
    print(SEP)
    winners, theme_details = per_theme_winners(resolved)
    win_counts = Counter(winners.values())
    for m in sorted(MODELS, key=lambda x: win_counts.get(x, 0), reverse=True):
        print(f"  {m:>8s}:  won {win_counts.get(m, 0):>2d} themes")
    print()
    for theme in sorted(winners.keys()):
        ranks_str = ", ".join(f"{m}={theme_details[theme][m]:.2f}" for m in MODELS)
        print(f"  {theme[:50]:<50s}  → {winners[theme]:>8s}  ({ranks_str})")

    # ── 8. Group Comparison ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  8. GROUP COMPARISON")
    print(SEP)
    for grp in [1, 2]:
        grp_resolved = [r for r in resolved if r["group"] == grp]
        grp_mean = compute_mean_rank(grp_resolved) if grp_resolved else {}
        theme_range = "11–30" if grp == 1 else "31–50"
        print(f"  Group {grp} (themes {theme_range}):")
        for m in sorted(MODELS, key=lambda x: grp_mean.get(x, 2)):
            print(f"    {m:>8s}:  mean rank = {grp_mean.get(m, float('nan')):.3f}")
        print()

    # ── 9. Statistical Tests ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  9. STATISTICAL TESTS")
    print(SEP)

    # Friedman test (non-parametric, related samples)
    # For each annotator-theme, we have ranks for 3 models
    # Group by (annotator, theme) as blocks
    rank_per_theme = defaultdict(lambda: defaultdict(list))
    for r in resolved:
        for rank_pos, model in enumerate(r["model_ranking"]):
            rank_per_theme[(r["annotator"], r["theme"])][model] = rank_pos + 1

    # Build arrays for Friedman test
    friedman_data = {m: [] for m in MODELS}
    for key in sorted(rank_per_theme.keys()):
        block = rank_per_theme[key]
        if len(block) == len(MODELS):
            for m in MODELS:
                friedman_data[m].append(block[m])

    if len(friedman_data[MODELS[0]]) >= 3:
        arrays = [np.array(friedman_data[m]) for m in MODELS]
        stat, p_val = stats.friedmanchisquare(*arrays)
        print(f"  Friedman χ² = {stat:.3f},  p = {p_val:.6f}")
        if p_val < 0.05:
            print("  → Significant difference between models (p < 0.05)")
        else:
            print("  → No significant difference between models (p ≥ 0.05)")
    else:
        print("  (Insufficient data for Friedman test)")

    # Wilcoxon signed-rank for pairwise comparisons
    print()
    for m1, m2 in itertools.combinations(MODELS, 2):
        r1 = np.array(friedman_data[m1])
        r2 = np.array(friedman_data[m2])
        diff = r1 - r2
        if np.any(diff != 0):
            stat, p_val = stats.wilcoxon(r1, r2)
            sig = "*" if p_val < 0.05 else ""
            print(f"  Wilcoxon {m1} vs {m2}:  W={stat:.0f},  p={p_val:.4f} {sig}")
        else:
            print(f"  Wilcoxon {m1} vs {m2}:  identical rankings")

    # ── 10. High-Agreement Subset Analysis ─────────────────────────────────
    print(f"\n{SEP}")
    print("  10. HIGH-AGREEMENT SUBSET ANALYSIS")
    print(SEP)
    subset_results = high_agreement_subset_analysis(resolved)

    # ── 11. Length Bias Analysis ───────────────────────────────────────────
    print(f"\n{SEP}")
    print("  11. LENGTH BIAS ANALYSIS")
    print(SEP)
    try:
        joke_lengths = {}
        for m in MODELS:
            with open(RESULTS_DIR / f"{m}.json") as f:
                joke_lengths[m] = {t: len(j) for t, j in json.load(f).items()}
        
        ranks_and_lengths = []
        for r in resolved:
            theme = r["theme"]
            ranking = r["model_ranking"]
            for pos, model in enumerate(ranking):
                length = joke_lengths.get(model, {}).get(theme, None)
                if length is not None:
                    ranks_and_lengths.append((length, pos + 1)) # lower pos is better rank
        
        if ranks_and_lengths:
            x = [item[0] for item in ranks_and_lengths]
            y = [item[1] for item in ranks_and_lengths]
            corr, p_val = stats.spearmanr(x, y)
            print(f"  Spearman correlation between length and assigned rank: ρ = {corr:.3f} (p = {p_val:.4f})")
            print("  (Negative correlation means longer jokes get rank closer to 1, i.e., better)")
            sig = "YES" if p_val < 0.05 else "NO"
            print(f"  Significant length bias? {sig}")
        else:
            print("  (Could not match themes to json files for length calculations)")
    except Exception as e:
        print(f"  (Failed to calculate length bias: {e})")

    # ── Plots ────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  GENERATING PLOTS")
    print(SEP)
    make_plots(resolved, borda_avg, rank_dist, pairwise, mean_ranks, theme_details, bootstrap_ci)
    make_subset_plots(resolved, subset_results)

    print(f"\n{'━' * 70}")
    print("  DONE")
    print(f"{'━' * 70}")


if __name__ == "__main__":
    main()
