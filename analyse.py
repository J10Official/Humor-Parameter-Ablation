"""
analyse.py — Statistical analysis of pairwise humor judgements.

Usage: python3 analyse.py              # prints report + saves plots
       python3 analyse.py --no-plots   # text-only (no matplotlib)

Reads results/judgements.json (however many themes are done).
Outputs:
  - Terminal report (global rankings, head-to-head, positional bias, etc.)
  - results/plots/  (bar charts, heatmaps, correlation plot)
"""

import json
import math
import argparse
import itertools
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_KEYS  = ["8B", "14B", "32B", "30A3", "235A22"]
PARAM_COUNT = {"8B": 8, "14B": 14, "32B": 32, "30A3": 30, "235A22": 235}
RESULTS_DIR = Path("results")
JUDGE_FILE  = RESULTS_DIR / "judgements.json"
PLOT_DIR    = RESULTS_DIR / "plots"

SEP = "═" * 70
THIN = "─" * 70


# ── Load data ────────────────────────────────────────────────────────────────
def load() -> dict:
    with open(JUDGE_FILE) as f:
        return json.load(f)


# ── Bradley-Terry (from all comparisons pooled) ─────────────────────────────
def bradley_terry_global(all_comps: list[dict], max_iter=2000, tol=1e-10) -> dict[str, float]:
    scores = {k: 1.0 for k in MODEL_KEYS}
    for _ in range(max_iter):
        new = {}
        for k in MODEL_KEYS:
            wins = sum(1 for c in all_comps if c["winner"] == k)
            denom = sum(
                1.0 / (scores[k] + scores[c["second"]])
                for c in all_comps if c["first"] == k
            ) + sum(
                1.0 / (scores[c["first"]] + scores[k])
                for c in all_comps if c["second"] == k
            )
            new[k] = wins / denom if denom > 0 else scores[k]
        total = sum(new.values())
        new = {k: v / total for k, v in new.items()}
        delta = max(abs(new[k] - scores[k]) for k in MODEL_KEYS)
        scores = new
        if delta < tol:
            break
    return scores


# ── Analysis ──────────────────────────────────────────────────────────────────

def bt_numpy(win_matrix, max_iter=200, n_models=5):
    scores = np.ones(n_models) / n_models
    wins = np.sum(win_matrix, axis=1)
    games_played = win_matrix + win_matrix.T
    for _ in range(max_iter):
        denom = np.zeros(n_models)
        for i in range(n_models):
            for j in range(n_models):
                if i != j and games_played[i, j] > 0:
                    denom[i] += games_played[i, j] / (scores[i] + scores[j])
        new_scores = np.zeros(n_models)
        for i in range(n_models):
            new_scores[i] = wins[i] / denom[i] if denom[i] > 0 else scores[i]
        new_scores /= new_scores.sum()
        if np.max(np.abs(new_scores - scores)) < 1e-10:
            break
        scores = new_scores
    return scores

def bradley_terry_se(themes: list[str], data: dict, n_bootstrap=1000) -> dict[str, float]:
    # Cluster bootstrap: resample at the theme level
    n_models = len(MODEL_KEYS)
    key_to_idx = {k: i for i, k in enumerate(MODEL_KEYS)}
    
    # Pre-compute win matrices for each theme
    n_themes = len(themes)
    theme_matrices = np.zeros((n_themes, n_models, n_models))
    for t_idx, t in enumerate(themes):
        for c in data[t]["comparisons"]:
            w = c["winner"]
            l = c["second"] if c["first"] == w else c["first"]
            theme_matrices[t_idx, key_to_idx[w], key_to_idx[l]] += 1
            
    boot_scores = np.zeros((n_bootstrap, n_models))
    for b in range(n_bootstrap):
        indices = np.random.randint(0, n_themes, size=n_themes)
        # Sum the win matrices for the sampled themes
        win_matrix = np.sum(theme_matrices[indices], axis=0)
        boot_scores[b] = bt_numpy(win_matrix, n_models=n_models)
        
    se = np.std(boot_scores, axis=0)
    return {MODEL_KEYS[i]: se[i] for i in range(n_models)}

def analyse(data: dict, make_plots: bool = True):
    themes = [t for t, v in data.items() if len(v.get("comparisons", [])) >= 19]
    n = len(themes)
    if n == 0:
        print("No fully-judged themes found.")
        return

    print(f"\n{SEP}")
    print(f"  HUMOR SCALING ANALYSIS  —  {n} theme(s)")
    print(SEP)

    # ── 1. Pool all comparisons ──────────────────────────────────────────────
    all_comps: list[dict] = []
    for t in themes:
        all_comps.extend(data[t]["comparisons"])

    total_comps = len(all_comps)
    print(f"\n  Total pairwise comparisons: {total_comps}  ({n} themes × 20)")

    # ── 2. Global win counts & win rates ─────────────────────────────────────
    wins   = Counter()
    played = Counter()
    for c in all_comps:
        wins[c["winner"]] += 1
        played[c["first"]]  += 1
        played[c["second"]] += 1

    global_wr = {k: wins[k] / played[k] if played[k] else 0 for k in MODEL_KEYS}

    print(f"\n{THIN}")
    print("  1. GLOBAL WIN RATES (pooled across all themes)")
    print(THIN)
    print(f"  {'Model':>8}  {'Wins':>6}  {'Played':>7}  {'Win Rate':>9}")
    print(f"  {THIN}")
    for k in sorted(MODEL_KEYS, key=lambda x: -global_wr[x]):
        bar = "█" * round(global_wr[k] * 30)
        print(f"  {k:>8}  {wins[k]:>6}  {played[k]:>7}  {global_wr[k]:>8.1%}   {bar}")

    # ── 3. Global Bradley-Terry ──────────────────────────────────────────────
    bt = bradley_terry_global(all_comps)
    print("  Calculating BT standard errors (bootstrap)...")
    bt_se = bradley_terry_se(themes, data)
    
    print(f"\n{THIN}")
    print("  2. GLOBAL BRADLEY-TERRY SCORES (MLE, normalised to sum=1)")
    print(THIN)
    print(f"  {'Model':>8}  {'BT Score':>10}  {'± SE':>7}   {'Relative':>10}")
    print(f"  {THIN}")
    max_bt = max(bt.values())
    for k in sorted(MODEL_KEYS, key=lambda x: -bt[x]):
        rel = bt[k] / max_bt if max_bt > 0 else 0
        bar = "█" * round(rel * 30)
        print(f"  {k:>8}  {bt[k]:>10.6f}  ±{bt_se[k]:>.4f}   {rel:>9.1%}   {bar}")

    # ── 4. Head-to-head matrix ───────────────────────────────────────────────
    h2h_wins = defaultdict(lambda: defaultdict(int))
    h2h_total = defaultdict(lambda: defaultdict(int))
    for c in all_comps:
        a, b, w = c["first"], c["second"], c["winner"]
        h2h_total[a][b] += 1
        h2h_total[b][a] += 1
        h2h_wins[w][a if w == a else b]  # just to track
        h2h_wins[a][b] += (1 if w == a else 0)
        h2h_wins[b][a] += (1 if w == b else 0)

    print(f"\n{THIN}")
    print("  3. HEAD-TO-HEAD WIN RATES  (row beats column)")
    print(THIN)
    header = "          " + "".join(f"{k:>8}" for k in MODEL_KEYS)
    print(f"  {header}")
    print(f"  {THIN}")

    h2h_matrix = np.zeros((len(MODEL_KEYS), len(MODEL_KEYS)))
    for i, a in enumerate(MODEL_KEYS):
        row = f"  {a:>8}"
        for j, b in enumerate(MODEL_KEYS):
            if a == b:
                row += "       —"
                h2h_matrix[i][j] = 0.5
            else:
                total = h2h_total[a][b]
                rate = h2h_wins[a][b] / total if total else 0
                h2h_matrix[i][j] = rate
                row += f"  {rate:>5.0%}  "
        print(row)

    # ── 5. Per-theme rankings ────────────────────────────────────────────────
    print(f"\n{THIN}")
    print("  4. PER-THEME RANKINGS (by win rate within each theme)")
    print(THIN)

    rank_accum = {k: [] for k in MODEL_KEYS}
    theme_winners = Counter()
    theme_losers  = Counter()

    for t in themes:
        wr = data[t].get("win_rates", {})
        # sort models by win rate descending
        ranked = sorted(MODEL_KEYS, key=lambda x: -(wr.get(x, 0)))
        for rank_idx, k in enumerate(ranked, start=1):
            rank_accum[k].append(rank_idx)
        theme_winners[ranked[0]] += 1
        theme_losers[ranked[-1]] += 1

    avg_rank = {k: np.mean(rank_accum[k]) for k in MODEL_KEYS}
    std_rank = {k: np.std(rank_accum[k])  for k in MODEL_KEYS}

    print(f"  {'Model':>8}  {'Avg Rank':>9}  {'Std':>6}  {'#1 Finishes':>12}  {'#5 Finishes':>12}")
    print(f"  {THIN}")
    for k in sorted(MODEL_KEYS, key=lambda x: avg_rank[x]):
        print(f"  {k:>8}  {avg_rank[k]:>9.2f}  {std_rank[k]:>6.2f}  {theme_winners[k]:>12}  {theme_losers[k]:>12}")

    # ── 6. Positional bias analysis ──────────────────────────────────────────
    a_wins = sum(1 for c in all_comps if c["winner"] == c["first"])
    b_wins = total_comps - a_wins

    print(f"\n{THIN}")
    print("  5. POSITIONAL BIAS (does being shown first/second matter?)")
    print(THIN)
    print(f"  Position A (first)  wins: {a_wins:>4}  ({a_wins/total_comps:.1%})")
    print(f"  Position B (second) wins: {b_wins:>4}  ({b_wins/total_comps:.1%})")

    # chi-square test for positional bias
    expected = total_comps / 2
    chi2 = (a_wins - expected)**2 / expected + (b_wins - expected)**2 / expected
    # p-value from chi2 with 1 df (using survival function approximation)
    from scipy.stats import chi2 as chi2_dist
    p_val = chi2_dist.sf(chi2, df=1)
    sig = "YES" if p_val < 0.05 else "No"
    print(f"  χ² = {chi2:.3f},  p = {p_val:.4f}  →  Significant bias? {sig}")

    # ── 6.5 Length Bias Analysis ─────────────────────────────────────────────
    # Load jokes to get lengths
    lengths = {}
    for m in MODEL_KEYS:
        try:
            with open(RESULTS_DIR / f"{m}.json") as f:
                lengths[m] = {t: len(j) for t, j in json.load(f).items()}
        except Exception:
            lengths[m] = {}

    longer_wins = 0
    shorter_wins = 0
    for t in themes:
        for c in data[t].get("comparisons", []):
            m1, m2 = c["first"], c["second"]
            w = c["winner"]
            l1, l2 = lengths.get(m1, {}).get(t, 0), lengths.get(m2, {}).get(t, 0)
            if l1 > l2:
                if w == m1:
                    longer_wins += 1
                else:
                    shorter_wins += 1
            elif l2 > l1:
                if w == m2:
                    longer_wins += 1
                else:
                    shorter_wins += 1

    total_unequal = longer_wins + shorter_wins
    print(f"\n{THIN}")
    print("  5.5 LENGTH BIAS")
    print(THIN)
    if total_unequal > 0:
        print(f"  Longer wins:  {longer_wins:>4}  ({longer_wins/total_unequal:.1%})")
        print(f"  Shorter wins: {shorter_wins:>4}  ({shorter_wins/total_unequal:.1%})")
        from scipy.stats import binomtest
        p_val_len = binomtest(longer_wins, total_unequal, 0.5).pvalue
        sig_len = "YES" if p_val_len < 0.05 else "No"
        print(f"  Binomial p_val = {p_val_len:.4f} → Significant bias? {sig_len}")

    # ── 7. Consistency check: does order-swap change winner? ─────────────────
    print(f"\n{THIN}")
    print("  6. ORDER CONSISTENCY (same-pair agreement across AB/BA swaps)")
    print(THIN)
    pair_results = defaultdict(list)
    for c in all_comps:
        pair_key = tuple(sorted([c["first"], c["second"]]))
        pair_results[(pair_key)].append(c["winner"])

    # group by theme first
    theme_pair_results = defaultdict(lambda: defaultdict(list))
    for t in themes:
        for c in data[t]["comparisons"]:
            pair_key = tuple(sorted([c["first"], c["second"]]))
            theme_pair_results[t][pair_key].append(c["winner"])

    agreements = 0
    disagreements = 0
    for t in themes:
        for pair_key, winners in theme_pair_results[t].items():
            if len(winners) == 2:
                if winners[0] == winners[1]:
                    agreements += 1
                else:
                    disagreements += 1

    total_pairs_checked = agreements + disagreements
    agree_pct = agreements / total_pairs_checked if total_pairs_checked else 0
    print(f"  Same winner in both orderings: {agreements}/{total_pairs_checked} ({agree_pct:.1%})")
    print(f"  Flipped by order swap:         {disagreements}/{total_pairs_checked} ({1-agree_pct:.1%})")

    # ── 8. Parameter count vs. humor correlation ─────────────────────────────
    from scipy.stats import spearmanr, pearsonr

    params = [PARAM_COUNT[k] for k in MODEL_KEYS]
    wrs    = [global_wr[k]   for k in MODEL_KEYS]
    bts    = [bt[k]          for k in MODEL_KEYS]
    log_params = [math.log(p) for p in params]

    print(f"\n{THIN}")
    print("  7. PARAMETER COUNT vs HUMOR QUALITY  (correlation)")
    print(THIN)

    r_sp, p_sp = spearmanr(params, wrs)
    r_pe, p_pe = pearsonr(params, wrs)
    r_lp, p_lp = pearsonr(log_params, wrs)
    r_bt_sp, p_bt_sp = spearmanr(params, bts)

    print(f"  Spearman (params vs win-rate):     ρ = {r_sp:+.4f}   p = {p_sp:.4f}")
    print(f"  Pearson  (params vs win-rate):      r = {r_pe:+.4f}   p = {p_pe:.4f}")
    print(f"  Pearson  (log-params vs win-rate):  r = {r_lp:+.4f}   p = {p_lp:.4f}")
    print(f"  Spearman (params vs BT-score):     ρ = {r_bt_sp:+.4f}   p = {p_bt_sp:.4f}")

    # MoE check: are MoE models (30A3, 235A22) better than their dense param equivalents?
    print(f"\n  MoE spotlight:")
    for moe in ["30A3", "235A22"]:
        print(f"    {moe}: WR={global_wr[moe]:.1%}, BT={bt[moe]:.4f}, active params={PARAM_COUNT[moe]}B")

    # ── 9. Theme difficulty variance ─────────────────────────────────────────
    print(f"\n{THIN}")
    print("  8. THEME DIFFICULTY (most/least decisive themes)")
    print(THIN)

    theme_entropy = {}
    for t in themes:
        wr = data[t].get("win_rates", {})
        vals = [wr.get(k, 0) for k in MODEL_KEYS]
        # normalise to distribution
        s = sum(vals) or 1
        probs = [v/s for v in vals]
        ent = -sum(p * math.log(p + 1e-12) for p in probs)
        theme_entropy[t] = ent

    sorted_themes = sorted(theme_entropy.items(), key=lambda x: x[1])
    print(f"\n  Most decisive (lowest entropy = one clear winner):")
    for t, e in sorted_themes[:5]:
        wr = data[t]["win_rates"]
        top = max(MODEL_KEYS, key=lambda k: wr.get(k, 0))
        print(f"    {e:.3f}  {top:>7} ({wr[top]:.0%})  ← {t}")

    print(f"\n  Most contested (highest entropy = close race):")
    for t, e in sorted_themes[-5:]:
        wr = data[t]["win_rates"]
        vals = sorted([(wr.get(k,0), k) for k in MODEL_KEYS], reverse=True)
        top2 = f"{vals[0][1]}({vals[0][0]:.0%}), {vals[1][1]}({vals[1][0]:.0%})"
        print(f"    {e:.3f}  {top2}  ← {t}")

    # ── Summary ──────────────────────────────────────────────────────────────
    ranked_models = sorted(MODEL_KEYS, key=lambda x: -bt[x])
    print(f"\n{SEP}")
    print("  FINAL RANKING  (by global Bradley-Terry)")
    print(SEP)
    for i, k in enumerate(ranked_models, 1):
        medal = ["🥇", "🥈", "🥉", " 4.", " 5."][i-1]
        print(f"  {medal} {k:>7}   BT={bt[k]:.4f}   WR={global_wr[k]:.1%}   AvgRank={avg_rank[k]:.2f}")
    print(SEP)

    # ── Plots ────────────────────────────────────────────────────────────────
    if make_plots:
        generate_plots(
            bt, global_wr, h2h_matrix, avg_rank, std_rank,
            params, wrs, bts, log_params, themes, data,
        )


# ── Plots ─────────────────────────────────────────────────────────────────────
def generate_plots(bt, global_wr, h2h_matrix, avg_rank, std_rank,
                   params, wrs, bts, log_params, themes, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # style — light theme for publication
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "#333333",
        "text.color":       "#111111",
        "axes.labelcolor":  "#111111",
        "xtick.color":      "#333333",
        "ytick.color":      "#333333",
        "grid.color":       "#cccccc",
        "font.size":        11,
    })
    accent = "#2563eb"
    colors = ["#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed"]

    # ── Log-param vs win-rate scatter (primary scaling result) ───────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(MODEL_KEYS):
        ax.scatter(math.log(PARAM_COUNT[k]), global_wr[k], s=120, c=colors[i],
                   zorder=5, edgecolors="black", linewidths=0.5)
        offset = (5, 8) if k != "235A22" else (-30, 10)
        ax.annotate(k, (math.log(PARAM_COUNT[k]), global_wr[k]),
                    textcoords="offset points", xytext=offset, fontsize=10, color=colors[i])
    # trend line
    z = np.polyfit(log_params, wrs, 1)
    xs = np.linspace(min(log_params)-0.2, max(log_params)+0.2, 50)
    ax.plot(xs, np.polyval(z, xs), "--", color=accent, alpha=0.5, linewidth=1)
    ax.set_xlabel("log(Parameter Count)")
    ax.set_ylabel("Win Rate")
    ax.set_title("log(Params) vs Humor Win Rate", fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "logparam_vs_winrate.png", dpi=150)
    plt.close(fig)

    # ── Per-theme win-rate heatmap (vertical: themes as rows, models as cols) ─
    n_themes = len(themes)
    short_labels = MODEL_KEYS
    fig, ax = plt.subplots(figsize=(8, max(6, n_themes * 0.35)))
    matrix = np.array([[data[t].get("win_rates", {}).get(k, 0) for k in MODEL_KEYS] for t in themes])
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(MODEL_KEYS)))
    ax.set_xticklabels(short_labels)
    theme_short = [t[:40] + "..." if len(t) > 40 else t for t in themes]
    ax.set_yticks(range(n_themes))
    ax.set_yticklabels(theme_short, fontsize=7)
    ax.set_title("Per-Theme Win Rates (LLM Judge)", fontweight="bold")
    for i in range(n_themes):
        for j in range(len(MODEL_KEYS)):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center",
                    fontsize=6, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Win Rate")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "theme_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  📊  2 plots saved to {PLOT_DIR}/")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse pairwise humor judgements")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    data = load()
    analyse(data, make_plots=not args.no_plots)
