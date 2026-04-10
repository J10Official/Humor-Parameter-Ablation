import json
from pathlib import Path
import numpy as np
from scipy import stats

RESULTS_DIR = Path("results")
MODEL_KEYS = ["8B", "14B", "32B", "30A3", "235A22"]

# Load jokes and calculate lengths (char lengths)
jokes = {}
lengths = {}
for m in MODEL_KEYS:
    with open(RESULTS_DIR / f"{m}.json") as f:
        jokes[m] = json.load(f)
        lengths[m] = {t: len(j) for t, j in jokes[m].items()}

# Average length per model
print("Average joke length per model:")
for m in MODEL_KEYS:
    avg_len = np.mean(list(lengths[m].values()))
    print(f"  {m}: {avg_len:.1f} chars")

# LLM Judge Length Bias
with open(RESULTS_DIR / "judgements.json") as f:
    judgements = json.load(f)

longer_wins = 0
shorter_wins = 0
equal_len = 0
length_diffs = []
wins_by_length = []

for t, data in judgements.items():
    if len(data.get("comparisons", [])) < 19:
        continue
    for c in data["comparisons"]:
        m1, m2 = c["first"], c["second"]
        w = c["winner"]
        l1, l2 = lengths[m1].get(t, 0), lengths[m2].get(t, 0)
        
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
        else:
            equal_len += 1

total_unequal = longer_wins + shorter_wins
print("\nLLM Judge Length Bias:")
print(f"  Longer wins: {longer_wins} ({longer_wins/total_unequal*100:.1f}%)")
print(f"  Shorter wins: {shorter_wins} ({shorter_wins/total_unequal*100:.1f}%)")
from scipy.stats import binomtest
p_val = binomtest(longer_wins, total_unequal, 0.5).pvalue
print(f"  Binomial test p-value: {p_val:.4f}")

# Human Evaluator Length Bias
# Read annotations
import csv
with open(RESULTS_DIR / "viewer_label_mapping.json") as f:
    mapping = json.load(f)

with open("prompts.txt") as f:
    all_themes = [line.strip().split(". ", 1)[-1] for line in f if line.strip()]

human_ranks = [] # list of (length, rank)
group1_themes = all_themes[10:30]
group2_themes = all_themes[30:50]

with open(RESULTS_DIR / "annotations.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        group = int(row["Group"])
        themes = group1_themes if group == 1 else group2_themes
        
        for i in range(1, 21):
            ranking_str = row[f"Set_{i}"]
            if not ranking_str:
                continue
            theme = themes[i-1]
            if theme not in mapping:
                continue
            
            label_to_model = mapping[theme]
            label_idx = {"A": 0, "B": 1, "C": 2}
            
            for rank_pos, ch in enumerate(ranking_str):
                idx = label_idx[ch]
                model = label_to_model[idx]
                human_ranks.append((lengths[model][theme], rank_pos + 1)) # rank 1, 2, 3

x = [r[0] for r in human_ranks]
y = [r[1] for r in human_ranks] # lower is better!
corr, p_val = stats.spearmanr(x, y)
print("\nHuman Annotator Length Bias:")
print(f"  Spearman correlation between length and rank: {corr:.3f} (p={p_val:.4f})")
print("  (Negative correlation means longer jokes get LOWER rank number = BETTER rank)")

