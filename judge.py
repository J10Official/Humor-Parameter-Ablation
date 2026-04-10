"""
Pairwise LLM judge for humor-vs-parameter-count study.

For each theme:
  - C(5,2) = 10 unique pairs
  - Each pair run twice with models swapped (positional bias mitigation)
  - 20 total comparisons per theme
  - Results: win-rate per model + Bradley-Terry scores (for tie-breaking)

Resume: progress saved after every single comparison.
"""

import json
import os
import re
import sys
import time
import argparse
import itertools
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


# ── Config ────────────────────────────────────────────────────────────────────
MODEL_KEYS     = ["8B", "14B", "32B", "30A3", "235A22"]
JUDGE_MODEL    = "deepseek/deepseek-v3.2"
JUDGE_PROVIDER = "atlas-cloud/fp8"
OR_BASE_URL    = "https://openrouter.ai/api/v1"
RESULTS_DIR    = Path("results")
JUDGE_FILE     = RESULTS_DIR / "judgements.json"
SEP            = "─" * 60

# 10 unique pairs × 2 orderings = 20 comparisons per theme
ALL_PAIRS: list[tuple[str, str]] = list(itertools.combinations(MODEL_KEYS, 2))
ALL_COMPARISONS: list[tuple[str, str]] = [
    ordering
    for pair in ALL_PAIRS
    for ordering in [pair, (pair[1], pair[0])]
]

JUDGE_SYSTEM = (
    "You are an expert comedy critic with a sharp, discerning sense of humor. "
    "Your task is to decide which of two jokes is funnier."
)

JUDGE_USER_TMPL = """\
Theme: "{theme}"

Joke A:
{joke_a}

Joke B:
{joke_b}

Which joke is funnier? Consider:
- Comedic timing and structure
- Originality of the idea
- Clarity and strength of the punchline
- How well it fits the theme

First, write 2-3 sentences explaining your reasoning. Then on the final line write ONLY:
WINNER: A
or
WINNER: B"""


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_jokes() -> dict[str, dict[str, str]]:
    """Return {theme: {model_key: joke}} for themes present in ALL model files."""
    per_model: dict[str, dict] = {}
    for key in MODEL_KEYS:
        path = RESULTS_DIR / f"{key}.json"
        if not path.exists():
            sys.exit(f"ERROR: results/{key}.json not found — run generate.py first.")
        with open(path) as f:
            per_model[key] = json.load(f)

    common = set(per_model[MODEL_KEYS[0]])
    for key in MODEL_KEYS[1:]:
        common &= set(per_model[key])

    return {
        theme: {k: per_model[k][theme] for k in MODEL_KEYS}
        for theme in per_model[MODEL_KEYS[0]]
        if theme in common
    }


def load_judgements() -> dict:
    if JUDGE_FILE.exists():
        with open(JUDGE_FILE) as f:
            return json.load(f)
    return {}


def save_judgements(data: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(JUDGE_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Bradley-Terry ─────────────────────────────────────────────────────────────
def bradley_terry(comparisons: list[dict], max_iter: int = 1000, tol: float = 1e-8) -> dict[str, float]:
    """
    Iterative MLE for Bradley-Terry scores from pairwise outcomes.
    Each entry in `comparisons`: {"first": key, "second": key, "winner": key}
    Returns {model_key: score} normalised to sum to 1.
    """
    scores = {k: 1.0 for k in MODEL_KEYS}
    for _ in range(max_iter):
        new_scores = {}
        for k in MODEL_KEYS:
            wins = sum(1 for c in comparisons if c["winner"] == k)
            denom = 0.0
            for c in comparisons:
                if c["first"] == k:
                    denom += 1.0 / (scores[k] + scores[c["second"]])
                elif c["second"] == k:
                    denom += 1.0 / (scores[c["first"]] + scores[k])
            new_scores[k] = wins / denom if denom > 0 else scores[k]

        total = sum(new_scores.values())
        new_scores = {k: v / total for k, v in new_scores.items()}
        delta = max(abs(new_scores[k] - scores[k]) for k in MODEL_KEYS)
        scores = new_scores
        if delta < tol:
            break

    return {k: round(scores[k], 6) for k in MODEL_KEYS}


def compute_win_rates(comparisons: list[dict]) -> dict[str, float]:
    wins   = {k: 0 for k in MODEL_KEYS}
    played = {k: 0 for k in MODEL_KEYS}
    for c in comparisons:
        wins[c["winner"]] += 1
        played[c["first"]]  += 1
        played[c["second"]] += 1
    return {
        k: round(wins[k] / played[k], 4) if played[k] else 0.0
        for k in MODEL_KEYS
    }


# ── Judge call ────────────────────────────────────────────────────────────────
def call_judge(
    client: OpenAI,
    theme: str,
    first: str,
    second: str,
    jokes: dict[str, str],
    comp_num: int,
    total: int,
) -> tuple[str, str] | None:
    """
    Ask the judge which of two jokes is funnier.
    Returns (winning_model_key, reasoning_text), or None if response is unparseable.
    """
    user_msg = JUDGE_USER_TMPL.format(
        theme=theme,
        joke_a=jokes[first],
        joke_b=jokes[second],
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_msg},
    ]

    # ── Log request ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  COMPARISON {comp_num}/{total}  [{JUDGE_MODEL} @ {JUDGE_PROVIDER}]")
    print(f"  A = {first}   vs   B = {second}")
    print(SEP)
    print(f"  [SYSTEM]  {JUDGE_SYSTEM}")
    print(f"\n  [USER]")
    print(user_msg)
    print(SEP)

    raw = ""
    for attempt in range(1, 7):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                max_tokens=256,
                extra_body={"provider": {"order": [JUDGE_PROVIDER], "allow_fallbacks": False}},
            )
            raw = (response.choices[0].message.content or "").strip()
            break
        except Exception as e:
            if "429" in str(e) and attempt < 6:
                wait = 30 * attempt
                print(f"  [rate-limited, retrying in {wait}s (attempt {attempt}/6)]", flush=True)
                time.sleep(wait)
            else:
                raise

    # strip thinking tokens if any
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    cleaned = re.sub(r"<think>.*",          "", cleaned, flags=re.DOTALL).strip()

    # ── Log response ───────────────────────────────────────────────────
    print(f"  RAW RESPONSE:  {raw!r}")
    if cleaned != raw:
        print(f"  CLEANED:       {cleaned!r}")

    letter = re.search(r"WINNER\s*:\s*([AB])", cleaned.upper())
    if not letter:
        # fallback: bare A/B on its own line at the end
        letter = re.search(r"^\s*([AB])\s*$", cleaned.upper(), re.MULTILINE)
    if not letter:
        print(f"  ⚠  Unparseable response — skipping.")
        print(SEP)
        return None

    winner = first if letter.group(1) == "A" else second
    # extract reasoning: everything before the WINNER line
    reasoning = re.sub(r"\bWINNER\s*:\s*[AB]\b.*", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    print(f"  Winner: {letter.group(1)} → {winner}")
    print(SEP)
    return winner, reasoning


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(theme: str, entry: dict):
    comps = entry.get("comparisons", [])
    wr    = entry.get("win_rates", {})
    bt    = entry.get("bradley_terry", {})
    print(f"\n  {'─' * 50}")
    print(f"  Results for '{theme}'  ({len(comps)}/20 comparisons)")
    print(f"  {'─' * 50}")
    print(f"  {'Model':>8}   {'Win-rate':>9}   {'BT score':>9}")
    print(f"  {'─' * 50}")
    for k in sorted(MODEL_KEYS, key=lambda x: -(wr.get(x) or 0)):
        bar = "█" * round((wr.get(k, 0)) * 20)
        print(f"  {k:>8}   {str(wr.get(k, '—')):>9}   {str(bt.get(k, '—')):>9}   {bar}")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(num_sets: int = 1):
    load_dotenv()
    or_key = os.getenv("OR_API_KEY")
    if not or_key:
        sys.exit("ERROR: OR_API_KEY not found in .env")

    client     = OpenAI(api_key=or_key, base_url=OR_BASE_URL)
    all_jokes  = load_jokes()
    judgements = load_judgements()

    done_count = sum(1 for v in judgements.values() if len(v.get("comparisons", [])) == 20)
    remaining  = max(0, num_sets - done_count)

    print(f"{'═' * 60}")
    print(f"  Judge     : {JUDGE_MODEL}  [{JUDGE_PROVIDER}]")
    print(f"  Models    : {' | '.join(MODEL_KEYS)}")
    print(f"  Pairs     : {len(ALL_PAIRS)} unique  ×  2 orderings  =  20 comparisons/theme")
    print(f"  Target    : {num_sets} theme(s)")
    print(f"  Done      : {done_count}  |  Remaining: {remaining}")
    print(f"{'═' * 60}")

    if remaining == 0:
        print("\nAll requested themes already judged. Nothing to do.")
        return

    todo = [
        t for t in all_jokes
        if len(judgements.get(t, {}).get("comparisons", [])) < 20
    ][:remaining]

    if not todo:
        print("\nNo eligible themes found.")
        return

    for set_num, theme in enumerate(todo, start=done_count + 1):
        print(f"\n{'═' * 60}")
        print(f"  Set [{set_num}/{num_sets}]  Theme: {theme}")
        print(f"{'═' * 60}")

        entry       = judgements.setdefault(theme, {"comparisons": []})
        comparisons = entry["comparisons"]
        done_so_far = len(comparisons)

        for idx, (first, second) in enumerate(ALL_COMPARISONS):
            if idx < done_so_far:
                print(f"  Comparison {idx+1}/20 ({first} vs {second}) — already done.")
                continue

            result = call_judge(
                client, theme, first, second,
                all_jokes[theme], idx + 1, 20,
            )
            if result is not None:
                winner, reasoning = result
                comparisons.append({"first": first, "second": second, "winner": winner, "reasoning": reasoning})

            entry["comparisons"] = comparisons
            entry["win_rates"]   = compute_win_rates(comparisons)
            if len(comparisons) >= 2:
                entry["bradley_terry"] = bradley_terry(comparisons)
            save_judgements(judgements)

        print_summary(theme, entry)

    print(f"\n{'═' * 60}")
    print(f"  Done. Judgements saved to {JUDGE_FILE}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pairwise LLM judge: ranks jokes across model sizes"
    )
    parser.add_argument(
        "-n", "--num-sets",
        type=int, default=1,
        help="Number of themes to judge (default: 1)",
    )
    args = parser.parse_args()
    run(args.num_sets)
