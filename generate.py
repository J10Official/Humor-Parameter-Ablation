import json
import os
import re
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ── Model registry ──────────────────────────────────────────────────────────
MODELS = {
    "8B":     "Qwen/Qwen3-8B",
    "14B":    "Qwen/Qwen3-14B",
    "32B":    "Qwen/Qwen3-32B",
    "30A3":   "Qwen/Qwen3-30B-A3B",
    "235A22": "Qwen/Qwen3-235B-A22B",
}

SYSTEM_PROMPT = (
    "You are a stand-up comedian known for sharp, original observational humor. "
    "When given a theme, write one short joke about it — either a classic setup/punchline "
    "or a punchy one-liner. The joke must be self-contained, land clearly, and end on the punchline. "
    "Output only the joke text, nothing else — no title, no explanation, no commentary."
)

MAX_TOKENS = 128
SEP = "─" * 60


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_prompts(path: str = "prompts.txt") -> list[str]:
    """Return list of prompt strings (strips numbering)."""
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # strip leading "1. " style numbering
            if ". " in line:
                line = line.split(". ", 1)[1]
            prompts.append(line)
    return prompts


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks (or unclosed <think>...) from output."""
    # closed blocks first, then unclosed trailing think block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def output_path_for(model_key: str) -> Path:
    """Return the JSON output path for a model."""
    return Path("results") / f"{model_key}.json"


def load_existing(model_key: str) -> dict:
    """Load existing results for resume support.

    Returns dict: {prompt_text: joke_string}
    """
    path = output_path_for(model_key)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(model_key: str, data: dict):
    path = output_path_for(model_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_joke(client: InferenceClient, model_id: str, theme: str) -> str:
    """Call HF Inference API and return the joke text."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"/no_think\n{theme}"},
    ]

    # ── Log request ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PROMPT  →  model: {model_id}:cheapest")
    print(f"{SEP}")
    for msg in messages:
        print(f"  [{msg['role'].upper()}]")
        print(f"  {msg['content']}")
    print(SEP)

    response = client.chat_completion(
        model=f"{model_id}:cheapest",
        messages=messages,
        max_tokens=MAX_TOKENS,
    )

    raw = response.choices[0].message.content
    joke = strip_thinking(raw)

    # ── Log response ────────────────────────────────────────────────────
    print(f"  RAW RESPONSE:")
    print(f"  {raw}")
    if raw != joke:
        print(f"\n  (thinking tokens stripped)")
        print(f"  CLEANED:")
        print(f"  {joke}")
    print(SEP)

    return joke


# ── Main ────────────────────────────────────────────────────────────────────
def run(model_key: str, num_jokes: int = 1):
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        sys.exit("ERROR: HF_TOKEN not found in .env")

    if model_key not in MODELS:
        sys.exit(f"ERROR: Unknown model key '{model_key}'. Choose from: {', '.join(MODELS)}")

    model_id = MODELS[model_key]
    client = InferenceClient(token=token)
    prompts = load_prompts()

    # resume support
    results = load_existing(model_key)

    # num_jokes = total jokes to generate (i.e. total prompts to use)
    already_done = len(results)
    remaining = max(0, num_jokes - already_done)

    print(f"{'═' * 60}")
    print(f"  Model  : {model_id}  ({model_key})")
    print(f"  Target : {num_jokes} joke(s) total")
    print(f"  Done   : {already_done}  |  Remaining: {remaining}")
    print(f"{'═' * 60}")

    if remaining == 0:
        print("\nAll jokes already generated. Nothing to do.")
        return

    # pick the next `remaining` prompts that haven't been done yet
    todo = [p for p in prompts if p not in results][:remaining]

    for i, prompt in enumerate(todo, start=already_done + 1):
        print(f"\n[{i}/{num_jokes}]  Theme: {prompt}")
        try:
            joke = generate_joke(client, model_id, prompt)
            results[prompt] = joke
            save_results(model_key, results)
            print(f"  ✓ saved")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            save_results(model_key, results)
            raise

    print(f"\n{'═' * 60}")
    print(f"  Done. {len(results)} jokes saved to {output_path_for(model_key)}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate jokes to study humor vs parameter count")
    parser.add_argument("model", choices=MODELS.keys(), help="Model size key")
    parser.add_argument("-n", "--num-jokes", type=int, default=1, help="Total number of jokes to generate (default: 1)")
    args = parser.parse_args()
    run(args.model, args.num_jokes)
