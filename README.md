# Does Bigger Mean Funnier? Evaluating Humor Generation Across the Qwen3 Model Family

**Published at CHum 2026 — Proceedings of the 2nd Workshop on Computational Humor.**

Jatin Agrawal and Radhika Mamidi, pages 81–94.

[ACL Anthology](https://aclanthology.org/2026.chum-1.7/) · [Published PDF](https://aclanthology.org/2026.chum-1.7.pdf) · [DOI](https://doi.org/10.18653/v1/2026.chum-1.7) · [Citation metadata](CITATION.cff) · [Google Scholar](https://scholar.google.com/citations?user=Pvycwx0AAAAJ&hl=en)  

This repository contains the official codebase, dataset, and analysis pipeline for the paper *"Does Bigger Mean Funnier? Evaluating Humor Generation Across the Qwen3 Model Family"*. This project conducts a controlled, within-family ablation study of parameter count on humor generation quality, alongside a comparative analysis of human annotators versus an automated LLM-as-a-judge metric.

## Main finding

The published study finds that the automated judge's ranking increases monotonically with model size, while human annotations show **no significant aggregate difference** in humor quality. A preference for the largest model appears within the subset of themes on which annotators agree. Positional and length biases make the distinction between these evaluation methods important.

## Overview

The study evaluates five instruction-tuned models from the Qwen3 family—three dense (8B, 14B, 32B) and two Mixture-of-Experts (30A3, 235A22)—generating short jokes across 50 distinct everyday themes. We present findings from:
1. **Automated LLM Evaluation**: Analyzing a perfect monotonic rank correlation with scaling alongside severe procedural artifacts (positional and length biases).
2. **Blinded Human Evaluation**: Highlighting a disconnected "quality floor" where scaling impacts are significant only in subset themes where annotators strongly agree.

## Repository Structure

The complete lifecycle—from generation to statistical analysis—is compartmentalized into the following scripts:

### 1. Generation & API Evaluation
- `generate.py`: Prompts the 5 Qwen3 models via the HuggingFace Inference API using the themes listed in `prompts.txt` (with thinking disabled, `max_tokens=128`).
- `judge.py`: Pairwise LLM-as-a-judge script deploying DeepSeek-V3.2 via OpenRouter to evaluate every generated joke pair symmetrically (A-first and B-first) to account for positional bias.

### 2. Human Annotation Utilities
- `build_viewer.py` & `viewer.html`: Generates an anonymized, split-screen viewing interface used by the human annotator groups.

### 3. Statistical Analysis Pipelines
- `analyse.py`: Computes aggregate automated metrics including Bradley-Terry (BT) scores, win rates, and cluster-level bootstrap standard errors for the LLM judge.
- `human_analysis.py`: Ingests `results/annotations.csv` to calculate Krippendorff's Alpha, Friedman test significance, subset analyses, and head-to-head Wilcoxon tests.
- `length_analysis.py`: Specific module measuring character lengths to calculate LLM length biases.

### 4. Data & Resources
- `/results/`: Contains the raw generation outputs (`8B.json`, `32B.json`, etc.), automated judgements (`judgements.json`), cleaned human rankings (`annotations.csv`), and generated plots.
- `prompts.txt`: The dataset containing 50 diverse everyday humor themes.

## Usage / Reproducibility

### Setup
Install the necessary dependencies using pip:
```bash
python -m pip install numpy scipy matplotlib
# Optional: only needed for new generation or automated judging
python -m pip install openai huggingface_hub python-dotenv
```
You will need a `.env` file containing your API keys if you wish to re-run generation or automated judging:
```env
HF_TOKEN=your_huggingface_token
OPENROUTER_API_KEY=your_openrouter_token
```

### Reproducing the Analysis
We provide the full logs of generated jokes and judge outputs inside the `/results/` directory, meaning you can directly run the analysis scripts without re-incurring API costs.

1. **Calculate Automated Judge Statistics & Plots:**
   ```bash
   python3 analyse.py
   ```
2. **Calculate Human Annotation Statistics:**
   ```bash
   python3 human_analysis.py
   ```
3. **Verify Length Biases:**
   ```bash
   python3 length_analysis.py
   ```

## Citation
Please cite the published paper using [CITATION.cff](CITATION.cff) or the [ACL Anthology citation](https://aclanthology.org/2026.chum-1.7/).
