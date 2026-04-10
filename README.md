# Does Bigger Mean Funnier? Evaluating Humor Generation Across the Qwen3 Model Family

[**📄 Read the Paper Here**](paper.pdf)  

This repository contains the official codebase, dataset, and analysis pipeline for the paper *"Does Bigger Mean Funnier? Evaluating Humor Generation Across the Qwen3 Model Family"*. This project conducts a controlled, within-family ablation study of parameter count on humor generation quality, alongside a comparative analysis of human annotators versus an automated LLM-as-a-judge metric.

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
- `clean_csv.py`: Formats and cleans the raw output data from human evaluators.

### 3. Statistical Analysis Pipelines
- `analyse.py`: Computes aggregate automated metrics including Bradley-Terry (BT) scores, win rates, and cluster-level bootstrap standard errors for the LLM judge.
- `human_analysis.py`: Ingests `results/annotations.csv` to calculate Krippendorff's Alpha, Friedman test significance, subset analyses, and head-to-head Wilcoxon tests.
- `length_analysis.py`: Specific module measuring character lengths to calculate LLM length biases.
- `fix_bootstrap.py`: Applies cluster-level bootstrapping (resampling at the theme level) to correct for intra-theme performance correlations.

### 4. Data & Resources
- `/results/`: Contains the raw generation outputs (`8B.json`, `32B.json`, etc.), automated judgements (`judgements.json`), cleaned human rankings (`annotations.csv`), and generated plots.
- `prompts.txt`: The dataset containing 50 diverse everyday humor themes.

## Usage / Reproducibility

### Setup
Install the necessary dependencies using pip:
```bash
pip install -r requirements.txt # (numpy, scipy, matplotlib, openai, huggingface_hub, python-dotenv)
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

## License and Citation
This code and dataset are made available for research and reproducibility purposes. *(Camera-ready citation block goes here).*
