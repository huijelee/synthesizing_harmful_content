# Beyond Static Benchmarks: Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation

This repository contains the code and data for **Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation**.

## Overview

Static benchmarks for harmful content detection are limited in scalability and diversity, and may also be affected by contamination from web-scale pre-training corpora. This project introduces a persona-guided simulation framework that synthesizes harmful content within real-world discussion threads for **robust stress-testing** of harmful content detection systems.

The framework constructs **two-dimensional personas**:

- **Intrinsic aspects**: user identity and topical interests
- **Extrinsic aspects**: situational harmful or trolling strategies

Persona-guided LLM agents are then instantiated inside discussion threads to generate harmful responses that are both **contextually grounded** and **behaviorally varied**.

We evaluate the framework along three dimensions:

- **Harmfulness**: whether generated comments are perceived as harmful by human and LLM evaluators
- **Challenge level**: whether the generated scenarios are harder for detection models than existing benchmarks
- **Diversity**: whether persona conditioning improves linguistic and categorical diversity

## Repository Contents

```text
synthesizing_harmful_content/
+-- data/
|   +-- reddit_random_threads/      # seed Reddit discussion threads
|   +-- reddit_policies/            # subreddit rules and policy text
|   +-- synthetic_userprofiles/     # synthesized intrinsic persona profiles
|   +-- simulation_outputs/         # generated harmful scenarios and outputs
|   +-- cadd/                       # CADD benchmark
|   +-- conan/                      # CONAN benchmark
|   +-- mtconan/                    # MT-CONAN benchmark
|   +-- qian_gab/                   # Qian-Gab benchmark
|   +-- qian_reddit/                # Qian-Reddit benchmark
|   +-- covid-hate/                 # COVID-HATE benchmark
|   +-- counter_trollingy/          # ELF/Counter-trolling style benchmark data
+-- experiments/
|   +-- simulation_template.yaml    # example experiment configuration
+-- src/
    +-- main_simulation.py          # main generation / evaluation pipeline
    +-- generate_userprofiles.py    # persona synthesis
    +-- evaluate_harmful_detection.py
    +-- evaluation.py
    +-- evaluation_prompts.py
    +-- dataset_classes.py
    +-- utils.py
    +-- private_keys.py             # API key placeholders
```

## Data Release

This repository includes:

- seed discussion threads sampled from Reddit
- a list of subreddit names used to synthesize intrinsic aspects
- synthesized user personas
- generated harmful scenarios
- static harmful content benchmarks used for comparison

### Data Notice

- Parts of this repository contain **Reddit-derived content** and **derivative simulation outputs** constructed from public discussion threads.
- If an original content owner requests removal of their content from this release, we will make reasonable efforts to remove or update the corresponding data.

### Ethical Note

- This repository contains **harmful, offensive, and abusive language** because the goal of the project is to study robust evaluation of harmful content detection systems.
- The framework is intended for **measurement, auditing, and stress-testing** of safety systems, not for deploying or amplifying harmful behavior.
- When using this repository, please handle the released data and generated outputs with appropriate care, access control, and institutional review procedures where applicable.

## Setup

The codebase is a lightweight research prototype rather than a packaged library. A typical environment will need:

- Python 3.10+
- `openai`
- `anthropic`
- `transformers`
- `datasets`
- `torch`
- `vllm`
- `pandas`
- `numpy`
- `scikit-learn`
- `omegaconf`
- `tqdm`
- `nltk`
- `sacrebleu`
- `tiktoken`
- `lingua-language-detector`
- `google-api-python-client`
- `filelock`
- `peft`
- `scipy`

You will also need to populate API keys in `src/private_keys.py`.

## Quickstart

### 1. Generate synthetic user profiles

```bash
python src/generate_userprofiles.py
```

This script synthesizes intrinsic persona profiles and writes them into `data/synthetic_userprofiles/`.

### 2. Run simulation

```bash
python src/main_simulation.py experiments/simulation_template.yaml
```

This runs the configured persona-guided harmful content generation pipeline and writes outputs to `data/simulation_outputs/`.

### 3. Evaluate harmful content detection

```bash
python src/evaluate_harmful_detection.py
```

This evaluates generated scenarios against moderation and harmful content detection models and saves detailed JSON/TSV outputs.

## Main Configuration

The example configuration is provided in:

```text
experiments/simulation_template.yaml
```

It defines:

- generator model backends
- sampling hyperparameters
- task types
- prompt variants (`vanilla`, `ours`, `ex_only`, `in_only`)
- evaluation metrics
- output directories

## Notes

- The repository contains **harmful and offensive text** for research purposes.
- The code is written as a research prototype and may require minor path or environment adjustments depending on your local setup.
- `main_simulation.py` can be run with an explicit YAML config; otherwise it falls back to a default development config path.

## Citation

If you use this repository, please cite:

```bibtex
@misc{lee2026synthesizingharmful,
  title        = {Synthesizing Harmful Content via Persona-based Simulation for Robust Evaluation},
  author       = {Lee, Huije and Shin, Jisu and Song, Hoyun and Ko, Changgeon and Park, Jong C.},
  year         = {2026},
  note         = {TBA}
}
```
