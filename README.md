# Personality or Instructions? Semantic Confounding in LLM Consumer Simulation

**Paper:** submitted to AIES 2026 (AAAI/ACM Conference on AI, Ethics, and Society)

## Abstract

Large language models are increasingly deployed as "synthetic consumers" to simulate marketing research at scale, yet no study has systematically tested whether these simulated agents reproduce demographic stereotypes or faithfully represent assigned personality traits. We conduct a 483,840-trial experiment across four open-weight LLMs (14--33B parameters), assigning 384 consumer profiles defined by socio-demographics and Big Five personality to binary product-choice tasks spanning five product categories. We find that the dominant personality effect -- Openness -- functions not as a personality trait but as a task instruction: its description contains 9 choice-relevant keywords (e.g., "innovative," "familiar"), while other traits contain none. Switching from consumer to general-psychology framing causes trait coefficients -- including Agreeableness -- to flip sign entirely, confirming that LLMs process prompt semantics rather than latent personality. Excluding the confounded Openness trait, LLMs over-stereotype income by approximately 2--4 times relative to published human benchmarks, with budget choice rates differing by 20--33 percentage points between low- and high-income profiles and near-deterministic behavior (Fleiss' kappa ~ 0.87) even at maximum temperature. Despite substantial income stereotyping, gender effects are absent in three of four models -- a positive finding suggesting dimension-specific rather than universal bias.

## Experimental Design

- **483,840 total API calls** across 4 open-weight LLMs (phi4, mistral-small3.2, gemma3-27b, qwen3-32b)
- **384 consumer profiles** = 24 demographic (Age x Gender x Income x Region) x 16 Big Five (2^(5-1) Resolution V fractional factorial)
- **5 product scenarios** spanning low-to-high involvement and price tiers
- **6 question forms** controlling for order and response format effects
- **3 temperatures** (0.0, 0.5, 1.0) for temperature sensitivity analysis
- **4 prompt conditions** (A: natural language consumer-framed, B: structured labels, C: effect codes, D: general psychology BFI-2-S)
- **All models run locally via Ollama** -- zero API cost, fully reproducible

Full experimental design: [`docs/design.md`](docs/design.md)

## Repository Structure

```
├── config/
│   ├── experiment.yaml            # Global settings (temperatures, replications, seed)
│   └── models/                    # Per-model configs (model name, API params)
├── data/
│   ├── design/                    # Experimental design matrices
│   │   ├── demographics_24.csv    # 24 demographic profiles (effect-coded)
│   │   ├── bigfive_16.csv         # 16 Big Five profiles (Resolution V)
│   │   └── personas_384.csv       # 384 = 24 x 16 full profiles
│   └── stimuli/                   # Scenarios, question forms, version assignment
├── prompts/
│   ├── system_header.txt          # Neutral system prompt
│   ├── traits/                    # Consumer-framed Big Five descriptions
│   └── ablation_d/               # General psychology (BFI-2-S) descriptions
├── scripts/
│   ├── 01_generate/               # Generate design matrices and stimuli
│   ├── 02_render/                 # Render and validate prompts
│   ├── 03_runners/                # Ollama API runners (resume, retry, JSONL)
│   ├── 04_parse/                  # Parse responses and quality control
│   └── 05_analysis/              # Statistical analysis and figure generation
├── tests/                         # Unit and integration tests
├── docs/
│   └── design.md                  # Full experimental design document (v6)
├── requirements.txt
└── .env.example                   # Template for Ollama server URLs
```

## Reproduction

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit with your Ollama server URLs
```

### 2. Generate Design

```bash
python scripts/01_generate/make_demographics.py
python scripts/01_generate/make_bigfive.py
python scripts/01_generate/make_personas.py
python scripts/01_generate/make_scenarios.py
python scripts/01_generate/make_assignment.py
```

### 3. Validate Prompts

```bash
python scripts/02_render/validate_prompts.py
```

### 4. Run Experiment

```bash
# Example: run phi4 at temperature 1.0
python scripts/03_runners/run_model.py --model phi4 --temp 1.0
```

### 5. Analysis

```bash
python scripts/05_analysis/preprocessing.py
python scripts/05_analysis/quality_control_all.py
python scripts/05_analysis/per_model_analysis.py
python scripts/05_analysis/mixed_effects.py
python scripts/05_analysis/temperature_analysis.py
python scripts/05_analysis/compute_s_index.py
python scripts/05_analysis/prompt_ablation_analysis.py
python scripts/05_analysis/generate_figures.py
python scripts/05_analysis/red_flags_analysis.py
python scripts/05_analysis/must_do_analysis.py
python scripts/05_analysis/must_do_4_7_analysis.py
python scripts/05_analysis/should_do_analysis.py
```

### 6. Tests

```bash
pytest tests/
```

## Models

| Model | Provider | Parameters | Architecture |
|-------|----------|------------|-------------|
| phi4 | Microsoft | 14.7B | Dense |
| mistral-small3.2 | Mistral AI | 24.0B | Dense |
| gemma3:27b | Google | 27.4B | Dense |
| qwen3:32b | Alibaba | 32.8B | Dense |

All models run via [Ollama](https://ollama.ai/) with Q4_K_M quantization.

## Key Findings

1. **Openness = semantic confound.** O descriptions contain 9 product-choice keywords; other Big Five traits contain 0. O accounts for 51--66% of explained variance.
2. **Prompt format changes everything.** Switching to general-psychology framing (Condition D) causes Agreeableness to flip sign. Effect codes (Condition C) collapse all personality effects.
3. **Income over-stereotyping.** Budget choice rates: $25K profiles 39--68% vs. $120K profiles 18--41% (20--33 pp gap). Approximately 2--4x published human benchmarks.
4. **Near-deterministic behavior.** Fleiss' kappa ~ 0.87 at temp=1.0. Temperature has negligible effect.
5. **No gender stereotyping.** 3/4 models show no significant gender effect on price sensitivity.

## License

[TBD]
