# Experimental Design & Implementation Plan v6.0

**Paper:** "Do LLMs Differentiate Between Consumer Segments? Socio-Demographics vs. Personality in Simulated Price Sensitivity Across Product Categories"
**Venue:** AIES 2026 (AAAI/ACM Conference on AI, Ethics, and Society)
**Deadlines:** Abstract — May 14, 2026; Paper — May 21, 2026
**Format:** AAAI 2026 Author Kit (aaai2026.sty), 10 pages + unlimited references, double-blind review
**Date:** 2026-02-24
**Status:** Final. Ready for implementation.

---

## PART I: EXPERIMENTAL DESIGN

---

## 1. Research Question and Motivation

### 1.1 Context

Large language models are increasingly used as "synthetic consumers" to simulate marketing research. A researcher assigns the model a demographic and personality profile (age, gender, income, region, Big Five) and asks it to make consumer choices on behalf of that profile. The implicit assumption is that LLMs differentiate between consumer segments in ways that mirror real human behavior.

### 1.2 Central Question

**What better predicts LLM price sensitivity: socio-demographic attributes or personality traits (Big Five)?** And does either work at all?

### 1.3 Dependent Variable

**Price sensitivity** — the tendency to choose the budget option (Option A) over the premium option (Option B). Measured as **Choice Rate for A** — the proportion of budget choices, marginalized across question forms and replications. Range: 0 (always premium) to 1 (always budget).

---

## 2. Experimental Design Overview

Full factorial design: **Profiles × Scenarios × Question Forms × Replications × Models**


```
384 profiles × 5 scenarios × 6 forms × 3 replications = 34,560 calls per model
34,560 × 4 models = 138,240 total responses per temperature
138,240 × 3 temperatures (1.0, 0.5, 0.0) = 414,720 main + ablation responses
```

---

## 3. Agent Profiles

384 profiles = 24 demographic × 16 personality configurations.

### 3.1 Demographic Profiles (24)

Full factorial of 4 factors:

| Factor | Levels | Count | Effect Coding | Prompt Description |
|---|---|---|---|---|
| **Age** | 25, 55 | 2 | Age_L: 25=−1, 55=+1 | "You are a 25-year-old..." / "You are a 55-year-old..." |
| **Gender** | Male, Female | 2 | G: Male=−1, Female=+1 | "...man..." / "...woman..." |
| **Income** | $25K, $65K, $120K | 3 | Inc_L (linear): −1, 0, +1; Inc_Q (quadratic): +1, −2, +1 | "...$25,000" / "...$65,000" / "...$120,000" |
| **Region** | USA, Japan | 2 | R: USA=−1, Japan=+1 | "...in the United States" / "...in Japan" |

**Total:** 2 × 2 × 3 × 2 = **24 demographic profiles** (D01–D24).

Income uses orthogonal polynomial contrasts:
- Inc_L (linear): Low=−1, Middle=0, High=+1
- Inc_Q (quadratic): Low=+1, Middle=−2, High=+1

**Demographic description principle:** Facts only, no behavioral cues. Income is stated as a number ("Your annual household income is approximately $25,000") without phrases like "you are budget-conscious" or "price is not a concern." This allows testing LLM stereotypical associations — the model must decide on its own that low income → budget preference.

### 3.2 Country Selection: Why USA and Japan?

The experiment requires exactly two countries (binary factor in factorial design). The pair must satisfy five constraints simultaneously:

| Constraint | Rationale |
|---|---|
| **C1: High cultural contrast** on consumer-relevant dimensions | Maximizes power to detect Region effects and Region × Personality interactions |
| **C2: Comparable income levels** ($25K/$65K/$120K meaningful in both) | Prevents income × region confound: $25K must mean "low" in *both* countries |
| **C3: Rich LLM training data** for both countries | Ensures LLM has sufficient knowledge to simulate consumers; avoids testing model *ignorance* |
| **C4: Published consumer behavior literature** | Both countries well-studied in marketing research, enabling literature-based benchmark |
| **C5: English proficiency** (practical constraint) | All prompts in English; human participants must be fluent |

#### Systematic Comparison of Candidate Pairs

| Pair | C1: Cultural contrast | C2: Income comparability | C3: Training data | C4: Consumer lit. | C5: English | Verdict |
|---|---|---|---|---|---|---|
| **USA + Japan** | **HIGH** (5/6 Hofstede axes gap >25) | **GOOD** (both high-income OECD; $25K=low in both) | Excellent (both) | Extensive (both) | USA native; Japan self-selected fluent | **SELECTED** |
| USA + UK | Minimal (same culture) | Excellent | Excellent | Extensive | Both native | No contrast |
| USA + Germany | Low (both Western, individualist) | Good | Excellent | Extensive | Germany good | Insufficient contrast |
| USA + India | High | **POOR**: $25K ≈ poverty in USA but top-15% in India | India weaker (Hindi/regional data) | Moderate | India good | **Income confound fatal** |
| USA + Brazil | Medium | **POOR**: $25K ≈ upper-middle in Brazil | Brazil weaker (Portuguese) | Limited (Portuguese) | Brazil limited | Income confound + language |
| USA + China | High | Medium ($25K ≈ emerging middle class) | China weaker (censored content, Mandarin-dominant) | Moderate (Mandarin) | China limited | Language barrier |
| USA + Nigeria | Very high | **POOR**: $25K ≈ wealthy in Nigeria | Nigeria weak | Sparse | Nigeria good | Income confound + data scarcity |
| USA + S. Korea | High (similar to Japan) | Good | Good but smaller than Japan | Moderate | S. Korea moderate | Smaller literature than Japan |

**The income comparability constraint (C2) is the most restrictive.** Developing countries are systematically excluded because the same dollar amounts carry radically different purchasing power and social status. Among high-income countries with sufficient cultural contrast, **Japan is the strongest candidate** opposite the USA.

#### Hofstede Cultural Dimensions: USA vs Japan

| Dimension | USA | Japan | Gap | Consumer Relevance |
|---|---|---|---|---|
| **Individualism (IDV)** | 91 | 46 | **45** | Social influence on purchases: conformity vs independent choice. Directly related to **Extraversion** trait. Prediction: Japan profiles → stronger social conformity; E effect should interact with Region. |
| **Uncertainty Avoidance (UAI)** | 46 | 92 | **46** | Risk aversion, brand loyalty, preference for proven products. Directly related to **Neuroticism** and **Openness**. Prediction: Japan profiles → more brand loyalty, less novelty-seeking. |
| **Long-Term Orientation (LTO)** | 26 | 88 | **62** | Planning purchases, investing in quality, delayed gratification. Directly related to **Conscientiousness**. Prediction: Japan profiles → more deliberate, quality-over-price. |
| **Masculinity (MAS)** | 62 | 95 | **33** | Achievement orientation, status consumption. Related to **Extraversion**. |
| **Power Distance (PDI)** | 40 | 54 | **14** | Respect for premium/luxury brands, hierarchical consumption. |
| **Indulgence (IVR)** | 68 | 42 | **26** | Impulse purchasing vs restrained consumption. Related to **Conscientiousness**. |

5 of 6 dimensions show gaps > 25 points.

#### Testable Predictions from Cultural Theory

| Prediction | Mechanism | Test in M4 |
|---|---|---|
| E effect stronger for Japan profiles | High IDV gap → social influence more salient in collectivist culture | E × Region interaction |
| N effect stronger for Japan profiles | High UAI gap → uncertainty avoidance amplifies purchase worry | N × Region interaction |
| O effect stronger for USA profiles | High UAI gap (inverse) → novelty-seeking more normative in low-UAI culture | O × Region interaction |
| C effect stronger for Japan profiles | High LTO gap → planning and deliberation more culturally valued | C × Region interaction |
| A+1 "trust brands" aligns with Japan norms | shinrai culture; high UAI → trust in established institutions | A × Region interaction (see Limitation #8) |

#### Acknowledged Limitations of Country Selection

1. **Japan ≠ "Asia."** Results do not generalize to India, China, Indonesia, or other Asian markets.
2. **Both are high-income OECD countries.** The pair cannot test how LLMs handle developing-market consumers.
3. **English prompts for Japan profiles.** Japanese consumers do not typically make purchasing decisions in English.
4. **Japan-specific consumer behavior data is sparse in English-language literature.** Literature-based benchmark effect sizes are predominantly from Western (USA/European) samples.

### 3.3 Big Five Personality Profiles (16)

16-run Resolution V fractional factorial: 2^(5−1), defining relation **I = OCEAN**. Fifth factor N generated as N = O × C × E × A.

| B5ID | O | C | E | A | N (=O×C×E×A) |
|------|---|---|---|---|---------------|
| B01 | −1 | −1 | −1 | −1 | +1 |
| B02 | +1 | −1 | −1 | −1 | −1 |
| B03 | −1 | +1 | −1 | −1 | −1 |
| B04 | +1 | +1 | −1 | −1 | +1 |
| B05 | −1 | −1 | +1 | −1 | −1 |
| B06 | +1 | −1 | +1 | −1 | +1 |
| B07 | −1 | +1 | +1 | −1 | +1 |
| B08 | +1 | +1 | +1 | −1 | −1 |
| B09 | −1 | −1 | −1 | +1 | −1 |
| B10 | +1 | −1 | −1 | +1 | +1 |
| B11 | −1 | +1 | −1 | +1 | +1 |
| B12 | +1 | +1 | −1 | +1 | −1 |
| B13 | −1 | −1 | +1 | +1 | +1 |
| B14 | +1 | −1 | +1 | +1 | −1 |
| B15 | −1 | +1 | +1 | +1 | −1 |
| B16 | +1 | +1 | +1 | +1 | +1 |

**Resolution V properties:**
- All 5 main effects estimated cleanly (aliased only with 4FI)
- All 10 two-factor interactions estimated cleanly (aliased only with 3FI)
- Balance: all 5 columns 8/8; all 10 pairwise combinations 4-4-4-4 (verified)

**Saturated design and error degrees of freedom:** 16 parameters (1 intercept + 5 main effects + 10 two-factor interactions) = 16 rows, leaving 0 df for error within the B5 sub-design. However, each B5 configuration is crossed with 24 demographic profiles × 5 scenarios × 6 forms × 3 replications = 2,160 observations per configuration per model. Total error df ≈ 172,800 − p_fixed − q_random.

### 3.4 Profile Combination

24 demographic × 16 personality = **384 profiles** (P001–P384).

---

## 4. Scenario Design

### 4.1 Selection Framework

Scenarios are selected along two axes covering the space of consumer decisions:

| | **Low Price Tier** (<$100 budget) | **Medium Price Tier** ($100–$300) | **High Price Tier** (>$300) |
|---|---|---|---|
| **Low Involvement** | **S5: Electric Toothbrush** | — | — |
| **Medium Involvement** | — | **S4: Winter Jacket** | — |
| **Medium–High Involvement** | — | — | **S3: Wireless Headphones** |
| **High Involvement** | — | — | **S1: Smartphone**, **S2: Laptop** |

**Axis justification:**

- **Price tier** is directly linked to income sensitivity: at $25K income, an $800 smartphone represents a significant budget share; a $35 toothbrush does not. If the LLM demonstrates an income effect only for expensive products, this is calibrated sensitivity. If the effect is uniform across price tiers, it indicates a heuristic (stereotype).
- **Involvement level** (Zaichkowsky, 1985) determines the depth of the decision-making process. High-involvement purchases involve research and deliberation (related to C and N). Low-involvement purchases are more impulsive (related to E and O).

### 4.2 Version Design Logic

For each scenario, 4 versions create systematic variation in tradeoff intensity:

| Version | Price Gap (A→B) | Attribute Gap (A→B) | Interpretation |
|---|---|---|---|
| **V1** | Small (~30–70% premium) | B slightly better | Weak tradeoff |
| **V2** | Medium (~70–150% premium) | B moderately better | Balanced tradeoff |
| **V3** | Large (~100–260% premium) | B substantially better | Strong tradeoff |
| **V4** | Negative (A cheaper) | A equal or better | **Dominance check: rational agent always picks A.** |

### 4.3 Full Scenario Specifications

#### S1: Smartphone

**Context:** "You are choosing a new smartphone for everyday use. Compare the options and pick the one you would buy."

**Fixed attribute:** Warranty: 2 years (both options).
**Varied attributes:** Price, Camera, Battery life.

| Version | Option A (Budget) | Option B (Premium) | Price Gap |
|---|---|---|---|
| V1 | $449 / Camera: 48 MP / Battery: 5,000 mAh | $599 / Camera: 64 MP / Battery: 5,500 mAh | +33% |
| V2 | $399 / Camera: 48 MP / Battery: 4,500 mAh | $749 / Camera: 108 MP / Battery: 5,500 mAh | +88% |
| V3 | $399 / Camera: 48 MP / Battery: 4,500 mAh | $899 / Camera: 200 MP / Battery: 6,000 mAh | +125% |
| V4 | $499 / Camera: 108 MP / Battery: 5,500 mAh | $699 / Camera: 64 MP / Battery: 5,000 mAh | **A dominates** |

#### S2: Laptop

**Context:** "You are choosing a new laptop for work and personal use. Compare the options and pick the one you would buy."

**Fixed attribute:** Screen: 15.6 inches, IPS display (both options).
**Varied attributes:** Price, Processor, RAM.

| Version | Option A (Budget) | Option B (Premium) | Price Gap |
|---|---|---|---|
| V1 | $699 / Processor: Intel i5 / RAM: 8 GB | $999 / Processor: Intel i7 / RAM: 16 GB | +43% |
| V2 | $699 / Processor: Intel i5 / RAM: 8 GB | $1,199 / Processor: Intel i7 / RAM: 32 GB | +72% |
| V3 | $649 / Processor: Intel i5 / RAM: 8 GB | $1,449 / Processor: Intel i9 / RAM: 32 GB | +123% |
| V4 | $799 / Processor: Intel i7 / RAM: 16 GB | $1,099 / Processor: Intel i5 / RAM: 8 GB | **A dominates** |

#### S3: Wireless Headphones

**Context:** "You are choosing new wireless headphones for daily use. Compare the options and pick the one you would buy."

**Fixed attribute:** Battery life: 30 hours (both options).
**Varied attributes:** Price, Sound driver size, Noise cancellation.

| Version | Option A (Budget) | Option B (Premium) | Price Gap |
|---|---|---|---|
| V1 | $69 / Driver: 30 mm / ANC: basic | $99 / Driver: 40 mm / ANC: adaptive | +43% |
| V2 | $59 / Driver: 30 mm / ANC: basic | $149 / Driver: 40 mm / ANC: adaptive, multipoint | +153% |
| V3 | $59 / Driver: 30 mm / ANC: none | $199 / Driver: 50 mm / ANC: adaptive, multipoint | +237% |
| V4 | $79 / Driver: 40 mm / ANC: adaptive | $129 / Driver: 30 mm / ANC: basic | **A dominates** |

#### S4: Winter Jacket

**Context:** "You are choosing a new winter jacket. Compare the options and pick the one you would buy."

**Fixed attribute:** Waterproof: yes, sealed seams (both options).
**Varied attributes:** Price, Insulation type, Weight.

| Version | Option A (Budget) | Option B (Premium) | Price Gap |
|---|---|---|---|
| V1 | $89 / Insulation: synthetic 150g / Weight: 850 g | $139 / Insulation: synthetic 200g / Weight: 650 g | +56% |
| V2 | $79 / Insulation: synthetic 150g / Weight: 900 g | $189 / Insulation: down 600-fill / Weight: 550 g | +139% |
| V3 | $79 / Insulation: synthetic 100g / Weight: 950 g | $269 / Insulation: down 800-fill / Weight: 450 g | +241% |
| V4 | $109 / Insulation: down 600-fill / Weight: 550 g | $179 / Insulation: synthetic 150g / Weight: 800 g | **A dominates** |

#### S5: Electric Toothbrush

**Context:** "You are choosing a new electric toothbrush. Compare the options and pick the one you would buy."

**Fixed attribute:** Includes 1 replacement brush head (both options).
**Varied attributes:** Price, Battery life, Cleaning modes.

| Version | Option A (Budget) | Option B (Premium) | Price Gap |
|---|---|---|---|
| V1 | $29 / Battery: 2 weeks / Modes: 2 (clean, sensitive) | $49 / Battery: 3 weeks / Modes: 3 (clean, sensitive, whitening) | +69% |
| V2 | $25 / Battery: 2 weeks / Modes: 1 (clean) | $59 / Battery: 4 weeks / Modes: 4 (clean, sensitive, whitening, gum care) | +136% |
| V3 | $25 / Battery: 2 weeks / Modes: 1 (clean) | $89 / Battery: 6 weeks / Modes: 5 + pressure sensor + travel case | +256% |
| V4 | $35 / Battery: 4 weeks / Modes: 3 (clean, sensitive, whitening) | $55 / Battery: 2 weeks / Modes: 2 (clean, sensitive) | **A dominates** |

### 4.4 Cross-Scenario Summary

| Scenario | A Price Range | B Price Range | Mean Premium V1–V3 | Domain | Involvement |
|---|---|---|---|---|---|
| S1: Smartphone | $399–$449 | $599–$899 | 82% | Tech | High |
| S2: Laptop | $649–$699 | $999–$1,449 | 79% | Tech | High |
| S3: Headphones | $59–$69 | $99–$199 | 144% | Tech/Lifestyle | Medium–High |
| S4: Winter Jacket | $79–$89 | $139–$269 | 145% | Apparel | Medium |
| S5: Toothbrush | $25–$29 | $49–$89 | 154% | Health/Hygiene | Low |

### 4.5 Version Assignment

Each ProfileID (P001–P384) is assigned one version (V1–V4) per scenario, balanced:
- 96 profiles per version per scenario (384 / 4 = 96)
- Assignment stratified by DemoID (each demographic group evenly split across versions)
- Deterministic: `version(ProfileID, ScenarioID) = hash(ProfileID, ScenarioID) mod 4 + 1`
- V4 used **only as quality filter**, excluded from primary analysis

### 4.6 Currency

All prices in USD for both USA and Japan profiles.

**Rationale:** Standardization — identical stimuli across regions ensures a clean Region effect estimate.

**Limitation:** Reduced ecological validity for Japan profiles. Addressed in Limitations (section 16, point 9).

### 4.7 Scenario Factor Coding for Analysis

| Coding | Definition | What It Captures |
|---|---|---|
| **Involvement** | S5=−2, S4=−1, S3=0, S1=+1, S2=+2 | Linear involvement effect |

Primary analysis uses Scenario as a 4-df factor. Exploratory analysis replaces Scenario with Involvement to test income × involvement interaction.

---

## 5. Question Forms

6 forms controlling for order effects and response format:

| Form ID | Type | Order | Question Text |
|---|---|---|---|
| AB_Afirst | AB | A first | "Which would you choose? Reply 'A' or 'B'." |
| AB_Bfirst | AB | B first | "Which would you choose? Reply 'A' or 'B'." |
| REP_Afirst | Repeat | A first | "Repeat exactly one option text (A or B)." |
| REP_Bfirst | Repeat | B first | "Repeat exactly one option text (A or B)." |
| CMP_AoverB | Compare | A over B | "Would you prefer option A over option B? Reply 'yes' or 'no'." |
| CMP_BoverA | Compare | B over A | "Would you prefer option B over option A? Reply 'yes' or 'no'." |

**Form decomposition for analysis:**

| Factor | Levels | What It Captures |
|---|---|---|
| **Order** | A-first = −1, B-first = +1 | Positional / primacy bias |
| **Format** | AB = [−1, −1], REP = [+1, −1], CMP = [0, +2] (Helmert) | Response format bias |

### 5.1 Response Parsing Rules

| Form | Expected Response | Parsing Rule | Mapped Choice |
|---|---|---|---|
| AB_Afirst | "A" or "B" | First character matching `[AaBb]` | A→A, B→B |
| AB_Bfirst | "A" or "B" | First character matching `[AaBb]` | A→A, B→B |
| REP_Afirst | Full option text | Levenshtein ratio to each option; closest match if max ratio ≥ 0.5; else `invalid` | Match→A or B |
| REP_Bfirst | Full option text | Levenshtein ratio to each option; closest match if max ratio ≥ 0.5; else `invalid` | Match→A or B |
| CMP_AoverB | "yes" or "no" | First token matching `[yY]es|[nN]o` | yes→A, no→B |
| CMP_BoverA | "yes" or "no" | First token matching `[yY]es|[nN]o` | yes→B, no→A |

Fallback: if response matches none of the expected patterns after lowercasing and stripping whitespace → status = `invalid`.

**REP-form specific QC:** Invalid rate is reported separately for REP-forms (where Levenshtein parsing is noisier) vs AB/CMP-forms. If REP invalid rate > 15% for a model, REP-forms are excluded from that model's analysis and this is documented.

---

## 6. Replications

**M = 3** repetitions of each combination (profile × scenario × form). Per model: 384 × 5 × 6 × 3 = **34,560 API calls**.

---

## 7. Prompt Design

### 7.1 Design Principles

1. No numerical codes — natural language only
2. Demographics = facts, no behavioral cues
3. Personality = purchasing behavior, equal consumer-relevance across traits
4. No price sensitivity cueing — no trait mentions price, budget, or premium
5. Discriminant validity — each trait describes a unique decision-making aspect
6. Trait order randomized per call
7. Standardized API placement (system vs user role)
8. System header does not mention any specific profile aspect

### 7.2 System Header

```
Imagine you are the person described below. Read the profile carefully
and fully adopt this identity.

You will see a product-choice scenario with two options. As this person,
choose the option you would buy. Your choice should reflect the priorities
and tendencies of the person described above.

Response rules:
1. Follow the exact reply format specified in the question.
2. No explanations, justifications, or commentary.
3. Treat each scenario independently.
4. Stay fully in character throughout.
```

**Critical design choice (v2 → v5 → v6):**
- v2: "reflect this person's financial situation, personality, and preferences" — **rejected**: explicitly cued income, creating a confound.
- v5: "consistent with the profile" — **revised**: too vague, risk that model relies on its defaults instead of attending to the profile.
- v6: "reflect the priorities and tendencies of the person described" — **adopted**: directs attention to the profile's content (priorities = what matters to them; tendencies = behavioral patterns), without naming any specific dimension (income, personality, age, etc.).

#### API Placement (Ollama)

All models use Ollama API:

```json
{
  "model": "<model_name>",
  "system": "<system_header>",
  "prompt": "<profile + scenario + question>",
  "stream": false,
  "options": {
    "temperature": 1.0,
    "top_p": 1.0,
    "num_predict": 16,
    "seed": 20250101
  }
}
```

If a model does not support system role in Ollama, the system header is concatenated with the user prompt via `\n\n`, and this is documented in the response metadata.

### 7.3 Profile Template

```
=== YOUR PROFILE ===

You are a {age}-year-old {gender} living in {region}.
Your annual household income is approximately ${income}.

Your personality:
- {trait_descriptions[shuffled_order[0]]}
- {trait_descriptions[shuffled_order[1]]}
- {trait_descriptions[shuffled_order[2]]}
- {trait_descriptions[shuffled_order[3]]}
- {trait_descriptions[shuffled_order[4]]}

=== END PROFILE ===
```

Trait order randomized per call: `shuffled_order` is a random permutation of [O, C, E, A, N], seeded by `hash(task_id)` for reproducibility. Because `task_id` depends on profile × scenario × form × replication, the same profile receives different trait orders across conditions. This ensures that trait position is uncorrelated with profile content across the full dataset, making positional effects noise rather than a confound. Positional bias is tested in a separate analysis (section 14.6); if significant, a sensitivity check adds positional covariates to the main model (section 14.2).

### 7.4 Big Five Trait Descriptions

| Trait | Decision Aspect | Uniqueness |
|---|---|---|
| **O** (Openness) | Novelty vs familiarity in product choice | What to choose (new vs known) |
| **C** (Conscientiousness) | Planning vs spontaneity in the buying process | How the decision process is organized |
| **E** (Extraversion) | Social vs private sources of influence | Where information comes from |
| **A** (Agreeableness) | Trust vs skepticism toward marketing claims | How to evaluate seller assertions |
| **N** (Neuroticism) | Confidence vs worry after purchase | Emotional relationship to the decision |

#### Openness (O)

**O = +1 (High):**
"You enjoy exploring new and unfamiliar products. You are drawn to innovative features and unconventional designs, and you like trying brands you haven't used before."

**O = −1 (Low):**
"You prefer products you are already familiar with. You tend to repurchase from brands you have used before and favor traditional, well-established designs over novel ones."

#### Conscientiousness (C)

**C = +1 (High):**
"You approach purchases methodically. You plan ahead, set a clear idea of what you need before shopping, and carefully evaluate whether each option meets your predetermined criteria."

**C = −1 (Low):**
"You make purchasing decisions spontaneously and in the moment. You rarely plan what to buy in advance and tend to decide based on what appeals to you."

#### Extraversion (E)

**E = +1 (High):**
"You often discuss potential purchases with friends and family. You pay attention to what is popular among people you know and enjoy products that enhance your social experiences."

**E = −1 (Low):**
"You make purchasing decisions privately and on your own. You are not particularly influenced by what others are buying and focus primarily on your own personal needs."

#### Agreeableness (A)

**A = +1 (High):**
"You tend to take product descriptions and brand promises at face value. You give companies the benefit of the doubt and are forgiving of minor product shortcomings."

**A = −1 (Low):**
"You approach product claims and marketing with skepticism. You look for independent verification of advertised features and hold products to strict standards before committing to a purchase."

#### Neuroticism (N)

**N = +1 (High):**
"You often worry about whether you are making the right purchasing decision. Even after buying something, you may wonder whether you should have chosen a different option."

**N = −1 (Low):**
"You rarely worry about your purchasing decisions. Once you have made a choice, you do not second-guess yourself and feel confident it was the right one."

### 7.5 Trait Description Verification

**Word count balance:** Mean 26.7 ± 0.8 (range 25–28). All descriptions are exactly 2 sentences.

**Consumer-relevance:** All 10 descriptions are strictly about purchasing behavior.

**No price sensitivity cueing:** No description mentions price, budget, premium, cheap, or expensive.

**Discriminant validity — critical pairs:**

| Pair | Trait 1 Keywords | Trait 2 Keywords | Overlap? |
|---|---|---|---|
| E−1 vs C+1 | privately, on your own, not influenced | methodically, plan ahead, criteria, evaluate | **NO** (E=social source, C=process rigor) |
| C−1 vs N−1 | spontaneously, rarely plan, in the moment | rarely worry, don't second-guess, confident | **NO** (C=planning process, N=emotional reaction) |
| A−1 vs C+1 | skepticism, marketing claims, independent verification | methodically, plan ahead, predetermined criteria | **NO** (A=trust of claims, C=systematic process) |

### 7.6 Scenario Block Format

```
=== SCENARIO S1: Smartphone ===
Context: You are choosing a new smartphone for everyday use.
Compare the options and pick the one you would buy.
Option A: Price: $399; Camera: 48 MP; Battery: 4,500 mAh; Warranty: 2 years.
Option B: Price: $749; Camera: 108 MP; Battery: 5,500 mAh; Warranty: 2 years.

=== QUESTION ===
Which would you choose? Reply 'A' or 'B'.
```

### 7.7 API Parameters

| Parameter | Value | Rationale |
|---|---|---|
| temperature | 1.0 (main) / 0.5 (mid) / 0.0 (greedy) | 3-level factorial |
| top_p | 1.0 | Default, no nucleus truncation |
| num_predict (max_tokens) | 16 | Safe for REP-forms (full option text ≈ 15 tokens) |
| seed | 20250101 | Reproducibility |
| stream | false | Batch processing |

**Logprobs:** Ollama support varies by model and version. Availability checked during smoke test. If available, collected for AB-forms only.

---

## 8. Complete Rendered Example

**Profile:** D01 (25, Male, $25K, USA) × B08 (O=+1, C=+1, E=+1, A=−1, N=−1)
**Trait order:** [E, N, A, O, C] (randomized)
**Scenario:** S1 Smartphone, V2, Form AB_Afirst

```
[SYSTEM]
Imagine you are the person described below. Read the profile carefully
and fully adopt this identity.

You will see a product-choice scenario with two options. As this person,
choose the option you would buy. Your choice should reflect the priorities
and tendencies of the person described above.

Response rules:
1. Follow the exact reply format specified in the question.
2. No explanations, justifications, or commentary.
3. Treat each scenario independently.
4. Stay fully in character throughout.

[USER]
=== YOUR PROFILE ===

You are a 25-year-old man living in the United States.
Your annual household income is approximately $25,000.

Your personality:
- You often discuss potential purchases with friends and family. You pay
  attention to what is popular among people you know and enjoy products
  that enhance your social experiences.
- You rarely worry about your purchasing decisions. Once you have made a
  choice, you do not second-guess yourself and feel confident it was the
  right one.
- You approach product claims and marketing with skepticism. You look for
  independent verification of advertised features and hold products to
  strict standards before committing to a purchase.
- You enjoy exploring new and unfamiliar products. You are drawn to
  innovative features and unconventional designs, and you like trying
  brands you haven't used before.
- You approach purchases methodically. You plan ahead, set a clear idea
  of what you need before shopping, and carefully evaluate whether each
  option meets your predetermined criteria.

=== END PROFILE ===

=== SCENARIO S1: Smartphone ===
Context: You are choosing a new smartphone for everyday use.
Compare the options and pick the one you would buy.
Option A: Price: $399; Camera: 48 MP; Battery: 4,500 mAh; Warranty: 2 years.
Option B: Price: $749; Camera: 108 MP; Battery: 5,500 mAh; Warranty: 2 years.

=== QUESTION ===
Which would you choose? Reply 'A' or 'B'.
```

---

## 9. Models

### 9.1 Model Selection

All models run locally via Ollama on dedicated GPU servers. No paid API services.

| # | Model | Provider | Parameters | Architecture | Quantization | Server | Est. Latency |
|---|---|---|---|---|---|---|---|
| 1 | phi4 | Microsoft | 14.7B | Dense | Q4_K_M | GPU2 | ~6.7 s/call |
| 2 | mistral-small3.2 | Mistral | 24.0B | Dense | Q4_K_M | GPU2 | TBD |
| 3 | gemma3:27b | Google (Gemma) | 27.4B | Dense | Q4_K_M | GPU2 | TBD |
| 4 | qwen3:32b | Alibaba | 32.8B | Dense | Q4_K_M | GPU3 | ~1.3 s/call |

**Dropped model:** gpt-oss:20b (OpenAI OSS, 20.9B MoE) was removed because it ignores `think:false` and generates ~181 internal reasoning tokens per call, making each call ~188s instead of ~2s. At 34,560 calls this would require ~75 days — infeasible.

**Model selection criteria:**
- 4 different providers (Microsoft, Mistral, Google, Alibaba)
- Parameter range: 14.7B to 32.8B (all Dense)
- All instruction-tuned and capable of following simple A/B choice format
- All available on local GPU servers (zero API cost)

### 9.2 GPU Server Allocation

| Server | URL | Models | Total Sequential Time |
|---|---|---|---|
| GPU2 (localhost:11434) | 3 models | phi4 → mistral-small3.2 → gemma3:27b | TBD |
| GPU3 (localhost:11434) | 1 model | qwen3:32b | ~12.8 hours |

Models on the same GPU run sequentially (VRAM limitation). Order: fastest first to get early results for pipeline validation.

### 9.3 Temperature Ablation (temperature = 0.0 and 0.5)

Three temperature levels: **1.0** (main, maximum sampling diversity), **0.5** (intermediate), **0.0** (greedy / near-deterministic). Same full design (34,560 calls per model) at each temperature.

**Rationale for three levels:** Two levels (1.0 vs 0.0) can only detect the presence of a temperature effect. Three levels enable testing whether the effect is **monotonic** (linear in temperature) or **nonlinear** (e.g., a threshold effect where sensitivity collapses only at temp=0.0). Temperature is coded as an ordered factor with orthogonal polynomial contrasts:
- Temp_L (linear): 0.0 = −1, 0.5 = 0, 1.0 = +1
- Temp_Q (quadratic): 0.0 = +1, 0.5 = −2, 1.0 = +1

A significant Temp_Q term would indicate that the midpoint (0.5) deviates from a linear interpolation between the extremes — evidence of a nonlinear temperature–sensitivity relationship.

**Determinism at temp=0.0:** M=3 replications retained because temperature=0 does not guarantee determinism due to GPU floating-point non-determinism and quantization effects. Intra-replicate agreement assessed via Fleiss' κ per model per temperature.

### 9.4 Changes from v5

| v5 (planned) | v6 (actual) | Reason |
|---|---|---|
| gpt-3.5-turbo (OpenAI API) | **Removed** | API cost |
| gpt-4o-mini (OpenAI API) | **Removed** | API cost |
| gemini-2.0-flash-lite (Gemini API) | **Removed** | API cost |
| qwen2.5:32b (Ollama) | **→ qwen3:32b** | qwen2.5:32b not installed; qwen3:32b available on GPU3 |
| gpt-oss:120b (Ollama) | **→ gemma3:27b + phi4** | gpt-oss:120b not installed; two smaller models add provider diversity |

**Impact on analysis:** K=4 instead of K=7. Random slope `(1 + Inc_L | Model)` has only 2 df for variance estimation — will likely be singular. Fall back to `(1 | Model)` for comparison models. Per-model analysis (primary) is unaffected.

---

## 10. Prompt Ablation Study

Four conditions testing prompt format effects:

| Condition | Demographics | Personality | What It Tests |
|---|---|---|---|
| **A (primary)** | NL facts | NL consumer-framed | Primary design |
| **B (structured)** | Labels | Labels (Age: 25, O: High...) | Structured vs NL |
| **C (coded)** | Effect codes | Effect codes (O=+1...) | Numeric codes |
| **D (general psych)** | NL facts | **NL general psychology (no consumer framing)** | **Instruction-following vs personality transfer** |

**Condition D descriptions** based on BFI-2-S item wording (Soto & John, 2017), with no consumer framing:

| Trait | +1 (High) | −1 (Low) |
|---|---|---|
| **O** | "You are original and come up with new ideas. You are curious about many different things and have an active imagination. You value artistic and aesthetic experiences." | "You prefer routine and familiar situations. You have few artistic interests and tend to think in conventional ways. You value practical outcomes over abstract ideas." |
| **C** | "You are dependable and self-disciplined. You tend to be organized, keep things in order, and persist with tasks until they are finished. You make plans and follow through on them." | "You can be somewhat careless and disorganized. You tend to put off tasks and have difficulty sticking to a plan. You prefer flexibility over structure in your daily life." |
| **E** | "You are talkative, outgoing, and sociable. You generate a lot of enthusiasm in social settings. You are assertive and energetic when interacting with others." | "You tend to be quiet and reserved. You prefer solitary activities over social gatherings and keep your thoughts and feelings mostly to yourself." |
| **A** | "You are helpful and unselfish with others. You have a forgiving nature and are generally trusting of people. You are considerate and kind to almost everyone you meet." | "You tend to find fault with others and can be cold and distant. You are sometimes blunt or dismissive and are skeptical of other people's intentions." |
| **N** | "You worry a lot and can be tense. You sometimes feel sad or blue and are easily upset by stressful situations. Your moods tend to change frequently." | "You are emotionally stable and rarely get upset. You handle stress well and remain calm in most situations. You seldom feel anxious or moody." |

**Design:** 24 demo × 4 B5 × 5 scenarios × 2 forms × 3 reps = 2,880 calls per condition per model.
4 conditions × 2 models (mistral-small3.2 + qwen3:32b) = **23,040 total**.

**Model choice rationale:** mistral-small3.2 (Mistral, 24B) and qwen3:32b (Alibaba, 32.8B) are chosen because (a) they come from different providers, (b) qwen3:32b runs on GPU3 (parallel with GPU2), and (c) qwen3 is a larger model more likely to show interesting sensitivity to prompt format variation. gpt-oss:20b was excluded from the study due to infeasible inference time (see Section 9.4).

---

## 11. Quality Control

| Status | Definition | Action |
|---|---|---|
| **valid** | Response parsed as A or B (with form mapping) | Include in analysis |
| **invalid** | Response not recognized (explanation, refusal, gibberish) | Exclude |
| **error** | API error (timeout, rate limit) | Retry up to 3 times, then exclude |

**Dominance check (V4):** V4 is used only as a quality filter. V4 pass rate is reported per model per scenario as a rationality metric. V4 is not included in the primary analysis.

---

## 12. Power Analysis

### 12.1 Approach

Simulation-based via `simr` package (Green & MacLeod, 2016).

### 12.2 Data-Generating Process

```
logit(P(Choice_A_ijk)) = β₀ + β_Inc_L × Inc_L + β_Inc_Q × Inc_Q +
                          β_Age × Age + β_Gender × Gender + β_Region × Region +
                          β_O × O + β_C × C + β_E × E + β_A × A + β_N × N +
                          β_Scenario × Scenario +
                          u_j + v_j × Inc_L + w_k

u_j ~ N(0, σ²_u)    # random intercept for Model j
v_j ~ N(0, σ²_v)    # random slope for Income by Model j
w_k ~ N(0, σ²_w)    # random intercept for Profile k
```

### 12.3 MDES (Minimum Detectable Effect Size)

| Analysis | MDES (log-odds) | MDES (OR) | MDES (Cohen's d) |
|---|---|---|---|
| Per-model (N=34,560) | 0.022 | 1.022 | 0.012 |
| Mixed-model, σ²_v=0 (K=4) | 0.028 | 1.028 | 0.016 |
| Mixed-model, σ²_v=0.30 (K=4) | 0.58 | 1.79 | 0.33 |

At homogeneous effects, the design has >99% power for even minimal B5 effects. At heterogeneous effects, per-model analysis detects effects within each model; mixed-model requires OR ≥ 1.68 for the average effect (slightly less powerful than K=7 design).

---

## 13. Literature-Based Benchmark

### 13.1 Rationale

Instead of a matched human sample (which would require IRB approval, Prolific recruitment, and ~$640–960 budget), we compare LLM behavior against **published effect sizes** from the consumer behavior and marketing literature. This makes the project fully self-contained (only GPU servers + researcher time) while still providing an empirical anchor for the S-index.

### 13.2 Reference Effect Sizes

Summary of published income and personality effects on price sensitivity / willingness-to-pay from meta-analyses and large-sample studies:

| Source | N | DV | Income Effect (β or d) | Big Five Effect (β or d) | S_literature |
|---|---|---|---|---|---|
| Lichtenstein et al. (1993) | 350 | Price consciousness | d = 0.38 (income) | C: d = 0.21 | 1.81 |
| Ailawadi et al. (2001) | 548 | Deal proneness | β = −0.19 (income) | — | — |
| Matz et al. (2016) | 625 | Spending–personality match | — | O: d = 0.18, C: d = 0.14 | — |
| Borghans et al. (2008) meta | >10K | Economic behavior | d = 0.25–0.40 | C: d = 0.10–0.20 | 1.5–2.5 |
| Donnelly et al. (2012) | 718 | Spending patterns | β = 0.31 (income) | A: β = 0.09, N: β = 0.12 | 2.6 |

**Note:** Effect sizes are approximate, extracted from different paradigms (survey scales, real spending, conjoint). They provide order-of-magnitude anchors, not exact benchmarks. **TODO (Phase 7):** Re-read each source paper and verify that d/β values are correctly extracted, converted, and attributed. This does not block implementation.

### 13.3 S_literature Computation

From the literature summary, a plausible range for the human income-to-personality ratio:

```
S_literature ≈ median(1.81, 1.5–2.5, 2.6) ≈ 2.0 (range: 1.5–2.5)
```

This means that in published studies, income effects on price-related behaviors are typically **1.5–2.5× larger** than the largest Big Five effect. An LLM with S(m) = 2.0 would be "calibrated to the literature." An LLM with S(m) >> 2.5 would show disproportionate income reliance even by human standards.

### 13.4 Comparability Caveats

- Published studies use different paradigms (survey scales, real spending, conjoint choice) — not binary product choice as in our experiment.
- Income is operationalized differently (continuous income vs. 3-level factorial).
- Big Five is measured (correlational) in human studies vs. assigned (experimental) in LLM study.
- S_literature is an order-of-magnitude anchor, not a precise benchmark. Confidence interval is wide.
- A matched human sample remains the gold standard; literature-based comparison is a pragmatic substitute.

---

## 14. Statistical Analysis Plan

### 14.1 Inferential Strategy (Pre-Registered)

1. **Primary: per-model logistic regression** — "Does model X respond to trait Y?"
   - 5 separate models, full fixed effects, FDR correction within each (10 tests)
   - Primary because: (a) high per-model power, (b) no random slope assumptions, (c) directly answers the applied question

2. **Secondary: mixed-effects logistic regression** — "Do LLMs in general respond to trait Y?"
   - All models pooled, Model as random effect
   - Generalization to LLM population; random slope variance informs heterogeneity

3. **Reconciliation protocol:**
   - Mixed n.s. but per-model significant in 3+ models → "heterogeneous effect"
   - Mixed significant but per-model significant in only 1 model → "driven by single model"
   - Both reported transparently

### 14.2 Primary Analysis: Per-Model

```r
glmer(Choice_A ~ Inc_L + Inc_Q + Age_L + Gender + Region +
                 O + C + E + A + N +
                 Scenario + Order + Format +
                 (1 | ProfileID),
      family = binomial, data = df_model_i)
```

If singular fit → fall back to `glm` with clustered SEs (sandwich estimator).

**Sensitivity check (positional bias):** If section 14.6 reveals significant trait-position effects, re-run the per-model model with `trait_position_O + ... + trait_position_N` as additional covariates. Report both results; if conclusions are unchanged, positional bias is negligible despite significance.

Results presented as a descriptive summary table: for each model × factor, report β̂, SE, 95% CI, p_raw, p_FDR, significance flag.

### 14.3 Secondary Analysis: Mixed-Effects

**Inference model (maximal random effects):**
```r
glmer(Choice_A ~ Inc_L + Inc_Q + Age_L + Gender + Region +
                 O + C + E + A + N +
                 Scenario + Order + Format +
                 (1 + Inc_L | Model) + (1 | ProfileID),
      family = binomial, data = df)
```

**Note:** With K=4 models, `(1 + Inc_L | Model)` has only 2 df for slope variance — likely singular. Default to `(1 | Model)` and report random slopes as infeasible.

**Comparison models (consistent random effects for nested LRTs):**

```r
M0: Choice_A ~ 1 + Scenario + Order + Format +
    (1 | Model) + (1 | ProfileID)                          # nuisance-adjusted baseline
M1: M0 + Inc_L + Inc_Q + Age_L + Gender + Region           # + demographics
M2: M0 + O + C + E + A + N                                 # + personality
M3: M0 + demographics + personality                         # full additive
M4: M3 + O:C + O:E + O:A + O:N + C:E + C:A + C:N +
    E:A + E:N + A:N                                        # + B5 interactions
```

LRTs: M1 vs M0 (demographics), M2 vs M0 (personality), M3 vs M1 (personality | demographics), M4 vs M3 (B5 interactions).

### 14.4 Variance Decomposition

R²_GLMM (Nakagawa & Schielzeth, 2013):
- R²_marginal: variance explained by fixed effects
- R²_conditional: variance explained by fixed + random effects
- Distribution-level variance: π²/3 ≈ 3.29 (logistic)

### 14.5 Multiple Comparisons

| Family | Tests | Scope | Correction |
|---|---|---|---|
| **F1: Main effects** | 10 (Inc_L, Inc_Q, Age, Gender, Region, O, C, E, A, N) | Primary hypotheses | BH-FDR, q < 0.05 |
| **F2: B5 interactions** | 10 (O:C, O:E, ..., A:N) in M4 | Exploratory | BH-FDR, q < 0.05 |
| **F3: Per-model** | 10 × 5 = 50 | Per-model inference | BH-FDR within each model (10 tests) |

### 14.6 Positional Bias Analysis

```r
glmer(Choice_A ~ trait_position_O + trait_position_C +
                 trait_position_E + trait_position_A + trait_position_N +
                 Scenario + Order + Format +
                 (1 | Model) + (1 | ProfileID),
      family = binomial, data = df)
```

### 14.7 Temperature Comparison

Three temperature levels (0.0, 0.5, 1.0) coded with orthogonal polynomial contrasts:
- Temp_L (linear): 0.0 = −1, 0.5 = 0, 1.0 = +1
- Temp_Q (quadratic): 0.0 = +1, 0.5 = −2, 1.0 = +1

**Primary temperature model:**
```r
glmer(Choice_A ~ (Temp_L + Temp_Q) * (Inc_L + Inc_Q + Age_L + Gender + Region +
                                       O + C + E + A + N) +
                 Scenario + Order + Format +
                 (1 | Model) + (1 | ProfileID),
      family = binomial, data = df_all_temps)
```

**Key tests:**
- Temp_L × Inc_L: Does the income effect change linearly with temperature?
- Temp_Q × Inc_L: Does the income effect show a nonlinear (threshold) response to temperature?
- Temp_L × B5 (O, C, E, A, N): Does personality sensitivity change linearly with temperature?
- Temp_Q × B5: Nonlinear temperature response for personality effects?

**S-index across temperatures:** S(m, t) computed per model per temperature. If S(m) decreases monotonically from temp=0.0 to temp=1.0, higher temperature diversifies responses and reduces stereotypical reliance on income. If S(m) is U-shaped or flat, temperature does not modulate stereotype strength.

**Determinism assessment:** Fleiss' κ per model per temperature. Expected: κ ≈ 1.0 at temp=0.0 (near-deterministic), decreasing at 0.5 and 1.0.

### 14.8 Logprobs Analysis

For models where logprobs are available (AB-forms only):

```r
# Primary: Beta mixed regression (bounded DV on (0,1))
glmmTMB(P_A ~ Inc_L + Inc_Q + Age_L + Gender + Region +
              O + C + E + A + N +
              Scenario + Order + Format +
              (1 + Inc_L | Model) + (1 | ProfileID),
        family = beta_family(), data = df_logprobs)

# Sensitivity: logit-transform + lmer, with residual diagnostics
```

### 14.9 Scenario-Level Analysis

**Income × Involvement interaction:**
```r
glmer(Choice_A ~ Inc_L * Involvement_linear +
                 Inc_Q + Age_L + Gender + Region +
                 O + C + E + A + N +
                 Order + Format +
                 (1 + Inc_L | Model) + (1 | ProfileID),
      family = binomial, data = df)
```

Tests whether income effect scales with product price tier.

### 14.10 Random Slopes Sensitivity

Primary inference model uses `(1 | Model)` (random intercept only). With K=4, random slopes `(1 + Inc_L | Model)` have only 2 df — expected to be singular. Report as limitation.

---

## 15. Operationalization of "Stereotype"

### 15.1 Definition

> Stereotypical reasoning: (a) a model's sensitivity to a demographic attribute (stated as a bare fact) is disproportionately large relative to its sensitivity to personality traits (described through explicit behavioral tendencies), and (b) the direction of the demographic effect follows a culturally conventional association.

### 15.2 Continuous S-Index

```
S(m) = |β̂_Inc_L(m)| / max(|β̂_O(m)|, |β̂_C(m)|, |β̂_E(m)|, |β̂_A(m)|, |β̂_N(m)|)
```

- S = 1: income and personality effects are equal
- S > 1: income dominates
- S = ∞: personality effects are zero

### 15.3 Literature-Anchored S-Index

```
S_anchored(m) = S(m) / S_literature
```

where S_literature ≈ 2.0 (range 1.5–2.5) from published effect sizes (section 13.3).

- S_anchored ≈ 1: income-to-personality ratio comparable to published human data
- S_anchored > 1: income overweighted relative to human literature (overstereotyping)
- S_anchored < 1: income underweighted relative to human literature

**Sensitivity:** Report S_anchored at S_literature = 1.5, 2.0, and 2.5 to reflect uncertainty in the benchmark.

### 15.4 Supplementary Classification

Models classified into archetypes based on natural breaks in S(m) distribution (k-means, k=3). Exploratory.

---

## 16. Known Limitations

1. **Binary gender only.** Does not cover non-binary identities.
2. **Two countries only.** USA and Japan: both high-income OECD. See section 3.2 for justification.
3. **Binary personality levels.** No middle level; nonlinear effects not detectable.
4. **English-only prompts.** Japan profiles prompted in English.
5. **K=4 models.** Random slopes infeasible (2 df). Random intercept `(1 | Model)` only. K ≥ 20 ideal for robust random slopes. Per-model analysis is primary to compensate.
6. **Simulated choice ≠ real purchase.** No real stakes, no WTP, no post-purchase behavior.
7. **Consumer-framed traits.** Ablation condition D tests generalization to non-consumer descriptions.
8. **A+1 × Japan confound.** Testable via A × Region interaction in M4.
9. **Currency: USD for all profiles.** Reduces ecological validity for Japan profiles.
10. **Product categories: 3 of 5 are tech.** Tech overrepresented.
11. **All models open-weight / quantized.** No proprietary frontier models (GPT-4o, Claude, Gemini Pro). Results may not generalize to larger proprietary models.
12. **No matched human sample.** Comparison relies on published effect sizes from different task paradigms (price consciousness scales, real spending, conjoint). S_literature is an order-of-magnitude anchor with wide confidence interval (~1.5–2.5). A matched human study using the same scenarios and forced-choice format remains the gold standard for future work.

---

## 17. Discussion Points

### 17.1 Downstream Harms

If LLMs exhibit stereotypical reasoning (S-index >> 1):

- **Marketing research bias:** Companies using LLMs as synthetic consumers may get recommendations that reinforce income stereotypes while ignoring personality-driven preferences.
- **Personalization harm:** LLM-powered recommendation systems may restrict product variety shown to users inferred as low-income, reducing consumer autonomy.
- **Feedback loops:** If LLM-generated synthetic survey data trains marketing models, stereotype amplification across iterations.
- **Representation harm:** Flattening consumer identity to income level erases personality, values, and individual preferences.

### 17.2 Mitigation Recommendations

**For practitioners:**
- Do not treat LLM-generated synthetic consumer data as a drop-in replacement for human surveys, especially when personality-driven segments are of interest.
- Validate against real respondent data on the specific demographic × product category of interest.
- Report which consumer dimensions the LLM is sensitive to (and which it ignores) as standard disclosure.

**For LLM developers:**
- Audit training data for overrepresentation of income → preference heuristics relative to personality → preference pathways.
- Include personality-diverse evaluation benchmarks for persona consistency.
- Provide documentation of known dimension-sensitivity biases (analogous to model cards for fairness).

**For regulators:**
- If synthetic consumer data from LLMs is used in consequential decisions, require disclosure and validation.
- Encourage audit standards for LLM-based consumer simulations, analogous to algorithmic impact assessments.
- Consider whether systematic income-stereotyping constitutes digital redlining when used to determine product access or pricing.

### 17.3 Scenario-Level Findings Framing

If Income × Involvement is significant → *partial calibration*. If not → *blanket heuristic*.

### 17.4 Comparison with Published Human Data

The S_literature benchmark (section 13) relies on published effect sizes from studies with different paradigms (price consciousness scales, real spending data, deal proneness surveys) rather than a matched human sample performing the same binary choice task. Key caveats:

- **Paradigm mismatch:** Published studies measure price sensitivity via Likert scales, real transaction data, or conjoint analysis — not forced binary product choice. Effect sizes may not transfer directly.
- **Operationalization differences:** Human studies use continuous income; our design uses 3-level factorial. Human Big Five is measured (correlational, attenuated by reliability); LLM Big Five is assigned (experimental, no attenuation).
- **S_literature is an interval, not a point:** We report S_anchored at three benchmark values (1.5, 2.0, 2.5) to reflect this uncertainty.
- **Direction, not magnitude:** The primary comparison is whether LLMs show income-to-personality *ratios* in the same range as published human data, not whether absolute effect sizes match.
- **Future work:** A matched human sample (same scenarios, same forced-choice format, BFI-2-S personality measurement) would provide a precise benchmark. The literature-based approach is a pragmatic substitute that avoids IRB/Prolific dependencies while still grounding the stereotype interpretation.

### 17.5 Region Effect Interpretation (USD Asymmetry)

The experiment uses USD prices for both USA and Japan profiles. This creates an asymmetry: for USA profiles, both income ($25,000) and prices ($399) are "native" numbers; for Japan profiles, both are foreign units. Any Region effect (or its absence) must therefore be interpreted as "LLM's cultural stereotype about Japanese consumers when prompted in English with USD prices," not as a reflection of real cross-cultural consumer differences. If Region is non-significant, this could mean either (a) the LLM has no Japan-specific consumer model, or (b) the USD framing suppresses cultural differentiation. If Region is significant, the LLM is applying a Japan stereotype *despite* the foreign currency context — which is arguably stronger evidence of stereotyping. This asymmetry is a fundamental limitation of the single-currency design and should be addressed in future work with PPP-adjusted local-currency conditions.

### 17.6 A+1 × Japan Cultural Note

Japanese consumer culture emphasizes brand trust and loyalty (shinrai). If A × Region is significant, this is a genuine cultural phenomenon, not a methodological artifact.

### 17.7 Open-Weight Models Framing

All models in this study are open-weight (14.7B–32.8B parameters, quantized). This is both a limitation (no proprietary frontier models) and a strength: (a) full reproducibility — any researcher can replicate results, (b) the models tested are those most likely to be deployed locally in cost-sensitive marketing research applications, (c) quantized open models represent the practical frontier of "LLM as synthetic consumer" use cases where API cost is a concern.

### 17.8 Tech-Dominated Scenario Set and Training Data Bias

Three of five product scenarios are technology products (wireless earbuds, smartwatch, laptop). This raises an important interpretive concern: the internet training data for LLMs is disproportionately rich in "best budget [tech product]" articles, price-comparison reviews, and deal aggregators that explicitly frame tech purchasing decisions through an income/budget lens. If LLMs show stronger income effects for tech scenarios than for apparel or health products, this could reflect either (a) genuine stereotypical reasoning (income → tech preference heuristic), or (b) faithful reproduction of the income-centric discourse that dominates tech product discussions in training corpora.

To partially disentangle these explanations, we examine the **Income × Involvement** interaction (section 14.9). If income effects scale monotonically with price tier (higher price → stronger income effect), this is consistent with rational calibration rather than stereotyping. If income effects are uniformly strong across all scenarios regardless of price tier, this supports the blanket heuristic interpretation. Additionally, per-scenario S-index values are reported (supplementary), allowing readers to assess whether the tech-heavy scenario mix inflates aggregate stereotyping metrics. Future work should include a broader category set (food, services, experiences, financial products) to improve ecological validity.

---

## PART II: IMPLEMENTATION PLAN

---

## 18. Budget

| Component | Calls | Cost |
|---|---|---|
| Main experiment (temp=1.0, 4 models) | 138,240 | $0 (Ollama) |
| Temperature ablation (temp=0.0, 4 models) | 138,240 | $0 (Ollama) |
| Temperature ablation (temp=0.5, 4 models) | 138,240 | $0 (Ollama) |
| Prompt ablation (4 conditions × 2 models) | 23,040 | $0 (Ollama) |
| **Grand total** | **437,760** | **$0** |

---

## 19. Contingency Plan

| Priority | Models | Rationale | Calls (temp=1.0) |
|---|---|---|---|
| **Tier 1 (must-have)** | mistral-small3.2, qwen3:32b | 2 providers, 2 GPUs | 69,120 |
| **Tier 2 (full)** | + phi4, gemma3:27b | 4 providers, wider size range | 138,240 |

Minimum K for mixed model with `(1|Model)`: K ≥ 3. With K=2 (Tier 1), no random effects — per-model only. With K=4 (Tier 2), random intercept feasible, random slopes infeasible.

### 19.1 Timeline Contingency: Compression Plan

If GPU delays push Phase 3 (main experiment) beyond Apr 3, the following phases compress in priority order:

| Trigger | Delay | Action | Savings |
|---|---|---|---|
| **Phase 3 finishes Apr 10** | +1 week | Phase 5 ablation: reduce to 2 conditions (A + C only) instead of 4. Drop B (structured) and D (general psych). | ~7 days |
| **Phase 3 finishes Apr 17** | +2 weeks | Above + Phase 4 temp=0.0/0.5: run only 2 models (Tier 1: mistral, qwen3) instead of 4. | ~10 days |
| **Phase 3 finishes Apr 24** | +3 weeks | Above + drop temp=0.5 entirely. Run temp=0.0 for Tier 1 only. | ~20 days |
| **Critical: Phase 3 not done by May 1** | +4 weeks | Drop gemma3:27b (lowest priority). Analysis with K=3. No temp ablation, no prompt ablation. Minimal viable paper. |

**Non-compressible phases:**
- Phase 7 (paper writing): fixed deadline May 21, cannot compress below 14 days
- Phase 3 Tier 1 models (mistral, qwen3): minimum for publishable results

**Early warning indicators:**
- Monitor GPU2 actual vs estimated time daily starting Phase 3
- If phi4 (first model, Day 1) takes >2× estimated → re-estimate all subsequent models and activate contingency

---

## 20. Timeline

### Phase 0: Infrastructure (Feb 24 – Feb 28)

- [ ] Create project directory structure
- [ ] Write `requirements.txt` and install dependencies
- [ ] Write `.env` with GPU server URLs
- [ ] Write `config/experiment.yaml` (global settings)
- [ ] Write `config/models/*.yaml` (5 model configs)

### Phase 1: Data Generation (Mar 1 – Mar 7)

- [ ] `scripts/01_generate/make_demographics.py` → `data/design/demographics_24.csv`
- [ ] `scripts/01_generate/make_bigfive.py` → `data/design/bigfive_16.csv`
- [ ] `tests/test_bigfive_design.py` — verify balance (all columns 8/8, all pairs 4-4-4-4)
- [ ] `scripts/01_generate/make_personas.py` → `data/design/personas_384.csv`
- [ ] `scripts/01_generate/make_scenarios.py` → `data/stimuli/scenarios.yaml`
- [ ] `data/stimuli/forms.yaml` — 6 question forms
- [ ] `scripts/01_generate/make_assignment.py` → `data/stimuli/assignment.csv`
- [ ] `tests/test_assignment.py` — verify 96 per version per scenario
- [ ] `prompts/system_header.txt`
- [ ] `prompts/traits/*.yaml` (consumer-framed) + `prompts/ablation_d/*.yaml` (BFI-2-S)

### Phase 2: Rendering + Runners (Mar 8 – Mar 14)

- [ ] `scripts/02_render/render_prompts.py` — trait order randomized per hash(task_id)
- [ ] `scripts/02_render/validate_prompts.py` — spot-check 100 prompts
- [ ] `scripts/04_parse/parse_responses.py` — AB/REP/CMP parsing
- [ ] `tests/test_response_parser.py` + `tests/test_prompt_renderer.py`
- [ ] `scripts/03_runners/base_runner.py` — resume logic, retry, JSONL output
- [ ] `scripts/03_runners/ollama_runner.py` — POST to Ollama API
- [ ] `scripts/03_runners/run_model.py` — CLI entry point
- [ ] `tests/smoke_test.py` — **2 calls × 4 models, verify parsing**

### Phase 3: Main Experiment, temp=1.0 (Mar 15 – Apr 3, 20 days)

**GPU2 (sequential, fastest first for validation):**

| Order | Model | Est. Duration | Cumulative | Priority Rationale |
|---|---|---|---|---|
| 1 | phi4 (~6.7s/call) | ~2.7 days | Feb 28 | Fastest; pipeline validation |
| 2 | mistral-small3.2 (TBD) | TBD | TBD | Narrative-critical |
| 3 | gemma3:27b (TBD) | TBD | TBD | Completes the set |

**GPU3 (parallel with GPU2):**

| Order | Model | Est. Duration | Finish |
|---|---|---|---|
| 1 | qwen3:32b (~1.3s/call) | ~12.8 hours | Feb 26 |

**Real-time monitoring:**
- `scripts/04_parse/quality_control.py`: invalid rate, error rate, V4 pass rate per model
- Alert if invalid > 5% or V4 pass < 80%

### Phase 4: Temperature Ablation, temp=0.0 and temp=0.5 (Feb 26 – Mar 3)

Same design, same GPU allocation, at temp=0.0 and temp=0.5.
GPU2: phi4 t=0.0 → mistral t=0.0 → phi4 t=0.5 → mistral t=0.5 (sequential).
GPU3: qwen3 t=0.0 → gemma3 t=0.0 → qwen3 t=0.5 → gemma3 t=0.5 (sequential).
Fleiss' κ per model per temperature after completion.

### Phase 5: Prompt Ablation (Apr 24 – Apr 30, 7 days)

4 conditions × 2 models (mistral-small3.2 on GPU2 + qwen3:32b on GPU3) = 23,040 calls.
Parallel execution on 2 GPUs → ~4-5 days.

### Phase 6: Data Analysis (Apr 4 – Apr 30, overlaps with Phases 4-5)

- [ ] `scripts/05_analysis/preprocessing.py` — merge JSONL, filter, code variables
- [ ] `scripts/05_analysis/per_model_analysis.R` — 5 × glmer + FDR
- [ ] `scripts/05_analysis/mixed_effects.R` — M0–M4, LRTs, R²_GLMM
- [ ] `scripts/05_analysis/logprobs_analysis.R` — beta regression if logprobs available
- [ ] `scripts/05_analysis/compute_s_index.py` — S-index, S_anchored (vs S_literature), archetypes
- [ ] `scripts/05_analysis/generate_figures.py` — forest plot, S-index bar, literature comparison

### Phase 7: Paper Writing (May 1 – May 21)

| Deadline | Section |
|---|---|
| May 1–3 | **Verify literature benchmark** (section 13.2): re-read each source, confirm d/β extraction, update table |
| May 7 | Abstract + Intro + Related Work + Methods |
| May 10 | Results (main + ablations) + Discussion |
| **May 14** | **ABSTRACT SUBMISSION** |
| May 17 | Full draft (10 pages AAAI format) |
| May 19 | Revision + figures |
| **May 21** | **PAPER SUBMISSION** |

### Timeline Gantt

```
              FEB      MARCH          APRIL          MAY
              24  1  8  15  22 29  5  12  19  26  3  10  17  21
Phase 0       ████
Phase 1            ██████
Phase 2                 ██████
Phase 3 t=1.0      ████████
  GPU2             ████████
  GPU3             ████
Phase 4 t=0.0  ████████
Phase 4b t=0.5     ████████
Phase 5 Ablat                                              ███████
Phase 6 Analysis        ███████████████████████████████████████████
Phase 7 Paper                                              ████████████████████
                                                           14▲         21▲
                                                        abstract     paper
```

---

## 21. Project Structure

```
pythonProject1/
├── config/
│   ├── experiment.yaml
│   └── models/
│       ├── phi4.yaml
│       ├── mistral-small3.2.yaml
│       ├── gemma3-27b.yaml
│       └── qwen3-32b.yaml
├── data/
│   ├── design/
│   │   ├── demographics_24.csv
│   │   ├── bigfive_16.csv
│   │   └── personas_384.csv
│   ├── stimuli/
│   │   ├── scenarios.yaml
│   │   ├── forms.yaml
│   │   └── assignment.csv
│   └── raw/{model_name}/responses.jsonl
├── prompts/
│   ├── system_header.txt
│   ├── traits/*.yaml
│   └── ablation_d/*.yaml
├── scripts/
│   ├── 01_generate/
│   ├── 02_render/
│   ├── 03_runners/
│   ├── 04_parse/
│   └── 05_analysis/
├── tests/
├── results/
├── figures/
├── requirements.txt
└── .env
```

---

## 22. Pre-Launch Checklist

### Infrastructure
- [ ] Verify all 4 models respond on designated servers (smoke test: 2 calls each)
- [ ] Verify logprobs availability per model (10 AB-form calls, check response)
- [ ] Record logprobs_available = true/false per model in config
- [ ] Verify Ollama supports system role for each model; document fallback
- [ ] Confirm GPU server availability and Ollama versions

### Data Generation
- [ ] Verify 384 profiles (24 demo × 16 B5)
- [ ] Verify balanced version assignment (96 per version per scenario)
- [ ] Verify rendered prompts:
  - [ ] System header: "reflect the priorities and tendencies of the person described above" (NOT "financial situation")
  - [ ] No numeric effect codes
  - [ ] Trait order varies between calls
  - [ ] All scenario attributes quantified
  - [ ] V4 included but flagged for exclusion
  - [ ] num_predict = 16 in all configs
- [ ] Spot-check 5 rendered prompts manually

### Analysis Pipeline
- [ ] Power simulation script runs and produces power curves
- [ ] All analysis scripts prepared: glmer per-model, Form decomposed, FDR three families, beta regression, S-index (with S_literature benchmark), Inc_L × Involvement interaction
- [ ] Literature benchmark table finalized with verified effect sizes and source citations

---

## 23. Verification Checklist

### Scenarios
- [x] 5 products span 3 domains (tech, apparel, health)
- [x] 5 products span involvement levels (low → high)
- [x] 5 products span price tiers ($25–$1,449)
- [x] All attributes quantified — no vague labels
- [x] V4 = clear dominance in all 5 scenarios
- [x] Currency: USD throughout, JPY note in limitations

### Country Selection
- [x] 5 selection constraints documented
- [x] 8 candidate pairs systematically compared
- [x] Hofstede dimensions with consumer relevance
- [x] 5 testable Region × B5 predictions

### System Header
- [x] No mention of financial situation, income, money, personality
- [x] "reflect the priorities and tendencies of the person described above" — neutral, action-oriented
- [x] No AI/study/experiment mentions

### Experimental Design
- [x] 16-run B5 Resolution V: balanced, pairwise balanced
- [x] Saturated design: error df justified
- [x] Power analysis specified (K=4)
- [x] num_predict = 16
- [x] Logprobs: checked per model via smoke test

### Models
- [x] 4 models, 4 providers, all Ollama (zero API cost)
- [x] Parameter range: 14.7B–32.8B (Dense + MoE)
- [x] GPU allocation: GPU2 (4 models), GPU3 (1 model)
- [x] Open-weight framing in Discussion (reproducibility + practical deployment)
- [x] Limitation #11: no proprietary frontier models

### Trait Descriptions
- [x] Consumer-relevant, word-balanced (26.7 ± 0.8), discriminant validity
- [x] Condition D (general psych, BFI-2-S wording) in ablation
- [x] A+1 × Japan culture note

### Analysis
- [x] Inferential strategy: per-model = primary, mixed = secondary
- [x] Per-model: glmer with (1|ProfileID)
- [x] Comparison models: (1|Model) for valid LRT
- [x] Inference model: (1 | Model) — K=4, random slopes infeasible
- [x] Form → Order + Format (Helmert)
- [x] Three FDR families (F1: main, F2: interactions, F3: per-model)
- [x] R²_GLMM
- [x] Beta regression for logprobs P(A)
- [x] Inc_L × Involvement interaction
- [x] Positional bias with Scenario + Order + Format covariates
- [x] Temperature: 3 levels (0.0, 0.5, 1.0), orthogonal polynomial contrasts (Temp_L, Temp_Q)
- [x] S-index per model per temperature for temperature–stereotype trajectory

### External Validity
- [x] Literature-based benchmark: effect sizes from 5+ published studies
- [x] S_literature ≈ 2.0 (range 1.5–2.5), sensitivity at 3 values
- [x] S-index: continuous, literature-anchored
- [x] Downstream harms + mitigation recommendations
- [x] Limitations: 12 points (added open-weight + no matched human sample)

### Budget & Operations
- [x] 437,760 API calls ($0), no external costs
- [x] Contingency: K=2 min (Tier 1), K=4 full (Tier 2)
- [x] Timeline: 12 weeks, buffer ~1 week
- [x] Timeline contingency compression plan (section 19.1)
- [x] Pre-launch and verification checklists consolidated
- [x] Prompt ablation models: mistral-small3.2 + qwen3:32b (parallel GPUs)
- [x] Tech scenario training data bias discussed (section 17.7)
