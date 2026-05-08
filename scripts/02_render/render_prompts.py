"""Render full prompts from profile + scenario + form.

Produces (system_message, user_message) tuples ready for Ollama API.
Trait order is randomized per task_id for reproducibility.
"""

import hashlib
import pathlib
from typing import Tuple

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT / "prompts"
STIMULI_DIR = ROOT / "data" / "stimuli"

# Load system header once
SYSTEM_HEADER = (PROMPTS_DIR / "system_header.txt").read_text().strip()

# Load trait descriptions
TRAITS_ORDER = ["O", "C", "E", "A", "N"]
TRAIT_KEY_MAP = {
    "O": "openness",
    "C": "conscientiousness",
    "E": "extraversion",
    "A": "agreeableness",
    "N": "neuroticism",
}


def _load_traits(condition: str = "A_nl_consumer") -> dict:
    """Load trait descriptions. condition A = consumer-framed, D = general psych."""
    traits_dir = PROMPTS_DIR / ("ablation_d" if condition == "D_general_psych" else "traits")
    traits = {}
    for code, fname in TRAIT_KEY_MAP.items():
        with open(traits_dir / f"{fname}.yaml") as f:
            data = yaml.safe_load(f)
        traits[code] = {"high": data["high"].strip(), "low": data["low"].strip()}
    return traits


# Pre-load both conditions
TRAITS_CONSUMER = _load_traits("A_nl_consumer")
TRAITS_GENERAL = _load_traits("D_general_psych")

# Load scenarios
with open(STIMULI_DIR / "scenarios.yaml") as f:
    SCENARIOS = yaml.safe_load(f)

# Load forms
with open(STIMULI_DIR / "forms.yaml") as f:
    FORMS = yaml.safe_load(f)

# Gender text map
GENDER_TEXT = {"Male": "man", "Female": "woman"}


def _shuffled_trait_order(task_id: str) -> list:
    """Deterministic random permutation of TRAITS_ORDER seeded by task_id hash."""
    h = int(hashlib.sha256(task_id.encode()).hexdigest(), 16)
    order = list(TRAITS_ORDER)
    # Fisher-Yates shuffle with deterministic seed from hash
    for i in range(len(order) - 1, 0, -1):
        j = h % (i + 1)
        order[i], order[j] = order[j], order[i]
        h //= (i + 1)
    return order


def get_trait_positions(task_id: str) -> dict:
    """Return {trait: position} (0-indexed) for a given task_id."""
    order = _shuffled_trait_order(task_id)
    return {t: i for i, t in enumerate(order)}


def render_prompt(
    profile: dict,
    scenario_id: str,
    version: str,
    form_id: str,
    replication: int,
    condition: str = "A_nl_consumer",
) -> Tuple[str, str]:
    """Render a complete (system, user) prompt pair.

    Args:
        profile: dict with keys Age, Gender, Income, Region_text, O, C, E, A, N
        scenario_id: e.g. "S1"
        version: e.g. "V2"
        form_id: e.g. "AB_Afirst"
        replication: 1-based replication index
        condition: prompt ablation condition

    Returns:
        (system_message, user_message) tuple
    """
    # Task ID for trait order randomization
    task_id = f"{profile['ProfileID']}:{scenario_id}:{form_id}:{replication}"
    trait_order = _shuffled_trait_order(task_id)

    # Select trait descriptions
    if condition == "D_general_psych":
        traits = TRAITS_GENERAL
    else:
        traits = TRAITS_CONSUMER

    # Build trait lines
    trait_lines = []
    for t in trait_order:
        level = "high" if profile[t] == 1 else "low"
        trait_lines.append(f"- {traits[t][level]}")

    # Income formatting
    income = profile["Income"]
    if income >= 1000:
        income_str = f"${income:,}"
    else:
        income_str = f"${income}"

    # Profile block
    gender_word = GENDER_TEXT[profile["Gender"]]
    profile_block = (
        f"=== YOUR PROFILE ===\n\n"
        f"You are a {profile['Age']}-year-old {gender_word} "
        f"living in {profile['Region_text']}.\n"
        f"Your annual household income is approximately {income_str}.\n\n"
        f"Your personality:\n"
        + "\n".join(trait_lines)
        + "\n\n=== END PROFILE ==="
    )

    # Scenario block
    sc = SCENARIOS[scenario_id]
    ver = sc["versions"][version]
    form = FORMS[form_id]

    # Option order based on form
    if form["option_order"][0] == "A":
        opt_first = f"Option A: {ver['option_a']}"
        opt_second = f"Option B: {ver['option_b']}"
    else:
        opt_first = f"Option B: {ver['option_b']}"
        opt_second = f"Option A: {ver['option_a']}"

    scenario_block = (
        f"=== SCENARIO {scenario_id}: {sc['name']} ===\n"
        f"Context: {sc['context']}\n"
        f"{opt_first}\n"
        f"{opt_second}\n\n"
        f"=== QUESTION ===\n"
        f"{form['question']}"
    )

    # Conditions B and C modify the profile block
    if condition == "B_structured":
        profile_block = _render_structured(profile, trait_order, traits)
    elif condition == "C_coded":
        profile_block = _render_coded(profile)

    user_message = f"{profile_block}\n\n{scenario_block}"
    return SYSTEM_HEADER, user_message


def _render_structured(profile: dict, trait_order: list, traits: dict) -> str:
    """Condition B: labels instead of NL."""
    trait_labels = []
    for t in trait_order:
        level = "High" if profile[t] == 1 else "Low"
        trait_labels.append(f"{t}: {level}")

    return (
        f"=== YOUR PROFILE ===\n\n"
        f"Age: {profile['Age']}\n"
        f"Gender: {profile['Gender']}\n"
        f"Income: ${profile['Income']:,}\n"
        f"Region: {profile['Region']}\n\n"
        f"Personality: {'; '.join(trait_labels)}\n\n"
        f"=== END PROFILE ==="
    )


def _render_coded(profile: dict) -> str:
    """Condition C: effect codes."""
    return (
        f"=== YOUR PROFILE ===\n\n"
        f"Age_L: {profile['Age_L']}\n"
        f"Gender: {profile['Gender_code']}\n"
        f"Inc_L: {profile['Inc_L']}\n"
        f"Inc_Q: {profile['Inc_Q']}\n"
        f"Region: {profile['Region_code']}\n\n"
        f"O: {profile['O']}; C: {profile['C']}; "
        f"E: {profile['E']}; A: {profile['A']}; N: {profile['N']}\n\n"
        f"=== END PROFILE ==="
    )
