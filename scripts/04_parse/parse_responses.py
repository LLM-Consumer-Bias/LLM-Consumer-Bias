"""Parse LLM responses into Choice A or Choice B.

Rules from design.md section 5.1:
  AB:  first char matching [AaBb]
  REP: Levenshtein ratio to each option; closest if max ratio >= 0.5
  CMP: first token matching yes/no
"""

import re
from Levenshtein import ratio as lev_ratio


def parse_ab(response: str) -> str:
    """Parse AB-form response. Returns 'A', 'B', or 'invalid'.

    Strategy: first try standalone A/B (word boundary or start of text),
    then fall back to first character if the entire response is just A or B.
    """
    text = response.strip()
    if not text:
        return "invalid"

    # 1. Check if entire response (ignoring trailing punctuation) is just A or B
    clean = re.sub(r'[.\s]+$', '', text)
    if clean.upper() in ("A", "B"):
        return clean[0].upper()

    # 2. Look for standalone A or B (word boundary)
    match = re.search(r'\b([AaBb])\b', text)
    if match:
        return match.group().upper()

    # 3. Check first character
    if text[0].upper() in ("A", "B"):
        return text[0].upper()

    return "invalid"


def parse_rep(response: str, option_a_text: str = "", option_b_text: str = "") -> str:
    """Parse REP-form response via Levenshtein. Returns 'A', 'B', or 'invalid'.

    Fallback: if response starts with 'Option A'/'Option B' or standalone A/B,
    use that directly (models often label their choice explicitly).
    """
    text = response.strip()
    if not text:
        return "invalid"

    # Fallback 1: explicit "Option A" / "Option B" prefix
    lower = text.lower()
    if re.match(r'option\s*a\b', lower):
        return "A"
    if re.match(r'option\s*b\b', lower):
        return "B"

    # Fallback 2: standalone A/B (same as AB parser)
    clean = re.sub(r'[.\s]+$', '', text)
    if clean.upper() in ("A", "B"):
        return clean[0].upper()

    # Levenshtein comparison (if option texts provided)
    if option_a_text and option_b_text:
        ratio_a = lev_ratio(text.lower(), option_a_text.lower())
        ratio_b = lev_ratio(text.lower(), option_b_text.lower())
        max_ratio = max(ratio_a, ratio_b)
        if max_ratio >= 0.5:
            return "A" if ratio_a >= ratio_b else "B"

    return "invalid"


def parse_cmp(response: str, form_id: str) -> str:
    """Parse CMP-form response. Returns 'A', 'B', or 'invalid'.

    CMP_AoverB: yes→A, no→B
    CMP_BoverA: yes→B, no→A
    """
    text = response.strip().lower()
    match = re.search(r'\b(yes|no)\b', text)
    if not match:
        # Also check for just y/n at start
        match = re.match(r'^(y|n)', text)
        if not match:
            return "invalid"
        answer = "yes" if match.group() == "y" else "no"
    else:
        answer = match.group()

    if form_id == "CMP_AoverB":
        return "A" if answer == "yes" else "B"
    elif form_id == "CMP_BoverA":
        return "B" if answer == "yes" else "A"
    else:
        return "invalid"


def parse_response(
    response: str,
    form_id: str,
    option_a_text: str = "",
    option_b_text: str = "",
) -> str:
    """Route to the correct parser based on form type.

    Returns: 'A', 'B', or 'invalid'
    """
    if not response or not response.strip():
        return "invalid"

    form_type = form_id.split("_")[0]  # AB, REP, CMP

    if form_type == "AB":
        return parse_ab(response)
    elif form_type == "REP":
        return parse_rep(response, option_a_text, option_b_text)
    elif form_type == "CMP":
        return parse_cmp(response, form_id)
    else:
        return "invalid"
