"""
Business Entity Resolution - Preprocessing and Normalization Module.
Provides pure, unit-tested normalization functions for business names,
addresses, and country fields across US, India, and France entities.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional


# =============================================================================
# Country Normalization
# =============================================================================

def normalize_country(country: Any) -> str:
    """
    Normalizes a country string.
    
    Open-string normalization: lowercases and strips whitespace.
    Does NOT map to a fixed enum or one-hot encoding, as the test set
    contains unseen countries such as 'France' (and potentially others).
    """
    if country is None:
        return ""
    if isinstance(country, float):
        import math
        if math.isnan(country):
            return ""
    return str(country).strip().lower()


# =============================================================================
# Business Name Normalization
# =============================================================================

# Full URLs with protocol or www
_URL_REGEX = re.compile(
    r"(?:https?://\S+|www\.\S+)",
    re.IGNORECASE,
)

# Standalone or embedded domain extensions to strip from company names (e.g., metrohealth.com -> metrohealth)
_TLD_EXTENSION_REGEX = re.compile(
    r"\.(?:co\.in|com|org|net|in|co|io|fr|biz|info|gov|edu|ai|me)\b",
    re.IGNORECASE,
)

# Canonical legal suffix expansions
_LEGAL_SUFFIX_MAP: Dict[str, str] = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "sarl": "sarl",  # French SARL
    "sas": "sas",    # French SAS
    "sa": "sa",      # French SA
    "gmbh": "gmbh",  # German/Intl GmbH
}

# Regex for legal suffixes surrounded by word boundaries
_LEGAL_SUFFIX_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _LEGAL_SUFFIX_MAP.keys()) + r")\b",
    re.IGNORECASE,
)

# Common non-informative symbols/prefixes at start/end: <<, >>, --, **, |, etc.
_NOISE_PUNCT_REGEX = re.compile(r"^[\s\-<>=*|#_~]+|[\s\-<>=*|#_~]+$")
_PUNCT_REGEX = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_REGEX = re.compile(r"\s+")


def _replace_suffix(match: re.Match[str]) -> str:
    token = match.group(1).lower()
    return _LEGAL_SUFFIX_MAP.get(token, token)


def normalize_name(name: Any) -> str:
    """
    Normalizes a business entity name:
    1. Handles None / non-string values safely.
    2. Strips URLs and domain extensions (e.g. .com, www., .in).
    3. Normalizes '&' and '+' to 'and'.
    4. Strips punctuation while preserving Unicode alphanumeric characters and combining marks
       (Devanagari matras, French accented characters, etc.).
    5. Expands/canonicalizes common legal suffix abbreviations:
       corp -> corporation, inc -> incorporated, pvt -> private, ltd -> limited, co -> company.
    6. Collapses redundant whitespace and returns trimmed lowercase string.
    """
    if name is None:
        return ""
    if isinstance(name, float):
        import math
        if math.isnan(name):
            return ""

    s = str(name).strip()
    if not s:
        return ""

    # Replace invisible and non-breaking spaces
    s = s.replace("\u00a0", " ").replace("\u200b", " ").replace("\ufeff", " ")

    # Strip noise prefixes/suffixes like <<, >>, --, |
    s = _NOISE_PUNCT_REGEX.sub(" ", s)

    # Strip web URLs (e.g. http://care.org, www.shivshakti.com)
    s = _URL_REGEX.sub(" ", s)

    # Strip domain TLD extensions (e.g. metrohealth.com -> metrohealth)
    s = _TLD_EXTENSION_REGEX.sub(" ", s)

    # Normalize '&' and '+' to 'and'
    s = s.replace("&", " and ").replace("+", " and ")

    # Convert to lowercase
    s = s.lower()

    # Filter characters: preserve Letters (L), Numbers (N), and Marks (M - Indic vowel signs/matras) + whitespace
    chars = [
        c if (unicodedata.category(c)[0] in ("L", "N", "M") or c.isspace()) else " "
        for c in s
    ]
    s = "".join(chars)
    s = s.replace("_", " ")

    # Canonicalize legal suffixes using token boundary matching
    s = _LEGAL_SUFFIX_REGEX.sub(_replace_suffix, s)

    # Collapse multiple whitespaces and strip
    s = _WHITESPACE_REGEX.sub(" ", s).strip()
    return s


# =============================================================================
# Address Normalization
# =============================================================================

# Street abbreviations expansion mapping
_STREET_ABBREV_MAP: Dict[str, str] = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "bldg": "building",
    "fl": "floor",
    "flr": "floor",
    "ste": "suite",
    "apt": "apartment",
    "unit": "unit",
    "no": "number",
    "ext": "extension",
    "cir": "circle",
    "sq": "square",
    "tr": "trail",
    "trail": "trail",
    "way": "way",
    # French street abbreviations
    "bd": "boulevard",
    "bvd": "boulevard",
    "r": "rue",
    "rte": "route",
    "all": "allee",
    "imp": "impasse",
    "pl": "place",
    "av": "avenue",
}

_STREET_ABBREV_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _STREET_ABBREV_MAP.keys()) + r")\b",
    re.IGNORECASE,
)


def _replace_street_abbrev(match: re.Match[str]) -> str:
    token = match.group(1).lower()
    return _STREET_ABBREV_MAP.get(token, token)


def _clean_address_text(t: str) -> str:
    t_clean = _PUNCT_REGEX.sub(" ", t)
    t_clean = t_clean.replace("_", " ").lower()
    t_expanded = _STREET_ABBREV_REGEX.sub(_replace_street_abbrev, t_clean)
    return _WHITESPACE_REGEX.sub(" ", t_expanded).strip()


# Landmark trigger phrases (English & Indian/Hindi)
# e.g., "near fortis hospital", "opposite sbi atm", "opp metro station", "next to iter college",
# "behind bus stop", "beside park", "pani ki tanki ke pas"
_LANDMARK_REGEX = re.compile(
    r"\b(?:near|opp(?:osite)?|behind|beside|next\s+to|in\s+front\s+of|close\s+to)\s+([^,]+)",
    re.IGNORECASE,
)
_HINDI_LANDMARK_REGEX = re.compile(
    r"([^,]+?\s+(?:ke\s+pas|ke\s+samne|ke\s+piche|ke\s+najdeek))\b",
    re.IGNORECASE,
)

# Common Postal Code Regexes
# US (5-digit or 5+4), India (6-digit starting 1-9), France (5-digit)
_POSTAL_CODE_REGEX = re.compile(
    r"\b([1-9][0-9]{5}|[0-9]{5}(?:-[0-9]{4})?)\b"
)

# US 2-letter state codes (uppercase & lowercase)
_US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
    "dc", "pr", "vi", "gu"
}


def normalize_address(address: Any) -> Dict[str, Any]:
    """
    Normalizes a business address into a structured dictionary:
    - raw_address: original string
    - clean_address: full normalized, expanded, lowercase address string
    - street_tokens: list of normalized street/unit/premise tokens
    - city: extracted city name or None
    - state: extracted state name / code or None
    - postal_code: extracted postal / zip / pin code or None
    - raw_landmark: extracted landmark phrase (e.g., 'near fortis hospital') or None

    Handles missing components gracefully (returns None, never throws).
    """
    if address is None:
        return {
            "raw_address": "",
            "clean_address": "",
            "street_tokens": [],
            "city": None,
            "state": None,
            "postal_code": None,
            "raw_landmark": None,
        }

    if isinstance(address, float):
        import math
        if math.isnan(address):
            return {
                "raw_address": "",
                "clean_address": "",
                "street_tokens": [],
                "city": None,
                "state": None,
                "postal_code": None,
                "raw_landmark": None,
            }

    raw = str(address).strip()
    if not raw:
        return {
            "raw_address": "",
            "clean_address": "",
            "street_tokens": [],
            "city": None,
            "state": None,
            "postal_code": None,
            "raw_landmark": None,
        }

    # Step 1: Extract landmark phrases before stripping commas
    raw_landmark: Optional[str] = None
    lm_match = _LANDMARK_REGEX.search(raw)
    if lm_match:
        raw_landmark = lm_match.group(0).strip().lower()
    else:
        hindi_lm = _HINDI_LANDMARK_REGEX.search(raw)
        if hindi_lm:
            raw_landmark = hindi_lm.group(1).strip().lower()

    # Step 2: Extract postal code from raw text
    postal_code: Optional[str] = None
    pc_match = _POSTAL_CODE_REGEX.search(raw)
    if pc_match:
        postal_code = pc_match.group(1).strip()

    # Step 3: Segment analysis for city, state, street
    # Split by comma to detect geographic hierarchy
    segments = [seg.strip() for seg in raw.split(",") if seg.strip()]

    city: Optional[str] = None
    state: Optional[str] = None
    street_segment: Optional[str] = None

    if len(segments) >= 2:
        # Check standard format: Street, City, State [Zip]
        last_seg = segments[-1].strip().lower()
        second_last_seg = segments[-2].strip().lower()

        # Check if last segment is a 2-letter US state or state + zip
        last_tokens = last_seg.split()
        if len(last_tokens) >= 1 and last_tokens[0] in _US_STATES:
            state = last_tokens[0]
            city = second_last_seg
            street_segment = ", ".join(segments[:-2]) if len(segments) > 2 else segments[0]
        # Check if first segment is state (permutation: OH, Columbus, 5559 Orville Avenue)
        elif segments[0].strip().lower() in _US_STATES:
            state = segments[0].strip().lower()
            city = segments[1].strip().lower()
            street_segment = ", ".join(segments[2:]) if len(segments) > 2 else None
        else:
            # Multi-segment Indian / French / International address
            # Often last segment is State (e.g. 'West Bengal', 'Maharashtra', 'Nouvelle-Aquitaine')
            # Second to last is City (e.g. 'Kolkata', 'Mumbai', 'Bordeaux')
            state = last_seg
            city = second_last_seg
            street_segment = ", ".join(segments[:-2]) if len(segments) > 2 else segments[0]
    elif len(segments) == 1:
        street_segment = segments[0]

    # Step 4: Expand abbreviations and clean street tokens
    clean_address = _clean_address_text(raw)

    street_tokens: List[str] = []
    if street_segment:
        cleaned_street = _clean_address_text(street_segment)
        street_tokens = [tok for tok in cleaned_street.split() if tok]

    # Clean city and state if extracted
    if city is not None:
        city = _clean_address_text(city)
        if not city:
            city = None

    if state is not None:
        state = _clean_address_text(state)
        if not state:
            state = None

    return {
        "raw_address": raw,
        "clean_address": clean_address,
        "street_tokens": street_tokens,
        "city": city,
        "state": state,
        "postal_code": postal_code,
        "raw_landmark": raw_landmark,
    }
