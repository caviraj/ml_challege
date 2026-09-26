"""
Blocking key extraction for business entity resolution.
Generates multi-attribute composite blocking keys across name, address, city, and postal code
partitioned by country to ensure high candidate recall with manageable cardinality.
"""

from typing import Set, Dict, Any, List
import re

LEGAL_PATTERN = re.compile(
    r'\b(?:incorporated|corporation|private limited|pvt ltd|limited|ltd|inc|corp|llc|llp|co|gmbh|sa|sarl|sas|bhd|plc|trust|society|foundation)\b'
)

COMMON_PREFIXES = ('the ', 'a ', 'an ', 'shri ', 'smt ', 'dr ', 'm/s ', 'ms ')

STOPWORDS = {
    'and', 'the', 'for', 'ltd', 'inc', 'corp', 'llc', 'llp', 'pvt', 'co', 'gmbh',
    'sa', 'sarl', 'sas', 'plc', 'company', 'services', 'enterprises', 'solutions',
    'international', 'group', 'holdings', 'global', 'india', 'usa', 'united', 'states',
    'france', 'de', 'du', 'des', 'le', 'la', 'les', 'en', 'et'
}

ADDR_STOP = {
    'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'suite', 'floor',
    'building', 'block', 'near', 'opp', 'opposite', 'behind', 'cross', 'main',
    'north', 'south', 'east', 'west', 'rue', 'chemin', 'route', 'allee', 'place',
    'nagar', 'colony', 'sector', 'marg', 'bhavan', 'tower', 'house', 'city', 'post', 'office'
}


def strip_legal_suffixes(name: str) -> str:
    """Strips legal entity suffixes to reveal the core distinctive business name."""
    prev = ''
    clean = name.strip()
    while prev != clean:
        prev = clean
        clean = LEGAL_PATTERN.sub('', clean).strip()
    return clean


def extract_blocking_keys(record: Dict[str, Any]) -> Set[str]:
    """
    Extracts multi-attribute blocking keys for an entity record.
    All keys are country-partitioned.
    
    Keys include:
      1. cn_p3: Compact alphanumeric name prefix (3 chars)
      2. cn_p4: Compact alphanumeric name prefix (4 chars)
      3. tok: Distinctive name tokens (>= 3 chars, not stopwords)
      4. addr_tok: Distinctive address tokens (>= 4 chars, not numbers, not generic)
      5. post_p2: Postal code + 2-char compact name prefix
      6. city_p2: City + 2-char compact name prefix
      7. city_num: City + street number (2-4 digits)
    """
    country = str(record.get('country_norm', '') or '').strip()
    name = str(record.get('name_norm', '') or '').strip()
    addr = str(record.get('clean_address', '') or '').strip()
    city = str(record.get('city', '') or '').strip()
    postal = str(record.get('postal_code', '') or '').strip()

    keys = set()
    
    # Strip common leading honorific prefixes
    clean_name = name
    for pfx in COMMON_PREFIXES:
        if clean_name.startswith(pfx):
            clean_name = clean_name[len(pfx):].strip()
            break

    compact_name = re.sub(r'[^a-z0-9]', '', clean_name)
    raw_compact = re.sub(r'[^a-z0-9]', '', name)

    # 1. Compact Name Prefixes
    if len(compact_name) >= 3:
        keys.add(f"cn_p3:{country}:{compact_name[:3]}")
    if len(compact_name) >= 4:
        keys.add(f"cn_p4:{country}:{compact_name[:4]}")
    if raw_compact != compact_name:
        if len(raw_compact) >= 3:
            keys.add(f"cn_p3:{country}:{raw_compact[:3]}")
        if len(raw_compact) >= 4:
            keys.add(f"cn_p4:{country}:{raw_compact[:4]}")

    # 2. Distinctive Name Tokens
    tokens = [t for t in name.split() if len(t) >= 3 and t not in STOPWORDS]
    for t in tokens[:4]:
        keys.add(f"tok:{country}:{t}")

    legal_stripped = strip_legal_suffixes(name)
    if legal_stripped != name:
        for t in [t for t in legal_stripped.split() if len(t) >= 3 and t not in STOPWORDS][:4]:
            keys.add(f"tok:{country}:{t}")

    # 3. Address Tokens (>= 4 chars, not purely digits, not generic)
    addr_tokens = [t for t in addr.split() if len(t) >= 4 and not t.isdigit() and t not in ADDR_STOP]
    for at in addr_tokens[:4]:
        keys.add(f"addr_tok:{country}:{at}")

    # 4. Exact Postal Code + 2-char Name Prefix
    if postal and postal != 'None' and len(postal) >= 3 and len(compact_name) >= 2:
        keys.add(f"post_p2:{country}:{postal}:{compact_name[:2]}")

    # 5. City + 2-char Name Prefix
    if city and city != 'None' and len(city) >= 3 and len(compact_name) >= 2:
        keys.add(f"city_p2:{country}:{city}:{compact_name[:2]}")

    # 6. City + Street Number
    nums = [t for t in addr.split() if t.isdigit() and 2 <= len(t) <= 4]
    if city and city != 'None' and len(city) >= 3 and nums:
        for n in nums[:2]:
            keys.add(f"city_num:{country}:{city}:{n}")

    return keys
