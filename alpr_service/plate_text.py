"""Indian license-plate text normalization, validation, OCR-error
correction, and the per-character weighted vote across multiple reads
of the same plate. Unchanged from R2 -- just relocated."""
import re
from collections import Counter, defaultdict

PLATE_PATTERNS = [
    re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$'),
    re.compile(r'^\d{2}BH\d{4}[A-Z]{1,2}$'),
]
BH_PATTERN = PLATE_PATTERNS[1]

TO_DIGIT = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'L': '1', 'Z': '2',
            'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8', 'J': '3'}

TO_ALPHA = {'0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A', '5': 'S',
            '6': 'G', '7': 'T', '8': 'B'}


def normalize(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def is_valid(text):
    return any(p.fullmatch(text) for p in PLATE_PATTERNS)


def fix_indian_plate(text):
    text = normalize(text)
    if BH_PATTERN.fullmatch(text):
        return text
    if len(text) < 8:
        return text
    c = list(text)
    c[0] = TO_ALPHA.get(c[0], c[0]); c[1] = TO_ALPHA.get(c[1], c[1])
    if len(c) >= 4:
        c[2] = TO_DIGIT.get(c[2], c[2]); c[3] = TO_DIGIT.get(c[3], c[3])
    for i in range(max(0, len(c) - 4), len(c)):
        c[i] = TO_DIGIT.get(c[i], c[i])
    for i in range(4, max(4, len(c) - 4)):
        c[i] = TO_ALPHA.get(c[i], c[i])
    fixed = ''.join(c)
    if is_valid(text) and not is_valid(fixed):
        return text
    return fixed


def weighted_vote(results):
    valid = [r for r in results if r['valid']]
    pool = valid if valid else results
    mode_len = Counter(len(r['plate']) for r in pool).most_common(1)[0][0]
    pool = [r for r in pool if len(r['plate']) == mode_len]
    votes = defaultdict(dict)
    for r in pool:
        for i, ch in enumerate(r['plate']):
            votes[i][ch] = votes[i].get(ch, 0) + r['conf']
    out = ''
    for i in range(mode_len):
        if i in votes:
            out += max(votes[i], key=votes[i].get)
    if not is_valid(out) and valid:
        out = max(valid, key=lambda r: r['conf'])['plate']
    return out
