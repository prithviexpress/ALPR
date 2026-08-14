"""Indian license-plate text normalization, validation, and OCR-error
correction."""
import re

PLATE_PATTERNS = [
    # State(2 letters) + RTO code(1-2 digits, some states/eras drop the
    # leading zero, e.g. Delhi "DL1LAJ8068") + series(0-3 letters, legacy
    # plates from some states carry none at all, e.g. "HR842403") +
    # number(4 digits).
    re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$'),
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
    # Already matches a known format (standard or BH-series) -- leave it
    # alone. The character-remapping below assumes a FIXED 2-digit RTO
    # code position, which is wrong for a 1-digit-RTO plate (e.g.
    # "DL1LAJ8068"): without this check it would "fix" the series' first
    # letter into a digit, corrupting an already-correct read.
    if is_valid(text):
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
