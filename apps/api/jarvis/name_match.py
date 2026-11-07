"""Sound-alike and near-spelling name matching, for names speech-to-text gets wrong ("Janice" for Janiece,
"Shreya" for Sreya).

`score(heard, known)` is 1.0 for the same letters and otherwise the better of two similarities:
- spelling: difflib's ratio of the plain letters;
- sound: the same ratio over a rough phonetic key (`sound`).
It is 0 when the words start with different sounds, when either has fewer than MIN_LETTERS letters (three-letter
names still match when their keys are equal: "Jon", "John"), or when the spelling ratio is under SPELLING_FLOOR (so
"Sorry" never matches "Sarah", whose keys sound alike). A word matches at THRESHOLD or above.
No model call and no network: it runs on every lookup.
"""

import re
import unicodedata
from difflib import SequenceMatcher

MIN_LETTERS = 4
SHORT_LETTERS = 3
THRESHOLD = 0.8
SPELLING_FLOOR = 0.65
# Words that open spoken sentences and so arrive capitalized; never read as a misheard name.
_COMMON = """
about actually after again also always anyway are aren before being both call can cannot could did does doing done
during each even ever every first from good great have having hello help here hey hold just keep know later let like
look made make maybe might more most much must never next nice none nope okay only other please pretty quick quite
really right said same says seems should show since some something sorry soon sounds start still such sure take tell
than thank thanks that then there these they thing think this those though through today tomorrow tonight wait want
well were what when where which while will with without would yeah yesterday you your
"""
COMMON = frozenset(_COMMON.split())

# Ordered: longer spellings first. Each maps a spelling to the sound it usually makes in a name.
SOUNDS = (("sch", "sk"), ("ph", "f"), ("ck", "k"), ("gh", "g"), ("th", "t"), ("sh", "s"), ("ch", "k"), ("qu", "kw"),
          ("q", "k"), ("x", "ks"), ("z", "s"), ("y", "i"))
VOWELS = re.compile(r"[aeiou]+")


def letters(word: str) -> str:
    """Lowercase a-z only, accents removed ("Zoë" -> "zoe")."""
    plain = unicodedata.normalize("NFKD", word or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", plain.casefold())


def sound(word: str) -> str:
    """A rough phonetic key: common spellings of one sound folded together, a soft c read as s, h after a letter
    dropped, doubled letters collapsed, and every run of vowels after the first letter written as "a"."""
    key = letters(word)
    if not key:
        return ""
    for spelled, said in SOUNDS:
        key = key.replace(spelled, said)
    key = re.sub(r"c(?=[eiy])", "s", key).replace("c", "k")
    key = key[0] + key[1:].replace("h", "")
    key = re.sub(r"(.)\1+", r"\1", key)
    # Any opening vowel sound is one sound ("Yvette", "Evette"); a vowel run after it is written "a".
    return ("a" if key[0] in "aeiouy" else key[0]) + VOWELS.sub("a", key[1:])


def score(heard: str, known: str) -> float:
    a, b = letters(heard), letters(known)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    sa, sb = sound(a), sound(b)
    if min(len(a), len(b)) < MIN_LETTERS:
        # Short names match only when they sound the same ("Jon", "John").
        return 0.9 if min(len(a), len(b)) >= SHORT_LETTERS and sa == sb else 0.0
    if sa[:1] != sb[:1]:
        return 0.0
    spelling = SequenceMatcher(None, a, b).ratio()
    if spelling < SPELLING_FLOOR:
        return 0.0
    return max(spelling, SequenceMatcher(None, sa, sb).ratio())


def alike(heard: str, known: str) -> bool:
    return letters(heard) not in COMMON and score(heard, known) >= THRESHOLD
