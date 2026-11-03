"""Assistant identity, separate from the Eridani product and protocol names."""

import re

ASSISTANT_NAME = "Dani"
# DAN-ee and DAH-nee can produce different spellings in browser/provider ASR.
SPOKEN_NAMES = ("dani", "danny", "danni", "donny", "donnie")
WAKE_PREFIXES = ("hey", "hi", "okay", "ok")
_WAKE = re.compile(
    rf"^(?:(?:{'|'.join(WAKE_PREFIXES)})[,\s]+)?(?:{'|'.join(SPOKEN_NAMES)})"
    r"(?=$|[.!?,\s])(?:[.!?,\s]*)(.*)$", re.IGNORECASE | re.DOTALL,
)


def wake_request(text: str) -> str | None:
    """An empty suffix is wake-only; None is ordinary speech, not activation."""
    match = _WAKE.match(text.strip())
    return match[1].strip() if match else None


def available_assistant_name(names) -> str:
    """Disambiguate the assistant without renaming people or custom actors."""
    used = {name.casefold() for name in names}
    candidate, number = ASSISTANT_NAME, 1
    while candidate.casefold() in used:
        suffix = "assistant" if number == 1 else f"assistant {number}"
        candidate = f"{ASSISTANT_NAME} ({suffix})"
        number += 1
    return candidate
