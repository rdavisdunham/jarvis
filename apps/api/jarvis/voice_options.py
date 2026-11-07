"""Provider voice names from the official voice guides, checked 2026-09-11."""

REALTIME_VOICES = ["marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
LIVE_VOICES = [
    "marin",
    "quartz",
    "ripple",
    "vesper",
    "willow",
    "stone",
    "gleam",
    "meridian",
    "bossa",
    "tempo",
    "beacon",
    "delta",
    "cinder",
]
OPTIONS = {
    "realtime": {"label": "Realtime", "voices": REALTIME_VOICES, "default_voice": "marin"},
    "live": {"label": "GPT-Live", "voices": LIVE_VOICES, "default_voice": "marin"},
}

# Realtime is temporarily paused by owner preference. Its catalog/controller stay
# intact; re-enable it here when it is ready for another device/provider trial.
DEFAULT_PROVIDER = "live"
ENABLED_PROVIDERS = ("live",)


def available_options():
    return {provider: OPTIONS[provider] for provider in ENABLED_PROVIDERS}


# Dani's pronouns follow the chosen voice. Presentation from OpenAI's GPT-Live voice table
# (guides/live-conversations#voice-options, checked 2026-11-07) and the Realtime voice guide.
FEMININE = {"marin", "quartz", "willow", "gleam", "bossa", "delta", "coral", "sage", "shimmer"}
MASCULINE = {"cedar", "ripple", "vesper", "stone", "meridian", "tempo", "beacon", "cinder", "ash", "ballad", "echo", "verse"}


def pronouns(voice=None):
    """she/her or he/him to match the voice; the default voice when none is known; they/them otherwise."""
    name = (voice or OPTIONS[DEFAULT_PROVIDER]["default_voice"]).strip().lower()
    return "she/her" if name in FEMININE else "he/him" if name in MASCULINE else "they/them"
