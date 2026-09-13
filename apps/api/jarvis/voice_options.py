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
