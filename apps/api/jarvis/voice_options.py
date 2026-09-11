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
