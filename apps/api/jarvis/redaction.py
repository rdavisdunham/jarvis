"""The secret patterns shared with remotea and the Python session collector (stage F, L6).

The canonical list lives in remotea's docs/remotea/redaction.md; remotea compiles the same strings in
remote-server/src/collector/redact.ts. Change all three in one release. Standard library only: the
collector imports this file next to itself (install.sh copies it) or from the repository.

Rules that keep JavaScript and Python identical: no inline flags (case is spelled with classes),
end of input is `(?![\\s\\S])`, and every match becomes `[redacted]`.
"""

import re

PATTERNS = (
    r"sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"AKIA[0-9A-Z]{16}",
    r"xox[abpr]-[A-Za-z0-9-]{10,}",
    r"eri_bot_[A-Za-z0-9_-]{20,}",
    r"lin_api_[A-Za-z0-9]{20,}",
    r"[Bb][Ee][Aa][Rr][Ee][Rr]\s+[A-Za-z0-9._~+/=-]{20,}",
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|(?![\s\S]))",
    r"://[^/\s:@]+:[^/\s@]+@",
    r"(?:[Pp][Aa][Ss][Ss][Ww][Oo][Rr][Dd]|[Ss][Ee][Cc][Rr][Ee][Tt]|[Tt][Oo][Kk][Ee][Nn]|[Aa][Pp][Ii]_[Kk][Ee][Yy])"
    r"\s*[=:]\s*\S{8,}",
)
SECRETS = re.compile("(" + "|".join(PATTERNS) + ")")
MARK = "[redacted]"


def redact(text):
    return SECRETS.sub(MARK, text or "")
