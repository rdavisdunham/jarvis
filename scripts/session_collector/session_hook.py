#!/usr/bin/env python3
"""Claude Code SessionStart hook: tell the agent its own session ID for Eridani's session_link tool."""

import json
import sys

try:
    event = json.load(sys.stdin)
except ValueError:
    sys.exit(0)
session = str(event.get("session_id") or "")
if session:
    print(
        f"Eridani session ledger: this session is tool=claude_code, session_id={session}. "
        "If the user asks to file this work under a Linear issue or client, call the Eridani session_link tool with them."
    )
