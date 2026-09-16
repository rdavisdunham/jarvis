"""Create an ignored cloud handoff file, preserving existing credential values."""

import argparse
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    target = root / ".env.cloud"
    if target.exists():
        print(".env.cloud already exists; no credentials were overwritten.")
        return target
    template = dotenv_values(root / ".env.cloud.example")
    legacy = {**dotenv_values(root / ".env"), **dotenv_values(root / ".env.upgrade")}
    copy = {
        "JARVIS_OWNER_ID",
        "JARVIS_OWNER_NAME",
        "JARVIS_TIMEZONE",
        "JARVIS_COST_TRACKING_ENABLED",
        "JARVIS_GOOGLE_CLIENT_ID",
        "JARVIS_GOOGLE_CLIENT_SECRET",
        "JARVIS_INTEGRATION_ENCRYPTION_KEY",
        "JARVIS_BACKUP_KEY",
        "JARVIS_VAPID_PRIVATE_KEY",
        "JARVIS_VAPID_PUBLIC_KEY",
        "JARVIS_VAPID_SUBJECT",
        "JARVIS_OPENAI_API_KEY",
        "JARVIS_GEMINI_API_KEY",
    }
    for key in copy:
        value = os.environ.get(key) or legacy.get(key)
        if key in {"JARVIS_OPENAI_API_KEY", "JARVIS_GEMINI_API_KEY"} and not value:
            alias = key.removeprefix("JARVIS_")
            value = os.environ.get(alias) or legacy.get(alias)
        if value:
            template[key] = value

    def quote(value):
        # python-dotenv single-quoted syntax; no shell interpolation.
        return "'" + (value or "").replace("\\", "\\\\").replace("'", "\\'") + "'"

    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write("# Cloud migration handoff. Private: never commit or share in chat.\n")
        stream.write("\n".join(f"{k}={quote(v)}" for k, v in template.items()) + "\n")
    print(
        "Created .env.cloud with existing credentials preserved. Fill the app URL, cloud database URL and R2 fields."
    )
    print("The current .env and .env.upgrade were not changed. No credentials were printed or uploaded.")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate .env.cloud without changing services.")
    parser.add_argument("--database", action="store_true", help="Also check the target database read-only.")
    args = parser.parse_args()
    if args.database and not args.check:
        parser.error("--database requires --check")
    if not args.check:
        prepare()
        return
    if not (ROOT / ".env.cloud").is_file():
        parser.error("Create .env.cloud first.")
    load_dotenv(ROOT / ".env.cloud", override=True)
    # Local dotenv files must never fill missing cloud variables during validation.
    os.environ["JARVIS_ENV_FILE"] = ""
    import json

    from jarvis.config import get_settings
    from jarvis.deploy import DeploymentConfigurationError, preflight

    get_settings.cache_clear()
    try:
        print(json.dumps(preflight(database=args.database)))
    except DeploymentConfigurationError as exc:
        raise SystemExit(str(exc)) from None
    except Exception:  # noqa: BLE001 - connection/decryption errors can contain credentials
        raise SystemExit(
            "Cloud check failed. Verify the target connection and preserved encryption keys."
        ) from None


if __name__ == "__main__":
    main()
