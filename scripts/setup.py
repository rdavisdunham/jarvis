"""Generate local secrets once. Never overwrite the legacy .env or existing upgrade settings."""

import base64
import secrets
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / ".env.upgrade"
    existing = dotenv_values(path) if path.exists() else {}
    key = ec.generate_private_key(ec.SECP256R1())
    encoded = lambda value: base64.urlsafe_b64encode(value).decode().rstrip("=")
    defaults = {
        "JARVIS_POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "JARVIS_BACKUP_KEY": Fernet.generate_key().decode(),
        "JARVIS_OWNER_TOKEN": secrets.token_urlsafe(32),
        "JARVIS_ORIGIN": "http://localhost:8765",
        "JARVIS_VAPID_PRIVATE_KEY": encoded(key.private_numbers().private_value.to_bytes(32, "big")),
        "JARVIS_VAPID_PUBLIC_KEY": encoded(
            key.public_key().public_bytes(
                serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
            )
        ),
        "JARVIS_TIMEZONE": "America/Chicago",
        "JARVIS_MONTHLY_BUDGET_USD": "150",
        "JARVIS_COST_TRACKING_ENABLED": "true",
    }
    password = existing.get("JARVIS_POSTGRES_PASSWORD") or defaults["JARVIS_POSTGRES_PASSWORD"]
    defaults["JARVIS_DATABASE_URL"] = f"postgresql+psycopg://jarvis:{password}@127.0.0.1:54329/jarvis"
    with path.open("a") as stream:
        for name, value in defaults.items():
            if not existing.get(name):
                stream.write(f"{name}={value}\n")
    path.chmod(0o600)
    runtime = ROOT / ".runtime"
    runtime.mkdir(mode=0o700, exist_ok=True)
    pairing = runtime / "pairing-code"
    pairing.write_text(dotenv_values(path)["JARVIS_OWNER_TOKEN"])
    pairing.chmod(0o600)
    recovery = runtime / "backup-key"
    recovery.write_text(dotenv_values(path)["JARVIS_BACKUP_KEY"])
    recovery.chmod(0o600)
    print("Upgrade configuration ready. Pairing code: .runtime/pairing-code (not printed).")
    print("Legacy data and .env were not changed.")


if __name__ == "__main__":
    main()
