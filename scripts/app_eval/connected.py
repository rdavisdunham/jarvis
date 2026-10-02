"""Opt-in disposable service probes. No production credentials are inferred."""

import json
import os
from pathlib import Path
from urllib.parse import quote, urlsplit
from uuid import uuid4

from .reporting import atomic_json


class PrerequisiteError(ValueError):
    pass


def secret(config, name):
    variable = config.get(name, "")
    if not variable.startswith("ERIDANI_EVAL_"):
        raise PrerequisiteError("Use dedicated ERIDANI_EVAL_ credential variables")
    value = os.environ.get(variable, "")
    if not value:
        raise PrerequisiteError("Missing dedicated credential variable: " + variable)
    return value


def google(config, resources, save):
    from jarvis.config import get_settings
    from jarvis.google_calendar import CalendarClient

    settings = get_settings()
    settings.external_services_enabled = True
    settings.google_client_id = secret(config, "client_id_env")
    settings.google_client_secret = secret(config, "client_secret_env")
    calendar = config["calendar_id"]
    if calendar == "primary":
        raise PrerequisiteError("A dedicated secondary test calendar is required")
    client = CalendarClient({"refresh_token": secret(config, "refresh_token_env")})
    path = "calendars/" + quote(calendar, safe="")
    event_id = "eridanieval" + uuid4().hex
    try:
        metadata = client.request(path)
        if not metadata.get("summary", "").lower().startswith("eridani eval"):
            raise PrerequisiteError("Test calendar name must start with Eridani Eval")
        resources.append({"service": "google", "container": calendar, "id": event_id, "state": "reserved"})
        save()
        body = {
            "id": event_id,
            "summary": "Eridani eval " + event_id,
            "description": "Synthetic disposable acceptance probe",
            "start": {"date": "2030-01-14"},
            "end": {"date": "2030-01-15"},
            "extendedProperties": {"private": {"eridaniEval": event_id}},
        }
        event_path = path + "/events/" + event_id
        try:
            created = client.request(path + "/events", params={"sendUpdates": "none"}, body=body)
            assert created["id"] == event_id
            resources[-1]["state"] = "created"
            save()
            read = client.request(event_path)
            assert read["start"]["date"] == "2030-01-14" and read["end"]["date"] == "2030-01-15"
            updated = client.request(
                event_path,
                method="PATCH",
                params={"sendUpdates": "none"},
                body={"summary": body["summary"] + " edited"},
                headers={"If-Match": read["etag"]},
            )
            assert updated["summary"].endswith(" edited")
            assert client.request(event_path)["summary"] == updated["summary"]
        finally:
            # Identity was generated and journaled before creation, so a lost response
            # never causes cleanup to target a pre-existing user event.
            try:
                client.request(event_path, method="DELETE", params={"sendUpdates": "none"})
                resources[-1]["state"] = "removed"
            except Exception as exc:
                resources[-1].update(state="cleanup_required", error=type(exc).__name__)
                raise
            finally:
                save()
    finally:
        client.close()


def linear(config, resources, save):
    from jarvis.config import get_settings
    from jarvis.linear_client import CREATE_QUERY, UPDATE_QUERY, LinearClient

    get_settings().external_services_enabled = True
    team_id = config["team_id"]
    client = LinearClient(secret(config, "api_key_env"))
    identity = str(uuid4())
    try:
        account = client.identity()
        if account["organization"]["id"] != config["organization_id"]:
            raise PrerequisiteError("Dedicated Linear organization mismatch")
        team = next((t for t in account["teams"] if t["id"] == team_id), None)
        if not team or not team["name"].lower().startswith("eridani eval"):
            raise PrerequisiteError("Dedicated team name must start with Eridani Eval")
        fields = client.query('query EvalCleanupCapability{__type(name:"Mutation"){fields{name}}}')["__type"][
            "fields"
        ]
        if "issueDelete" not in {f["name"] for f in fields}:
            raise PrerequisiteError("Cannot verify issue cleanup capability")
        resources.append({"service": "linear", "container": team_id, "id": identity, "state": "reserved"})
        save()
        title = "Eridani eval " + identity
        try:
            created = client.query(
                CREATE_QUERY,
                {
                    "input": {
                        "id": identity,
                        "teamId": team_id,
                        "title": title,
                        "description": "Synthetic disposable acceptance probe",
                    }
                },
            )["issueCreate"]
            assert created["success"] and created["issue"]["id"] == identity
            resources[-1]["state"] = "created"
            save()
            assert client.issue(identity)["title"] == title
            changed = client.query(UPDATE_QUERY, {"id": identity, "input": {"title": title + " edited"}})[
                "issueUpdate"
            ]
            assert changed["success"] and client.issue(identity)["title"] == title + " edited"
        finally:
            try:
                removed = client.query(
                    "mutation EvalRemove($id:String!){issueDelete(id:$id){success}}", {"id": identity}
                )
                assert removed["issueDelete"]["success"]
                resources[-1]["state"] = "removed"
            except Exception as exc:
                resources[-1].update(state="cleanup_required", error=type(exc).__name__)
                raise
            finally:
                save()
    finally:
        client.close()


def r2(config, resources, save):
    try:
        import boto3
    except ImportError as exc:
        raise PrerequisiteError("Install the backup runtime's boto3 dependency for R2 probes") from exc
    from cryptography.fernet import Fernet

    endpoint = config["endpoint"]
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not (parsed.hostname or "").endswith(".r2.cloudflarestorage.com")
        or parsed.username
    ):
        raise PrerequisiteError("Use an HTTPS R2 endpoint")
    prefix = config.get("prefix", "eridani-eval").strip("/")
    if not prefix.startswith("eridani-eval/") and prefix != "eridani-eval":
        raise PrerequisiteError("R2 probe prefix must be eridani-eval or its child")
    key = prefix + "/" + uuid4().hex + ".enc"
    bucket = config["bucket"]
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=secret(config, "access_key_env"),
        aws_secret_access_key=secret(config, "secret_key_env"),
        config=Config(connect_timeout=5, read_timeout=20, retries={"max_attempts": 1}),
    )
    cipher = Fernet(Fernet.generate_key())
    plaintext = b"Synthetic Eridani backup round-trip probe"
    encrypted = cipher.encrypt(plaintext)
    resources.append({"service": "r2", "container": bucket, "id": key, "state": "reserved"})
    save()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=encrypted, Metadata={"eridani-eval": "true"})
        resources[-1]["state"] = "created"
        save()
        read = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        assert read != plaintext and cipher.decrypt(read) == plaintext
    finally:
        try:
            client.delete_object(Bucket=bucket, Key=key)
            resources[-1]["state"] = "removed"
        except Exception as exc:
            resources[-1].update(state="cleanup_required", error=type(exc).__name__)
            raise
        finally:
            save()
            client.close()


def execute(config, trace):
    source = config.get("live_config")
    if not source:
        return {"status": "blocked", "reason": "Dedicated live-service configuration is not supplied"}
    settings = json.loads(Path(source).read_text())
    service = config["job"]["target"]
    options = settings.get(service)
    if not options or options.get("dedicated_test_resources") is not True:
        return {"status": "blocked", "reason": "Declare dedicated test resources for " + service}
    resources = []
    destination = Path(config["attempt"]) / "resources.json"

    def save():
        atomic_json(destination, resources)

    try:
        {"google": google, "linear": linear, "r2": r2}[service](options, resources, save)
    except PrerequisiteError as exc:
        return {"status": "blocked", "reason": str(exc)}
    except Exception as exc:
        return {
            "status": "infra_error",
            "reason": type(exc).__name__,
            "cleanup_required": any(r["state"] != "removed" for r in resources),
        }
    return {"status": "passed", "resources_cleaned": all(r["state"] == "removed" for r in resources)}
