"""Preflight dollar ceilings for synthetic text-only eval requests, including retries."""
import json
from decimal import Decimal


class EvalLimit(RuntimeError):
    pass


class SpendingMeter:
    def __init__(self, limit, path):
        self.limit = Decimal(str(limit))
        if not self.limit.is_finite() or not 0 < self.limit <= 10:
            raise ValueError("Per-model cap must be greater than zero and at most $10")
        self.path = path
        self.calls = []

    def total(self, profile):
        return sum((Decimal(c["charged_bound_usd"]) for c in self.calls if c["profile"] == profile), Decimal(0))

    def save(self):
        self.path.write_text(json.dumps({
            "per_model_cap_usd": str(self.limit),
            "models": {p: {
                "estimated_usd": str(sum((Decimal(c.get("estimated_usd", "0")) for c in self.calls if c["profile"] == p), Decimal(0))),
                "including_uncertain_usd": str(self.total(p)),
                "requests": sum(c["profile"] == p for c in self.calls),
            } for p in sorted({c["profile"] for c in self.calls})},
            "calls": self.calls,
        }, indent=2))

    def reserve(self, agent, request):
        payload = json.loads(request.content)
        output = payload.get("max_output_tokens", payload.get("max_completion_tokens", payload.get("max_tokens")))
        if output != agent.max_output_tokens or payload.get("model") != agent.model:
            raise EvalLimit("Unexpected model or output bound")
        # Text-only payload. UTF-8 bytes + transport/framing allowance overbound token count,
        # including tools and encrypted reasoning replay; long-context premiums are included.
        bound = Decimal(str(agent.reserve_cost(len(request.content) + 4096)))
        if self.total(agent.profile_id) + bound > self.limit:
            raise EvalLimit("Per-model dollar ceiling reached")
        row = {"profile": agent.profile_id, "model": agent.model, "charged_bound_usd": str(bound),
               "state": "unconfirmed", "bound_usd": str(bound)}
        self.calls.append(row)
        self.save()  # Persist the reservation before any network traffic.
        return row

    def settle(self, row, agent, response):
        try:
            data = response.json()
            usage = data["usage"]
            if agent.api == "responses":
                usage = {
                    "prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"],
                    "prompt_tokens_details": usage.get("input_tokens_details") or {},
                }
            for key in ("prompt_tokens", "completion_tokens"):
                if type(usage[key]) is not int or usage[key] < 0:
                    raise ValueError("Invalid usage")
            cost = Decimal(str(agent.usage_cost(usage)))
            if not cost.is_finite() or cost < 0:
                raise ValueError("Invalid cost")
        except (ValueError, KeyError, TypeError):
            return  # Missing/error/uncertain responses retain their entire preflight bound.
        row.update(estimated_usd=str(cost), charged_bound_usd=str(cost), state="recorded")
        self.save()
        if cost > Decimal(row["bound_usd"]):
            raise EvalLimit("Provider usage exceeded conservative estimate; stop for investigation")
