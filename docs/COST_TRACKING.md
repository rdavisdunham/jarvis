# AI cost tracking

Tracking is independent of spending-limit enforcement. Development uses
JARVIS_COST_TRACKING_ENABLED=true and JARVIS_BUDGET_ENFORCEMENT_ENABLED=false.
Old uncertain reservations stay visible but cannot pause Eri in this mode.
Enabling enforcement restores monthly limits and optional-work thresholds.

Settings → System → Model usage shows rolling last-7-day and last-30-day totals
by feature, plus a calendar-month total for budgets. Costs come from reported
tokens or GPT-Live session seconds. Each Usage row stores tokens.feature;
concurrent feature contexts stay independent. Memory queries and extraction,
note organization, field/rule learning and search indexing have separate buckets.
Local deterministic work adds no AI charge. CPU/database/hosting costs are excluded.

Set JARVIS_COST_TRACKING_SINCE to an explicit UTC timestamp when recording resumes
after a gap. Incomplete periods are labeled and monthly projections are suppressed.
No historical charges are invented. Earlier usage without attribution stays in
an unclassified bucket. A future user-facing tracking toggle must update this marker.

Usage is an estimate, not a provider invoice. Credits, taxes, infrastructure and
isolated eval runs are excluded. Missing usage remains uncertain. Known usage
from an incomplete answer is recorded; repeated usage IDs cannot double-charge.

Prices verified September 19, 2026:
- Luna: $0.20/M input, $0.02/M cached input, $0.25/M cache writes, $1.20/M output.
  Above 272K input, input rates double and output is 1.5 times.
  https://developers.openai.com/api/docs/models/gpt-5.6-luna
- Gemini 3.8 Flash: $0.75/M input and $3.75/M output through 2026, then $1.50/$7.50.
  Estimates conservatively do not deduct Gemini caching.
  https://ai.google.dev/gemini-api/docs/latest-model
- GPT-Live 1: $0.05/minute, billed by the second, plus backend usage.
  https://developers.openai.com/api/docs/models/gpt-live-1
- text-embedding-3-small: $0.02/M tokens, unchanged configured rate.

Paid evals use disposable synthetic databases with app accounting disabled.
Their separate spending.json journal and committed summary keep eval cost out of
normal usage. The September 19 20-case run cost about $0.52 combined. The full
acceptance catalog is not a claim every feature has been paid-evaluated.

Validated through isolated budget tests, concurrent attribution, incomplete and
missing usage, frontend checks and phone/fold/desktop browser coverage.
