# Background model candidates

Checked September 12, 2026 against official model and pricing documentation.
Selection discussion only: Eridani's configured gpt-5.4-mini agent is unchanged.

Standard text prices, USD per million tokens, uncached input / output:

- GPT-5.4 mini: $0.75 / $4.50. Current proven integration.
  https://developers.openai.com/api/docs/models/gpt-5.4-mini
- GPT-5.6 Luna: $0.20 / $1.20. OpenAI positions it in the nano-like,
  cost-sensitive tier; a newer name does not establish superior accuracy.
  https://developers.openai.com/api/docs/models/gpt-5.6-luna
- GPT-5.6 Terra: $2 / $12. OpenAI positions it in the balanced, mini-like tier.
  https://developers.openai.com/api/docs/models/gpt-5.6-terra
- Gemini 3.8 Flash: $0.75 / $3.75 through December 31, 2026, then $1.50 / $7.50
  starting January 1, 2027. Google lists output as including thinking tokens.
  https://ai.google.dev/gemini-api/docs/pricing
  https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash

Illustration: 1,000 calls using 10,000 uncached input and 1,000 billed output
tokens each cost $12 with mini, $3.20 with Luna, $32 with Terra, or $11.25 with
Flash's current introductory price ($22.50 at its announced standard price).
These are equal-token illustrations, not measured Eridani costs. Reasoning,
retries, tool rounds, caching and provider tokenization change actual cost.
Luna/Terra also have higher rates for prompts above 272K input tokens.

Recommendation: evaluate Luna first for bounded extraction, classification,
summaries and routine task operations. Keep 5.4 mini as the working baseline.
Consider Terra for difficult planning/escalation if measured improvements justify
its cost. Flash is a credible alternative for evaluation, but switching providers
adds integration work and its introductory pricing should not be treated as
permanent.

An evaluation should measure cost per correct completed workflow, not only token
price: exact record resolution, safe multi-task edits, date/time interpretation,
many-to-many goal/project changes, evidence-grounded note extraction, memory
contradiction handling, tool-result accuracy, latency, retries and failures.
Use synthetic or explicitly approved cases and prohibit writes to real external
calendars/issues during comparative tests. No candidate has been benchmarked on
Eridani yet, so this is a proposed evaluation order rather than a proven quality
ranking.
