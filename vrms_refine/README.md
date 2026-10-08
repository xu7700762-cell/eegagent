# V2 case evidence

Case RAG retrieves the genuine distance Top 5, with at most one case per internal meta subject and no class quotas. The query fingerprint uses raw deep/temporal summaries and QC, so requesting cases never requires running the current path's Biomarker or Covariance tools. Distances use an internal-bank scaler. Missing shared measurements remain unavailable.

`CaseRetriever` returns distances, actual high/low counts, distinct subjects, mixed evidence and retrieval quality. These counts are descriptive neighborhood evidence, not calibrated symptom probabilities. It chooses a distance cutoff from meta-subject LOSO retrieval predictions, refitting the scaler without each heldout subject. No cutoff is invented if the predeclared accuracy/coverage gates fail. Nested meta-subject LOSO evaluates range selection as well as retrieval without the query subject; the frozen encoder itself is not retrained in this audit.

The former up-to-4-high plus 4-low similarity policy remains `balanced_few_shot_control`. It uses the same fingerprint so the comparison isolates label quotas and subject caps. Forced balance can produce tied votes by design; the vote comparison is a retrieval diagnostic, not a claim about LLM usefulness. Explanation quality and final Supervisor outcomes need their own experiment.

```bash
# Needs corrected V2 cloud deep/QC snapshots; no training or API calls:
python -m vrms_refine.prepare --cloud-out outputs/vrms_cloud/v2_seed2026
python -m vrms_refine.validate
# Actual requests:
python -m vrms_refine.run --vendor gpt --workers 4
python -m vrms_refine.run --vendor deepseek --workers 4
# Cached retrieval/coverage audit; LLM outputs never enter probability fusion:
python -m vrms_refine.evaluate
python -m vrms_refine.export_replies
```

Default outputs: `outputs/vrms_refine/v2_seed2026`. Preparation requires `uncertainty_agent_v2` sources and refuses existing outputs. Source hashes and data-dependent request counts are frozen before calls. Runtime tools are acquired lazily through `ToolReplay`; the preparation CLI is a separate offline comparison of case prompts.

`run` stores supporting/conflicting/missing evidence and explanations. `evaluate` reports per-fold nested retrieval diagnostics and API coverage, including a `not_run` status before APIs exist. It cannot promote an LLM score or mix one with the deep probability. Historical probability-fusion evaluation requires the historical Git version and its preserved outputs. V2 code and synthetic tests do not establish a new numerical benefit.
