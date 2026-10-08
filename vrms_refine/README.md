# Evidence refinement

Refinement adds similar meta examples, calibration-head crossfit reliability, internal fusion/calibration selection and selective publication. MIL epoch selection saw meta labels, so head crossfit is not full encoder crossfit.

`analysis_plan.json` freezes candidate ordering, blend grid and calibration/selective thresholds. Preparation copies it to the new output before API requests. Freeze a new plan before a new study, not after viewing outer outcomes.

```bash
# Needs completed pilot/cloud/DeepSeek/weight-audit artifacts:
python -m vrms_refine.prepare
python -m vrms_refine.validate
# Actual requests:
python -m vrms_refine.run --vendor gpt --workers 4
python -m vrms_refine.run --vendor deepseek --workers 4
# Cached evaluation; fits only small internal heads:
python -m vrms_refine.evaluate
python -m vrms_refine.export_replies
```

Default outputs: `outputs/vrms_refine/20261008_seed2026`. Preparation refuses existing output directories. Runners reuse matching logs and retain failure history. Historical plans/model guards require `gpt-6.1-sol` and `deepseek-flash`; a new model requires a new predeclared experiment and updated guards.

Historical requests completed for both vendors, but accuracy/calibration interventions did not pass promotion gates. Keep coverage separate from selective accuracy. The historical data-completeness problem also applies; see [protocol](../docs/PROTOCOL.md).
