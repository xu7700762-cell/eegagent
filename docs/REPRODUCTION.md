# Reproduction order

## Offline contracts

Install the package and cloud/plot extras as in the root README. Contract tests and `python examples/assess_path.py --dry-run` need no data, credentials or API calls. The example is synthetic.

## Historical data pipeline

The existing study expects24 subjects,147 candidates and fixed downstream request counts. Read [the event-completeness problem](PROTOCOL.md): a corrected146-path study requires coordinated guards, caches and protocols. This release does not claim corrected results exist.

Set the `EEG_DATA_ROOT` environment variable to the directory containing `raw/` and `labels/`. Run at the repository root. Downstream modules depend on fixed historical output locations: start in a fresh checkout/output tree and preserve completed artifacts. Changing `--out` on one stage does not automatically change other stages.

```bash
# Numerical preparation/training
python -m vrms_pilot.experiment --stage prepare --data-root /path/to/data
python -m vrms_pilot.experiment --stage smoke
python -m vrms_pilot.experiment --stage run
python -m vrms_pilot.validate_results
python -m vrms_pilot.export_window_oof

# Evidence and candidate training
python -m vrms_cloud.prepare
python -m vrms_cloud.improve

# Actual Responses API calls; separate batch/independent contexts
python -m vrms_cloud.cloud --modes zero_shot few_shot improved_few_shot --workers 3
python -m vrms_cloud.independent --workers 4
python -m vrms_cloud.evaluate
python -m vrms_cloud.validate

# Actual DeepSeek calls, followed by cached evaluation
python -m vrms_deepseek.experiment --workers 4
python -m vrms_deepseek.evaluate

# Cache-only scan and protected-source manifest
python -m vrms_weights.scan

# Freeze new requests and the analysis plan
python -m vrms_refine.prepare
python -m vrms_refine.validate
# Actual requests for both vendors
python -m vrms_refine.run --vendor gpt --workers 4
python -m vrms_refine.run --vendor deepseek --workers 4
# Cached evaluation and internal small-head fitting
python -m vrms_refine.evaluate
```

Raw files are read-only. Caches/models use local disk; API runners may incur charges. Smoke, `--limit` and `--limit-folds` are partial runs. Original data/artifacts are excluded from this repo, so these full-data commands were not executed during publication.

## Release scope

Network architectures, losses, features, split/selection logic, adaptive policy, system prompt, schema and fusion rules are retained. Only configuration/path wiring, the standalone client and existing analysis-plan template are repackaged. Historical manuscript-sync scripts are omitted. `source_snapshot.json` records original core hashes; `release_manifest.json` lists release modifications.

Historical artifacts have old source hashes and machine paths. New release sources are intentionally not byte-identical to those records: create a new protocol rather than disabling hash checks. Client packaging tests use mocks and establish no real endpoint connectivity or fresh numerical results.
