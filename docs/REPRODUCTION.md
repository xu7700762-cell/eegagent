# V2 reproduction order

## Offline checks

Install dependencies and the package using `python -m pip install -e ".[cloud,plot]"`. Run the explicit V2 unittest module list in the root README and `python -m examples.assess_path --dry-run`. The Git release also supports `python -m unittest discover -v`; local experimental folders can retain historical modules/caches outside this release. Their frozen-source mismatches must not be bypassed. Synthetic tests do not require data/keys/network or establish scientific model performance.

## New eligible data and models

Use a data root containing raw/ and labels/ as described in PROTOCOL.md. Preserve historical outputs. Defaults are outputs/vrms_pilot/v2_seed2026 and outputs/vrms_cloud/v2_seed2026. Existing cache/training directories are rejected. Set EEG_DATA_ROOT or pass --data-root.

```bash
python -m vrms_pilot.experiment --stage audit --data-root /path/to/data
python -m vrms_pilot.experiment --stage prepare --data-root /path/to/data
# Optional one-fold smoke: this is not a finished experiment
python -m vrms_pilot.experiment --stage smoke
# Full new 24-fold training, only when intended
python -m vrms_pilot.experiment --stage run
python -m vrms_pilot.validate_results --data-root /path/to/data
python -m vrms_cloud.prepare
python -m vrms_cloud.independent --dry-run
python -m vrms_cloud.evaluate --offline
python -m vrms_cloud.validate --offline
```

The cloud pipeline requires full results/ checkpoints. Smoke artifacts live in smoke/ and are not automatically promoted into results/. A fresh V2 model must include the separately fitted probability calibrator and internal gate; V1 checkpoints cannot be reused.

Custom paths: pilot --out /fresh/pilot; cloud.prepare --pilot-out /fresh/pilot --out /fresh/cloud; all cloud downstream commands --out /fresh/cloud. The prepared cloud protocol freezes the pilot path and source/checkpoint hashes. After a code change, create a new protocol/run rather than disabling frozen-source validation.

## Case RAG comparisons without network

```bash
python -m vrms_refine.prepare
python -m vrms_refine.validate
python -m vrms_refine.evaluate
```

Custom paths: refine.prepare --cloud-out /fresh/cloud --out /fresh/refine; subsequent refine commands --out /fresh/refine. This stage derives true Top-K and balanced Few-Shot controls from frozen deep/QC snapshots, records nested internal retrieval LOSO diagnostics, and requires no physiology/LLM calls. It does not prove Top-K is more effective before real outcome analysis.

## Explicit real API runs

Configure project .env or environment values from .env.example. These commands actually call APIs and may incur charges; they were not executed during the V2 code change.

```bash
python -m vrms_cloud.independent --workers 4
python -m vrms_cloud.evaluate
python -m vrms_cloud.validate
python -m vrms_deepseek.experiment --workers 4
python -m vrms_deepseek.evaluate
python -m vrms_refine.run --vendor gpt --workers 4
python -m vrms_refine.run --vendor deepseek --workers 4
python -m vrms_refine.evaluate
python -m vrms_refine.export_replies
```

DeepSeek defaults to outputs/vrms_deepseek/v2_seed2026 and prepares its own frozen deep/QC evidence folder from the same pilot models if absent. --dry-run avoids provider configuration and requests. LLM responses are explanation-only; probability-fusion evaluation and the historical weight scan are excluded from V2. Failed requests retain p_cal and refusal state. Counts come from eligible manifests; no fixed147/1309 request guard is used in V2 entrypoints.

## Historical artifacts

V1 algorithms can be inspected at Git commit205abc7. source_snapshot.json and release_manifest.json describe that release, not the current V2 source hashes. Keep old results for clearly labelled comparisons; V2 requires new full eligibility caches, base models, head fitting, probability calibration and policy gates. No V2 real-data training result or performance improvement is claimed by contract-test success.
