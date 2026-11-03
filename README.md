## Separate automatic-recording clustering experiment

[CLUSTERING.md](CLUSTERING.md) documents the independent schema-11, generated-data
pipeline. It builds full-coverage five-minute windows, fits development-only
standardized K-means, and keeps activity labels/ratings out of clustering inputs.
Unseen-person and personal later-day models are separate. No cluster is named or
validated as fatigued/rested. The existing supervised pipeline below is unchanged
and remains work-only, supporting its original schema contract.

# FocusFix Backend

Train experimental models from FocusFix recordings and export Apple **Core ML** `.mlmodel` files for later use in the app. All processing runs locally. Nothing uploads recordings, contacts Supabase, or modifies the app.

## Setup

Use Python **3.13** for the native macOS Core ML runtime. The repository includes a dependency lockfile.

```sh
uv sync --python 3.13
uv run pytest -q
```

On macOS tests compile and run every exported model and compare it with Python. Compilation needs access to macOS's temporary compiler directory. On other systems the exporter saves model specifications but reports `runtime_verified: false`; perform the macOS tests before app integration.

## Process your recordings

The default local recording folder is `~/Library/Application Support/FocusFix/Sessions`. You can also supply one exported session folder, a folder containing many sessions, or a research export ZIP. ZIP paths are validated and files are unpacked into a temporary directory.

For recordings from one person:

```sh
uv run python train.py prepare "$HOME/Library/Application Support/FocusFix/Sessions" \
  --participant-id local-owner --skip-invalid --output artifacts/prepared-01
```

`training_windows.csv` contains one row per eligible answered work check-in. `ingestion.json` explains exclusions. `--skip-invalid` explicitly allows older, unfinished or malformed sessions to be excluded; without it ingestion stops on the first invalid session. Use a fresh output directory for each iteration. Recordings and generated artifacts are ignored by Git.

For multiple people, create a **private** CSV mapping every session to the same pseudonym for its participant, even across different days. Do not use a new participant ID for every session.

```csv
session_id,participant_id
session-uuid-1,person-001
session-uuid-2,person-001
session-uuid-3,person-002
```

```sh
uv run python train.py train /path/to/recordings \
  --manifest /private/path/participants.csv --skip-invalid --output artifacts/study-01
```

At least four participants with eligible labels are required: some remain untouched for testing, and at least three development participants allow grouped model comparison. Four is a technical minimum, **not sufficient evidence of useful accuracy**. More people and many varied work sessions are needed.

For a personal experiment, `--participant-id local-owner --group-by session` evaluates different sessions from the same person, requiring at least four eligible sessions. This does not establish accuracy for other people or future sessions. The split is by session, not chronological.

Optional controls:

```sh
uv run python train.py train /path/to/recordings --manifest /private/path/participants.csv \
  --models ridge random_forest --window-seconds 300 --output artifacts/study-02
# Add --no-coreml to train without exporting.
```

## What the models learn

Two independent models predict the app's **self-reported mental fatigue** and **self-reported productivity**, each on its own 0–10 scale. These are neither NASA-TLX workload scores nor objective measures of productivity. They do not establish when taking a break will help.

We compare five learned approaches and a benchmark:

| Model | Simple explanation | Why include it? |
| --- | --- | --- |
| Average benchmark | Always predicts the training set's average rating | Checks whether activity adds value at all |
| Linear regression | Adds up weighted activity signals | Small, fast, easy to understand |
| Ridge regression | Linear regression with limits on extreme weights | A sensible starting point for small, correlated datasets |
| Small decision tree | Asks a short sequence of questions | Finds simple thresholds and combinations |
| Random forest | Averages many small trees | Learns more complicated patterns with less dependence on a single tree |
| Gradient boosting | Adds small trees that correct previous mistakes | Another compact approach to nonlinear patterns |

We select the lowest development cross-validation mean absolute error (MAE); the benchmark wins exact ties. MAE of 1.5 means predictions are about 1.5 rating points away from recorded answers on average. Each development fold keeps a person's sessions together. Scaling is fitted inside each fold. The selected candidate and benchmark are then evaluated on untouched people. **A candidate can still fail that final test.** Do not choose another candidate by inspecting the final test results; use a new study or fresh test data for another iteration.

Final models are refitted on all data after evaluation. Their training fit is not reported as evidence of accuracy. `deployment_ready` stays false: deployment is a separate decision requiring enough representative data and app testing. Neural networks and automatic large model searches are premature for this small tabular dataset; they increase complexity without evidence of benefit. See [MODEL_GUIDE.md](MODEL_GUIDE.md) for ten methods and tradeoffs.

## Feature contract for the app

Feature version: `focusfix_work_window_v1`. The default observation window is the **300 seconds before a check-in is shown**, ending before the check-in itself. Its size is embedded in each model. Ratings, answer duration, survey context, future break activity and participant IDs are never model inputs.

Only complete mouse bins starting at or after `max(0, cutoff - window)` and ending at or before `cutoff` with `activity_phase == work` count as observed work. Include stationary bins. Exclude `check_in`, `awaiting_return`, planned breaks and missing coverage. Require **30 observed work seconds**. Include an event only if it is marked work and falls inside an observed bin `[start, end)`. Events at the cutoff are excluded. This prevents gaps from inflating activity rates. The Python reference is `window_features` in `focusfix_data.py`.

Use this exact feature order in the `features` MLMultiArray:

| Index | Name | Calculation |
| --- | --- | --- |
| 0 | elapsed_minutes | cutoff / 60 |
| 1 | work_fraction | observed work seconds / (cutoff - window start) |
| 2 | keys_per_minute | (key_down + delete_down) count × 60 / observed work seconds; includes repeats |
| 3 | deletes_per_minute | delete_down count × 60 / observed work seconds |
| 4 | clicks_per_minute | left/right/other click_down count × 60 / observed work seconds |
| 5 | switches_per_minute | app_switch count × 60 / observed work seconds |
| 6 | scrolls_per_minute | scroll event count × 60 / observed work seconds |
| 7 | mouse_points_per_minute | Sum of bin distances × 60 / observed work seconds |
| 8 | mouse_events_per_minute | Sum of bin movement_events × 60 / observed work seconds |
| 9 | mean_key_hold_seconds | Mean nonblank hold duration on key_up/delete_up events; 0 when none |

Round every feature to float32, then store it as a Double in the array. The output `raw_rating` must be clipped to `[0, 10]`, as in evaluation. Scalers and learned parameters are already part of exported models; do not scale inputs again. Missing or insufficient recording coverage means **no prediction**, not a zero rating. No-input work may still be meaningful work; Secure Input, device settings and mouse sensitivity can change measurements.

Supported recordings are schemas **5–8** with `focusfix_check_in_v2` and the matching activity columns. Version 1 workload responses are rejected rather than relabelled as productivity. Session-start ratings have no preceding activity and are excluded. Skipped/cancelled responses remain missing. Invalid ranges, duplicate check-ins, overlapping mouse bins, unsupported phases and inconsistent session IDs are rejected. A one-microsecond tolerance accounts for app CSV timestamp rounding.

## Outputs and Core ML integration

A training run writes:

- `report.json`: selected model for each target, benchmark comparisons, held-out test errors, feature contract, provenance and export verification.
- `model_comparison.csv`: development results for every candidate, suitable for opening in Numbers or Excel.
- `held_out_predictions.csv` and `split_assignments.csv`: auditable evaluation rows and development/test membership. These contain private study identifiers; keep them local.
- One `.joblib` and `.mlmodel` per target and candidate, including the benchmark. Load only trusted joblib files.
- `ingestion.json`: included/excluded session audit.

Use the selected model's filename listed in `report.json`. Add it to the Xcode target and review [examples/FocusFixPredictor.swift](examples/FocusFixPredictor.swift). The helper checks feature order/version, expected window, target and synthetic provenance, runs inference, and clips outputs. It does not collect features or enable predictions in the current app; that future integration must reproduce the feature contract above. It rejects synthetic models by default. Test with `allowSynthetic: true` only for an explicit demo.

Exports use Core ML's public linear and tree model formats, avoiding dependence on an aging sklearn converter. Each export carries feature metadata. On macOS it is reloaded and compared against Python on up to 128 examples. The test suite also compares all six model families on varied synthetic inputs. This verifies conversion, not real-world predictive value.

## End-to-end demo and legacy SWELL experiment

```sh
uv run python train.py demo --output artifacts/demo-01
uv run python train_swell.py '../DATA-(not_for_Github)/swell.xlsx' --output artifacts/swell-01
```

The demo creates deterministic synthetic app-format recordings, compares models and exports twelve verified Core ML files. All demo models are marked synthetic. Good demo scores only prove that the pipeline can recover an invented relationship.

SWELL remains a separate workload experiment with participant-held-out evaluation and four original methods. It cannot be pooled with FocusFix labels or used as a fatigue/productivity model. Its models remain Python artifacts because the app has different feature definitions and labels.

See [DATA_CATALOG.md](DATA_CATALOG.md) for candidate research data and [VALIDATION.md](VALIDATION.md) for the verified runs and current data limits.

## Independent cluster evaluation and safe integration boundary

`activity_validation.py` evaluates independent prompt-linked fatigue ratings against
mean and context/time/prior-work baselines, with new-person and later-time protocols
kept separate. It reports insufficient evidence for small samples and never equates
cluster stability, fatigue prediction and intervention benefit. See [CLUSTERING.md](CLUSTERING.md).
The app has only a disabled synthetic preview provider with exact seven-rate parity;
its shipped timer remains active. No model, randomization or real collection is enabled.
