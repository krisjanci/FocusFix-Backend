# Verification and results

Verified locally on macOS, October 30, 2026, using Python 3.13.15, scikit-learn 1.5.2, pandas 2.3.3 and coremltools 9.0. The committed lockfile records dependencies.

## Automated checks

- `python -m pytest -q`: **25 passed**. Tests cover causal feature windows, phase/coverage filtering, missing labels, invalid measurements, schema validation, ZIP ingestion, group separation, benchmark fallback, output overwrite protection, native model conversion and an end-to-end CLI run.
- All six regressor families produce saved Core ML files that are reloaded and executed on macOS. Their raw predictions match Python within the export tolerance (`atol=1e-4`, `rtol=1e-5`).
- The reference Swift inference helper passed `swiftc -typecheck` with CoreML/Foundation.
- The comparison workbook was inspected for values, scanned for formula errors, and rendered for both sheets. It is a snapshot of the runs below and contains no raw participant identifiers.
- A macOS GitHub Actions workflow is committed. Hosted CI has not been run from this chat; commits have not been pushed.

## Actual FocusFix recordings

Read 25 local session folders without changing originals. The compatibility review accepts recording schemas 5–8 with v2 fatigue/productivity questions, records exclusions for older or unfinished sessions, and applies a one-microsecond tolerance for the app's CSV timestamp rounding.

Five answered work check-ins across five sessions have at least 30 observed work seconds. All belong to the same local owner. A personal-session experiment used four development sessions and one untouched test session. This is a pipeline pilot, not a reliable accuracy estimate.

| Target | Selected model | Final test MAE (0–10 rating points) | Mean benchmark MAE |
| --- | --- | --- | --- |
| Fatigue | Mean benchmark | 2.000 | 2.000 |
| Self-reported productivity | Mean benchmark | 1.000 | 1.000 |

Both targets select the benchmark from development validation. There is no demonstrated advantage from learned activity patterns in these data. All twelve real-data experimental Core ML artifacts passed native parity checks; maximum observed export error was less than 0.000001 rating points.

Private outputs are in ignored `artifacts/local-prepared-v2/` and `artifacts/local-experiment-v1/`. The latter contains the full model comparison, evaluation predictions, session exclusion audit, Python models and Core ML files. No recording data or study identifiers were committed.

## Synthetic end-to-end demonstration

Six synthetic participants, two sessions each and four check-ins per session produced 48 examples. Development validation selected ridge for fatigue and linear regression for productivity. Final test MAE was 0.446 for synthetic fatigue and less than 0.000001 for synthetic productivity, compared with mean benchmarks of 1.953 and 2.000.

Those relationships were deliberately invented to exercise learning and export. These numbers are **not evidence about real users**. All twelve synthetic exports passed native Core ML parity checks. Models carry synthetic metadata, which the Swift helper rejects by default. Outputs are in ignored `artifacts/demo-v2/models/`.

## Legacy SWELL workload experiment

The existing private spreadsheet produced 72 work blocks from 24 participants after cleaning. Leave-one-participant-out results:

| Method | NASA-TLX MAE | Beats mean benchmark? |
| --- | --- | --- |
| Mean benchmark | 12.103 | Reference |
| Linear regression | 13.246 | No |
| Ridge regression | 13.210 | No |
| Random forest | 13.203 | No |
| Support vector regression | 12.274 | No |

This reproduces the earlier finding: those four aggregate activity features do not beat the average workload score in this evaluation. SWELL labels and feature definitions differ from FocusFix; they remain a separate experiment. Outputs are in ignored `artifacts/swell-v2/`.

## What remains before app deployment

Collect more varied, labelled work sessions and participants. Keep participant identity consistent across sessions. Run a new experiment with enough untouched data to assess useful error, and reproduce the versioned feature contract in the app before adding the selected `.mlmodel`. A good synthetic score or a successful export does not replace those steps. The backend intentionally reports `deployment_ready: false`.
