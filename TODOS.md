# Backend task status

## Delivered

- [x] Catalog relevant research datasets and explain compatibility in `DATA_CATALOG.md` (screened shortlist, not an exhaustive worldwide inventory).
- [x] Define the ideal app training table and private raw/derived/run organization.
- [x] Read app session folders and research ZIPs with validation and exclusion audits.
- [x] Keep v1 workload labels separate from v2 fatigue/productivity ratings.
- [x] Make training configurable by method, input path, feature window and evaluation group.
- [x] Compare linear, ridge, shallow tree, random forest and gradient boosting against the average benchmark.
- [x] Export all app candidates and benchmark models to Core ML and verify native predictions.
- [x] Provide a Swift reference loader and document the exact app feature contract.
- [x] Explain ten algorithms with pros, cons and decisions in `MODEL_GUIDE.md` and the delivered comparison workbook.
- [x] Save model comparisons as CSV, with actual/synthetic/SWELL results documented in `VALIDATION.md` and the workbook.
- [x] Explain when the mean is useful and possible future median/personal-history benchmarks.
- [x] Test the complete workflow using synthetic app-format recordings.
- [x] Run the legacy SWELL experiment and a small pilot on actual app recordings.
- [x] Add native macOS automated tests and CI configuration.

## Later study and app work

- Collect enough varied participant recordings to measure real predictive value. Current five eligible local answers do not support deployment claims.
- Implement the documented rolling feature extraction in the app, then integrate a validated model and test prediction behavior on-device.
- Reassess additional datasets or methods only if their features/labels match a concrete research question.
