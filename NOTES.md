# Notes

## Linear Regression

Tested Linear Regression model and it did not beat a Mean Baseline in predicting NasaTLX scores with the four selected features: Keystrokes, Error-key presses, Left clicks and Application changes.

Model MAE was: 13.246
Baseline MAE was: 12.103

### Failure Hypothesis

Failure to beat the baseline may mean:

- the relationship between the 4 features and NasaTLX scores is non-linear

- four activity features are not sufficient

- NasaTLX score vary widely between people


## Ridge Regression

### Failure Hypothesis 

- NasaTLX score vary widely between people

- I averaged minutes instead of looking at how each person progressed

(Problem) NasaTLX score was per block

## Random Forest Regression

## Support Vector Regression


# Questions

- Would hard coding a path in my public repository pose a security risk to my private data?
- https://numpy.org/ - machine learning library suggested by issei
- Why do I need these: 

from sklearn.base import clone # makes a clean copy of a model

from sklearn.pipeline import make_pipeline # connects scaling and a model into one unit

from sklearn.preprocessing import StandardScaler # puts differently sized features on comparable scales


## meeting notes

auto ML - FLAML (microsoft)

per projects ml 

### mini meeting notes

- check app/server functionality from synthetically generated data
## Backend implementation — October 30, 2026

The app now has its own ingestion/training/export workflow. See README.md for commands, MODEL_GUIDE.md for plain-language decisions, DATA_CATALOG.md for research data suitability, and VALIDATION.md for measured results. The original SWELL experiment remains available as train_swell.py.

Five real local work check-ins were usable. The personal-session pilot selected the mean benchmark for both targets; more labelled data is needed before claiming accuracy. Synthetic data verifies the pipeline separately. Core ML conversion and native inference were checked for every app candidate. Generated recordings, reports and model files remain ignored by Git.

## November 2, 2026 — separate local activity clustering

I added a separate schema-11 clustering command instead of changing my existing
supervised/SWELL pipelines. It uses nonoverlapping five-minute windows with complete
observed normal-phase coverage. Observed stationary bins count; missing coverage,
check-in interaction, breaks, return waiting and partial file-boundary windows do
not become zero activity. Work/non-work/unknown remain evaluation metadata; the old
work phase is never treated as a work label.

I predeclared seven input rates and k=2–4. Scaling, candidate choice and stability
checks use development data only. Population models hold out participants; personal
models use only earlier days and hold out the last day. The saved model stays the
development fit. Cluster IDs are stable within its version, with descriptive names,
not fatigued/rested. Reports include exclusions, splits, profiles, coverage, constant
features and bootstrap diagnostics. Synthetic data is clearly marked and is the
CLI default; I did not open participant data.

Generated end-to-end/leakage tests cover changed held-out values, mixed context,
missing/no mouse observations, explicit breaks/check-ins, constant features, too
little data, invalid schema, half-open events and personal time ordering. I found
the existing .venv used Python 3.14/sklearn 1.9, so I created a separate /tmp Python
3.13 environment from uv.lock (sklearn 1.5.2) for final checks without replacing it.
The generated run is technical feasibility, not fatigue accuracy or intervention
benefit. Device/task/participant confounding still needs independent evaluation.

## November 2, 2026 — independent cluster validation

I added a separate evaluator joining independent prompt IDs/context/recall bounds to
preceding full-coverage windows. It compares a development mean, fixed context/UTC-
time/known-prior-work baseline and that baseline plus cluster IDs. Later-time personal
features use only strictly earlier ratings; new-person predictions do not calibrate
on held-out people's answers. Later history corrections are excluded from prior-work
features when they were not available at the prompt. Missing/skip ratings stay missing.

I predeclared technical sample gates, report missingness by context and use participant
resampling for new-person uncertainty when feasible. These gates do not establish
statistical adequacy. The small six-person generated demo reports insufficient
evidence because it has only two held-out people; it does not print an accuracy claim.
Larger generated tests exercise all three models, later-time ordering and unchanged
training-mean predictions after held-out changes. Productivity evaluation stays separate.

The app's synthetic provider reproduces the seven-rate float32 contract, checks
versions/coverage/model age and falls back to the existing fixed policy for unknown,
missing or ineligible input. It returns descriptive cluster IDs, never fatigue scores.
Experimental selection is off, with no real model or randomization activated. My
private research-plan draft separates sparse independent validation from a later
reviewed randomized comparison; it does not change the course contract or prior IRB
draft. Cluster structure, fatigue prediction and benefit remain three different claims.
