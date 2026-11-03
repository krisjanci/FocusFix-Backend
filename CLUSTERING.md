# Separate activity-clustering experiment

I keep this separate from `train.py`, the original supervised work-only ratings
pipeline and SWELL. It reads schema 11 only. The default CLI accepts explicitly
marked generated fixtures, with no default participant folder or service access.
The non-synthetic override is a separate deliberate future analysis action, not
permission to use real data in this implementation task.

```sh
uv run --python 3.13 python activity_clustering.py demo --output /tmp/my-new-clustering-demo
uv run --python 3.13 python activity_clustering.py fit /tmp/my-new-clustering-demo/recordings \
  --manifest /tmp/my-new-clustering-demo/participants.csv --mode personal \
  --output /tmp/my-new-personal-demo
```

Use new/empty output folders. `demo` is reproducible and clearly synthetic. Personal
mode fits each participant separately using earlier days and holds out the last day.
Population mode holds out the last sorted participant pseudonyms (25%, at least one),
requiring at least four people as a technical minimum, not statistical sufficiency.
Use a stable private manifest pseudonym across all of a person's sessions/days.

## Windows and inputs

[ACTIVITY-FEATURE-CONTRACT.json](ACTIVITY-FEATURE-CONTRACT.json) is the exact versioned
contract. Five-minute windows align to the logical recording grid and never overlap.
Partial file-boundary windows are excluded, rather than joining across a gap or
inventing coverage. Every window needs all 300 observed normal-phase seconds from
nonoverlapping mouse bins, including observed stationary bins. Check-in interaction,
awaiting return, explicit breaks and missing coverage exclude the window. Recovered
raw tails after the durable checkpoint are ignored. No observations means no window,
not low activity. Strict full coverage is an engineering default, not an approved
study criterion; it may exclude many real recordings and needs prospective review.

Seven input-rate features start the experiment. They use counts/timing exposure,
not app names, letter identities or trajectories. Labels, ratings, participant IDs,
time of day, elapsed/prior work and future break information are metadata only.
`activity_phase=work` means normal intervention phase; context comes exclusively
from validated context intervals. Mixed work/non-work/unknown fractions are retained
for evaluation. Missing key-hold measures are not imputed because hold means are
not part of this first feature contract. There is no coverage imputation.

## Fitting and outputs

Scaling fits development data only. K-means candidates are predeclared as 2–4;
development silhouette selects k, with smaller k winning ties. Held-out windows do
not fit scaling, clusters, candidate selection or personal calibration. The saved
model remains that development fit, rather than being refit on final test data.
Constant individual features are harmless; entirely constant/insufficient data
produce `insufficient_evidence` and no model. No large search or fatigue inference.

Outputs include ingestion/exclusion audit, feature windows, development/test
assignments, fitted scaler/model (`clustering.joblib`, trusted local loading only),
portable numeric `model.json`, descriptive cluster profiles, constant-feature and
bootstrap stability diagnostics. IDs are deterministic within a model version;
retraining does not establish stable meanings across versions. Profiles say things
like low observed input and frequent typing, never fatigued/rested. Stability is a
development structural diagnostic, not held-out fatigue accuracy or intervention
benefit. Task/device/participant confounding remains; normalization cannot prove it
is removed. Secure Input/mouse sensitivity and capture limitations remain relevant.

API references: [StandardScaler](https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.StandardScaler.html)
and [KMeans](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html).
These describe algorithms, not evidence that this application predicts fatigue.

## Independent validation (separate from structural clustering)

```sh
uv run --python 3.13 python activity_validation.py demo \
  --cluster-run /tmp/my-new-clustering-demo/population --output /tmp/my-new-validation
```

The default generated demo can legitimately report `insufficient_evidence`: two
held-out participants do not meet the predeclared three-person technical gate.
The evaluator compares a development-rating mean, fixed Ridge context/UTC-time/
known-prior-declared-work baseline, and that baseline plus frozen cluster IDs.
Later-time personal evaluation adds strictly earlier personal ratings with an
explicit missing-history indicator. New-person evaluation never uses held-out
people's ratings as personal calibration. Predictors stop before the request;
future answer interaction or later label corrections cannot enter prior-work inputs.

The real-data adapter requires schema-11 independent study prompt snapshots, a
stable manifest and an exact reviewed schedule version. Legacy work-only ratings
are not silently treated as independent cross-context measurements. Skips/missing
answers remain missing. Whole preceding cluster windows must fit inside the prompt's
recall bounds; misaligned/short windows are excluded and reported rather than
extrapolated. Timing/recall-grid compatibility therefore needs prospective review.

Outputs include joining exclusions, context-specific missingness, sample counts,
eligible ratings and, only after technical gates pass, held-out comparisons and
participant-resampled uncertainty where feasible. A single-person comparison cannot
provide participant-level uncertainty. The gates are engineering minima, not a
power calculation or sufficient statistical evidence. No candidates are chosen on
final-test performance. Productivity is not evaluated here: neither input nor
declared minutes establishes quality-adjusted output. No intervention benefit has
been tested. See the private planning draft for unresolved protocol/reader decisions.
