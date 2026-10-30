# Data sources and suitability

Reviewed October 30, 2026. This is a screened shortlist, not an exhaustive inventory of every computer-behavior dataset. External datasets have not been downloaded or pooled into FocusFix training. Access and reuse conditions must be checked with each provider before obtaining data.

| Source | Recorded signals and labels | Fit for FocusFix | Handling |
| --- | --- | --- | --- |
| FocusFix recordings | Generic keyboard/click/switch/scroll events, mouse summaries, phases and optional fatigue/productivity answers | Best alignment with the actual app and target questions | Local session folders are the raw source. Use a private participant manifest and the versioned ingestion pipeline |
| [SWELL Knowledge Work](https://www.cs.ru.nl/~skoldijk/Papers/ICMI%202014%20paper_final_cr.pdf) | Computer interaction plus other modalities and workload/stress measures in knowledge-work conditions | Useful research comparison; NASA-TLX is a different target from the app ratings | Existing `swell.xlsx` stays in the sibling `DATA-(not_for_Github)` directory. `train_swell.py` keeps the experiment separate |
| [CMU Keystroke Dynamics Benchmark](https://www.cs.cmu.edu/~keystroke/) | Password typing timing and participant identities | Useful for understanding timing and person-specific patterns; lacks the required fatigue/productivity labels | Research reference only, not a supervised FocusFix training source |
| [KEasyLogger](https://zenodo.org/records/6535004) | Labeled desktop free-text typing behavior | Natural typing is relevant to input measurement; its labels must not be assumed to mean fatigue | Inspect protocol and available labels before proposing any new task |
| [MEU-Mobile KSD](https://archive-beta.ics.uci.edu/dataset/399/meu%2Bmobile%2Bksd) | Touch-device keystroke dynamics for authentication | Device and task differ from macOS work; identity is not a fatigue label | Not selected for app prediction training |
| [WESAD](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf) | Wearable physiological signals for stress/affect research | App does not collect matching sensor inputs; stress is a separate target | Background research only |
| [BB-MAS](https://arxiv.org/abs/1912.02736) | Typing, gait and swipe data across desktop/tablet/phone | Useful for studying device differences; biometric context differs from app check-ins | Background research only |

Suitability is our assessment based on the linked original dataset descriptions/papers. A large dataset with different inputs and labels does not automatically improve this app's prediction task. Do not manufacture fatigue labels from authentication identity, stress conditions, or mouse inactivity.

## Ideal study table

One row represents the observed work window immediately before an answered check-in. Keep stable pseudonymous participant and session IDs for splitting, the ten versioned activity features, and separate fatigue/productivity answers. Keep phase/timing information in original recordings so cleaning can be audited. Treat missing answers as missing and distinguish synthetic examples from participant recordings. Schema changes need a deliberate compatibility review.

## Private organization

- Raw FocusFix data: original `FocusFix/Sessions` folders in Application Support, or a separately received export folder/ZIP. The backend reads originals without changing them.
- Existing external raw data: sibling `DATA-(not_for_Github)/swell.xlsx`.
- Study identity mapping: private `participants.csv`, outside tracked source.
- Derived windows: ignored `artifacts/<iteration>/training_windows.csv`.
- Models, metrics and exclusions: ignored `artifacts/<new-run-name>/`.

Use unique run folders rather than overwriting an experiment. Keep participant data out of Git. A path in source code does not publish the underlying file, but a hard-coded personal path is brittle; commands now accept explicit paths instead.
