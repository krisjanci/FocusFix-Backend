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
