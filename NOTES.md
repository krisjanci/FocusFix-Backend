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

## Random Forest Regression

## Support Vector Regression


# Questions

- Would hard coding a path in my public repository pose a security risk to my private data?
- https://numpy.org/ - machine learning library suggested by issei
- Why do I need these: 

from sklearn.base import clone # makes a clean copy of a model

from sklearn.pipeline import make_pipeline # connects scaling and a model into one unit

from sklearn.preprocessing import StandardScaler # puts differently sized features on comparable scales