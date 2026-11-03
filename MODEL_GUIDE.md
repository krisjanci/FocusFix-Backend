# Ten methods considered for FocusFix

These are practical judgments for small tables of activity summaries paired with 0–10 self-reports, not a universal ranking. The app pipeline compares five learned methods and the mean benchmark. Additional methods are considered here rather than adding complexity before data supports it.

| Method | How it works | Advantages | Limitations | Decision |
| --- | --- | --- | --- | --- |
| Linear regression | Adds weighted input signals | Fast, tiny export, understandable | Misses complicated relationships; correlated signals can destabilize weights | Train and export |
| Ridge regression | Penalizes very large linear weights | Useful with small datasets and correlated activity rates | Still mostly linear; penalty chosen in advance | Train and export |
| Lasso regression | Pushes some linear weights to zero | Can remove unhelpful inputs | Can choose inconsistently between correlated signals | Consider later if many more features are added |
| Elastic net | Combines ridge and lasso penalties | Can balance stability and feature selection | Adds tuning choices without clear benefit for ten inputs | Consider later |
| Decision tree | Splits data using simple questions | Finds thresholds; easy to inspect when shallow | Unstable with small samples; can memorize | Train shallow trees and export |
| Random forest | Averages many independently fitted trees | Captures input combinations; averaging improves stability | Larger artifact; cannot extrapolate beyond learned outcomes | Train depth-limited forest and export |
| Gradient boosting | Adds trees to correct prior errors | Strong candidate for tabular nonlinear patterns | Can overfit; settings need restraint | Train small, slow-learning trees and export |
| Support vector regression | Fits a smooth function with a tolerance for small errors | Useful nonlinear comparison on small datasets | Sensitive to scaling and tuning; separate export work | Retained in legacy SWELL experiment; not in app candidate set |
| k-nearest-neighbor regression | Averages ratings from similar examples | Simple explanation; little training cost | Retains study examples; sensitive to scaling and irrelevant inputs | Not selected for app deployment |
| Neural network regression | Learns multiple layers of combinations | Flexible when there are many representative examples | Needs more data and tuning; harder to interpret | Defer until data justifies it |

The mean benchmark is not counted among these ten learned methods. It predicts the average training rating and is essential: if learned activity patterns do not improve held-out error, the benchmark remains selected. Neither benchmark nor learned models automatically qualify for use in the app.

The current mean benchmark is a reasonable population reference for testing new participants. A median predictor is another useful future MAE benchmark. For personal forecasting, a prior-rating or personal-history benchmark would be useful, but needs a separate time-ordered protocol and sufficient earlier ratings. Do not compute a personal average using a held-out person's future answers.

Model settings are fixed before the final test. The comparison uses grouped development validation; only the selected model and mean benchmark are scored on the final test. This limits selecting a model because it happened to match the test answers. The same logic is described in [scikit-learn's cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).


## Separate unsupervised experiment

The schema-11 activity experiment starts with standardized K-means, k=2–4, selected
on development silhouette only. It uses seven observed input rates and retains
labels/ratings for evaluation, not inputs. See [CLUSTERING.md](CLUSTERING.md) and the
machine-readable feature contract. Cluster structure/stability, independent fatigue
prediction and intervention benefit are separate questions. Generated results do
not establish any of those claims for real users.
