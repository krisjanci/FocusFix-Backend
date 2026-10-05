from pathlib import Path   # to find files

import pandas   # to work with tables
import joblib   # to save and load my trained models

from sklearn.base import clone # makes a clean copy of a model
from sklearn.linear_model import LinearRegression, Ridge # imports the Linear and Ridge models
from sklearn.ensemble import RandomForestRegressor # imports Random Forrest model
from sklearn.svm import SVR # imports the support vector model
from sklearn.pipeline import make_pipeline # connects scaling and a model into one unit
from sklearn.preprocessing import StandardScaler # puts differently sized features on comparable scales
from sklearn.dummy import DummyRegressor   # imports a prediction tool for my model to compete against
from sklearn.model_selection import LeaveOneGroupOut   # tool for seperating training data and testing data
from sklearn.metrics import mean_absolute_error   # tool for measuring how far my preediction is from reality on average


""" Hard coding a path to my data """
backend_folder = Path(__file__).resolve().parent
workplace_folder = backend_folder.parent

data_file = (
    workplace_folder
    /"DATA-(not_for_Github)"
    /"swell.xlsx"
)

print("Looking for data at:", data_file)
print("Does the file exist?", data_file.exists())


""" Loading the spreadsheet data """
data = pandas.read_excel(
    data_file,
    sheet_name = "SWELLdata",
)

print("Rows and columns:", data.shape)
print(data.head())


""" Choosing cues and target """
features = [
    "SnKeyStrokes",
    "SnErrorKeys",
    "SnLeftClicked",
    "SnAppChange"
    ]

target = "NasaTLX"

print(data[features + [target]].head(10))


""" Clean data """
work = data[
    data["Condition"].isin(["N","I","T"]) # N = neutral, I = interaptions, T = time pressure
].copy()

required_columns = [
    "PP",
    "Blok",
    "Condition",
] + features + [target]

print("Missing values:")
print(work[required_columns].isna().sum())

clean = work.dropna(
    subset = required_columns
).copy()

print("Work rows before cleaning:", len(work))
print("Work rows after cleaning:", len(clean))

""" Check if each block's rating is consistent """
group_columns = ["PP", "Blok", "Condition"]

rating_counts = (
    work.groupby(group_columns)[target].nunique()
)

if (rating_counts > 1).any():
    raise ValueError(
        "A block contains a different workload ratings. Inspect the data."
    )


""" Building block table """
activity = (
    clean.groupby(group_columns)[features]
    .mean()
    )
ratings = (
    clean.groupby(group_columns)[target]
    .first()
)
minutes = (
    clean.groupby(group_columns)
    .size()
    .rename("minutes_used")
)
blocks = (
    activity
    .join(ratings)
    .join(minutes)
    .reset_index()
)

print(blocks.head())
print("Work blocks:", len(blocks))
print("Participants:", blocks["PP"].nunique())


""" Seperate cues from answers """
X = blocks[features] # activity inputs
y = blocks[target] # worload answers


""" Always leaving out one person to predict workload """
splitter = LeaveOneGroupOut()


""" Models available for testing """
models = {
    "linear_regression": LinearRegression(),
    "ridge_regression": make_pipeline( # creates an ordered process
        StandardScaler(), # rescales input so they're comparable
        Ridge(alpha=1.0) # alpha controls how strongly Ridge prevents extreme coefficients
    )
}


""" Prepare a place to store every models prediction """
results = blocks[
    group_columns + [target]
].copy()

results["baseline"] = float("nan")

for model_name in models:
    results[model_name] = float("nan")


""" Test every model on the missing person"""
for train_rows, test_rows in splitter.split(
    X,
    y,
    groups=blocks["PP"]
):
    X_train = X.iloc[train_rows]
    X_test = X.iloc[test_rows]

    y_train = y.iloc[train_rows]

    # test the mean baseline
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    results.loc[
        test_rows,
        "baseline",
    ] = baseline_predictions

    # test every ML model in the dictionary 
    for model_name, model_template in models.items():
        model = clone(model_template)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        results.loc[
            test_rows,
            model_name,
        ] = predictions


""" Evaluate every model by measuring its mistakes """
prediction_columns = ["baseline"] + list(models.keys())

if results[prediction_columns].isna().any().any():
    raise ValueError("Some predictions are missing.")

baseline_error = mean_absolute_error(
    results[target],
    results["baseline"]
)

metrics_rows = [
    {
        "model": "dummy_mean_baseline",
        "participants": blocks["PP"].nunique(),
        "work_blocks": len(blocks),
        "mae": baseline_error,
        "baseline_mae": baseline_error,
        "beat_baseline": False
    }
]

print()
print("Model Comparison")
print("----------------")
print("Baseline MAE:", round(baseline_error, 3))

for model_name in models:
    model_error = mean_absolute_error(
        results[target],
        results[model_name]
    )

    beat_baseline = model_error < baseline_error

    metrics_rows.append(
        {
            "model": model_name,
            "participants": blocks["PP"].nunique(),
            "work_blocks": len(blocks),
            "mae": model_error,
            "baseline_mae": baseline_error,
            "beat_baseline": beat_baseline
        }
    )

print(
    model_name,
    "MAE:",
    round(model_error,3),
    "| Beat baseline:",
    beat_baseline
)

metrics = pandas.DataFrame(metrics_rows)


""" Save experiment results """
artifacts_folder = backend_folder / "artifacts"
metrics_folder = artifacts_folder / "metrics"
models_folder = artifacts_folder / "models"
predictions_folder = artifacts_folder / "predictions"

# Held out predictions
results.to_csv(
    predictions_folder / "held_out_predictions.csv",
    index=False
)

# Model comparison
metrics.to_csv(
    metrics_folder / "model_comparison.csv",
    index=False
)

# train final versions

for model_name, model_template in models.items():
    final_model = clone(model_template)
    final_model.fit(X,y)

    joblib.dump(
        final_model,
        models_folder / f"{model_name}.joblib"
    )

print("The comparison results and final models were saved.")