from pathlib import Path   # to find files

import pandas   # to work with tables
import joblib   # to save and load my trained models

from sklearn.linear_model import LinearRegression   # imports the Linear ML model
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

results = blocks[
    group_columns + [target]
].copy()

results["prediction"] = float("nan")
results["baseline"] = float("nan")


""" Create ML model and a baseline model to compare against """
for train_rows, test_rows in splitter.split(
    X,
    y,
    groups=blocks("PP"),
):
    X_train = X.iloc[train_rows]
    X_test = X.iloc[test_rows]

    y_train = y.iloc[train_rows]

    model = LinearRegression() # create an untrained model
    model.fit(X_train, y_train) # learn the relationship between activity and workload

    predictions = model.predict(X_test) # use the relationship to predict the missing person's workload score

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    results.loc[test_rows, "prediction"] = predictions
    results.loc[test_rows, "baseline_prediction"] = baseline_predictions


""" Evaluate the model by measuring the mistakes """
if results[["prediction","baseline"]].isna().any().any():
    raise ValueError("Some predictions are missing.")

model_error = mean_absolute_error(
    results[target], 
    results["prediction"]
)
baseline_error = mean_absolute_error(
    results[target],
    results["baseline"]
)

print("Model MAE:", round(model_error, 3))
print("Baseline MAE:", round(baseline_error, 3))

if model_error < baseline_error:
    print("The model beat the baseline.")
else:
    print("The model did not beat the baseline.")