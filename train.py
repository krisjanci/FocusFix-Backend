from pathlib import Path   # to find files

import pandas   # to work with tables
import joblib   # to save and load my trained models

from sklearn.linear_model import LinearRegression   # imports the Linear ML model
from sklearn.dummy import DummyRegressor   # imports a prediction tool for my model to compete against
from sklearn.model_selection import LeaveOneGroupOut   # tool for seperating training data and testing data
from sklearn.metrics import mean_absolute_error   # tool for measuring how far my preediction is from reality on average


backend_folder = Path(__file__).resolve().parent
workplace_folder = backend_folder.parent

data_file = (
    workplace_folder
    /"DATA-(not_for_Github)"
    /"swell.xlsx"
)

print("Looking for data at:", data_file)
print("Does the file exist?", data_file.exists())