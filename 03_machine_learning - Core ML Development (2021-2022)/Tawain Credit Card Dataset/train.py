import datetime
import logging
import pathlib
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import pandas as pd

from config.config_utils import load_config
from models.xgb_models import xgb_dmat_single, xgb_sklearn_cv, xgb_sklearn_single
from models.xgb_search import xgb_gridsearch, xgb_hyperopt
from src.utils.clean import clean_data

# log = logging.getLogger(__name__)

# log.debug("Debug level message")
# log.info("Info level message")
# log.warning("Warning level message")

config = load_config("config.yaml")


PROJECT_DIR = pathlib.Path().resolve()
DATA_DIR = PROJECT_DIR / "data/data_artifacts"
MODEL_PATH = PROJECT_DIR / "artifacts/final_artifacts"

clean_data(no_pay=True, bill=False, upsample=False)

# Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
feature_names = X_train.columns.values.tolist()
X_train = X_train.to_numpy()

X_test = pd.read_csv(DATA_DIR / "X_test.csv").to_numpy()
y_train = pd.read_csv(DATA_DIR / "y_train.csv").to_numpy()
y_train_one_hot = pd.read_csv(DATA_DIR / "y_train_one_hot.csv").to_numpy()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").to_numpy()
y_test_one_hot = pd.read_csv(DATA_DIR / "y_test_one_hot.csv").to_numpy()
X_full = pd.read_csv(DATA_DIR / "X_full.csv").to_numpy()
y_full = pd.read_csv(DATA_DIR / "y_full.csv").to_numpy()
# Train gridsearch


if __name__ == "__main__":
    # Uses xgb scikit learn interface to evaluate a single model with train/test split
    # xgb_sklearn_single(
    #     X_train, y_train, X_test, y_test, feature_names, "xgb_hyperopt", "test",
    # )

    # Uses Dmatrix to define single model and evaluate
    # xgb_dmat_single(X_train, y_train, X_test, y_test)

    # Uses xgb scikit learn interface to evaluate a single model with cross-validation
    # xgb_sklearn_cv(
    #     X_train, y_train, X_test, y_test, model_name="xgb_hyperopt", exp_name="test"
    # )
    # Uses gridsearch to optimize model
    # xgb_gridsearch(X_full, y_full, log_name="gs_recall_opt")

    # Uses Hyperopt to optimize model
    xgb_hyperopt(X_full, y_full, log_name="fixed_bugs", no_runs=5)

    xgb_sklearn_single(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        "xgb_hyperopt",
        "test",
        save_on="ON",
        save_name=MODEL_PATH / "xgb_best.pkl",
    )

