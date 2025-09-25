import pathlib

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler  # ,StandardScaler
from src.utils.util_file import pre_pipeline_process, upsample_data

PROJECT_DIR = pathlib.Path().resolve()
DATA_DIR = PROJECT_DIR / "data/data_src"
SRC_DATA = DATA_DIR / "data.csv"


def clean_data(no_pay=True, bill=True, upsample=True):
    """[summary]

    Args:
        no_pay (bool, optional): [description]. Defaults to True.
        bill (bool, optional): [description]. Defaults to True.
        upsample (bool, optional): [description]. Defaults to True.
    """
    # Load data make X and y
    source_data = pd.read_csv(SRC_DATA)
    label_data = source_data[["default.payment.next.month"]]
    features_data = source_data.loc[
        :, source_data.columns != "default.payment.next.month"
    ]

    # Split data
    (X_train, X_test, y_train, y_test) = train_test_split(
        features_data, label_data, test_size=0.2, random_state=42, stratify=label_data
    )

    # Upsample data
    if upsample:
        X_train, y_train = upsample_data(X_train, y_train)

    # Pre pipeline process
    X_train = pre_pipeline_process(X_train, no_pay=no_pay, bill=bill)
    
    # Get col types
    num_cols = X_train.select_dtypes(include=["number"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns

    feature_arr = num_cols.to_list() + cat_cols.to_list()
    label_cols = ["default.payment.next.month"]

    # Apply preprocess
    preprocessor = ColumnTransformer(
        transformers=[
            ("nums", RobustScaler(), num_cols),
            ("cats", OneHotEncoder(sparse=False), cat_cols),
        ]
    )
    # ('pass','passthrough', num_cols)

    # Transform data
    pipe = Pipeline(steps=[("preprocessor", preprocessor)])
    X_train_pipe = pipe.fit_transform(X_train)

    # Save pipe
    fname_pipe = PROJECT_DIR / "artifacts/work_artifacts/pipe.joblib"
    joblib.dump(pipe, fname_pipe)

    # Make list of feature names after one-hot
    feature_list = []
    for name, estimator, features in preprocessor.transformers_:
        if hasattr(estimator, "get_feature_names"):
            if isinstance(estimator, OneHotEncoder):
                f = estimator.get_feature_names_out(features)
                feature_list.extend(f)
        else:
            feature_list.extend(features)

    # Pre pipeline process
    X_test = pre_pipeline_process(X_test, no_pay=no_pay, bill=bill)
    # Transform test
    X_test_pipe = pipe.transform(X_test)

    # Put in df to save
    X_train_pipe_save = pd.DataFrame(X_train_pipe, columns=feature_list)
    y_train_save = pd.DataFrame(y_train, dtype=int)
    y_train_one_hot = pd.get_dummies(y_train["default.payment.next.month"], dtype=int)

    # Print X_train data
    # print("X_train shape: {}".format(X_train_pipe_save.shape))
    # print("X_train columns: {}".format(X_train_pipe_save.columns))

    # Put in df to save
    X_test_pipe_save = pd.DataFrame(X_test_pipe, columns=feature_list)
    y_test_save = pd.DataFrame(y_test, dtype=int)
    y_test_one_hot = pd.get_dummies(y_test["default.payment.next.month"], dtype=int)

    # Full dataset
    X_full = pd.concat([X_train_pipe_save, X_test_pipe_save])
    y_full = pd.concat([y_train_save, y_test_save])

    # Print test data
    # print("X_test shape: {}".format(X_test_pipe_save.shape))
    # print("X_test columns: {}".format(X_test_pipe_save.columns))

    # Save data
    X_train_pipe_save.to_csv(
        PROJECT_DIR / "data/data_artifacts/X_train.csv", index=False
    )
    X_test_pipe_save.to_csv(PROJECT_DIR / "data/data_artifacts/X_test.csv", index=False)
    y_train_save.to_csv(PROJECT_DIR / "data/data_artifacts/y_train.csv", index=False)
    y_train_one_hot.to_csv(
        PROJECT_DIR / "data/data_artifacts/y_train_one_hot.csv", index=False
    )
    y_test_save.to_csv(PROJECT_DIR / "data/data_artifacts/y_test.csv", index=False)
    y_test_one_hot.to_csv(
        PROJECT_DIR / "data/data_artifacts/y_test_one_hot.csv", index=False
    )
    X_full.to_csv(PROJECT_DIR / "data/data_artifacts/X_full.csv", index=False)
    y_full.to_csv(PROJECT_DIR / "data/data_artifacts/y_full.csv", index=False)
    return
