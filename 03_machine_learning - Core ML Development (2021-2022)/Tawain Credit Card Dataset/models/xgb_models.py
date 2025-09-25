import datetime
import logging
import pathlib
import pickle
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from config.config_utils import load_config
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from src.utils.clean import clean_data
from src.utils.mlflow_utils import log_results
from src.utils.plot_utils import (
    plot_conf_mat,
    plot_training,
    plot_xgb_roc,
    plot_xgb_var_imp,
)
from xgboost import XGBClassifier

config = load_config("config.yaml")


def xgb_sklearn_single(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    model_name: str,
    exp_name: str,
    save_on="OFF",
    save_name="xgb_default",
):
    """ Fits an xgb model with config_best params. Uses XGBClassifier and .fit method. Track result with mlflow, plot result.

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
        model_name (str): [description]
        exp_name (str): [description]
    """

    config = load_config("config_best.yaml")
    # print(config)
    # mlflow.set_tracking_uri(
    #     "http://localhost:5000/"
    # )  # Actual Server URI instead of localhost
    mlflow.set_experiment(exp_name)
    experiment = mlflow.get_experiment_by_name(exp_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # model parameters
        eval_metric = ["logloss", "auc"]  # config["eval_metric"]
        eval_set = [(X_train, y_train), (X_test, y_test)]

        # log model params
        for key in config[model_name]:
            mlflow.log_param(key, config[model_name][key])

        clf = XGBClassifier(**config[model_name], use_label_encoder=False)
        xgb_model = clf.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=eval_metric,
            early_stopping_rounds=10,
            verbose=False,
        )

        if save_on == "ON":
            pickle.dump(xgb_model, open(save_name, "wb"))

        # importances = xgb_baseline.get_booster().get_fscore()
        # print(importances)
        # rec = cross_val_score(clf, X_train, y_train, scoring="recall", cv=3)

        # Plot training
        results = xgb_model.evals_result()
        plot_training(results, train_metric="logloss")

        # get predictions
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {}".format(round(accuracy * 100.0, 2)))
        recall = recall_score(y_test, y_pred)
        print("Recall: {}".format(round(recall * 100.0, 2)))
        precision = precision_score(y_test, y_pred)
        print("Precision: {}".format(round(precision * 100.0, 2)))
        F1 = f1_score(y_test, y_pred)
        print("F1: {}".format(round(F1 * 100.0, 2)))
        # log accuracy metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)

        # Plot confusion matrix
        plot_conf_mat(y_test, y_pred)
        # Log plot artifact
        mlflow.log_artifact("conf_mat.png")

        # Plot variable importance
        plot_xgb_var_imp(xgb_model, feature_names)
        # Log plot artifact
        mlflow.log_artifact("importance.png")

        # Plot ROC curve
        y_pred_proba = xgb_model.predict_proba(X_test)
        plot_xgb_roc(y_pred_proba, y_test)

        # log model
        mlflow.sklearn.log_model(xgb_model, "model")
        return


def xgb_sklearn_cv(X_full, y_full, model_name: str, exp_name: str):
    """Use cross-validate to evaluate config_best params. 

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
        X_test ([type]): [description]
        y_test ([type]): [description]
        model_name (str): [description]
        exp_name (str): [description]
    """
    config = load_config("config_best.yaml")

    xgb_model = xgb.XGBClassifier(
        **config[model_name], n_jobs=-1, verbosity=0, use_label_encoder=False,
    )

    scoring = {
        "acc": "accuracy",
        "prec_macro": "precision_macro",
        "rec_micro": "recall_macro",
        "f1_score": "f1",
        "roc_curve": "roc_auc",
        "pred_neg_log_loss": "neg_log_loss",
    }
    kfold = StratifiedKFold(n_split=3, random_state=42)
    eval_metric = cross_validate(xgb_model, X_full, y_full, scoring=scoring, cv=kfold)

    # Average run metrics
    acc = eval_metric["test_acc"].mean()
    prec = eval_metric["test_prec_macro"].mean()
    rec = eval_metric["test_rec_micro"].mean()
    f1_bin = eval_metric["test_f1_score"].mean()
    roc_bin = eval_metric["test_roc_curve"].mean()
    pred_neg_log_loss = eval_metric["test_pred_neg_log_loss"].mean()
    print("Accuracy: {}".format(acc))

    return


def xgb_dmat_single(X_train, y_train, X_test, y_test):
    # mlflow.set_tracking_uri(
    #     "http://localhost:5000/"
    # )  # Actual Server URI instead of localhost
    mlflow.set_experiment("experiment_Dmat")
    experiment = mlflow.get_experiment_by_name("experiment_Dmat")

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # model parameters
        xg_train = xgb.DMatrix(X_train, label=y_train)
        xg_test = xgb.DMatrix(X_test, label=y_test)
        watchlist = [(xg_test, "eval"), (xg_train, "train")]

        # log model params
        for key in config["XGB_Dmat"]:
            mlflow.log_param(key, config["XGB_Dmat"][key])

        xgb_baseline = xgb.train(
            config["XGB_Dmat"],
            xg_train,
            1000,
            watchlist,
            early_stopping_rounds=20,
            verbose_eval=True,
        )

        # importances = xgb_baseline.get_fscore()
        # print(importances)

        # get predictions
        y_pred = xgb_baseline.predict(xgb.DMatrix(X_test))
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy: {}".format(round(accuracy * 100.0, 2)))
        recall = recall_score(y_test, y_pred)
        print("Recall: {}".format(round(recall * 100.0, 2)))
        # log accuracy metric
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)

        # Log confusion matrix
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["No default", "Default"]
        )
        # plt.show()
        plt.savefig("conf_mat.png", dpi=200)
        # Log plot artifact
        mlflow.log_artifact("conf_mat.png")

        # Plot variable f1 score
        sns.set(font_scale=1.5)
        xgb.plot_importance(xgb_baseline)
        plt.savefig("importance.png", dpi=200, bbox_inches="tight")
        # Log plot artifact
        mlflow.log_artifact("importance.png")

        # log model
        mlflow.sklearn.log_model(xgb_baseline, "model")
        return
