import datetime
import logging
import pathlib
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
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from src.utils.clean import clean_data
from src.utils.mlflow_utils import log_results
from xgboost import XGBClassifier
from xgboost.sklearn import xgboost_model_doc


# Plot training train and test to check overfit
def plot_training(results, train_metric):
    epochs = len(results["validation_0"][train_metric])
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"][train_metric], label="Train")
    ax.plot(x_axis, results["validation_1"][train_metric], label="Test")
    ax.legend()
    plt.ylabel(str(train_metric))
    plt.title("XGBoost Baseline - " + str(train_metric))
    plt.show()
    # plot classification error
    fig, ax = plt.subplots()
    ax.plot(x_axis, results["validation_0"]["auc"], label="Train")
    ax.plot(x_axis, results["validation_1"]["auc"], label="Test")
    ax.legend()
    plt.ylabel("auc")
    plt.title("XGBoost Baseline - auc ")
    plt.show()
    return


# Plot confusion matrix
def plot_conf_mat(y_test, y_pred):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["No default", "Default"]
    )
    plt.show()
    plt.savefig("conf_mat.png", dpi=200)
    return


# Plot variable importance
def plot_xgb_var_imp(xgb_baseline, feature_names):
    xgb_baseline.get_booster().feature_names = feature_names
    ax = xgb.plot_importance(xgb_baseline.get_booster())
    # fig = plt.gcf()
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")
    plt.savefig("importance.png", dpi=200)
    return


# Plot ROC curve
def plot_xgb_roc(y_pred_proba, y_test):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    # Plot roc-curve
    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.show()
    return
