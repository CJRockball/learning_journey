import datetime
import pathlib
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from config.config_utils import load_config
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from src.utils.mlflow_utils import log_results
from xgboost.sklearn import XGBClassifier

config = load_config("config.yaml")


def xgb_gridsearch(X_train, y_train, log_name="gs"):
    PROJECT_DIR = pathlib.Path().resolve()
    xgb_reg = xgb.XGBClassifier(
        random_state=config["XGB_Options"]["seed"],
        n_jobs=cpu_count() // 2,
        eval_metric=config["eval_metric"],
        verbosity=0,
        use_label_encoder=False,
    )
    start = datetime.datetime.now()
    print(
        "Starting with low learning rate and tuning: max_depth, min_child_weight, n_estimators"
    )

    params = {
        "learning_rate": config["XGB_Options"]["learning_rate"],
        "max_depth": config["XGB_Options"]["max_depth"],
        "min_child_weight": config["XGB_Options"]["min_child_weight"],
        "n_estimators": config["XGB_Options"]["n_estimators"],
        "colsample_bytree": config["XGB_Options"]["colsample_bytree"],
        "subsample": config["XGB_Options"]["subsample"],
        "gamma": config["XGB_Options"]["gamma"],
        "scale_pos_weight": config["XGB_Options"]["scale_pos_weight"],
        "reg_alpha": config["XGB_Options"]["reg_alpha"],
        "reg_lambda": config["XGB_Options"]["reg_lambda"],
    }

    GSCV = GridSearchCV(
        xgb_reg,
        params,  # config["XGB_Options_search"])
        cv=config["XGB_Options"]["cv"],
        scoring=config["XGB_Options"]["scoring"],
        n_jobs=cpu_count() // 2,
        verbose=config["XGB_Options"]["verbose"],
    )

    GSCV.fit(X_train, y_train.ravel())
    end = datetime.datetime.now()

    print("Time to fit", (end - start))
    print("best_params_:", GSCV.best_params_)
    print("best_score_:", GSCV.best_score_)

    df = pd.DataFrame(GSCV.cv_results_)
    df.to_csv(PROJECT_DIR / "data/data_artifacts/CV_result.csv", index=False)

    log_results(
        gridsearch=GSCV,
        experiment_name=log_name,
        model_name="xgb_baseline_gscv",
        tags={},
        log_only_best=False,
    )
    return


def xgb_hyperopt(X_train, y_train, log_name="gs", no_runs=10):

    space = {
        "n_estimators": hp.choice("n_estimators", np.arange(50, 300, 5, dtype=int)),
        "learning_rate": hp.quniform("learning_rate", 0.025, 0.5, 0.025),
        # A problem with max_depth casted to float instead of int with
        # the hp.quniform method.
        "max_depth": hp.choice("max_depth", np.arange(5, 20, dtype=int)),
        "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
        "subsample": hp.quniform("subsample", 0.5, 1, 0.05),
        "gamma": hp.quniform("gamma", 0.5, 5, 0.05),
        "colsample_bytree": hp.quniform("colsample_bytree", 0.3, 1, 0.05),
        "objective": "binary:logistic",
        "reg_lambda": hp.quniform("reg_lambda", 5, 30, 1),
        "reg_alpha": hp.choice("reg_alpha", [1e-6, 1e-5, 1e-4, 0.001, 0.01]),
        "scale_pos_weight": hp.choice("scale_pos_weight", np.arange(5, 20, dtype=int)),
        # Increase this number if you have more cores. Otherwise, remove it and it will default
        # to the maxium number.
        # "nthread": 4,
        # "booster": "gbtree",
        # "tree_method": "exact",
        # "silent": 1,
        "seed": 42,
    }

    mlflow.set_experiment(log_name)
    experiment = mlflow.get_experiment_by_name(log_name)

    def hyperopt_fcn(params):
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            # log model params
            for key in params:
                mlflow.log_param(key, params[key])

            clf = xgb.XGBClassifier(
                **params, n_jobs=-1, verbosity=0, use_label_encoder=False,
            )

            scoring = {
                "acc": "accuracy",
                # "prec_macro": "precision_macro",
                # "rec_micro": "recall_macro",
                # "f1_score": "f1",
                # "roc_curve": "roc_auc",
                # "pred_neg_log_loss": "neg_log_loss",
            }
            kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            eval_metric = cross_validate(
                clf, X_train, y_train, scoring=scoring, cv=kfold
            )
            # Average run metrics
            acc = eval_metric["test_acc"].mean()
            # prec = eval_metric["test_prec_macro"].mean()
            # rec = eval_metric["test_rec_micro"].mean()
            # f1_bin = eval_metric["test_f1_score"].mean()
            # roc_bin = eval_metric["test_roc_curve"].mean()
            # pred_neg_log_loss = eval_metric["test_pred_neg_log_loss"].mean()
            # Log metrics
            mlflow.log_metric("accuracy", acc)
            # mlflow.log_metric("precision", prec)
            # mlflow.log_metric("recall", rec)
            # mlflow.log_metric("f1", f1_bin)
            # mlflow.log_metric("roc_bin", roc_bin)

        return {"loss": (1 - acc), "status": STATUS_OK}

    # Use the fmin function from Hyperopt to find the best hyperparameters
    trials = Trials()
    best = fmin(
        fn=hyperopt_fcn,
        space=space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=no_runs,
    )

    # Get best params from hyperopt
    best_space = space_eval(space, best)
    # Print best score
    print("Best: {}".format(best_space))

    # Get best params and write to a yaml file
    # Move data to df then dict to prepare for yaml write
    df = pd.DataFrame(best_space, index=["a"])
    df_dict = df.to_dict(orient="records")
    best_setting = {"xgb_hyperopt": df_dict[0]}
    # Save file
    SAVE_PATH = pathlib.Path(__file__).resolve().parent.parent
    CONFIG_PATH = SAVE_PATH / "config/config_best.yaml"
    with open(CONFIG_PATH, "w") as f:
        f.write(yaml.dump(best_setting))

    # Plot hyperopt score
    hyperopt_scores = [x["result"]["loss"] for x in trials.trials]
    plt.figure()
    plt.plot(hyperopt_scores)
    plt.ylabel("Metric")
    plt.xlabel("Trial")
    plt.grid()
    plt.show()

    return

