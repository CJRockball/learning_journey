import datetime
import os
import pathlib
import tempfile
import warnings

import mlflow
import pandas as pd

# from config.config_utils import load_config
from sklearn.model_selection import GridSearchCV

# config = load_config("config.yaml")


def log_run(
    gridsearch: GridSearchCV,
    experiment_name: str,
    model_name: str,
    run_index: int,
    conda_env,
    tags={},
):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        conda_env (str): A dictionary that describes the conda environment (MLFlow Format)
        tags (dict): Dictionary of extra data and tags (usually features)
    """

    cv_results = gridsearch.cv_results_
    with mlflow.start_run(run_name=str(run_index)) as run:

        mlflow.log_param("folds", gridsearch.cv)

        print("Logging parameters")
        params = list(gridsearch.param_grid.keys())
        for param in params:
            mlflow.log_param(param, cv_results["param_%s" % param][run_index])

        print("Logging metrics")
        for score_name in [score for score in cv_results if "mean_test" in score]:
            mlflow.log_metric(score_name, cv_results[score_name][run_index])
            mlflow.log_metric(
                score_name.replace("mean", "std"),
                cv_results[score_name.replace("mean", "std")][run_index],
            )

        print("Logging model")
        mlflow.sklearn.log_model(
            gridsearch.best_estimator_, model_name, conda_env=conda_env
        )

        print("Logging CV results matrix")
        tempdir = tempfile.TemporaryDirectory().name
        os.mkdir(tempdir)
        timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":", ".")
        filename = "%s-%s-cv_results.csv" % (model_name, timestamp)
        csv = os.path.join(tempdir, filename)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pd.DataFrame(cv_results).to_csv(csv, index=False)

        mlflow.log_artifact(csv, "cv_results")

        print("Logging extra data related to the experiment")
        mlflow.set_tags(tags)

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run_id)
        return


def log_results(
    gridsearch: GridSearchCV, experiment_name, model_name, tags={}, log_only_best=False,
):
    """Logging of cross validation results to mlflow tracking server
    
    Args:
        experiment_name (str): experiment name
        model_name (str): Name of the model
        tags (dict): Dictionary of extra tags
        log_only_best (bool): Whether to log only the best model in the gridsearch or all the other models as well
    """
    conda_env = {
        "name": "mlflow-env",
        "channels": ["defaults"],
        "dependencies": [
            "python=3.7.0",
            "scikit-learn>=0.21.3",
            {"pip": ["xgboost==1.0.1"]},
        ],
    }

    best = gridsearch.best_index_

    # mlflow.set_tracking_uri("http://kubernetes.docker.internal:5000")
    mlflow.set_experiment(experiment_name)

    if log_only_best:
        log_run(gridsearch, experiment_name, model_name, best, conda_env, tags)
        return
    else:
        for i in range(len(gridsearch.cv_results_["params"])):
            log_run(gridsearch, experiment_name, model_name, i, conda_env, tags)
            return
