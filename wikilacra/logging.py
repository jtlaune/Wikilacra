"""
Custom logging functions for mlflow experiment tracking during training in dvc
pipelines.
"""

from pandas import DataFrame
from matplotlib.pyplot import subplots
from sklearn.metrics import ConfusionMatrixDisplay
from mlflow import (
    log_metric,
    log_figure,
    log_params,
)
from mlflow.sklearn import log_model
from wikilacra.scoring import scoring


def log_sklearn(clf, X_test, y_test, metric_name):
    # Confusion matrix on the test data
    fCMD = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=["NONE", "EVENT"],
    ).figure_

    # Feature importances from the algorithm
    try:
        fFE, axFE = subplots(
            figsize=(6, len(clf.best_estimator_.feature_importances_) * 0.2)
        )
        axFE.barh(X_test.columns, clf.best_estimator_.feature_importances_)
        fFE.tight_layout()
        log_figure(fFE, "FeatureImportances.png")
    except AttributeError:
        pass

    # Unpack the cross validation results to log
    cv_results = DataFrame(clf.cv_results_)

    log_figure(fCMD, "ConfusionMatrixDisplay.png")

    log_params(clf.best_params_)
    # Get the results for the model that performed the best at the chose metric
    best = cv_results.loc[cv_results[f"rank_test_{metric_name}"] == 1].squeeze()

    log_model(clf, name="model")

    # Save the best cross-validation metrics
    for _metric in scoring.keys():
        mean = float(best[f"mean_test_{_metric}"])
        std = float(best[f"std_test_{_metric}"])
        log_metric(f"cross_val/{_metric}", mean)
        log_metric(f"cross_val/{_metric}-std", std)
        log_metric(f"test/{_metric}", scoring[_metric](clf, X_test, y_test))
