import os
import sys
from ast import literal_eval
import pandas as pd
from matplotlib.pyplot import subplots

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)

import xgboost as xgb

from dvclive.live import Live

from wikilacra.scoring import scoring
from wikilacra.data import create_parameter_grid

if __name__ == "__main__":
    # Directory of the data
    engineered_dir = sys.argv[1]
    # Base name the engineered data
    engineered_basename = sys.argv[2]
    # Metric to optimize in cross-validation
    metric_name = sys.argv[3]  # precision, recall, fpr, tpr, f1
    # Random state to set in the appropriate sklearn functions
    random_state = int(sys.argv[4])
    # Number of concurrent jobs for cross-validation grid search
    n_jobs = int(sys.argv[5])

    # generate the max_depths grid? determines format of next argument
    max_depths_gen = int(sys.argv[6])
    if max_depths_gen:
        # grid bounds & number of values to search. Must be a literal tuple
        # e.g., '(1,5,5,"lin")'
        max_depth1, max_depth2, max_depth_n, max_depth_type = literal_eval(sys.argv[7])
        max_depths = create_parameter_grid(
            max_depth1, max_depth2, max_depth_n, max_depth_type, int
        )
    else:
        max_depths = literal_eval(sys.argv[7])

    # Proportion of test data to be held out
    test_prop = float(sys.argv[8])
    # Type of cross validation (time series or KFold)
    CV_type = str(sys.argv[9])
    # Number of cross validation splits in the time series
    N_fold_cv = int(sys.argv[10])

    # Load the engineered and cleaned features
    engineered = pd.read_csv(
        os.path.join(engineered_dir, engineered_basename + ".csv"), index_col=0
    )
    # Read the endog/exog column lists
    with open(
        os.path.join(engineered_dir, engineered_basename + "_exog_cols.txt"), "r"
    ) as f:
        exog_cols = literal_eval(f.readline())
    with open(
        os.path.join(engineered_dir, engineered_basename + "_endog_cols.txt"), "r"
    ) as f:
        endog_cols = literal_eval(f.readline())
    # X is the exog cols, y is the endog cols
    X = engineered[exog_cols]
    y = engineered[endog_cols].astype(int)

    # Split the test set off, shuffle=False which means we're getting the last
    # entries in test
    X, X_test, y, y_test = train_test_split(X, y, test_size=test_prop, shuffle=False)

    # Random forest classifier with the correct random state set
    bt = xgb.XGBClassifier(random_state=random_state, enable_categorical=True)
    # Grid search parameters
    parameters = {
        "max_depth": max_depths,
    }
    # Do time series cross-validation split
    if CV_type == "TimeSeries":
        cv_splitter = TimeSeriesSplit(n_splits=N_fold_cv)
    elif CV_type == "KFold":
        cv_splitter = KFold(n_splits=N_fold_cv, shuffle=True)
    else:
        raise Warning("Supported CV_type: TimeSeries, KFold")

    # Grid search, optimizing for the refit metric
    clf = GridSearchCV(
        bt,
        parameters,
        n_jobs=n_jobs,
        cv=cv_splitter,
        scoring=scoring,
        refit=metric_name,
    )
    clf.fit(X, y.values.ravel())

    # Confusion matrix on the test data
    fCMD = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=["NONE", "EVENT"],
    ).figure_

    # Feature importances from the RF algorithm
    fFE, axFE = subplots(
        figsize=(6, len(clf.best_estimator_.feature_importances_) * 0.2)
    )
    axFE.barh(X.columns, clf.best_estimator_.feature_importances_)
    fFE.tight_layout()

    # Unpack the cross validation results to log
    cv_results = pd.DataFrame(clf.cv_results_)

    with Live("dvclive/XGBT/") as live:
        # Log images and params into dvclive
        live.log_image("FeatureImportances.png", fFE)
        live.log_image("ConfusionMatrixDisplay.png", fCMD)
        live.log_params(clf.best_params_)
        # Get the results for the model that performed the best at the chose metric
        best = cv_results.loc[cv_results[f"rank_test_{metric_name}"] == 1].squeeze()
        # Save the best cross-validation metrics
        for _metric in scoring.keys():
            mean = float(best[f"mean_test_{_metric}"])
            std = float(best[f"std_test_{_metric}"])
            live.log_metric(f"cross_val/{_metric}", mean)
            live.log_metric(f"cross_val/{_metric}-std", std)
