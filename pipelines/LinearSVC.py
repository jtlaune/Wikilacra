import os
import sys
from pickle import dump
from ast import literal_eval
import pandas as pd
from matplotlib.pyplot import subplots

from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

from dvclive.live import Live

from wikilacra.scoring import scoring
from wikilacra.data import create_parameter_grid
from wikilacra.scaling import scaler

if __name__ == "__main__":
    # Directory of the data
    engineered_dir = str(sys.argv[1])
    # Base name the engineered data
    engineered_basename = str(sys.argv[2])
    # Metric to optimize in cross-validation
    metric_name = str(sys.argv[3])  # precision, recall, fpr, tpr, f1
    # Random state to set in the appropriate sklearn functions
    random_state = int(sys.argv[4])
    # Number of concurrent jobs for cross-validation grid search
    n_jobs = int(sys.argv[5])
    # Proportion of test data to be held out
    test_prop = float(sys.argv[6])
    # Type of cross validation (time series or KFold)
    CV_type = str(sys.argv[7])
    # Number of cross validation splits in the time series
    N_fold_cv = int(sys.argv[8])
    # Scaler type to apply after the custom scaling (see wikilacra.scaling)
    scale_type = str(sys.argv[9])
    # generate the C grid? determines format of next argument
    C_gen = int(sys.argv[10])
    if C_gen:
        # grid bounds & number of values to search. Must be a literal tuple
        # e.g., '(1,5,5,"lin")'
        C1, C2, C_n, C_type = literal_eval(sys.argv[11])
        C = create_parameter_grid(C1, C2, C_n, C_type, float)
    else:
        C = literal_eval(sys.argv[11])
    loss = literal_eval(sys.argv[12])

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

    # Custom scaler function from wikilacra.scaling that handles the count and
    # slope data. New features should be manually added there.
    count_slope_scaling = FunctionTransformer(scaler)

    # Normalization scaling from sklearn
    if scale_type == "standard":
        norm_scaler = StandardScaler()
    elif scale_type == "robust":
        norm_scaler = RobustScaler()
    else:
        raise Warning(f"scale-type options are [standard,robust], not {scale_type}")

    # Classifier and pipeline. Pipeline is necessary for cross validation consistency.
    svc = LinearSVC()
    pipe = Pipeline(
        [
            ("count_slope_scaling", count_slope_scaling),
            ("norm_scaler", norm_scaler),
            ("svc", svc),
        ]
    )
    # Parameter grid. Parameters for the pipeline are prefixed with "[stepname]__"
    print(C)
    parameters = {
        "svc__C": C,
        "svc__loss": loss,
    }

    if CV_type == "TimeSeries":
        # Do time series cross-validation split
        cv_splitter = TimeSeriesSplit(n_splits=N_fold_cv)
    elif CV_type == "KFold":
        # Do KFold cross-validation split
        cv_splitter = KFold(n_splits=N_fold_cv, shuffle=True, random_state=random_state)
    else:
        raise Warning("Supported CV_type: TimeSeries, KFold")

    # Grid search, optimizing for the refit metric
    clf = GridSearchCV(
        pipe, parameters, cv=cv_splitter, scoring=scoring, refit=metric_name
    )
    clf.fit(X, y.values.ravel())

    # Confusion matrix on the test data
    fCMD = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=["NONE", "EVENT"],
    ).figure_

    # Unpack the cross validation results to log
    cv_results = pd.DataFrame(clf.cv_results_)

    with Live("dvclive/LinSVC/") as live:
        # Log images and params into dvclive
        live.log_image("ConfusionMatrixDisplay.png", fCMD)
        live.log_params(clf.best_params_)
        # Get the results for the model that performed the best at the chose metric
        best = cv_results.loc[cv_results[f"rank_test_{metric_name}"] == 1].squeeze()
        
        with open("outputs/models/LinSVC-model.pkl", "wb") as f:
            dump(clf, f)
        live.log_artifact("outputs/models/LinSVC-model.pkl", name="LinSVC-model")

        # Save the best cross-validation metrics
        for _metric in scoring.keys():
            mean = float(best[f"mean_test_{_metric}"])
            std = float(best[f"std_test_{_metric}"])
            live.log_metric(f"cross_val/{_metric}", mean, plot=False)
            live.log_metric(f"cross_val/{_metric}-std", std, plot=False)
            live.log_metric(f"test/{_metric}", scoring[_metric](clf, X_test, y_test), plot=False)
