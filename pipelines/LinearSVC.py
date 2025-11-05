import os
import sys
from ast import literal_eval
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler

from wikilacra.scoring import scoring
from wikilacra.training import create_parameter_grid, get_cv_splitter
from wikilacra.scaling import scaler
from wikilacra.logging import log_sklearn_metrics

from mlflow import start_run, set_tracking_uri, log_params

if __name__ == "__main__":
    set_tracking_uri("http://localhost:5000")
    with start_run():
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
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=test_prop, shuffle=False
        )

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
        parameters = {
            "svc__C": C,
            "svc__loss": loss,
        }

        # Get cross validation splitter
        cv_splitter = get_cv_splitter(CV_type, N_fold_cv, random_state=random_state)

        # Grid search, optimizing for the refit metric
        clf = GridSearchCV(
            pipe, parameters, cv=cv_splitter, scoring=scoring, refit=metric_name
        )
        clf.fit(X, y.values.ravel())

        # Log to MLflow
        log_params({"CV_" + key: val for key, val in parameters.items()})
        log_sklearn_metrics(clf, X_test, y_test, metric_name)
