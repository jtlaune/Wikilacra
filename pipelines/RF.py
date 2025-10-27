import sys
from ast import literal_eval
from dvclive import Live
import pandas as pd
import plotly.express as px
from io import BytesIO
from matplotlib.pyplot import subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from wikilacra.scoring import scoring
from wikilacra.data import (
    clean_for_training,
    create_parameter_grid,
)
from wikilacra.features import engineer_common_training

if __name__ == "__main__":
    labels_fp = sys.argv[1]
    metric_name = sys.argv[2]  # precision, recall, fpr, tpr, f1
    random_state = int(sys.argv[3])

    n_jobs = int(sys.argv[4])

    # generate the max_depths grid? determines format of next argument
    max_depths_gen = int(sys.argv[5])
    if max_depths_gen:
        # grid bounds & number of values to search. Must be a literal tuple
        # e.g., '(1,5,5,"lin")'
        max_depth1, max_depth2, max_depth_n, max_depth_type = literal_eval(sys.argv[6])
        max_depths = create_parameter_grid(
            max_depth1, max_depth2, max_depth_n, max_depth_type, int
        )
    else:
        max_depths = literal_eval(sys.argv[6])

    # generate the n_estimators grid? determines format of next argument
    n_estimators_gen = int(sys.argv[7])
    if n_estimators_gen:
        # grid bounds & number of values to search. Must be a literal tuple
        # e.g., '(1,5,5,"lin")'
        n_estimator1, n_estimator2, n_estimator_n, n_estimator_type = literal_eval(
            sys.argv[8]
        )
        n_estimators = create_parameter_grid(
            n_estimator1, n_estimator2, n_estimator_n, n_estimator_type, int
        )
    else:
        n_estimators = literal_eval(sys.argv[8])

    test_prop = float(sys.argv[9])
    K_fold_cv = int(sys.argv[10])

    labels = pd.read_csv(labels_fp, index_col=0)
    labels = engineer_common_training(labels)
    X, y = clean_for_training(labels)

    X, X_test, y, y_test = train_test_split(X, y, test_size=test_prop, shuffle=False)

    rf = RandomForestClassifier(random_state=random_state)
    parameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depths,
    }

    cv_splitter = TimeSeriesSplit(n_splits=5)
    clf = GridSearchCV(
        rf,
        parameters,
        n_jobs=n_jobs,
        cv=cv_splitter,
        scoring=scoring,
        refit=metric_name,
    )

    clf.fit(X, y)

    fCMD = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        display_labels=["NONE", "EVENT"],
    ).figure_

    fFE, axFE = subplots(
        figsize=(6, len(clf.best_estimator_.feature_importances_) * 0.2)
    )
    axFE.barh(X.columns, clf.best_estimator_.feature_importances_)
    fFE.tight_layout()

    cv_results = pd.DataFrame(clf.cv_results_)

    with Live() as live:
        live.log_image("FeatureImportances.png", fFE)
        live.log_image("ConfusionMatrixDisplay.png", fCMD)
        live.log_params(clf.best_params_)
        best = cv_results.loc[cv_results[f"rank_test_{metric_name}"] == 1].squeeze()

        for _metric in scoring.keys():
            mean = float(best[f"mean_test_{_metric}"])
            std = float(best[f"std_test_{_metric}"])
            live.log_metric(f"cross_val/{_metric}", mean)
            live.log_metric(f"cross_val/{_metric}-std", std)
