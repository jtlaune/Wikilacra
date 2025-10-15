import sys
from ast import literal_eval
from dvclive import Live
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from wikilacra.scoring import scoring
from wikilacra.data import (
    clean_for_training,
    engineer_common_training,
    create_parameter_grid,
)

if __name__ == "__main__":
    labels_fp = sys.argv[1]
    metric_name = sys.argv[2]  # precision, recall, fpr, tpr, f1
    random_state = int(sys.argv[3])
    shuffle = int(sys.argv[4])  # shuffle data during cross validation
    if shuffle: 
        shuffle = True
    else:
        shuffle = False

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

    # generate the n_estimators grid? determines format of next argument
    n_estimators_gen = int(sys.argv[8])
    if n_estimators_gen:
        # grid bounds & number of values to search. Must be a literal tuple
        # e.g., '(1,5,5,"lin")'
        n_estimator1, n_estimator2, n_estimator_n, n_estimator_type = literal_eval(
            sys.argv[9]
        )
        n_estimators = create_parameter_grid(
            n_estimator1, n_estimator2, n_estimator_n, n_estimator_type, int
        )
    else:
        n_estimators = literal_eval(sys.argv[9])

    test_prop = float(sys.argv[10])
    K_fold_cv = int(sys.argv[11])

    labels = pd.read_csv(labels_fp)
    labels = engineer_common_training(labels)
    X, y = clean_for_training(labels)

    X, X_test, y, y_test = train_test_split(X, y, test_size=test_prop, shuffle=shuffle)

    rf = RandomForestClassifier(random_state=random_state)
    parameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depths,
    }
    cv_splitter = KFold(n_splits=K_fold_cv, shuffle=shuffle, random_state=random_state)
    clf = GridSearchCV(
        rf, parameters, n_jobs=n_jobs, cv=cv_splitter, refit=metric_name, scoring=scoring
    )
    clf.fit(X, y)

    fCMD = ConfusionMatrixDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        labels=["1", "0"],
        display_labels=["EVENT", "NONE"],
    ).figure_

    with Live() as live:
        live.log_image("ConfusionMatrixDisplay.png", fCMD)
        live.log_params(clf.best_params_)
        live.log_metric(f"test/{metric_name}", clf.score(X_test, y_test))
        live.log_metric(f"cross_val/{metric_name}", clf.best_score_)

