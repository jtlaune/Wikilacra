import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from wikilacra.data import clean_for_training, engineer_common_training, train_val_test

if __name__ == "__main__":
    labels_fp = sys.argv[0]
    shuffle = bool(sys.argv[1])
    max_depth = int(sys.argv[2])
    random_state = int(sys.argv[3])
    n_estimators = int(sys.argv[4])
    val_prop = float(sys.argv[5])
    test_prop = float(sys.argv[6])

    labels = pd.read_csv(labels_fp)

    X, y = clean_for_training(labels)

    X = engineer_common_training(X)


clf = GridSearchCV(RandomForestClassifier(), parameters)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test(
        X, y, 0.2, 0.1, shuffle
    )

    clf = RandomForestClassifier(
        max_depth=max_depth, n_estimators=n_estimators, random_state=random_state
    )
    clf.fit(X_train, y_train)
    ConfusionMatrixDisplay.from_estimator(
        clf,
        X_val,
        y_val,
        labels=[1.0, 0.0],
        display_labels=["EVENT", "NONE"],
    )
