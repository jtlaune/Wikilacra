"""
Helper functions for training models in pipelines.

train_val_test: split into training, validation, and test set.

create_parameter_grid: Create a geometrically or linearly spaced grid of
parameters.

get_cv_splitter: create a cv_splitter based on options typically passed from the
command line
"""

from numpy import linspace, geomspace
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit


def train_val_test(X, y, val_prop, test_prop, shuffle):
    """Split the dataset into train, validation, and test sets.

    Args:
        X (_type_): Train data.
        y (_type_): Target data.
        val_prop (_type_): Proportion used for validation.
        test_prop (_type_): Proportion used for test
        shuffle (_type_): Shuffle (same for both splits)

    Returns:
        tuple [DataFrame, DataFrame, DataFrame, DataFrame, DataFrame,
        DataFrame]: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_train, _X, y_train, _y = train_test_split(
        X, y, test_size=val_prop + test_prop, shuffle=shuffle
    )
    X_val, X_test, y_val, y_test = train_test_split(
        _X, _y, test_size=test_prop / (val_prop + test_prop), shuffle=shuffle
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_parameter_grid(x1, x2, n, spacing, dtype):
    """Create a parameter grid.

    Args:
        x1: Start value
        x2: End value
        n: Number of points
        spacing: Spacing type, lin or geom
        dtype: Type of data (e.g., int or float)
    """
    if spacing == "geom":
        _ = geomspace(x1, x2, n, dtype=dtype)
        # For dvclive reasons, need to make sure its in a native Python type
        return [dtype(x) for x in _]
    elif spacing == "lin":
        _ = linspace(x1, x2, n, dtype=dtype)
        # For dvclive reasons, need to make sure its in a native Python type
        return [dtype(x) for x in _]
    else:
        raise Warning(f"parameter grid type is [geom,lin], not {spacing}")


def get_cv_splitter(CV_type, N_fold_cv, random_state=None):
    """Do time series cross-validation split or k-fold cross-validation split
    based on CV_type. N_fold_cv sets the number of folds. random_state is unused
    if TimeSeriesSplit."""
    if CV_type == "TimeSeries":
        if random_state is not None:
            print("Warning... random_state is unused since you are using a TimeSeriesSplit")
        cv_splitter = TimeSeriesSplit(n_splits=N_fold_cv)
    elif CV_type == "KFold":
        cv_splitter = KFold(n_splits=N_fold_cv, shuffle=True, random_state=random_state)
    else:
        raise Warning("Supported CV_type: TimeSeries, KFold")
    return cv_splitter
