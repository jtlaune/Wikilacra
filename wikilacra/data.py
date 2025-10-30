"""
Helper functions for (simple) feature engineering/data cleaning. More involved
engineering/manipulation is put into a respective pipeline.

clean_for_training: drop columns that are commonly unused
engineer_common_training: engineer common features, e.g. taking the logarithm of
                          page share
"""

from numpy import linspace, geomspace
from sklearn.model_selection import train_test_split


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
        X, y, val_prop + test_prop, shuffle=shuffle
    )
    X_val, X_test, y_val, y_test = train_test_split(
        _X, _y, test_prop / (val_prop + test_prop), shuffle=shuffle
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
