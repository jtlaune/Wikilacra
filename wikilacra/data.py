"""
Helper functions for (simple) feature engineering/data cleaning. More involved
engineering/manipulation is put into a respective pipeline.

clean_for_training: drop columns that are commonly unused
engineer_common_training: engineer common features, e.g. taking the logarithm of
                          page share
"""

from numpy import log


def clean_for_training(df):
    """Drop columns unused for training (e.g., the page history url)

    Args:
        df (DataFrame): proto-training data

    Returns:
        DataFrame: X
        DataFrame: y
    """
    df["target"] = (df["Category"] == "EVENT") * 1
    df["target"] = df["target"].astype("string")

    X = df.drop(
        columns=[
            "page_share_cur",
            "page_id",
            "SECOND_CLASS",
            "COMMENT",
            "page_is_deleted",
            "page_url",
            "event_timestamp",
        ]
    )
    y = X["target"]
    X = X.drop(columns="target")
    return X, y


def engineer_common_training(df):
    df["log_page_share_cur"] = log(df["page_share_cur"])
    df.drop("page_share_cur", inplace=True)
    return df