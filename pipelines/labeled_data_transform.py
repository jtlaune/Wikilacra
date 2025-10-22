"""
Transform raw labeled data into binary classification in correct timeframe.
"""

import sys
import pandas as pd
from numpy import log


def load_labeled_data(filepath, start_dt, rolling_hrs):
    """Load the labels, preprocess, and return.

    Args:
        filepath (str): Path of input labeled data file.
        start_dt (str): Datetime string of start time, e.g. 2025-08-01 00:00:00
        offset_hrs (int): Number of hours for rolling averages
    """
    labels = pd.read_csv(filepath, index_col=0)
    labels.rename(
        columns={
            "EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED": "Category",
        },
        inplace=True,
    )
    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])

    # Since we're going to be using rolling_hrs-hour moving averages
    labels = labels[
        labels.event_timestamp
        >= pd.to_datetime(start_dt) + pd.Timedelta(hours=rolling_hrs)
    ]

    return labels


if __name__ == "__main__":
    data_start_dt = sys.argv[1]
    rolling_avg_hrs = int(sys.argv[2])
    labels_input_file = sys.argv[3]
    output_path = sys.argv[4]

    non_event = {"EDIT_WAR", "VANDALISM", "NONE", "MOVED_OR_DELETED"}

    labels = load_labeled_data(labels_input_file, data_start_dt, rolling_avg_hrs)

    labels.to_csv(output_path)
