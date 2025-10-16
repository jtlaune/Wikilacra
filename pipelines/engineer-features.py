"""
Engineer features.
"""

import sys
from extract_data_to_label import load_and_clean
import pandas as pd
from wikilacra.features import (
    create_feature_columns,
    engineer_features_by_event,
    entropy_from_series,
)

if __name__ == "__main__":
    rolling_avg_hrs = int(sys.argv[1])
    dump_input_file = sys.argv[2]
    labels_only_input_file = sys.argv[3]
    counts_path = sys.argv[4]
    output_path = sys.argv[5]

    labels = pd.read_csv(labels_only_input_file, index_col=0)

    columns_to_keep = [
        "event_timestamp",
        "page_title",
        "event_user_id",
        "event_user_text",
        "page_id",
        "page_is_deleted",
        "event_comment",
    ]
    columns_to_read = [
        "event_entity",
        "page_namespace",
        *columns_to_keep,
    ]
    revisions = load_and_clean(
        dump_input_file,
        columns_to_keep,
        columns_to_read,
    )
    counts = pd.read_csv(counts_path)
    counts["event_timestamp"] = pd.to_datetime(
        counts["event_timestamp"]
    )

    colnames = create_feature_columns(labels, rolling_avg_hrs)

    page_entropies = counts.groupby("event_timestamp").agg(
        test=("revision_count", entropy_from_series)
    )

    labels["event_timestamp"] = pd.to_datetime(labels["event_timestamp"])

    for idx, row in labels.iterrows():
        engineer_features_by_event(
            row,
            idx,
            counts,
            page_entropies,
            revisions,
            labels,
            rolling_avg_hrs,
            **colnames,
        )

    labels.to_csv(output_path)
