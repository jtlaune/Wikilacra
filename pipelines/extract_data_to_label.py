import sys
from wikilacra.stream import bin_and_count, read_data_chunked
from pandas import read_csv, to_datetime


def load_and_clean(fn, columns_to_keep, columns_to_read):
    revisions = read_data_chunked(fn, columns_to_keep, columns_to_read)
    revisions["event_user_id"] = revisions["event_user_id"].fillna("")
    return revisions


def filter_for_manual_labeling(counts):
    """In future should parameterize this and treat as a hyperparameter"""
    return counts[(counts.num_unique_users > 1) & (counts.revision_count >= 15)]


def generate_url(counts):
    """Generate history link for easy labeling"""
    counts["page_url"] = (
        "https://en.wikipedia.org/w/index.php?title="
        + counts["page_title"].astype(str)
        + "&action=history&offset=&limit=500"
    )


if __name__ == "__main__":
    # Call from a DVC stage.
    filepath = sys.argv[1]
    freq = sys.argv[2]
    pre_labels_path = sys.argv[3]
    outpath = sys.argv[4]
    counts_outpath = sys.argv[5]

    columns_to_keep = [
        "event_timestamp",
        "page_title",
        "event_user_id",
        "page_id",
        "page_is_deleted",
    ]
    columns_to_read = [
        "event_entity",
        "page_namespace",
        *columns_to_keep,
    ]
    pre_labels = read_csv(
        pre_labels_path,
        index_col=0,
    )
    labeled_events = list(
        zip(to_datetime(pre_labels["event_timestamp"]), pre_labels["page_title"])
    )

    revisions = load_and_clean(filepath, columns_to_keep, columns_to_read)
    counts = bin_and_count(revisions, freq)
    generate_url(counts)
    filtered_counts = filter_for_manual_labeling(counts)

    filtered_counts = filtered_counts.reset_index(names="idx").set_index(
        ["event_timestamp", "page_title"]
    )
    filtered_counts.loc[
        labeled_events,
        [
            "EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED",
            "SECOND_CLASS",
            "COMMENT",
        ],
    ] = pre_labels[
        [
            "EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED",
            "SECOND_CLASS",
            "COMMENT",
        ]
    ].values
    filtered_counts[
        [
            "EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED",
            "SECOND_CLASS",
            "COMMENT",
        ]
    ] = filtered_counts[
        [
            "EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED",
            "SECOND_CLASS",
            "COMMENT",
        ]
    ].fillna(
        ""
    )

    filtered_counts = filtered_counts.reset_index().set_index("idx")
    filtered_counts.index.name = None

    counts.to_csv(counts_outpath)
    filtered_counts.to_csv(outpath)
