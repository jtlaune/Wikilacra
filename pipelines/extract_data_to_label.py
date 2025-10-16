import sys
import pandas as pd
from wikilacra import MEDIAWIKI_HISTOR_DUMP_COL_NAMES
from wikilacra.stream import bin_and_count, read_data_chunked


def load_and_clean(fn, columns_to_keep, columns_to_read):
    revisions = read_data_chunked(fn, columns_to_keep, columns_to_read)
    revisions = revisions[revisions["event_user_id"].notna()]
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
    outpath = sys.argv[3]
    counts_outpath = sys.argv[4]

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

    revisions = load_and_clean(filepath, columns_to_keep, columns_to_read)
    counts = bin_and_count(revisions, freq)
    generate_url(counts)
    filtered_counts = filter_for_manual_labeling(counts)
    counts.to_csv(counts_outpath)
    filtered_counts.to_csv(outpath)
