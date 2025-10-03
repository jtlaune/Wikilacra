import sys
import numpy as np
import pandas as pd
import plotly.express as px
from wikilacra import MEDIAWIKI_HISTOR_DUMP_COL_NAMES


def load_and_clean(fn, chunksize=250_000):
    # Stream and filter the dump to stay within memory limits.
    # Read dump file and retrieve edits.

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

    read_csv_kwargs = dict(
        sep="	",
        names=MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
        header=None,
        usecols=columns_to_read,
        on_bad_lines="warn",
        parse_dates=["event_timestamp"],
    )

    if chunksize:
        reader = pd.read_csv(fn, chunksize=chunksize, **read_csv_kwargs)
    else:
        reader = [pd.read_csv(fn, **read_csv_kwargs)]

    filtered_chunks = []

    for chunk in reader:
        title = chunk["page_title"]
        mask = chunk["event_entity"].eq("revision") & chunk["page_namespace"].eq(0)
        mask &= ~title.str.contains(r"/sandbox", na=False)
        mask &= ~title.str.fullmatch(r"Sandbox", na=False)
        mask &= ~title.str.fullmatch(r"Undefined/junk", na=False)
        mask &= ~title.str.fullmatch(r"Wiki", na=False)
        mask &= chunk["event_user_id"].notna()

        filtered = chunk.loc[mask, columns_to_keep]
        if not filtered.empty:
            filtered_chunks.append(filtered)

    if filtered_chunks:
        revisions = pd.concat(filtered_chunks, ignore_index=True)
    else:
        revisions = pd.DataFrame(columns=columns_to_keep).astype(
            {"event_timestamp": "datetime64[ns]"}
        )

    return revisions


def bin_and_count(revisions, freq):
    # Bin number of edits by hour and compute number of unique users
    # editing the page during that window.
    counts = (
        revisions[revisions.event_user_id.notna()]
        .groupby([pd.Grouper(key="event_timestamp", freq=freq), "page_id"])
        .agg(
            revision_count=("event_timestamp", "size"),
            page_title=("page_title", "first"),
            page_is_deleted=("page_is_deleted", "last"),
        )
        .reset_index()
    )
    user_counts = (
        revisions[revisions.event_user_id.notna()]
        .groupby(
            [pd.Grouper(key="event_timestamp", freq=freq), "page_id", "event_user_id"]
        )
        .agg(user_counts=("event_timestamp", "size"))
    ).reset_index()
    user_counts = (
        user_counts.groupby(["event_timestamp", "page_id"])
        .agg(num_unique_users=("event_timestamp", "size"))
        .reset_index()
    )
    counts = counts.merge(user_counts, on=["event_timestamp", "page_id"])
    return counts


def filter_for_manual_labeling(counts):
    # In future should parameterize this and treat as a hyperparameter
    return counts[(counts.num_unique_users > 1) & (counts.revision_count >= 15)]


def generate_url(counts):
    # Generate history link for easy labeling
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

    revisions = load_and_clean(filepath)
    counts = bin_and_count(revisions, freq)
    generate_url(counts)
    filtered_counts = filter_for_manual_labeling(counts)
    filtered_counts.to_csv(outpath)
