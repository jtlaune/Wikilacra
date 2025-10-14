"""
Module of helper functions for the WikiMedia History dump files and EventStream
data streams.

read_data_chunked: read large dump file chunked bin_and_count: bin the revisions
by frequency
"""

import pandas as pd
from wikilacra import MEDIAWIKI_HISTOR_DUMP_COL_NAMES


def read_data_chunked(
    fn, columns_to_keep, columns_to_read, edit_type="revision", chunksize=250_000
):
    """Stream and filter the dump to stay within memory limits. Read dump file
    and retrieve edits."""

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
        mask = chunk["event_entity"].eq(edit_type) & chunk["page_namespace"].eq(0)
        mask &= ~title.str.contains(r"/sandbox", na=False)
        mask &= ~title.str.fullmatch(r"Sandbox", na=False)
        mask &= ~title.str.fullmatch(r"Undefined/junk", na=False)
        mask &= ~title.str.fullmatch(r"Wiki", na=False)

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


def bin_and_count(stream, freq):
    """Bin number of edits by hour and compute number.

    Counts number of edits and unique users editing the page during window of
    length freq. Intended to be used with WikiMedia Dumps data files, but will
    probably be extended to EventStream data.

    Args:
        stream (DataFrame): DataFrame with "event_timestamp", "page_title",
                            "page_is_deleted", "page_id"
        freq (Timedelta): Timedelta with which to count.
    """
    counts = (
        stream[stream.event_user_id.notna()]
        .groupby([pd.Grouper(key="event_timestamp", freq=freq), "page_id"])
        .agg(
            revision_count=("event_timestamp", "size"),
            page_title=("page_title", "first"),
            page_is_deleted=("page_is_deleted", "last"),
        )
        .reset_index()
    )
    user_counts = (
        stream[stream.event_user_id.notna()]
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
