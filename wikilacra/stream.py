import pandas as pd


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
