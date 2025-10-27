import pandas as pd
from numpy import log


def entropy_from_series(s):
    """Compute entropy from a Series

    Args:
        s (Series): Series of counts where p=count/Sum(counts)

    Returns:
        float: Entropy (base e)
    """
    tot = s.sum()
    p = s.div(tot)
    user_entropy = (-p * log(p)).sum()
    return user_entropy


def compute_intracounts_stats(df, intra_freq, max_burst_roll):
    """df = revisions, intra_freq =
    intra-hour bin frequency, e.g. 1min, should divide 60,
    max_burst_roll=rolling window for max_burst calculation in units of
    intra_freq, i.e. 15"""
    # Do it by resampling the whole thing at intra_freq frequency
    sel_df = df.copy()
    sel_df["intra_time"] = sel_df["event_timestamp"]
    out = (
        sel_df.groupby(
            [
                "page_title",
                pd.Grouper(key="event_timestamp", freq="1h"),  # the one all others use
                pd.Grouper(
                    key="intra_time", freq=intra_freq
                ),  # the intra-hour frequency
            ]
        )
        .size()
        .reset_index(name="count")
        .groupby(["page_title", "event_timestamp"])[["intra_time", "count"]]
        .apply(
            lambda d: (
                d.set_index("intra_time")["count"]
                .resample("1min")
                .sum()
                .reindex(
                    pd.date_range(
                        d["intra_time"].min().floor("h"),
                        d["intra_time"].min().floor("h")
                        + pd.Timedelta(hours=1)
                        - pd.Timedelta(intra_freq),
                        freq="1min",
                    ),
                    fill_value=0,
                )
            )
        )
        .reset_index()
    ).rename(columns={"level_2": "intra_time"})
    intracounts = pd.DataFrame([])
    intracounts["max_burst_score"] = (
        out.groupby(["page_title", "event_timestamp"])["count"]
        .rolling(max_burst_roll, min_periods=1)
        .sum()
        .groupby(level=[0, 1])
        .max()
    )
    # add more intra-hour statistics here...
    intracounts.reset_index(inplace=True)
    return intracounts


def get_glob_quants(df, bin_pd):
    """df = revisions"""
    dt_glob = (
        df.groupby(
            [
                pd.Grouper(key="event_timestamp", freq=bin_pd),
                "page_title",
            ]
        )
        .agg(rev_cnt=("event_timestamp", "count"))
        .groupby(level="event_timestamp")
        .agg(
            num_pages=("rev_cnt", "size"),
            page_ent=("rev_cnt", entropy_from_series),
            rev_vol=("rev_cnt", "sum"),
        )
    )
    # Could put in top 25 page stats here... Like average number of revisions
    # Can use this to get meaningful page shares due to long tail
    return dt_glob


def get_page_quants(df, bin_pd):
    """df = revisions of selected pages, bin_pd=freq of binning counts, usually
    1hr, adds new columns, including for tags"""
    df = df.copy(deep=False)

    df["rev_mobile_edit"] = df["revision_tags"].fillna("").str.contains("mobile edit")
    df["rev_mobile_web_edit"] = (
        df["revision_tags"].fillna("").str.contains("mobile web edit")
    )

    df["event_user_is_permanent_clean"] = (
        df["event_user_is_permanent"].replace("true", True).fillna(0) * 1
    )

    counts_usr = df.groupby(
        [
            pd.Grouper(key="event_timestamp", freq=bin_pd),
            "page_title",
            "event_user_text_historical",
        ]
    ).agg(
        user_rev_cnt=("event_timestamp", "count"),
        user_anon_cnt=("event_user_is_anonymous", "sum"),
        user_perm_cnt=("event_user_is_permanent_clean", "sum"),
        user_revert_cnt=("revision_is_identity_revert", "sum"),
        user_minor_rev_cnt=("revision_minor_edit", "sum"),
        page_creation=("page_creation_timestamp", "first"),
        user_rev_diff_bytes_tot=("revision_text_bytes_diff", "sum"),
        user_rev_bytes_tot=("revision_text_bytes", "sum"),
        user_mob_edit=("rev_mobile_edit", "sum"),
        user_web_mob_edit=("rev_mobile_web_edit", "sum"),
        # Insert more stats here ...
        # =("", ""),
    )

    dt_page = counts_usr.groupby(level=["event_timestamp", "page_title"]).agg(
        user_ent=("user_rev_cnt", entropy_from_series),
        rev_cnt=("user_rev_cnt", "sum"),
        user_cnt=("user_rev_cnt", "count"),
        max_cnt_user=("user_rev_cnt", "max"),
        anon_cnt=("user_anon_cnt", "sum"),
        perm_cnt=("user_perm_cnt", "sum"),
        revert_cnt=("user_revert_cnt", "sum"),
        minor_cnt=("user_minor_rev_cnt", "sum"),
        page_creation=("page_creation", "first"),
        rev_diff_bytes_total=("user_rev_diff_bytes_tot", "sum"),
        rev_bytes_total=("user_rev_bytes_tot", "sum"),
        mob_edits=("user_mob_edit", "sum"),
        web_mob_edits=("user_web_mob_edit", "sum"),
        # Insert more stats here ...
    )

    dt_page = dt_page.reset_index()
    dt_page["hrs_since_page_creation"] = (
        dt_page["event_timestamp"]
        + pd.Timedelta("1h")
        - pd.to_datetime(dt_page["page_creation"])
    )

    # Normalizing bool sum columns
    dt_page["anon_cnt"] = dt_page["anon_cnt"] * 1
    dt_page["perm_cnt"] = dt_page["perm_cnt"] * 1
    dt_page["revert_cnt"] = dt_page["revert_cnt"] * 1
    dt_page["minor_cnt"] = dt_page["minor_cnt"] * 1

    # Insert more stats here...

    return dt_page.set_index(["event_timestamp", "page_title"])


def construct_counts(df, start_dt, end_dt, bin_pd):
    """df = dt_page (for threshold events only), start_dt,end_dt=timestamps,
    bin_pd=binning for avg/var/..., roll_pd=rolling window length"""

    grid = pd.MultiIndex.from_product(
        [
            pd.date_range(start_dt, end_dt, freq=bin_pd),
            pd.Index(df.index.get_level_values(1).unique()),
        ],
        names=["event_timestamp", "page_title"],
    )
    counts_filled = df.reindex(grid, fill_value=0)

    return counts_filled


def compute_windows(df, roll_pd_hrs, cols):
    """df=counts, roll_pd_hrs=e.g. 6, cols=list of columns to window
    df is indexed [timestamp, name] on a regular grid
    """
    roll = (
        df.reset_index()
        .set_index(
            "event_timestamp"
        )  # unclear why this is so painful.. not doing it this way results in duplicate columns
        .groupby("page_title")
        .rolling(f"{roll_pd_hrs}h", min_periods=roll_pd_hrs)[cols]
        .agg(["sum", "mean", "var", "skew", "kurt", "corr", "cov"])
    )
    expwin = (
        df.reset_index()
        .set_index("event_timestamp")
        .groupby("page_title")[df.columns]
        .ewm(span=roll_pd_hrs, min_periods=roll_pd_hrs)[cols]
        .agg(["mean", "var", "std", "corr", "cov"])
    )
    return roll, expwin


def get_lagged_and_slopes(df, N_lags, cols):
    """df = counts, N_lags= e.g. 6
    df is indexed [timestamp, name] on a regular grid"""
    df_lags = pd.concat(
        [
            df[cols]
            .groupby("page_title")
            .shift(l)
            .rename(columns=lambda c: f"lag{l}_{c}" if l > 0 else c)
            for l in range(1, N_lags + 1)
        ],
        axis=1,
    )
    slopes = pd.concat(
        [
            (
                (
                    df[cols]
                    - df_lags[f"lag{s}_" + df[cols].columns].set_axis(
                        df[cols].columns, axis=1
                    )
                )
                / s
            ).set_axis(f"slope{s}_" + df[cols].columns, axis=1)
            for s in range(1, N_lags + 1)
        ],
        axis=1,
    )
    return df_lags, slopes


def create_feature_columns(labels, rolling_hrs):
    """_summary_

    Args:
        labels (DataFrame): Labeled data DataFrame to be modified.
        rolling_hrs (int): Rolling average length in hours.

    Returns:
        dict: dictionary of column names (to be passed as kwarg)
    """
    colnames = {
        "hist_user_entropy_colname": f"avg_user_entropy_past_{rolling_hrs}_hrs",
        "hist_user_count_colname": f"avg_user_count_past_{rolling_hrs}_hrs",
        "hist_page_entropy_colname": f"avg_page_entropy_past_{rolling_hrs}_hrs",
        "hist_revision_count_colname": f"avg_revision_count_past_{rolling_hrs}_hrs",
    }

    labels["user_entropy_cur"] = None
    labels[colnames["hist_user_entropy_colname"]] = None

    labels["user_count_cur"] = None
    labels[colnames["hist_user_count_colname"]] = None

    labels["page_entropy_cur"] = None
    labels[colnames["hist_page_entropy_colname"]] = None

    labels[colnames["hist_revision_count_colname"]] = None

    labels["num_pages_cur"] = None
    labels["page_share_cur"] = None
    labels["revision_vol_cur"] = None

    labels["max_burst_score"] = None

    return colnames


def engineer_common_training(df):
    """Do common transformations on columns, e.g. taking logs.

    Args:
        df (DataFrame): The training data.

    Returns:
        DataFrame: The transformed training data (in place).
    """
    df["log_page_share_cur"] = log(df["page_share_cur"])
    return df


def engineer_features_by_event(
    row, idx, counts, page_entropies, revisions, labels, rolling_hrs, **colnames
):
    """Engineer the features for a row (event) at index idx.

    Args:
        row (Series): Row's Series from labels.iterrows()
        idx (int): Index from labels.iterrows()
        counts (DataFrame): Binned/counted data.
        page_entropies (DataFrame): Pre-computed page entropy statistics.
        revisions (DataFrame): Dictionary of revision data.
        labels (DataFrame): labeled data
        rolling_hrs (int): rolling average span in hours
        colnames (dict): names of the historical average columns,
                         generated by create_feature_columns
    """
    ts2 = row["event_timestamp"]
    ts1 = ts2 - pd.Timedelta(hours=rolling_hrs - 1)

    # Revisions for this page in this time period
    filtered = revisions[
        (
            revisions.event_timestamp.between(
                ts1, ts2 + pd.Timedelta(hours=1), inclusive="right"
            )
        )
        & (revisions.page_id == row.page_id)
    ]

    # Revisions binned by user and hourly timestamp
    binned_filtered = (
        filtered.groupby(
            [pd.Grouper(key="event_timestamp", freq="h"), "event_user_text_historical"]
        )
        .agg(num_edits_by_user=("event_timestamp", "size"))
        .reset_index()
    )

    # Count the number of unique users in each timestamp and compute historical average
    unique_users = binned_filtered.groupby("event_timestamp").agg(
        num_unique_users=("event_timestamp", "size")
    )["num_unique_users"]
    labels.loc[idx, colnames["hist_user_count_colname"]] = unique_users.iloc[
        :-1
    ].sum() / (rolling_hrs - 1)
    labels.loc[idx, "user_count_cur"] = unique_users.iloc[-1]

    # Compute & set user entropy
    binned_filtered["user_entropy_per_hour"] = binned_filtered.groupby(
        "event_timestamp"
    )["num_edits_by_user"].transform(entropy_from_series)
    labels.loc[idx, "user_entropy_cur"] = binned_filtered.tail(1)[
        "user_entropy_per_hour"
    ].values

    _ = (
        binned_filtered.groupby("event_timestamp")
        .agg(test=("user_entropy_per_hour", "first"))
        .reset_index()["test"]
        .iloc[:-1]
        .values
    )
    if _.size == 0:
        labels.loc[idx, colnames["hist_user_entropy_colname"]] = 0
    else:
        labels.loc[idx, colnames["hist_user_entropy_colname"]] = _.mean()

    # Retrieve the corresponding current page entropy from page_entropies
    labels.loc[idx, "page_entropy_cur"] = page_entropies.loc[
        row["event_timestamp"]
    ].values
    labels.loc[idx, colnames["hist_page_entropy_colname"]] = page_entropies.loc[
        ts1 : ts2 - pd.Timedelta(hours=1)
    ].values.mean()

    # Retrieve the total number of pages being edited from counts
    labels.loc[idx, "num_pages_cur"] = (
        counts.groupby(["event_timestamp"])
        .agg(num_page_cur=("page_title", "size"))["num_page_cur"]
        .loc[ts2]
    )

    # Retrieve the total revision volume in this timestamp from counts
    # Compute the historical revision volume
    # Compute this page's share of the revision volume
    labels.loc[idx, "revision_vol_cur"] = (
        counts.groupby(["event_timestamp"])
        .agg(revision_vol_cur=("revision_count", "sum"))["revision_vol_cur"]
        .loc[ts2]
    )

    labels.loc[idx, colnames["hist_revision_count_colname"]] = counts[
        (counts.page_title == row["page_title"])
        & (counts.event_timestamp.between(ts1, ts2 - pd.Timedelta(hours=1)))
    ].num_unique_users.sum() / (rolling_hrs - 1)

    # TODO: fix the typing of the columns so Pylance doesn't throw an error here
    labels.loc[idx, "page_share_cur"] = labels.loc[idx, "revision_count"].astype(
        "float"
    ) / labels.loc[idx, "revision_vol_cur"].astype("float")

    # Compute the max burstiness score for 5 minute periods in the past hour
    filtered_past1hr = filtered[
        filtered.event_timestamp.between(
            ts2 - pd.Timedelta(hours=1), ts2 + pd.Timedelta(hours=1)
        )
    ]
    # TODO: Make this mutable w/ arguments passed
    binned_past1hr = filtered_past1hr.groupby(
        pd.Grouper(key="event_timestamp", freq="5Min")
    ).agg(revision_count=("event_timestamp", "size"))
    labels.loc[idx, "max_burst_score"] = (
        binned_past1hr["revision_count"].div(labels.loc[idx, "revision_count"]) - 1 / 12
    ).max()
