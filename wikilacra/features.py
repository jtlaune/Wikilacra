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
        user_perm_cnt=("event_user_is_permanent_clean", "sum"),
        user_revert_cnt=("revision_is_identity_revert", "sum"),
        user_minor_rev_cnt=("revision_minor_edit", "sum"),
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
        perm_cnt=("user_perm_cnt", "sum"),
        revert_cnt=("user_revert_cnt", "sum"),
        minor_cnt=("user_minor_rev_cnt", "sum"),
        rev_diff_bytes_total=("user_rev_diff_bytes_tot", "sum"),
        rev_bytes_total=("user_rev_bytes_tot", "sum"),
        mob_edits=("user_mob_edit", "sum"),
        web_mob_edits=("user_web_mob_edit", "sum"),
        # Insert more stats here ...
    )

    dt_page = dt_page.reset_index()

    # Normalizing bool sum columns
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


def coalesce_select(
    engineered,
    start_dt,
    end_dt,
    dt_glob,
    dt_page,
    slopes,
    lags,
    roll,
    expwin,
    intracounts,
):
    # Select the date range, ensuring event_timestamp is a datetime (otherwise
    # it will default to lexicographical)
    engineered["event_timestamp"] = pd.to_datetime(engineered["event_timestamp"])
    engineered = engineered[
        (engineered["event_timestamp"] >= start_dt)
        & (engineered["event_timestamp"] <= end_dt)
    ]

    ###########
    # Globals #
    ###########
    X = dt_glob.loc[engineered["event_timestamp"]].reset_index(drop=True)
    engineered[["num_pages_cur", "page_ent_cur", "rev_vol_cur"]] = X[
        ["num_pages", "page_ent", "rev_vol"]
    ]
    engineered["avg_rev_per_page_cur"] = engineered["rev_vol_cur"].div(
        engineered["num_pages_cur"], axis=0
    )

    #############
    # Page data #
    #############
    X = dt_page.loc[
        pd.MultiIndex.from_frame(engineered[["event_timestamp", "page_title"]])
    ].reset_index(drop=True)
    engineered[
        [
            "rev_cnt_cur",
            "user_count_cur",
            "user_ent_cur",
            "max_user_edits_cur",
        ]
    ] = X[
        [
            "rev_cnt",
            "user_cnt",
            "user_ent",
            "max_cnt_user",
        ]
    ]

    # Percentages
    engineered[
        [
            "perm_revs_pct",
            "revert_revs_pct",
            "minor_revs_pct",
            "web_mob_revs_pct",
            "mob_revs_pct",
            "rev_diff_avg_bytes",
            "rev_avg_bytes_tot",
        ]
    ] = X[
        [
            "perm_cnt",
            "revert_cnt",
            "minor_cnt",
            "web_mob_edits",
            "mob_edits",
            "rev_diff_bytes_total",
            "rev_bytes_total",
        ]
    ].divide(
        X["rev_cnt"], axis=0
    )
    engineered["page_share_cur"] = engineered["rev_cnt_cur"].div(
        engineered["rev_vol_cur"], axis=0
    )

    #####################
    # Page title string #
    #####################
    engineered["cur_yr_in_title"] = engineered["page_title"].str.contains(
        str(pd.Timestamp.now().year)
    )

    ######################
    # Exponential Window #
    ######################
    X = expwin.loc[
        pd.MultiIndex.from_frame(engineered[["page_title", "event_timestamp"]])
    ].reset_index(drop=True)
    engineered[
        [
            "ema_count_mean",
            "ema_count_var",
        ]
    ] = X[
        [
            ("rev_cnt", "mean"),
            ("rev_cnt", "var"),
        ]
    ]

    ##########
    # Slopes #
    ##########
    X = slopes.loc[
        pd.MultiIndex.from_frame(engineered[["event_timestamp", "page_title"]])
    ].reset_index(drop=True)
    engineered[
        [
            "count_1hr_slope",
            "count_2hr_slope",
            "count_3hr_slope",
        ]
    ] = X[
        [
            "slope1_rev_cnt",
            "slope2_rev_cnt",
            "slope3_rev_cnt",
        ]
    ]

    ########
    # Lags #
    ########
    X = lags.loc[
        pd.MultiIndex.from_frame(engineered[["event_timestamp", "page_title"]])
    ].reset_index(drop=True)
    engineered[
        [
            "count_1hr_lag",
            "count_2hr_lag",
            "count_3hr_lag",
        ]
    ] = X[
        [
            "lag1_rev_cnt",
            "lag2_rev_cnt",
            "lag3_rev_cnt",
        ]
    ]

    ###########
    # Rolling #
    ###########
    X = roll.loc[
        pd.MultiIndex.from_frame(engineered[["page_title", "event_timestamp"]])
    ].reset_index(drop=True)
    engineered[
        [
            "roll_count_mean",
            "roll_count_var",
            "roll_count_skew",
            "roll_count_kurt",
            "roll_user_ent_mean",
            "roll_user_count_mean",
            "roll_revert_count_mean",
        ]
    ] = X[
        [
            ("rev_cnt", "mean"),
            ("rev_cnt", "var"),
            ("rev_cnt", "skew"),
            ("rev_cnt", "kurt"),
            ("user_ent", "mean"),
            ("user_cnt", "mean"),
            ("revert_cnt", "mean"),
        ]
    ]

    ###############
    # Intracounts #
    ###############
    X = (
        intracounts.set_index(["event_timestamp", "page_title"])
        .loc[pd.MultiIndex.from_frame(engineered[["event_timestamp", "page_title"]])]
        .reset_index(drop=True)
    )
    engineered["max_burst_cur"] = X["max_burst_score"]

    #######################################
    # Drop NaNs from rolling windows, etc #
    #######################################
    engineered = engineered[~engineered.isna().any(axis=1)]

    ##########################
    # Get useful column sets #
    ##########################
    drop_cols = []
    engineered = engineered.drop(columns=drop_cols)

    cat_cols = {
        "cur_yr_in_title",
    }
    num_cols = list(
        set(engineered.columns)
        - {
            "event_timestamp",
            "target",
            "page_title",
        }
        - cat_cols
    )
    cat_cols = list(cat_cols)
    exog_cols = list(
        set(engineered.columns)
        - {
            "event_timestamp",
            "page_title",
            "target",
        }
    )
    endog_cols = ["target"]

    ##########
    # Typing #
    ##########
    engineered[num_cols] = engineered[num_cols].astype("float")
    engineered[cat_cols] = engineered[cat_cols].astype("category")

    return exog_cols, endog_cols, engineered