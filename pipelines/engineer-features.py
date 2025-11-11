"""
Engineer features.
"""

import sys
import os
from extract_data_to_label import load_and_clean
import pandas as pd
from wikilacra import MEDIAWIKI_HISTOR_DUMP_COL_NAMES
from wikilacra.features import (
    get_glob_quants,
    get_page_quants,
    construct_counts,
    compute_windows,
    get_lagged_and_slopes,
    compute_intracounts_stats,
)
from wikilacra.stream import read_data_chunked

if __name__ == "__main__":
    ##########
    # Inputs #
    ##########
    # Timespan to take the rolling window and exponential window statistics
    rolling_avg_hrs = int(sys.argv[1])
    # period to bin for intra-hour count statistics, like max burst score
    intra_pd = str(sys.argv[2])
    # window in units of intra_pd for intra-hour count statistics, like max burst score
    intra_win = int(sys.argv[3])
    # input file for all revisions
    dump_input_file = sys.argv[4]
    # labeled data
    labels_only_input_file = sys.argv[5]
    # output path for the engineered features for the labeled data
    output_dir = sys.argv[6]
    output_basename = sys.argv[7]
    start_dt = pd.to_datetime(sys.argv[8])
    end_dt = pd.to_datetime(sys.argv[9])

    ################
    # Loading Data #
    ################
    # Get the labeled data
    labels = pd.read_csv(labels_only_input_file, index_col=0)
    # Selected titles for reading all columns
    sel_titles = set(labels["page_title"].unique())
    # Read in only 2 columns for the whole revisions list to save memory
    # TODO: Make this use the counts outname from the label_data_extract pipeline
    revisions_glob = read_data_chunked(
        dump_input_file,
        ["event_timestamp", "page_title"],
        MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
        chunksize=100_000,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    # Read all the columns in for those in the labeled set
    revisions_sel = read_data_chunked(
        dump_input_file,
        MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
        MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
        page_titles=sel_titles,
        chunksize=100_000,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    ##################
    # Computing Data #
    ##################
    # Global count quantities
    dt_glob = get_glob_quants(revisions_glob, "1h")
    # Labeled-page-specific count quantities
    dt_page = get_page_quants(revisions_sel, "1h")
    counts = construct_counts(
        dt_page,
        start_dt,
        end_dt,
        "1h",
    )
    # Numerical columns selection (for rolling, lagged, expwin, and intracounts)
    num_cols = list(set(counts.columns))
    # Rolling & exponential windows
    roll, expwin = compute_windows(counts, rolling_avg_hrs, num_cols)
    # Lags and slopes
    lags, slopes = get_lagged_and_slopes(counts, rolling_avg_hrs, num_cols)
    # Intra-hour statistics, e.g. max burst score
    intracounts = compute_intracounts_stats(revisions_sel, intra_pd, intra_win)

    ###################
    # Engineered Data #
    ###################
    CLASSES = ["EVENT", "EDIT_WAR", "VANDALISM", "NONE", "MOVED_OR_DELETED"]

    engineered = labels.copy()
    # Select the date range, ensuring event_timestamp is a datetime (otherwise
    # it will default to lexicographical)
    engineered["event_timestamp"] = pd.to_datetime(engineered["event_timestamp"])
    engineered = engineered[
        (engineered["event_timestamp"] >= start_dt)
        & (engineered["event_timestamp"] <= end_dt)
    ]

    # Drop typo-labeled data
    to_drop = engineered[
        ~engineered["EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED"].isin(CLASSES)
    ]
    engineered = engineered.drop(to_drop.index)

    # Get targets for events
    engineered["target"] = (
        engineered["EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED"] == "EVENT"
    ) * 1
    engineered = engineered.drop(
        columns="EVENT, EDIT_WAR, VANDALISM, NONE, MOVED_OR_DELETED"
    )
    # Drop everything but the index columns and target column
    engineered = engineered[["event_timestamp", "page_title", "target"]]
    engineered = engineered.reset_index(drop=True)

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

    # Save to directory output_dir with filename output_basename.csv
    engineered.to_csv(os.path.join(output_dir, output_basename + ".csv"))

    # Write the endog/exog columns to files to be loaded in other pipelines,
    # prefixed by output_basename_*
    with open(os.path.join(output_dir, output_basename + "_exog_cols.txt"), "w") as f:
        f.write(str(exog_cols))
    with open(os.path.join(output_dir, output_basename + "_endog_cols.txt"), "w") as f:
        f.write(str(endog_cols))
