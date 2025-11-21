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
    coalesce_select,
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

    exog_cols, endog_cols, engineered = coalesce_select(
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
    )

    # Save to directory output_dir with filename output_basename.csv
    engineered.to_csv(os.path.join(output_dir, output_basename + ".csv"))

    # Write the endog/exog columns to files to be loaded in other pipelines,
    # prefixed by output_basename_*
    with open(os.path.join(output_dir, output_basename + "_exog_cols.txt"), "w") as f:
        f.write(str(exog_cols))
    with open(os.path.join(output_dir, output_basename + "_endog_cols.txt"), "w") as f:
        f.write(str(endog_cols))
