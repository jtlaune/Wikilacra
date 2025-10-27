"""
Engineer features.
"""

import sys
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
    # Inputs
    rolling_avg_hrs = int(sys.argv[1])
    dump_input_file = sys.argv[2]
    labels_only_input_file = sys.argv[3]
    counts_path = sys.argv[4]
    output_path = sys.argv[5]

    # Get the labeled data
    labels = pd.read_csv(labels_only_input_file, index_col=0)
    # Selected titles for reading all columns
    sel_titles = set(labels["page_title"].unique())

    # Read in only 2 columns for the whole revisisions list to save memory
    revisions_glob = read_data_chunked(
        dump_input_file,
        ["event_timestamp","page_title"],
        MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
    )

    # Read all the columns in for those in the labeled set
    revisions_sel = read_data_chunked(
        "../inputs/label_data/2025-08.enwiki.2025-08.tsv",
        MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
        MEDIAWIKI_HISTOR_DUMP_COL_NAMES,
        page_titles=sel_titles,
    )

    # Global count quantities
    dt_glob = get_glob_quants(revisions_glob, "1h")

    # Labeled-page-specific count quantities
    dt_page = get_page_quants(revisions_sel,"1h")
    counts = construct_counts(
        dt_page,
        "2025-08-01 00:00:00", # TODO: make these inputtable 
        "2025-08-31 23:59:59",
        "1h",
    )

    # Numerical columns selection (for rolling, lagged, expwin, and intracounts)
    num_cols = list(set(counts.columns)-{"page_creation","hrs_since_page_creation"})

    # Rolling & exponential windows
    roll, expwin = compute_windows(counts, ROLLING_HRS, num_cols)

    # Lags and slopes
    lags, slopes = get_lagged_and_slopes(counts, ROLLING_HRS, num_cols)

    # Intra-hour statistics, e.g. max burst score
    intracounts = compute_intracounts_stats(
        revisions_sel, INTRA_PD, INTRA_WIN
    )

    # TODO: write all of it into `engineered` dataframe.

    engineered.to_csv(output_path)
