"""Scaling helper functions for classifiers.
"""
from numpy import tanh, log1p


def scaler(df):
    """
    Takes a dataframe, copies it, scales count columns with log1p, and slope
    columns with tanh. It turns boolean columns into integers.

    This was created manually with a list of the feature columns. If you add
    more, they need to be handled here. Unscaled columns are left in for manual
    inspection.
    """
    scaled = df.copy()
    scaled["avg_rev_per_page_cur"] = log1p(scaled["avg_rev_per_page_cur"])
    scaled["count_1hr_lag"] = log1p(scaled["count_1hr_lag"])
    scaled["count_1hr_slope"] = tanh(scaled["count_1hr_slope"])
    scaled["count_2hr_lag"] = log1p(scaled["count_2hr_lag"])
    scaled["count_2hr_slope"] = tanh(scaled["count_2hr_slope"])
    scaled["count_3hr_lag"] = log1p(scaled["count_3hr_lag"])
    scaled["count_3hr_slope"] = tanh(scaled["count_3hr_slope"])
    scaled["cur_yr_in_title"] = scaled["cur_yr_in_title"].astype(int)
    scaled["ema_count_mean"] = log1p(scaled["ema_count_mean"])
    scaled["ema_count_var"] = log1p(scaled["ema_count_var"])
    scaled["max_burst_cur"] = scaled["max_burst_cur"]
    scaled["max_user_edits_cur"] = log1p(scaled["max_user_edits_cur"])
    scaled["minor_revs_pct"] = scaled["minor_revs_pct"]
    scaled["mob_revs_pct"] = scaled["mob_revs_pct"]
    scaled["num_pages_cur"] = log1p(scaled["num_pages_cur"])
    scaled["older_1day"] = scaled["older_1day"].astype(int)
    scaled["older_2hrs"] = scaled["older_2hrs"].astype(int)
    scaled["older_8hrs"] = scaled["older_8hrs"].astype(int)
    scaled["page_ent_cur"] = scaled["page_ent_cur"]
    scaled["page_share_cur"] = log1p(scaled["page_share_cur"])
    scaled["perm_revs_pct"] = scaled["perm_revs_pct"]
    scaled["rev_avg_bytes_tot"] = scaled["rev_avg_bytes_tot"]
    scaled["rev_cnt_cur"] = scaled["rev_cnt_cur"]
    scaled["rev_diff_avg_bytes"] = tanh(scaled["rev_diff_avg_bytes"])
    scaled["rev_vol_cur"] = log1p(scaled["rev_vol_cur"])
    scaled["revert_revs_pct"] = scaled["revert_revs_pct"]
    scaled["roll_count_kurt"] = log1p(scaled["roll_count_kurt"])
    scaled["roll_count_mean"] = log1p(scaled["roll_count_mean"])
    scaled["roll_count_skew"] = log1p(scaled["roll_count_skew"])
    scaled["roll_count_var"] = log1p(scaled["roll_count_var"])
    scaled["roll_revert_count_mean"] = log1p(scaled["roll_revert_count_mean"])
    scaled["roll_user_count_mean"] = log1p(scaled["roll_user_count_mean"])
    scaled["roll_user_ent_mean"] = scaled["roll_user_ent_mean"]
    scaled["user_count_cur"] = log1p(scaled["user_count_cur"])
    scaled["user_ent_cur"] = scaled["user_ent_cur"]
    scaled["web_mob_revs_pct"] = scaled["web_mob_revs_pct"]
    return scaled
