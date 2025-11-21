from os import getenv
import requests
import json
from mlflow.sklearn import load_model
from mlflow import set_registry_uri, set_tracking_uri, get_tracking_uri
from urllib.parse import urlparse
import pandas as pd
from time import sleep
from wikilacra.features import (
    get_glob_quants,
    coalesce_select,
    get_page_quants,
    construct_counts,
    compute_windows,
    get_lagged_and_slopes,
    compute_intracounts_stats,
)
from sqlite3 import connect
from ast import literal_eval
from datetime import datetime, timedelta, timezone

HEADERS = {"User-Agent": "Wikilacra/0.1 (https://jtlaune.github.io; jtlaune@gmail.com)"}
DB_PATH = getenv("SSE_DB_PATH")

model_name = "sklearn-random-forest"
model_version = "latest"

set_tracking_uri("http://mlflow:5000")
set_registry_uri("http://mlflow:5000")

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = load_model(model_uri)
print(model)

con = connect(DB_PATH)
cur = con.cursor()
now_dt = datetime.now(timezone.utc)
start_dt = now_dt - timedelta(hours=16)

tables_timestamps = {
    "revisioncreate": "rev_timestamp",
    "recentchange": "meta_dt",
    "recenttags": "rev_timestamp",
}

for table, ts_col in tables_timestamps.items():
    cur.execute(
        f"""
        DELETE FROM {table}
        WHERE {ts_col} < '{str(start_dt.isoformat(sep="T")).replace("+00:00","Z")}';
        """,
    )

con.commit()

revisions = pd.read_sql_query(
    f"""
    WITH t AS (
        SELECT rev_id, GROUP_CONCAT(prior_state_tags, ", ") as prior_state_tags, GROUP_CONCAT(tags, ", ") as tags
        FROM recenttags
        GROUP BY rev_id
    ) 
    SELECT r.*, t.tags, t.prior_state_tags, c.length_new, c.length_old 
    FROM revisioncreate AS r
    LEFT JOIN t ON r.rev_id = t.rev_id
    LEFT JOIN recentchange as c ON r.rev_id = c.revision_new
    WHERE datetime(r.rev_timestamp) >= datetime('{start_dt.isoformat(sep=" ")}') AND datetime(r.rev_timestamp) < datetime('{now_dt.isoformat(sep=" ")}');
    """,
    con,
)

con.close()

revisions["event_user_is_permanent"] = ~revisions["performer_user_groups"].str.contains(
    '"temp"'
)
revisions["revision_is_identity_revert"] = (
    revisions["tags"].str.contains("mw-undo")
    | revisions["tags"].str.contains("mw-rollback")
    | revisions["tags"].str.contains("mw-manual-revert")
)
revisions["revision_text_bytes_diff"] = (
    revisions["length_new"] - revisions["length_old"]
)
revisions["rev_mobile_web_edit"] = revisions["prior_state_tags"].str.contains(
    "mobile web edit"
) | revisions["tags"].str.contains("mobile web edit")
revisions["rev_mobile_edit"] = revisions["prior_state_tags"].str.contains(
    "mobile edit"
) | revisions["tags"].str.contains("mobile edit")

revisions["rev_minor_edit"] = revisions["rev_minor_edit"].astype(int)
rename_cols = {
    "tags": "revision_tags",
    "performer_user_text": "event_user_text_historical",
    "rev_timestamp": "event_timestamp",
    "rev_minor_edit": "revision_minor_edit",
    "rev_len": "revision_text_bytes",
}
revisions.rename(columns=rename_cols, inplace=True)
revisions["event_timestamp"] = pd.to_datetime(revisions["event_timestamp"])
revisions.columns
dt_glob = get_glob_quants(revisions, "1h")
counts_all = (
    revisions.groupby([pd.Grouper(key="event_timestamp", freq="1h"), "page_title"])
    .agg(
        rev_cnt=("event_timestamp", "size"),
        user_cnt=("event_user_text_historical", "nunique"),
        page_id=("page_id", "first"),
    )
    .reset_index()
)
sel_pages = set(
    counts_all[(counts_all["rev_cnt"] >= 15) & (counts_all["user_cnt"] >= 2)][
        "page_title"
    ]
)

revisions_sel = revisions[revisions["page_title"].isin(sel_pages)]
dt_page = get_page_quants(revisions_sel, "1h")
start_dt = dt_page.reset_index()["event_timestamp"].min()
end_dt = dt_page.reset_index()["event_timestamp"].max()
counts = construct_counts(dt_page, start_dt, end_dt, "1h")
num_cols = list(set(counts.columns))
rolling_avg_hrs = 6
intra_pd = "1Min"
intra_win = 10
roll, expwin = compute_windows(counts, rolling_avg_hrs, num_cols)
lags, slopes = get_lagged_and_slopes(counts, rolling_avg_hrs, num_cols)
intracounts = compute_intracounts_stats(revisions_sel, intra_pd, intra_win)

_ = counts[(counts["rev_cnt"] >= 15) & (counts["user_cnt"] >= 2)]
engineered = _.reset_index()[["event_timestamp", "page_title"]]
engineered = (
    engineered.sort_values(by="event_timestamp", ascending=True)
    .groupby(by=["page_title"])
    .first()
    .reset_index()
)

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


with open("/app/engineered_aug25_exog_cols.txt", "r") as f:
    model_exog_cols = literal_eval(f.readline())
preds = engineered[["event_timestamp", "rev_cnt_cur", "page_title"]]
preds["prediction"] = model.predict(engineered[model_exog_cols])

url = "https://en.wikipedia.org/w/rest.php/v1/search/page"

to_json_list = []
for idx, row in preds.iterrows():
    sleep(0.5)
    params = {"q": row["page_title"], "limit": "1"}
    response = requests.get(url, headers=HEADERS, params=params)
    data = response.json()["pages"][0]
    to_json_list += [
        {
            "prediction": int(row["prediction"]),
            "title": data["title"],
            "description": data["description"],
            "url": "https://en.wikipedia.org/wiki/" + row["page_title"],
            "event_timestamp": str(row["event_timestamp"]),
        }
    ]
    if data["thumbnail"]:
        to_json_list[-1]["thumbnail"] = data["thumbnail"]

with open("/app/output/items.json", "w") as f:
    json.dump(to_json_list, f, indent=2)
