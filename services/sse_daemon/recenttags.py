import sys
from wikilacra.listen import stream_listen, recenttag_filter, RECENTTAGS_COLS

isodt_start = sys.argv[1]

stream_listen(
    f"https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change?since={isodt_start}",
    "/app/data/db.sqlite",
    recenttag_filter,
    ["meta", "prior_state"],
    ["performer"],
    RECENTTAGS_COLS,
    "recenttags",
    batch_size=100,
)
