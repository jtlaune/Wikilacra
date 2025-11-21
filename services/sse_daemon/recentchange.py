import sys
from wikilacra.listen import stream_listen, recentchange_filter, RECENTCHANGE_COLS

isodt_start = sys.argv[1]

stream_listen(
    f"https://stream.wikimedia.org/v2/stream/recentchange?since={isodt_start}",
    "/app/data/db.sqlite",
    recentchange_filter,
    ["meta", "length", "revision"],
    [],
    RECENTCHANGE_COLS,
    "recentchange",
    batch_size=100,
)
