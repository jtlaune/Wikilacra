import sys

sys.path.insert(0, "/workspaces/Wikilacra/")
import wikilacra.listen
from wikilacra.listen import RECENTCHANGE_COLS

isodt_start = sys.argv[1]

wikilacra.listen.stream_listen(
    f"https://stream.wikimedia.org/v2/stream/recentchange?since={isodt_start}",
    "test.sqlite",
    wikilacra.listen.recentchange_filter,
    ["meta", "length", "revision"],
    [],
    RECENTCHANGE_COLS,
    "recentchange",
    batch_size=100,
)
