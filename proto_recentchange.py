import sys
sys.path.insert(0, "/workspaces/Wikilacra/")
import wikilacra.listen
from wikilacra.listen import RECENTCHANGE_COLS

wikilacra.listen.stream_listen(
    "https://stream.wikimedia.org/v2/stream/recentchange",
    "1hrtest.sqlite",
    wikilacra.listen.recentchange_filter,
    ["meta", "length", "revision"],
    [],
    RECENTCHANGE_COLS,
    "recentchange",
    batch_size=100,
)