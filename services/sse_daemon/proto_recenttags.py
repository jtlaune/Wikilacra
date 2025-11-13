import sys

sys.path.insert(0, "/workspaces/Wikilacra/")
import wikilacra.listen
from wikilacra.listen import RECENTTAGS_COLS

isodt_start = sys.argv[1]

wikilacra.listen.stream_listen(
    f"https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change?since={isodt_start}",
    "test.sqlite",
    wikilacra.listen.recenttag_filter,
    ["meta", "prior_state"],
    ["performer"],
    RECENTTAGS_COLS,
    "recenttags",
    batch_size=100,
)
