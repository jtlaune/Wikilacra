import sys

sys.path.insert(0, "/workspaces/Wikilacra/")
import wikilacra.listen
from wikilacra.listen import RECENTTAGS_COLS

wikilacra.listen.stream_listen(
    "https://stream.wikimedia.org/v2/stream/mediawiki.revision-tags-change",
    "1hrtest.sqlite",
    wikilacra.listen.recenttag_filter,
    ["meta", "prior_state"],
    ["performer"],
    RECENTTAGS_COLS,
    "recenttags",
    batch_size=100,
)
