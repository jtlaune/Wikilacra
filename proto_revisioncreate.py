import sys
sys.path.insert(0, "/workspaces/Wikilacra/")
import wikilacra.listen
from wikilacra.listen import REVISIONCREATE_COLS

wikilacra.listen.stream_listen(
    "https://stream.wikimedia.org/v2/stream/revision-create/",
    "1hrtest.sqlite",
    wikilacra.listen.recentchange_filter,
    ["meta", "length", "revision"],
    [],
    ,
    "recentchange",
    batch_size=100,
)