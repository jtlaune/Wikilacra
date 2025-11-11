import sys

sys.path.insert(0, "/workspaces/Wikilacra/")
import wikilacra.listen
from wikilacra.listen import REVISIONCREATE_COLS

isodt_start = sys.argv[1]

wikilacra.listen.stream_listen(
    f"https://stream.wikimedia.org/v2/stream/revision-create?since={isodt_start}",
    "test.sqlite",
    wikilacra.listen.revisioncreate_filter,
    ["meta", "performer"],
    ["rev_slots"],
    REVISIONCREATE_COLS,
    "revisioncreate",
    batch_size=100,
)
