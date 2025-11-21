import sys
from wikilacra.listen import stream_listen, revisioncreate_filter, REVISIONCREATE_COLS

isodt_start = sys.argv[1]

stream_listen(
    f"https://stream.wikimedia.org/v2/stream/revision-create?since={isodt_start}",
    "/app/data/db.sqlite",
    revisioncreate_filter,
    ["meta", "performer"],
    ["rev_slots"],
    REVISIONCREATE_COLS,
    "revisioncreate",
    batch_size=100,
)
