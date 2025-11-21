import os
from wikilacra.listen import create_tables

if os.path.exists("/app/data/db.sqlite"):
    os.remove("/app/data/db.sqlite")
create_tables("/app/data/db.sqlite")