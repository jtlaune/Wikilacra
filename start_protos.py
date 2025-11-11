import os, subprocess, time
from datetime import datetime, timezone
import pytz

dt = datetime.strptime("2025-11-11 03:00:00", "%Y-%m-%d %H:%M:%S")
local_tz = pytz.timezone("America/Chicago")
dt_local = local_tz.localize(dt)
dt_utc = dt_local.astimezone(timezone.utc)
isodt_start = dt_utc.isoformat().replace("+00:00", "Z")

# remove old log files if they exist
for fname in ("changes.log", "revisions.log", "tags.log"):
    if os.path.exists(fname):
        os.remove(fname)

# Open log files for each process
change_log = open("changes.log", "a")
rev_log = open("revisions.log", "a")
tag_log = open("tags.log", "a")

procs = [
    subprocess.Popen(
        ["python", "-u", "proto_recentchange.py", f"{isodt_start}"],
        stdout=change_log,
        stderr=subprocess.STDOUT,
    ),
    subprocess.Popen(
        ["python", "-u", "proto_revisioncreate.py", f"{isodt_start}"],
        stdout=rev_log,
        stderr=subprocess.STDOUT,
    ),
    subprocess.Popen(
        ["python", "-u", "proto_recenttags.py", f"{isodt_start}"],
        stdout=tag_log,
        stderr=subprocess.STDOUT,
    ),
]

try:
    # keep main process alive until Ctrl-C
    i = 0
    while True:
        time.sleep(10)
        for p in procs:
            if p.poll() is not None:
                print(f"{p.args} exited with code {p.returncode}")
        i += 1
        print(f"Keepalive: {i*10/60:0.2f}min", end="\r")
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
