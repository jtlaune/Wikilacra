import os, subprocess, time
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone

parser = ArgumentParser()
parser.add_argument("--fresh", action="store_true")
args = parser.parse_args()

# remove old log files if they exist
for fname in ("/app/logs/changes.log", "/app/logs/revisions.log", "/app/logs/tags.log"):
    if os.path.exists(fname):
        os.remove(fname)

# Open log files for each process (create if missing)
change_log = open("/app/logs/changes.log", "a+")
rev_log = open("/app/logs/revisions.log", "a+")
tag_log = open("/app/logs/tags.log", "a+")


def start_time_first() -> datetime:
    now_utc = datetime.now(timezone.utc)
    if args.fresh:
        print("Fresh start!")
        return now_utc - timedelta(hours=16)
    return now_utc


def start_time_restart() -> datetime:
    # restarts always use "now"
    return datetime.now(timezone.utc)


def spawn_proc(script: str, log_file, isodt_start: datetime) -> subprocess.Popen:
    return subprocess.Popen(
        ["python", "-u", script, str(isodt_start.isoformat()).replace("+00:00", "Z")],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


scripts = [
    ("recentchange.py", change_log),
    ("revisioncreate.py", rev_log),
    ("recenttags.py", tag_log),
]

# initial start time (fresh logic only applies here)
initial_start = start_time_first()
procs = [spawn_proc(script, log_file, initial_start) for script, log_file in scripts]

try:
    while True:
        time.sleep(10)
        for idx, p in enumerate(procs):
            if p.poll() is not None:
                print(f"{p.args} exited with code {p.returncode}")
                # restart with current-time start (no fresh offset)
                script, log_file = scripts[idx]
                procs[idx] = spawn_proc(script, log_file, start_time_restart())
except KeyboardInterrupt:
    for p in procs:
        p.terminate()
