import os, subprocess, time

# remove old log files if they exist
for fname in ("revisions.log", "tags.log"):
    if os.path.exists(fname):
        os.remove(fname)

# Open log files for each process
rev_log = open("revisions.log", "a")
tag_log = open("tags.log", "a")

procs = [
    subprocess.Popen(
        ["python", "proto_recentchange.py"],
        stdout=rev_log,
        stderr=subprocess.STDOUT,
    ),
    subprocess.Popen(
        ["python", "proto_recenttags.py"],
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
