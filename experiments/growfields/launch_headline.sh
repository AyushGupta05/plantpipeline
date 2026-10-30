#!/usr/bin/env bash
# Headline reproduction (Table 1/2 protocol: all frames, fixed point split, aligned leaves), 1 seed,
# ORIGINAL GrowFields then the monotone variant, sequentially so CPU use stays at P*THREADS = 12 cores.
# Detached from the launching shell: `python -c` with start_new_session=True + nohup. Progress:
#   tail runs/*/jobs.log ; grep -c "^Step" runs/<config>/<seq>_seed43.log   (one line per 200 steps; 250 lines = 50k steps)
# Afterwards: .venv/bin/python collect_results.py shared_continuous ; .venv/bin/python collect_results.py monotone_continuous
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; cd "$D"
P=${P:-4}; THREADS=${THREADS:-3}; N_SEEDS=${N_SEEDS:-1}
mkdir -p runs
"$D/.venv/bin/python" - "$D" "$P" "$THREADS" "$N_SEEDS" <<'PY'
import subprocess, sys, os
d, p, th, n = sys.argv[1:5]
cmd = (f"bash {d}/run_test_split.sh shared_continuous {n} {p} {th} && "
       f"{d}/.venv/bin/python {d}/collect_results.py shared_continuous ; "
       f"bash {d}/run_test_split.sh monotone_continuous {n} {p} {th} && "
       f"{d}/.venv/bin/python {d}/collect_results.py monotone_continuous")
log = open(f"{d}/runs/launch_headline.log", "a")
proc = subprocess.Popen(["nohup", "bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, start_new_session=True, cwd=d)
print(f"launched detached master pid={proc.pid}\n  {cmd}")
open(f"{d}/runs/launch_headline.pid", "w").write(str(proc.pid) + "\n")
PY
