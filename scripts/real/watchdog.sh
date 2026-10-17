#!/bin/zsh
# Usage: watchdog.sh <pid> <logfile>. Kills the process tree of <pid> when free memory < 5% or swap grows > 10 GB.
PID=$1; LOG=$2
swap_mb() { sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
BASE=$(swap_mb)
kill_tree() { local p=$1; for c in $(pgrep -P $p); do kill_tree $c; done; kill -TERM $p 2>/dev/null; }
echo "watchdog start pid=$PID swap_base=${BASE}M" >> $LOG
MINFREE=100; MAXSWAP=$BASE
while kill -0 $PID 2>/dev/null; do
  FREE=$(memory_pressure | tail -1 | sed -E 's/.*: ([0-9]+)%/\1/')
  SW=$(swap_mb)
  (( FREE < MINFREE )) && MINFREE=$FREE
  (( SW > MAXSWAP )) && MAXSWAP=$SW
  if (( FREE < ${FORCE_FREE:-5} )) || (( SW - BASE > 10240 )); then
    echo "watchdog KILL free=${FREE}% swap=${SW}M" >> $LOG
    # Only the server this run started, and its child processes. Another mlx-imagine server on the Mac keeps running.
    kill_tree $PID; sleep 3
    kill -0 $PID 2>/dev/null && kill -KILL $PID
    break
  fi
  sleep 2
done
echo "watchdog end min_free=${MINFREE}% max_swap=${MAXSWAP}M (base ${BASE}M)" >> $LOG
