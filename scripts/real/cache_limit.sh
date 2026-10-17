#!/bin/zsh
# Measure one MLX cache limit in a fresh server (ROADMAP T-10). Usage: cache_limit.sh LIMIT_GB MODEL JOBS CONFIG
# The server uses a copy of CONFIG with cache_limit_gb set. The memory watchdog stops it, as in run.sh.
set -u
LIMIT=$1; MODEL=$2; JOBS=$3; CONFIG=${4:a}
HERE=${0:a:h}; ROOT=${HERE:h:h}; LOGDIR=$ROOT/.real-logs; LOG=$LOGDIR/cache-$LIMIT.log; U=http://127.0.0.1:8765
mkdir -p $LOGDIR; : > $LOG
if curl -sf $U/v1/health >/dev/null; then echo "A server already runs at $U. Stop it first."; exit 1; fi
TMPCONFIG=$LOGDIR/config-cache-$LIMIT.yaml
$ROOT/.venv/bin/python - "$CONFIG" "$TMPCONFIG" "$LIMIT" <<'PY'
import sys, yaml
source, target, limit = sys.argv[1:4]
data = yaml.safe_load(open(source))
data["cache_limit_gb"] = float(limit)
yaml.safe_dump(data, open(target, "w"))
PY
echo "limit=$LIMIT GB | memory before: $(memory_pressure | tail -1) | $(sysctl -n vm.swapusage)" | tee -a $LOG
cd $ROOT
env -u VIRTUAL_ENV .venv/bin/mlx-imagine --config $TMPCONFIG serve > $LOGDIR/server-cache-$LIMIT.log 2>&1 &
SRV=$!
zsh $HERE/watchdog.sh $SRV $LOG &
WD=$!
for i in {1..120}; do curl -sf $U/v1/health >/dev/null && break; sleep 1; done
.venv/bin/python $HERE/cache_limit.py "$MODEL" "$JOBS" $U 2>&1 | tee -a $LOG
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null; wait $WD 2>/dev/null
grep watchdog $LOG
echo "memory after: $(memory_pressure | tail -1) | $(sysctl -n vm.swapusage)" | tee -a $LOG
