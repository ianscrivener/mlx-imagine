#!/bin/zsh
# Real-weight test run: start the server under the memory watchdog, run suite.py, stop the server.
# Usage: run.sh STEPS MODEL IMAGE CONFIG
# The watchdog kills the server when free memory drops below 5% or swap grows by more than 10 GB.
set -u
STEPS=$1; MODEL=$2; IMAGE=${3:a}; CONFIG=${4:a}
HERE=${0:a:h}; ROOT=${HERE:h:h}; LOGDIR=$ROOT/.real-logs; LOG=$LOGDIR/suite.log; U=http://127.0.0.1:8765
mkdir -p $LOGDIR; : > $LOG
if curl -sf $U/v1/health >/dev/null; then echo "A server already runs at $U. Stop it first."; exit 1; fi
echo "memory before: $(memory_pressure | tail -1) | $(sysctl -n vm.swapusage)" | tee -a $LOG
cd $ROOT
env -u VIRTUAL_ENV .venv/bin/mlx-imagine --config $CONFIG serve > $LOGDIR/server.log 2>&1 &
SRV=$!
zsh $HERE/watchdog.sh $SRV $LOG &
WD=$!
for i in {1..120}; do curl -sf $U/v1/health >/dev/null && break; sleep 1; done
.venv/bin/python $HERE/suite.py "$STEPS" "$MODEL" "$IMAGE" $U 2>&1 | tee -a $LOG
kill -TERM $SRV 2>/dev/null; wait $SRV 2>/dev/null; wait $WD 2>/dev/null
grep watchdog $LOG
echo "memory after: $(memory_pressure | tail -1) | $(sysctl -n vm.swapusage)" | tee -a $LOG
grep -q "^FAIL" $LOG && exit 1 || exit 0
