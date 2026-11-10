# Watchdog for a replica string run.  Source this from an sbatch script.
#
# WHY. On picotte job 26741257 a 48-rank, two-node run finished its work in
# 27.7 s -- every replica wrote NAMD's "End of program" and NO namd3 process
# remained on either node -- and then mpirun and its orted helper sat sleeping
# for 22 minutes without returning.  The science was complete; the launcher
# never reaped it.  A production string that does this burns the rest of its
# walltime, and inside a chunked restart campaign it can stall the campaign.
#
# Seen once, at the full 48-rank case, and not at 24, 32 or 40 ranks, so treat
# it as something that happens rather than something that always happens.
#
# WHAT. run_string_watched passes its arguments straight to ./runstring.sh,
# then watches the per-replica logs.  Once EVERY replica has written the
# completion marker it gives the launcher a grace period; if it still has not
# exited, the launcher is terminated and the run is reported as successful,
# because by then NAMD has done everything it was asked to do.
#
# The launcher is killed BY RECORDED PID, never by name.  A pattern kill such
# as `pkill -f namd3` also matches other people's jobs on a shared node, and
# can match the killing script's own shell.
#
# runstring.sh ends in `exec mpirun`, so the pid captured here IS the
# launcher; there is no intermediate shell to kill around.
#
# Tunables (environment):
#   CFACV_WATCHDOG_GRACE  seconds to allow for teardown after NAMD finishes (60)
#   CFACV_WATCHDOG_POLL   seconds between checks (5)
#   CFACV_WATCHDOG_OFF    set to 1 to run without the watchdog

CFACV_WATCHDOG_GRACE=${CFACV_WATCHDOG_GRACE:-60}
CFACV_WATCHDOG_POLL=${CFACV_WATCHDOG_POLL:-5}

run_string_watched() {
  local conf=job0.conf outdir=output smconf=alad_stringmethod.conf a
  for a in "$@"; do
    case $a in
      -c=*|--conf=*)                  conf=${a#*=} ;;
      -o=*|--output-dir=*)            outdir=${a#*=} ;;
      -i=*|--string-method-config=*)  smconf=${a#*=} ;;
    esac
  done
  # Match only THIS job's logs.  A resume writes job1.*.log beside the
  # already-complete job0.*.log of the first leg; counting both would see the
  # target reached immediately and kill a run that had not started.
  local base=${conf%.conf}
  local nrep
  nrep=$(awk '$1=="set" && $2=="num_replicas" {print $3; exit}' "$smconf" 2>/dev/null)
  if [ -z "$nrep" ] || [ "${CFACV_WATCHDOG_OFF:-0}" = "1" ]; then
    [ -n "$nrep" ] || echo "watchdog: no num_replicas in $smconf; running unwatched" >&2
    ./runstring.sh "$@"
    return $?
  fi

  ./runstring.sh "$@" &
  local pid=$!
  local marked_at="" n rc

  while kill -0 "$pid" 2>/dev/null; do
    n=$(grep -l "End of program" "$outdir"/*/"$base".*.log 2>/dev/null | wc -l)
    if [ "$n" -ge "$nrep" ]; then
      if [ -z "$marked_at" ]; then
        marked_at=$SECONDS
        echo "watchdog: all $nrep replicas reported End of program; allowing ${CFACV_WATCHDOG_GRACE}s for teardown"
      elif [ $(( SECONDS - marked_at )) -ge "$CFACV_WATCHDOG_GRACE" ]; then
        echo "watchdog: launcher pid $pid has not exited ${CFACV_WATCHDOG_GRACE}s after NAMD finished; terminating it"
        kill -TERM "$pid" 2>/dev/null
        local w=0
        while kill -0 "$pid" 2>/dev/null && [ $w -lt 15 ]; do sleep 1; w=$(( w + 1 )); done
        kill -KILL "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        echo "watchdog: launcher terminated; the run itself completed"
        return 0
      fi
    fi
    sleep "$CFACV_WATCHDOG_POLL"
  done

  wait "$pid"; rc=$?
  return $rc
}
