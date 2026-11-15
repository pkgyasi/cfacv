#!/bin/bash -l
#SBATCH --job-name=namd-array
#SBATCH --partition=def
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# One independent NAMD run per array task: task N runs line N of a list file.
# Used for the mean-force runs, the string warm-ups and the committor shots.
#
#   sbatch --account=<account> --array=1-200%20 namd_array.sbatch mf/runs.list
#
# NO ACCOUNT IS SET HERE ON PURPOSE.  Which billed account a job charges is
# the PI's decision and nothing on the cluster flags a wrong one; pass
# --account on the command line every time.
#
# Submit from the directory the list's paths are relative to, and keep that
# directory on persistent scratch (/scratch/$USER/...), not on /ifs or /home:
# these runs write restraint reports and restart files continuously.
#
# A config may start with lines "#PRE <other.conf>"; those configs are run
# first, in order (the warm-up uses this to step the force constant up).
# A run whose log ends cleanly AND is newer than its config is skipped, so
# resubmitting the same array only redoes what is missing, failed, or was set
# up again.  Blank lines and lines starting with # in the list do not count.
#
# HUNG RUNS.  NAMD does not always exit after a Tcl error: it can sit at 0%
# CPU until the walltime, and the allocation is billed for all of it.  A run
# none of whose output files has changed for HANG_MINUTES (default 30; 0
# turns the guard off) is killed and counted as failed.  Set it longer than
# the longest quiet stretch of a healthy run: the gap between two outputs.
#   sbatch --export=ALL,HANG_MINUTES=60 ...
set -u
LIST=${1:?usage: sbatch --account=... --array=1-N namd_array.sbatch <list file>}
: "${CFACV_BASEDIR:?export CFACV_BASEDIR (the cfacv checkout) before submitting}"
export CFACV_BASEDIR
HANG_MINUTES=${HANG_MINUTES:-30}
case $HANG_MINUTES in ''|*[!0-9]*) echo "HANG_MINUTES must be a whole number of minutes, got '$HANG_MINUTES'" >&2; exit 2;; esac
HANG_MINUTES=$((10#$HANG_MINUTES))        # 08 is not octal
: "${SLURM_ARRAY_TASK_ID:?submit with --array=1-N}"

module purge
module load abramsGrp-namd/3.0.3/multicore-gcc
module load gsl/gcc/2.7.1

# blank lines and comments do not count, as in thalweg_run.py
CONF=$(tr -d '\r' < "$LIST" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' | grep -v '^\(#\|$\)' | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$CONF" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2; }

run_one () {
    local conf=$1 log=${1%.conf}.log
    # "End of program" is NOT a success marker: NAMD prints it after a FATAL
    # ERROR too.  Done means: finished AND no fatal error.
    # ... AND the log is not older than the config: a config rewritten by a
    # new setup is a new run, whatever the old log says.
    if [ -s "$log" ] && [ -e "$conf" ] && [ "$log" -nt "$conf" ] && grep -q "End of program" "$log" && ! grep -q "FATAL ERROR" "$log"; then
        echo "skip (already complete): $conf"; return 0
    fi
    echo "run: $conf  ($(date))"
    namd3 +p"$SLURM_CPUS_PER_TASK" +setcpuaffinity "$conf" > "$log" 2>&1 < /dev/null &
    local pid=$! stem=${conf%.conf} last=0 seen now t poll=2 rc
    seen=$(date +%s)
    while kill -0 $pid 2>/dev/null; do
        sleep $poll; [ $poll -lt 30 ] && poll=$((poll * 2))
        [ "$HANG_MINUTES" -gt 0 ] || continue
        # newest change among this run's own files (log, cvout, dcd, restart)
        t=$(stat -c %Y "$stem".* "$stem"_* 2>/dev/null | sort -n | tail -1); t=${t:-0}
        now=$(date +%s)
        if [ "$t" -gt "$last" ]; then last=$t; seen=$now; fi
        if [ $((now - seen)) -ge $((HANG_MINUTES * 60)) ]; then
            echo "HUNG: no output from $conf for $HANG_MINUTES minutes; killing it" >&2
            tail -5 "$log" >&2
            kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
            return 3
        fi
    done
    wait $pid; rc=$?
    if grep -q "FATAL ERROR" "$log"; then
        echo "NAMD FATAL ERROR in $log:" >&2
        grep -A2 "FATAL ERROR" "$log" | head -6 >&2
        return 1
    fi
    [ $rc -eq 0 ] || { echo "namd3 exit $rc for $conf" >&2; return $rc; }
    grep -q "End of program" "$log" || { echo "$log did not finish" >&2; return 1; }
}

for pre in $(grep '^#PRE ' "$CONF" | awk '{print $2}'); do
    run_one "$pre" || exit 1
done
run_one "$CONF" || exit 1
echo "done: $CONF  ($(date))"
