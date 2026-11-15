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
# A run whose log already ends cleanly is skipped, so resubmitting the same
# array only redoes what is missing or failed.
set -u
LIST=${1:?usage: sbatch --account=... --array=1-N namd_array.sbatch <list file>}
: "${CFACV_BASEDIR:?export CFACV_BASEDIR (the cfacv checkout) before submitting}"
export CFACV_BASEDIR

module purge
module load abramsGrp-namd/3.0.3/multicore-gcc
module load gsl/gcc/2.7.1

CONF=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$CONF" ] || { echo "no line $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2; }

run_one () {
    local conf=$1 log=${1%.conf}.log
    # "End of program" is NOT a success marker: NAMD prints it after a FATAL
    # ERROR too.  Done means: finished AND no fatal error.
    if [ -s "$log" ] && grep -q "End of program" "$log" && ! grep -q "FATAL ERROR" "$log"; then
        echo "skip (already complete): $conf"; return 0
    fi
    echo "run: $conf  ($(date))"
    namd3 +p"$SLURM_CPUS_PER_TASK" +setcpuaffinity "$conf" > "$log" 2>&1 < /dev/null
    local rc=$?
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
