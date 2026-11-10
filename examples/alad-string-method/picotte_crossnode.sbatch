#!/bin/bash
#SBATCH --job-name=cfacv-crossnode
#SBATCH --partition=def-sm
#SBATCH --account=abramsfreeprj
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Does the replica barrier get more expensive when it crosses nodes?
#
# The earlier scaling run measured 4-24 replicas, all of which fit on one
# 48-core node, so every barrier was intra-node.  A 16-20 image string in
# dual mode needs 32-40 replicas and spans more than one node, which is a
# different regime: the barrier goes over InfiniBand.
#
# Stage 1 is the controlled comparison.  Same 24 replicas, same 24 ranks,
# same initial string, same iteration count -- only the PLACEMENT changes,
# 24 ranks on one node against 12 on each of two.  Everything else is held
# fixed, so the difference is the interconnect and nothing else.  Both run
# inside ONE allocation, so it is also the same two nodes in both arms.
#
# Stage 2 then extends the replica-count scaling into the multi-node regime
# at 32, 40 and 48 replicas, which is where the T4L string will sit.
set -euo pipefail

module purge
module load abramsGrp-namd/3.0.3/mpi-gcc

SRC=$SLURM_SUBMIT_DIR
if [ -z "${CFACV_BASEDIR:-}" ]; then CFACV_BASEDIR=$(cd "$SRC/../.." && pwd); fi
[ -d "$CFACV_BASEDIR/tcl" ] || { echo "ERROR: CFACV_BASEDIR='$CFACV_BASEDIR' is not a cfacv tree." >&2; exit 1; }
export CFACV_BASEDIR

EXAMPLE=$CFACV_BASEDIR/examples/alad-string-method

# Terminate a launcher that outlives its NAMD ranks; see watchdog.sh.
source "$EXAMPLE/watchdog.sh"
WORK=/scratch/$USER/$SLURM_JOB_ID
RESULTS=$SRC/crossnode.$SLURM_JOB_ID
NRUNS=2000
mkdir -p "$WORK" "$RESULTS"

CSV=$WORK/crossnode.csv
echo "label,replicas,nodes,ranks_per_node,num_runs,wallclock_s,ms_per_iteration" > "$CSV"
stage_out() { cp -a "$CSV" "$RESULTS"/ 2>/dev/null || true
              for d in "$WORK"/*/; do n=$(basename "$d"); [ -f "$d/output/0/job0.0.log" ] || continue
                mkdir -p "$RESULTS/$n"; cp -a "$d/output/0/job0.0.log" "$RESULTS/$n"/ 2>/dev/null || true; done; }
trap stage_out EXIT

echo "=== allocation: $SLURM_NNODES nodes, $SLURM_NTASKS tasks"
scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' '; echo

prep() {   # prep <dir> <num_replicas>
  local d=$1 nrep=$2
  mkdir -p "$WORK/$d"; cd "$WORK/$d"
  cp -a "$EXAMPLE"/. .
  rm -rf output
  sed -i "s|^set num_replicas .*|set num_replicas $nrep  ; # crossnode|"  alad_stringmethod.conf
  sed -i "s|^set steps_per_run .*|set steps_per_run 1  ; # crossnode|"    alad_stringmethod.conf
  sed -i "s|^set num_runs .*|set num_runs $NRUNS  ; # crossnode|"         alad_stringmethod.conf
}

run_case() {   # run_case <label> <dir> <replicas> <ranks_per_node> <nodes>
  local label=$1 d=$2 nrep=$3 ppn=$4 nodes=$5
  cd "$WORK/$d"
  run_string_watched --cfacv-base-dir="$CFACV_BASEDIR" --num-procs="$nrep" \
                 --mpirun-opts="--map-by ppr:${ppn}:node --bind-to core" > run.log 2>&1 || true
  local w ms
  w=$(grep -h "WallClock:" output/0/job0.0.log 2>/dev/null | tail -1 | awk '{print $2}')
  if [ -z "$w" ]; then
    echo "  $label: FAILED (see $WORK/$d/run.log)"; tail -3 run.log || true
    echo "$label,$nrep,$nodes,$ppn,$NRUNS,NA,NA" >> "$CSV"; return
  fi
  ms=$(awk -v t="$w" -v n="$NRUNS" 'BEGIN{printf "%.3f", 1000*t/n}')
  echo "  $label: ${w}s   ${ms} ms/iteration"
  echo "$label,$nrep,$nodes,$ppn,$NRUNS,$w,$ms" >> "$CSV"
}

######################################################################
echo "########## STAGE 1 - same 24 replicas, one node vs two"
######################################################################
for arm in "1node 24 1" "2node 12 2"; do
  set -- $arm
  prep "s1_$1" 24
  cp -a "$SRC"/initial_string/. .          # identical initial string in both arms
  echo "--- 24 replicas, $3 node(s), $2 rank(s) per node"
  run_case "s1_$1" "s1_$1" 24 "$2" "$3"
done

######################################################################
echo "########## STAGE 2 - replica scaling across two nodes"
######################################################################
for N in 32 40 48; do
  prep "s2_rep$N" "$N"
  echo "--- building $N initial systems"
  module purge; module load abramsGrp-namd/3.0.3/multicore-gcc
  ./mkinitimages.sh --cfacv-base-dir="$CFACV_BASEDIR" --num-procs=8 > mkinit.log 2>&1
  module purge; module load abramsGrp-namd/3.0.3/mpi-gcc
  echo "--- $N replicas across 2 nodes"
  run_case "s2_rep$N" "s2_rep$N" "$N" $((N/2)) 2
done

echo "########## complete"
cat "$CSV"
