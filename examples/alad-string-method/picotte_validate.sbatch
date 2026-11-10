#!/bin/bash
#SBATCH --job-name=cfacv-validate
#SBATCH --partition=def-sm
#SBATCH --account=abramsfreeprj
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Three checks the string method needs before a production run.
#
#  A  restart       A 4000-iteration string run, against 2000 + restart +
#                   2000.  Production strings cannot fit in one job (def is
#                   2 days, long is 8), so the restart path is on the
#                   critical path and has never been executed.
#  B  CV types      Unbiased MD reporting cfacv's own CV values alongside a
#                   trajectory, so the online values can be checked against
#                   independent post hoc measurement.  Exercises BOND and
#                   ANGLE, which no example here has ever run.
#  C  replica cost  How the 6.22 ms per-update synchronization cost measured
#                   at 24 replicas scales with replica count, since the T4L
#                   string wants a different number of images.
#
# Submit with CFACV_BASEDIR exported:
#   sbatch --export=ALL,CFACV_BASEDIR=/ifs/groups/abramsGrp/$USER/cfacv \
#          picotte_validate.sbatch
set -euo pipefail

# Stage A launches replicas, so the MPI build has to be loaded before it --
# the later stages switch modules for their own needs and switch back.
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
RESULTS=$SRC/validate.$SLURM_JOB_ID
mkdir -p "$WORK" "$RESULTS"

stage_out() {
  for d in A_continuous A_restart B_cvtypes; do
    [ -d "$WORK/$d" ] || continue
    mkdir -p "$RESULTS/$d"
    cp -a "$WORK/$d"/output/0/*.history "$RESULTS/$d"/ 2>/dev/null || true
    cp -a "$WORK/$d"/output/0/job*.log   "$RESULTS/$d"/ 2>/dev/null || true
    cp -a "$WORK/$d"/*.log "$WORK/$d"/*.cvout "$WORK/$d"/*.dcd "$WORK/$d"/*.psf "$RESULTS/$d"/ 2>/dev/null || true
  done
  cp -a "$WORK"/replica_scaling.csv "$RESULTS"/ 2>/dev/null || true
}
trap stage_out EXIT

prep() {   # prep <dir> <num_replicas> <steps_per_run> <num_runs>
  local d=$1 nrep=$2 spr=$3 nruns=$4
  mkdir -p "$WORK/$d"; cd "$WORK/$d"
  cp -a "$EXAMPLE"/. .
  rm -rf output
  sed -i "s|^set num_replicas .*|set num_replicas $nrep  ; # validate|"   alad_stringmethod.conf
  sed -i "s|^set steps_per_run .*|set steps_per_run $spr  ; # validate|"  alad_stringmethod.conf
  sed -i "s|^set num_runs .*|set num_runs $nruns  ; # validate|"          alad_stringmethod.conf
}

######################################################################
echo "########## STAGE A - restart path"
######################################################################
# Both arms share one initial string so the comparison is clean.
prep A_continuous 24 1 4000
cp -a "$SRC"/initial_string/. .
echo "--- A1: continuous, 4000 iterations"
run_string_watched --cfacv-base-dir="$CFACV_BASEDIR" --num-procs=24 > run.log 2>&1
grep -h "WallClock:" output/0/job0.0.log | tail -1

prep A_restart 24 1 2000
cp -a "$SRC"/initial_string/. .
echo "--- A2a: first leg, 2000 iterations"
run_string_watched --cfacv-base-dir="$CFACV_BASEDIR" --num-procs=24 > run.a.log 2>&1
echo "--- restart files written:"
ls output//*.tcl 2>/dev/null || echo "  NONE -- restart was not written"
echo "--- A2b: resume to 4000 iterations"
# Raise the target in the config rather than exporting CFACV_NUM_RUNS:
# mpirun does not reliably forward arbitrary environment variables to the
# ranks, and a silently unforwarded value would stop the resume at 2000
# and look like a successful restart.
sed -i "s|^set num_runs .*|set num_runs 4000  ; # validate: resume target|" alad_stringmethod.conf
run_string_watched --cfacv-base-dir="$CFACV_BASEDIR" \
    --num-procs=24 --conf=job1.conf > run.b.log 2>&1
grep -hE "resuming from|CFACV/SM\) run (2000|2001) begins" output/0/job1.0.log | head -3

######################################################################
echo "########## STAGE B - BOND and ANGLE, online vs post hoc"
######################################################################
mkdir -p "$WORK/B_cvtypes"; cd "$WORK/B_cvtypes"
cp -a "$EXAMPLE"/. .
module purge; module load abramsGrp-namd/3.0.3/multicore-gcc
namd3 +p8 cvcheck.conf > cvcheck.log 2>&1
echo "--- CV report lines: $(grep -c 'CFACV/C) Th' cvcheck.cvout 2>/dev/null || echo 0)"
grep "CFACV/C) Th" cvcheck.cvout 2>/dev/null | head -3 || true
module purge; module load abramsGrp-namd/3.0.3/mpi-gcc

######################################################################
echo "########## STAGE C - synchronization cost vs replica count"
######################################################################
CSV=$WORK/replica_scaling.csv
echo "replicas,images,num_runs,wallclock_s,ms_per_iteration" > "$CSV"
for N in 4 8 12 16 24; do
  d=C_rep$N
  prep "$d" "$N" 1 2000
  echo "--- building $N initial systems"
  module purge; module load abramsGrp-namd/3.0.3/multicore-gcc
  ./mkinitimages.sh --cfacv-base-dir="$CFACV_BASEDIR" --num-procs=8 > mkinit.log 2>&1
  module purge; module load abramsGrp-namd/3.0.3/mpi-gcc
  echo "--- string, $N replicas"
  run_string_watched --cfacv-base-dir="$CFACV_BASEDIR" --num-procs="$N" > run.log 2>&1
  w=$(grep -h "WallClock:" output/0/job0.0.log | tail -1 | awk '{print $2}')
  if [ -z "$w" ]; then echo "$N,$((N/2)),2000,NA,NA" >> "$CSV"; echo "  FAILED, see $d/run.log"; continue; fi
  ms=$(awk -v t="$w" "BEGIN{printf \"%.3f\", 1000*t/2000}")
  echo "  $N replicas: ${w}s  ${ms} ms/iteration"
  echo "$N,$((N/2)),2000,$w,$ms" >> "$CSV"
done

echo "########## all stages complete"
cat "$CSV"
