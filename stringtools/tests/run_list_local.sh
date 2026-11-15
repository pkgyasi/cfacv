#!/bin/bash
# Run every config of a list file here, one after another, with the same
# rules as sbatch/namd_array.sbatch (#PRE chains, skip-if-complete, FATAL
# check).  For tests on a workstation.   run_list_local.sh <list> [ncores]
set -u
LIST=$1; NP=${2:-4}; NAMD=${NAMD3:-namd3}
run_one () {
    local conf=$1 log=${1%.conf}.log
    if [ -s "$log" ] && grep -q "End of program" "$log" && ! grep -q "FATAL ERROR" "$log"; then return 0; fi
    $NAMD +p$NP "$conf" > "$log" 2>&1 < /dev/null
    if grep -q "FATAL ERROR" "$log"; then grep -A2 "FATAL ERROR" "$log" | head -4 >&2; return 1; fi
    grep -q "End of program" "$log" || { echo "$log did not finish" >&2; tail -3 "$log" >&2; return 1; }
}
fail=0
while read -r CONF; do
    [ -n "$CONF" ] || continue
    for pre in $(grep '^#PRE ' "$CONF" | awk '{print $2}'); do run_one "$pre" || { fail=1; continue 2; }; done
    run_one "$CONF" || fail=1
done < "$LIST"
exit $fail
