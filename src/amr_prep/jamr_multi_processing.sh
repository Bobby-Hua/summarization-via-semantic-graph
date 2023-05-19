#!/bin/sh

# number of threads
N=12
# number of scenes
MAX=$1 
for ((i=0;i<=MAX;i++)); do
    scripts/ALIGN.sh<"$2"/jamr_input_"$i".txt>"$3"/jamr_output_"$i" &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

echo "all done"
