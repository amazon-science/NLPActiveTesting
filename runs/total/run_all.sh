#!/bin/bash
set +e

> errors.txt

run_and_log() {
    local script=$1
    error_output=$(bash "$script" 2>&1 >/dev/null)
    if [ $? -ne 0 ]; then
        {
            echo "----------------------------------------"
            echo "[$(date)] Error in $script"
            echo "Error message:"
            echo "$error_output"
            echo "----------------------------------------"
        } >> errors.txt
    fi
}

run_and_log "runs/total/run_dbpedia.sh"
run_and_log "runs/total/run_imdb.sh"
run_and_log "runs/total/run_rotten.sh"
run_and_log "runs/total/run_pubmed.sh"
run_and_log "runs/total/run_agnews.sh"
run_and_log "runs/total/run_banking.sh"
run_and_log "runs/total/run_fnc1.sh"
run_and_log "runs/total/run_qnli.sh"
run_and_log "runs/total/run_sst2.sh"
run_and_log "runs/total/run_trec6.sh"
run_and_log "runs/total/run_trec6.sh"
run_and_log "runs/total/run_emotions.sh"