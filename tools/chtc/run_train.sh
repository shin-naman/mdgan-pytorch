#!/bin/bash
# HTCondor job wrapper for mdgan.train (see docs/cluster.md).
#
#   run_train.sh NAME SEED [extra mdgan.train arguments...]
#
# Expects in the job sandbox: src/ (the package), pe_paper.json, PE.npz.
# Writes run_NAME/. Exit code 85 = "saved at the time limit, restart me" (HTCondor
# checkpoint_exit_code); 0 = training finished; anything else = a real failure.
set -euo pipefail
NAME="$1"
SEED="$2"
shift 2
export PYTHONPATH="$PWD/src"

echo "host $(hostname)  start $(date -Is)  run_${NAME}  seed ${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi unavailable"
python -c "import sys, torch, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, \
'| cuda available', torch.cuda.is_available(), '| numpy', numpy.__version__)"
if [ -f "run_${NAME}/latest.pt" ]; then echo "resuming from run_${NAME}/latest.pt"; fi

set +e
python -m mdgan.train PE.npz --out "run_${NAME}" --config pe_paper.json --set "seed=${SEED}" \
    --device cuda --auto-resume --max-hours "${MAX_HOURS:-4}" --log-every 100 "$@"
status=$?
set -e
echo "exit ${status}  at $(date -Is)"
exit ${status}