#! /bin/bash

export CUDA_VISIBLE_DEVICES=0
nvidia-smi dmon -s mu -d 5 -o TD > "${LOG_DIR}/gpu_load-${EXP_ID}-${PPID}.log" &
echo python -u "$@" --node_rank $SLURM_NODEID
python -u "$@" --node_rank $SLURM_NODEID
