#!/bin/bash

#SBATCH --job-name=dummy
#SBATCH --partition=batch
#SBATCH --time=2-0:00:00
#SBATCH --output out/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=256G


# TRAINING
export MAMMOTH=`pwd`/mammoth
export CONFIG_DIR=`pwd`/configs

# info for model and log saving
export SAVE_DIR=`pwd`/models/phoenix14
export LOG_DIR=${SAVE_DIR}/logs
export EXP_ID=pheonix2014t-1-node

mkdir -p  ${SAVE_DIR}/{logs,models}

srun \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    wrapper.sh -u ${MAMMOTH}/train.py \
    -config ${CONFIG_DIR}/slt-1node-1gpu.yml \
    -save_model ${SAVE_DIR}/models/${EXP_ID}


# TRANSLATION
export model=`pwd`/models/phoenix14t-signjoey/models/pheonix2014t-1-node_step_2000
export MAMMOTH=`pwd`/mammoth
export CONFIG_DIR=`pwd`/configs

srun \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    wrapper.sh -u ${MAMMOTH}/translate.py \
       -config ${CONFIG_DIR}/slt-1node-1gpu.yml \
       -model ${model} \
       -src "`pwd`/data/PHOENIX2014T/phoenix14t.pami0.test" \
       -tgt "`pwd`/data/PHOENIX2014T/phoenix14t.pami0.test" \
       -task_id train_bsl-de \
       -verbose true

# Compute BLEU-score
srun \
    --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
    --container-workdir="`pwd`" \
    --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,"`pwd`":"`pwd`" \
    --task-prolog="`pwd`/install.sh" \
    python compute_bleu.py