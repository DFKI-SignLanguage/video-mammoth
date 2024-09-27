#!/bin/bash

# make sure only first task per node installs stuff, others wait
DONEFILE="/tmp/install_done_${SLURM_JOBID}"
if [[ $SLURM_LOCALID == 0 ]]; then
  
  # put your install commands here
  # git clone https://github.com/Helsinki-NLP/mammoth.git
  pip install -e mammoth --index-url=https://pypi.python.org/simple/
  pip install sentencepiece
  pip install sacrebleu # for compute_bleu.py
  # pip install -r "`pwd`/requirements.txt"
  
  # Tell other tasks we are done installing
  touch "${DONEFILE}"
else
  # Wait until packages are installed
  while [[ ! -f "${DONEFILE}" ]]; do sleep 1; done
fi