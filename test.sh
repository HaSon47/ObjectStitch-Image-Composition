#!/bin/bash

python scripts/inference_FSC_k.py \
--outdir /mnt/disk2/hachi/Output/many_insert/Reference-based/objectstitch/FSC_final/FSC_easy_no_overlap \
--testdir /mnt/disk2/hachi/data/FSC_final/FSC_easy_no_overlap \
--rootpath /mnt/disk2/hachi/data/workdir/radish/hachi/OBJ_INS/phase2_V2 \
--ckpt_dir /mnt/disk2/hachi/checkpoints \
--num_samples 3 \
--sample_steps 45 \
--gpu 1

# python scripts/inference.py \
# --outdir results \
# --testdir examples \
# --num_samples 3 \
# --sample_steps 50 \
# --plms \
# --gpu 0