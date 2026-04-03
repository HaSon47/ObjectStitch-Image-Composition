#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

# python scripts/inference_FSC_hw.py \
# --outdir /workdir/radish/hachi/Output/FSC_final/ObjectStitch/Turn_1/FSC_difficult_overlap \
# --testdir /workdir/radish/hachi/OBJ_INS/FSC_final/FSC_difficult_overlap \
# --rootpath /workdir/radish/hachi/OBJ_INS/phase2_V2 \
# --ckpt_dir ./checkpoints \
# --num_samples 3 \
# --sample_steps 45 \
# --gpu 2
# python scripts/inference_location.py \
# --outdir /workdir/radish/hachi/Output/Location_phase1/ObjectStitch/ \
# --testdir /workdir/radish/hachi/Data/Location/Location_with_anno_test \
# --rootpath /workdir/radish/hachi/OBJ_INS/phase2_V2 \
# --ckpt_dir ./checkpoints \
# --num_samples 3 \
# --sample_steps 35 \
# --gpu 2 \
# --turn 1 

# python scripts/inference_FSC_k_loop.py \
# --outdir /mnt/disk2/hachi/Output/many_insert/Reference-based/objectstitch_k_loop_v4/FSC_final/FSC_easy_overlap \
# --testdir /mnt/disk2/hachi/data/FSC_final/FSC_easy_overlap \
# --rootpath /mnt/disk2/hachi/data/workdir/radish/hachi/OBJ_INS/phase2_V2 \
# --ckpt_dir /mnt/disk2/hachi/checkpoints \
# --num_samples 1 \
# --sample_steps 45 \
# --gpu 2

python scripts/inference_location_squared.py \
--outdir /mnt/disk2/hachi/Output/Location_squared/Turn_1/Objectstitch \
--testdir /mnt/disk2/hachi/data/Location_squared/Turn_1 \
--rootpath /mnt/disk2/hachi/data/workdir/radish/hachi/OBJ_INS/phase2_V2 \
--ckpt_dir /mnt/disk2/hachi/checkpoints \
--num_samples 1 \
--sample_steps 45 \
--gpu 2
