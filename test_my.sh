#!/bin/bash
# python scripts/inference_FSC_hw.py \
# --outdir /workdir/radish/hachi/Output/FSC_final/ObjectStitch/Turn_1/FSC_difficult_overlap \
# --testdir /workdir/radish/hachi/OBJ_INS/FSC_final/FSC_difficult_overlap \
# --rootpath /workdir/radish/hachi/OBJ_INS/phase2_V2 \
# --ckpt_dir ./checkpoints \
# --num_samples 3 \
# --sample_steps 45 \
# --gpu 2
python scripts/inference_location.py \
--outdir /workdir/radish/hachi/Output/Location_phase1/ObjStit_Finetune \
--testdir /workdir/radish/hachi/Data/Location/Location_with_anno_test \
--rootpath /workdir/radish/hachi/OBJ_INS/phase2_V2 \
--ckpt_dir ./checkpoints \
--ckpt_objstit /workdir/radish/hachi/Output/experiments/objectstitch_15k_mix2/2026-02-27T08-43-50/checkpoints/epoch=000012.ckpt \
--num_samples 3 \
--sample_steps 45 \
--gpu 2 \
--turn 2 