#!/bin/bash
# python scripts/inference_FSC_hw.py \
# --outdir /workdir/radish/hachi/Output/FSC_final/ObjectStitch/Turn_1/FSC_difficult_overlap \
# --testdir /workdir/radish/hachi/OBJ_INS/FSC_final/FSC_difficult_overlap \
# --rootpath /workdir/radish/hachi/OBJ_INS/phase2_V2 \
# --ckpt_dir ./checkpoints \
# --num_samples 3 \
# --sample_steps 45 \
# --gpu 2
python scripts/inference_real.py \
--outdir /workdir/radish/hachi/Output/Location_real_add/ObjStitch/difficult \
--testdir /workdir/radish/hachi/OBJ_INS/Real/Real_difficult \
--locationdir /workdir/radish/hachi/Data/box_results/difficult \
--rootpath /workdir/radish/hachi/OBJ_INS/phase2_V2 \
--ckpt_dir ./checkpoints \
--ckpt_objstit /home/hachi/ObjectStitch-Image-Composition/checkpoints/ObjectStitch.pth \
--num_samples 3 \
--sample_steps 45 \
--gpu 3 \
