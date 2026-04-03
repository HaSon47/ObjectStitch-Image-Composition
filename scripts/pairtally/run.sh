#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2

python scripts/inference_location.py \
--outdir /workdir/radish/hachi/Output/Location_phase1/ObjectStitch/ \
--imgdir /mnt/disk2/hachi/data/PairTally \
--annodir /mnt/disk2/hachi/data/PairTally \
--order_log /mnt/disk1/aiotlab/hachi/ObjectStitch-Image-Composition/scripts/pairtally/logs/turn_order.json \
--ckpt_dir ./checkpoints \
--num_samples 1 \
--sample_steps 45 \
--gpu 2 \
--turn 0 
