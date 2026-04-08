#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2
# cd /mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition

# pwd 
export PYTHONPATH=/mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition/src/taming-transformers:$PYTHONPATH


# python scripts/pairtally/inference_pairtally_maskSam_blend.py \
# --outdir /mnt/disk1/hachi/ImgEdit/Output/Pairtally/ObjStit_Finetune_crop256_maskSam_blend/ \
# --imgdir /mnt/disk1/hachi/ImgEdit/data/PairTally \
# --annodir /mnt/disk1/hachi/ImgEdit/data/PairTally \
# --order_log /mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition/scripts/pairtally/logs/turn_order.json \
# --ckpt_dir /mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition/checkpoints \
# --ckpt_objstit /mnt/disk1/hachi/ImgEdit/checkpoint/ObjectStitch_Finetune/objectstitch_15k_mix2/2026-02-27T08-43-50/checkpoints/epoch=000012.ckpt \
# --num_samples 1 \
# --sample_steps 45 \
# --gpu 0 \
# --turn 0 \
# --blend

python scripts/pairtally/inference_pairtally_10_turns.py \
--outdir /mnt/disk1/hachi/ImgEdit/Output/Pairtally/ObjStit_Finetune_10_turns_final/ \
--imgdir /mnt/disk1/hachi/ImgEdit/data/PairTally \
--annodir /mnt/disk1/hachi/ImgEdit/data/PairTally \
--order_log /mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition/scripts/pairtally/logs/turn_order.json \
--ckpt_dir /mnt/disk1/hachi/ImgEdit/code/ObjectStitch-Image-Composition/checkpoints \
--ckpt_objstit /mnt/disk1/hachi/ImgEdit/checkpoint/ObjectStitch_Finetune/objectstitch_15k_mix2/2026-02-27T08-43-50/checkpoints/epoch=000012.ckpt \
--num_samples 1 \
--sample_steps 45 \
--gpu 0 \
--num_turn 10 \
--blend