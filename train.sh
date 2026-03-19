export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --logdir /workdir/radish/hachi/Output/experiments/objectstitch_15k_mix2 \
    --num_workers 8 \
    --devices 2 \
    --batch_size 8 \
    --num_nodes 1 \
    --base configs/train.yaml \
    --pretrained_model "./checkpoints/ObjectStitch.pth" \
    --train_from_scratch False