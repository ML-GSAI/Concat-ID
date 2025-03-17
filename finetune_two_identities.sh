#! /bin/bash

echo "RUN on $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 train_video.py --base configs/cogvideox_5b.yaml configs/training/sft_two_identities.yaml --seed 42"

echo ${run_cmd}
eval ${run_cmd}

echo "DONE on `hostname`"