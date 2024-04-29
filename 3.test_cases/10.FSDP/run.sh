#!/bin/bash

ip=$(curl -s ifconfig.me)
if [ "$ip" == "64.110.102.13" ]; then
    node_rank=0
else
    node_rank=1
fi

declare -a TORCHRUN_ARGS=(
    --nproc_per_node=8
    --nnodes=2
    --master_addr=64.110.102.13
    --master_port=1234
    --node_rank=$node_rank
)

declare -a TRAINING_ARGS=(
    --max_context_width=4096
    --num_key_value_heads=32 # 7b: 32 13b: 40 70b: 8
    --llama_intermediate_size=11008 # 7b: 11008 13b: 13824 70b: 28672
    --hidden_width=4096 # 7b: 4096 13b: 5120 70b: 8192
    --num_layers=32 # 7b: 32 13b: 40 70b: 80
    --num_heads=32 # 7b: 32 13b: 40 70b: 64
    --model_type=llama_v2
    --tokenizer="hf-internal-testing/llama-tokenizer"
    --checkpoint_freq=5000
    --validation_freq=500
    --max_steps=5000
    --checkpoint_dir=./checkpoints
    --dataset='c4'
    --dataset_config_name='en'
    --resume_from_checkpoint=./checkpoints
    --train_batch_size=1
    --val_batch_size=1
    --sharding_strategy="hybrid" # https://pytorch.org/docs/stable/fsdp.html
    --offload_activations=1
)


export TRAIN_SCRIPT=./train.py

torchrun "${TORCHRUN_ARGS[@]}" $TRAIN_SCRIPT "${TRAINING_ARGS[@]}"
