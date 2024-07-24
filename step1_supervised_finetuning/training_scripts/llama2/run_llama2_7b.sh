#!/bin/bash
export NCCL_P2P_DISABLE=1
export TZ="Asia/Seoul"
set -e
set -x

# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=2
# DATA_PATH="Dahoas/full-hh-rlhf"
DATA_PATH="Dahoas/template-full-hh-rlhf"
MODEL_NAME="Llama-2-7b"
MODEL_PATH="~/workspace/models/llama-2-7b-hf"
SEED=2023

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log/step1_sft-${MODEL_NAME/'/'/_}-$TIME_STEP-$SEED"
fi
mkdir -p $OUTPUT

deepspeed --master_port 12354 main.py \
   --data_path $DATA_PATH \
   --data_output_path "~/workspace/RLHF-TLCR/data_files/llama" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 2 \
   --num_inner_step 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --gradient_checkpointing \
   --output_dir $OUTPUT \
   --print_loss \
   --offload \
   --bf16 \
   --enable_tensorboard \
   &> $OUTPUT/training.log
