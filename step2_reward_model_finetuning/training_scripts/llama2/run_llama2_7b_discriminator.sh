#!/bin/bash

set -e 
set -x
export TZ="Asia/Seoul"
# export HF_DATASETS_OFFLINE=1 
# export TRANSFORMERS_OFFLINE=1
# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=2
DATA_PATH="Dahoas/full-hh-rlhf"
MODEL_NAME="Llama-2-7b"
# MODEL_PATH="~/workspace/models/llama-2-7b-hf" # Llama-2-7b model path in the local directory
MODEL_PATH="SFT_MODEL_CKPT_PATH" # started from sft model
SEED=2023
MODIFICATION_TRAIN_DATA_PATH="GPT GENERATED MODIFICATED TRAIN DATA PATH"
MODIFICATION_EVAL_DATA_PATH='GPT GENERATED MODIFICATED EVAL DATA PATH'

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log/step2_discriminator_token_reward_${MODEL_NAME/'/'/_}-$TIME_STEP-$SEED"
fi
mkdir -p $OUTPUT


deepspeed --master_port 12348 main_discriminator.py \
   --data_path $DATA_PATH \
   --modification_train_data_path $MODIFICATION_TRAIN_DATA_PATH \
    --modification_eval_data_path $MODIFICATION_EVAL_DATA_PATH \
   --data_output_path "~/workspace/RLHF-TLCR/data_files/llama" \
   --data_split 2,4,4 \
   --model_name_or_path $MODEL_PATH \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 32 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0.1 \
   --num_padding_at_beginning 0 \
   --num_train_epochs 2 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --print_loss \
    --gradient_checkpointing \
    --bf16 \
    --offload \
    --discriminate_reward \
    --uni_discriminator_head \
    --enable_tensorboard \
   &> $OUTPUT/training.log
