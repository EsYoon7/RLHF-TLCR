#!/bin/bash
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export TZ="Asia/Seoul"
set -e
set -x

# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# DeepSpeed Team
DATA_PATH="Dahoas/full-hh-rlhf"

ACTOR_MODEL_PATH="SFT_MODEL_CKPT_PATH"
CRITIC_MODEL_PATH="REWARD_MODEL_CKPT_PATH"

ACTOR_ZERO_STAGE=2
CRITIC_ZERO_STAGE=2
REFERENCE_ZERO_STAGE=3
REWARD_ZERO_STAGE=3
OUTPUT=$1
# LOG_PATH="/hy-tmp/log"
SEED=1235

if [ "$OUTPUT" == "" ]; then
    TIME_STEP=`date "+%Y-%m-%d-%H-%M-%S"`
    OUTPUT="./log/step3_ppo_tlcr-Llama2_7b-full_hh_rlhf-$TIME_STEP-$SEED"
fi

mkdir -p $OUTPUT

ACTOR_LR=1e-6 # 9.65e-6
CRITIC_LR=1e-6

# --offload \
# --offload_reference_model \
# --bf16 \

deepspeed --master_port 1296 main.py \
   --algo "ppo" \
   --data_path $DATA_PATH \
   --data_output_path "~/workspace/RLHF-TLCR/data_files/llama" \
   --data_split 2,4,4 \
   --actor_model_name_or_path $ACTOR_MODEL_PATH \
   --critic_model_name_or_path $CRITIC_MODEL_PATH \
   --num_padding_at_beginning 0 \
   --per_device_generation_batch_size 30 \
   --per_device_training_batch_size 30 \
   --per_device_eval_batch_size 30 \
   --generation_batches 1 \
   --ppo_epochs 1 \
   --max_answer_seq_len 256 \
   --max_prompt_seq_len 256 \
   --actor_learning_rate ${ACTOR_LR} \
   --critic_learning_rate ${CRITIC_LR} \
   --actor_weight_decay 0.1 \
   --critic_weight_decay 0.1 \
   --num_train_epochs 1 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --disable_actor_dropout \
   --disable_critic_dropout \
   --num_warmup_steps 0 \
   --penalty 'full_kl' \
   --kl_ctl 0.1 \
   --gamma 1.0 \
   --lam 0.95 \
   --deepspeed \
   --actor_bf16 \
   --critic_bf16 \
   --offload \
   --offload_critic_model \
   --offload_reference_model \
   --offload_reward_model \
   --seed $SEED \
   --actor_zero_stage $ACTOR_ZERO_STAGE \
   --critic_zero_stage $CRITIC_ZERO_STAGE \
   --reference_zero_stage $REFERENCE_ZERO_STAGE \
   --reward_zero_stage $REWARD_ZERO_STAGE \
   --enable_hybrid_engine \
   --output_dir $OUTPUT \
   --print_answers \
   --save_answers \
   --save_model \
   --discriminate_reward \
   --uni_discriminator_head \
   --sum_to_one_reward \
    --enable_tensorboard \
   &> $OUTPUT/training.log  #     --random_dense_reward \ #    --sum_to_one_reward \
