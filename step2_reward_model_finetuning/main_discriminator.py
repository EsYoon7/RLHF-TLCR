#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
import json
import time

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

import wandb
import ipdb

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
)
from utils.model.model_utils import *
from utils.data.data_utils import *
from utils.utils import (
    print_rank_0,
    to_device,
    save_hf_format,
    set_random_seed,
    save_code,
    get_all_reduce_mean,
    get_optimizer_grouped_parameters,
    save_zero_three_model,
    load_hf_tokenizer,
)
from utils.ds_utils import get_train_ds_config
from utils.module.lora import (
    convert_linear_layer_to_lora,
    convert_lora_to_linear_layer,
    only_optimize_lora_parameters,
    make_model_gradient_checkpointing_compatible,
)
from utils.perf import print_throughput
from utils.gpu_utils import print_machine_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--data_path",
        nargs="*",
        default=["Dahoas/rm-static"],
        help="Path to the training dataset. Accepted format:"
        "1) a single data path, 2) multiple datasets in the"
        "form: dataset1-path dataset2-path ...",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="2,4,4",
        help="Comma-separated list of proportions for training"
        "phase 1, 2, and 3 data. For example the split `2,4,4`"
        "will use 60%% of data for phase 1, 20%% for phase 2"
        "and 20%% for phase 3.",
    )
    parser.add_argument(
        "--data_output_path",
        type=str,
        default="/tmp/data_files/",
        help="Where to store the data-related files such as shuffle index.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help="OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the model."
    )
    parser.add_argument(
        "--seed", type=int, default=2023, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--data_seed", type=int, default=1234, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable HF gradient checkpointing for Actor model.",
    )
    parser.add_argument(
        "--disable_dropout",
        action="store_true",
        help="Disable the dropout of the model.",
    )
    # deepspeed features
    parser.add_argument(
        "--offload", action="store_true", help="Enable ZeRO Offload techniques."
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16.")
    parser.add_argument(
        "--zero_stage",
        type=int,
        default=0,
        help="ZeRO optimization stage for Actor model (and clones).",
    )
    ## LoRA for efficient training setting
    parser.add_argument(
        "--lora_dim",
        type=int,
        default=0,
        help="If > 0, use LoRA for efficient training.",
    )
    parser.add_argument(
        "--lora_module_name",
        type=str,
        default="decoder.layers.",
        help="The scope of LoRA.",
    )
    parser.add_argument(
        "--only_optimize_lora",
        action="store_true",
        help="Only optimize the LoRA parameters.",
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help="Initial LoRA learning rate (after the potential warmup period) to use.",
    )
    ## Print loss
    parser.add_argument(
        "--print_loss", action="store_true", help="Prints loss at each step."
    )
    ## Tensorboard logging
    parser.add_argument(
        "--enable_tensorboard", action="store_true", help="Enable tensorboard logging"
    )
    parser.add_argument("--tensorboard_path", type=str, default="step2_tensorboard")

    # added by esyoon 2024-01-13-15:42:50
    parser.add_argument(
        "--token_level_reward",
        action="store_true",
        help="Token level reward",
    )
    parser.add_argument(
        "--reward_aggregate_top_k",
        type=float,
        default=1.0,
        help="Top k tokens (%) to aggregate",)
    
    parser.add_argument(
        "--bidirectional_reward",
        action="store_true",
        help="Remove causal mask for top layers of reward model",
    )
    parser.add_argument(
        "--bidirectional_layer_num",
        type=int,
        default=1,
        help="Number of layers to remove causal mask",
    )

    parser.add_argument(
        "--discriminate_reward",
        action="store_true",
        help="Discriminator reward",
    )
    parser.add_argument(
        "--holistic_reward",
        action="store_true",
        help="Both Disctrimiator + Holistic reward",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Only evaluate",
    )
    parser.add_argument(
        "--uni_discriminator_head",
        action="store_true",
        help="Use one discriminator head",
    )
    parser.add_argument(
        "--modification_train_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--modification_eval_data_path",
        type=str,
        default=None,
    )

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


def save_model(rm_model, tokenizer, args):
    if args.output_dir is not None:
        print_rank_0("saving model ...", args.global_rank)
        rm_model = convert_lora_to_linear_layer(rm_model)

        if args.global_rank == 0:
            save_hf_format(rm_model, tokenizer, args)
        if args.zero_stage == 3:
            # for zero stage 3, each gpu only has a part of the model, so we need to save the model on each gpu by using DS-Engine
            save_zero_three_model(
                rm_model, args.global_rank, args.output_dir, zero_stage=args.zero_stage
            )


def main():
    args = parse_args()
    args.tensorboard_path = args.output_dir

    global do_eval
    do_eval = args.do_eval

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    if torch.distributed.get_rank() == 0:
        if args.enable_tensorboard:
            #writer = SummaryWriter(
            #    f"{args.tensorboard_path}/step3_tensorboard_logs")
            # wandb.init(project="ReMax",name='opt1.3b_rm_hh_rlhf')
            wandb.init(project="ReMax",name='llama2-7b_discriminator_rm_hh_rlhf')
        else:
            wandb.init(project="ReMax",name='debug12/26/23', mode="disabled")


    ds_config = get_train_ds_config(
        offload=args.offload,
        stage=args.zero_stage,
        enable_tensorboard=args.enable_tensorboard,
        bf16=args.bf16,
        tb_path=args.tensorboard_path,
        tb_name="",
    )
    ds_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
    ds_config["train_batch_size"] = (
        args.per_device_train_batch_size
        * torch.distributed.get_world_size()
        * args.gradient_accumulation_steps
    )

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    if args.global_rank == 0:
        with open(
            os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            for key, value in args.__dict__.items():
                json.dump({key: value}, f, ensure_ascii=False)
                f.write("\n")
        save_code(args.output_dir)

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    if args.token_level_reward:
        rm_model = create_token_critic_model(
            args.model_name_or_path,
            tokenizer,
            ds_config,
            args.num_padding_at_beginning,
            disable_dropout=args.disable_dropout,
            bidir_attn=args.bidirectional_reward,
            bidir_layer_num=args.bidirectional_layer_num,
            top_k=args.reward_aggregate_top_k
        )
    elif args.discriminate_reward:
        if args.holistic_reward:
            rm_model = create_discriminator_w_holistc_critic_model(
                args.model_name_or_path,
                tokenizer,
                ds_config,
                args.num_padding_at_beginning,
                disable_dropout=args.disable_dropout,
                bidir_attn=args.bidirectional_reward,
                bidir_layer_num=args.bidirectional_layer_num
            )
        if args.uni_discriminator_head:
            rm_model = create_discriminator_uni_critic_model(
                args.model_name_or_path,
                tokenizer,
                ds_config,
                args.num_padding_at_beginning,
                disable_dropout=args.disable_dropout,
                bidir_attn=args.bidirectional_reward,
                bidir_layer_num=args.bidirectional_layer_num
            )
        else:
            rm_model = create_discriminator_critic_model(
                args.model_name_or_path,
                tokenizer,
                ds_config,
                args.num_padding_at_beginning,
                disable_dropout=args.disable_dropout,
                bidir_attn=args.bidirectional_reward,
                bidir_layer_num=args.bidirectional_layer_num
            )
    else:
        rm_model = create_critic_model(
            args.model_name_or_path,
            tokenizer,
            ds_config,
            args.num_padding_at_beginning,
            disable_dropout=args.disable_dropout,
            bidir_attn=args.bidirectional_reward,
            bidir_layer_num=args.bidirectional_layer_num
        )
        
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(
            rm_model, args.lora_module_name, args.lora_dim
        )
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
            rm_model = make_model_gradient_checkpointing_compatible(rm_model)

    train_phase = 2
    if args.discriminate_reward:
        import pickle
        # train_dataset = pickle.load(open('/data/kakao/workspace/KU_KaKao/prompt_generate/gpt4_generation_token_processed/data_final.pkl', 'rb'))
        train_dataset = pickle.load(open(args.modification_train_data_path, 'rb'))
        import datasets
        
        train_dataset  =  datasets.Dataset.from_dict(train_dataset)
        train_dataset = create_discriminator_dataset(
            args.local_rank,
            train_dataset,
            args.data_seed,
            tokenizer,
            args.max_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
        )
        
        eval_dataset = create_discriminator_eval_dataset(
            args.local_rank,
            args.data_path,
            args.data_seed,
            tokenizer,
            args.max_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
        )
        # eval_discriminator_dataset = pickle.load(open('/data/kakao/workspace/KU_KaKao/prompt_generate/gpt4_generation_token_processed/eval_data_final.pkl', 'rb'))
        eval_discriminator_dataset = pickle.load(open(args.modification_eval_data_path, 'rb'))
        eval_discriminator_dataset = datasets.Dataset.from_dict(eval_discriminator_dataset)
        eval_discriminator_dataset = create_discriminator_token_eval_dataset(
            args.local_rank,
            eval_discriminator_dataset,
            args.data_seed,
            tokenizer,
            args.max_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
        )

        data_collator = DataCollatorReward()
        discriminator_data_collator = DataCollatorDiscriminatorReward()
        eval_data_collator = DataCollatorDiscriminatorEvalReward()
        eval_discriminator_data_collator = DataCollatorDiscriminatorTokenEvalReward()


    else:
        train_dataset, eval_dataset = create_prompt_dataset(
            args.local_rank,
            args.data_path,
            args.data_split,
            args.data_output_path,
            train_phase,
            args.data_seed,
            tokenizer,
            args.max_seq_len,
            end_of_conversation_token=tokenizer.eos_token,
            reload=True
        )
        data_collator = DataCollatorReward()

    # DataLoaders creation:
    
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    
    if args.discriminate_reward:
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=discriminator_data_collator,
            sampler=train_sampler,
            batch_size=args.per_device_train_batch_size,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            collate_fn=data_collator,
            sampler=train_sampler,
            batch_size=args.per_device_train_batch_size,
        )
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=eval_data_collator,
        sampler=eval_sampler,
        batch_size=args.per_device_eval_batch_size,
    )
    if args.discriminate_reward:
        eval_discriminator_sampler = SequentialSampler(eval_discriminator_dataset)
        eval_discriminator_dataloader = DataLoader(
            eval_discriminator_dataset,
            collate_fn=eval_discriminator_data_collator,
            sampler=eval_discriminator_sampler,
            batch_size=args.per_device_eval_batch_size,
        )
        
    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model.forward_eval(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()

            if not do_eval:
                if step == 99:  # For faster evaluation and debugging
                    break
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        return scores, acc    
    
    def evaluation_discriminator_preference_eval(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model.forward_discriminator_preference_eval(**batch)

            chosen = outputs["chosen_mean_token_scores"]
            rejected = outputs["rejected_mean_token_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_token_scores"].mean().float()
            if not do_eval:
                if step == 99:  # For faster evaluation and debugging
                    break
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        try:
            acc = get_all_reduce_mean(acc).item()
            scores = get_all_reduce_mean(scores).item()
        except:
            pass
        return scores, acc

    def evaluation_discriminator_eval(model, eval_dataloader):
        model.eval()
        positive_acc = 0
        negative_acc = 0

        reject_pred_overlap = 0
        modification_pred_overlap = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model.forward_discriminator_eval(**batch)

            positive_acc += outputs["positive_acc"]
            negative_acc += outputs["negative_acc"]
            if not args.uni_discriminator_head:
                reject_pred_overlap += outputs["num_reject_pred_both_ratio"]
                modification_pred_overlap += outputs["num_modification_pred_both_ratio"]

            if not do_eval:
                if step == 99:  # For faster evaluation and debugging
                    break

        total_positive_acc = positive_acc / (step + 1)
        total_negative_acc = negative_acc / (step + 1)
        if not args.uni_discriminator_head:
            total_reject_pred_overlap = reject_pred_overlap / (step + 1)
            total_modification_pred_overlap = modification_pred_overlap / (step + 1)
        try:
            total_positive_acc = get_all_reduce_mean(total_positive_acc).item()
            total_negative_acc = get_all_reduce_mean(total_negative_acc).item()
            if not args.uni_discriminator_head:
                total_reject_pred_overlap = get_all_reduce_mean(total_reject_pred_overlap).item()
                total_modification_pred_overlap = get_all_reduce_mean(total_modification_pred_overlap).item()
        except:
            pass
        return total_positive_acc, total_negative_acc   
        # return total_reject_pred_overlap, total_modification_pred_overlap
        

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        rm_model, args.weight_decay, args.lora_learning_rate
    )

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(
        optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95)
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True,
    )

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    print_machine_info(args.global_rank)
    print_rank_0(
        f"***** Evaluating reward, Epoch {1}/{args.num_train_epochs} *****",
        args.global_rank,
    )
    if args.holistic_reward:
        reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
    else:
        reward_score, acc = evaluation_discriminator_preference_eval(rm_model, eval_dataloader)
    if args.discriminate_reward:
        positive_acc, negative_acc = evaluation_discriminator_eval(rm_model, eval_discriminator_dataloader)
    print_rank_0(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank,
    )
    if args.discriminate_reward:
        print_rank_0(
            f"positive_acc (higher is better) : {positive_acc}, negative_acc (higher is better) : {negative_acc}",
            args.global_rank,
        )
    if args.do_eval:
        return

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank,
        )
        rm_model.train()
        mean_loss = 0
        mean_holistic_loss = 0
        mean_positive_loss = 0
        mean_negative_loss = 0
        t0 = time.time()
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            batch = to_device(batch, device)
            outputs = rm_model(**batch, use_cache=False)
            loss = outputs["loss"]
            if args.holistic_reward:
                holistic_loss = outputs["holistic_loss"]
            positive_loss = outputs["positive_loss"]
            negative_loss = outputs["negative_loss"]
            if args.print_loss:
                if args.holistic_reward:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}, holistic loss = {holistic_loss}, positive loss = {positive_loss}, negative loss = {negative_loss}"
                    )
                else:
                    print(
                        f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, loss = {loss}, positive loss = {positive_loss}, negative loss = {negative_loss}"
                    )

            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
            if args.holistic_reward:
                mean_holistic_loss += holistic_loss.item()
            mean_positive_loss += positive_loss.item()
            mean_negative_loss += negative_loss.item()

            end = time.time()
            if torch.distributed.get_rank() == 0:
                print_throughput(rm_model.module, args, end - start, args.global_rank)

            if (step + 1) % int(len(train_dataloader) // 10) == 0:
                if args.discriminate_reward:
                    if args.holistic_reward:
                        print_rank_0(
                            f"Epoch {epoch + 1}/{args.num_train_epochs} Step {step + 1}/{len(train_dataloader)} with loss: {mean_loss / (step + 1)}, with holistic loss: {mean_holistic_loss / (step + 1)}, with positive loss: {mean_positive_loss / (step + 1)}, with negative loss: {mean_negative_loss / (step + 1)}, time used: {(time.time() - t0) / 60:.0f} minutes",
                            args.global_rank,
                        )
                    else:
                        print_rank_0(
                            f"Epoch {epoch + 1}/{args.num_train_epochs} Step {step + 1}/{len(train_dataloader)} with loss: {mean_loss / (step + 1)}, with positive loss: {mean_positive_loss / (step + 1)}, with negative loss: {mean_negative_loss / (step + 1)}, time used: {(time.time() - t0) / 60:.0f} minutes",
                        )
                else:
                    print_rank_0(
                        f"Epoch {epoch + 1}/{args.num_train_epochs} Step {step + 1}/{len(train_dataloader)} with loss: {mean_loss / (step + 1)}, time used: {(time.time() - t0) / 60:.0f} minutes",
                        args.global_rank,
                    )
                # Evaluate reward_loss on the validation set.
                print_rank_0(
                    f"***** Evaluating reward, Epoch {epoch + 1}/{args.num_train_epochs} Step {step}/{len(train_dataloader)} *****",
                    args.global_rank,
                )
                if args.discriminate_reward:
                    positive_acc, negative_acc = evaluation_discriminator_eval(rm_model, eval_discriminator_dataloader)
                    print_rank_0(
                        f"positive_acc (higher is better) : {positive_acc}, negative_acc (higher is better) : {negative_acc}",
                        args.global_rank,
                    )
                    reward_score, acc = evaluation_discriminator_preference_eval(rm_model, eval_dataloader)
                    print_rank_0(
                        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
                        args.global_rank,
                    )
                else:
                    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
                    print_rank_0(
                        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
                        args.global_rank,
                    )
                print_machine_info(args.global_rank)
                if rm_model.monitor.enabled and rm_model.global_rank == 0:
                    summary_events = [
                        ("Test/loss", mean_loss / (step + 1), rm_model.global_samples),
                        ("Test/reward_score", reward_score, rm_model.global_samples),
                        ("Test/acc", acc, rm_model.global_samples),
                    ]
                    # rm_model.monitor.write_events(summary_events)

                    # added by esyoon 2024-01-09-22:22:33
                    if args.enable_tensorboard:
                        if args.discriminate_reward:
                            if args.holistic_reward:
                                    wandb.log({"rm_train/loss": mean_loss / (step + 1), "rm_train/holistic_loss": mean_holistic_loss / (step + 1), "rm_train/positive_loss": mean_positive_loss / (step + 1), "rm_train/negative_loss": mean_negative_loss / (step + 1), "rm_eval/reward score": reward_score, "rm_eval/acc": acc})
                            else:
                               wandb.log({"rm_train/loss": mean_loss / (step + 1), "rm_train/positive_loss": mean_positive_loss / (step + 1), "rm_train/negative_loss": mean_negative_loss / (step + 1), "rm_eval/reward score": reward_score, "rm_eval/acc": acc ,"rm_eval/positive_acc": positive_acc, "rm_eval/negative_acc": negative_acc})
                        else:
                            wandb.log({"rm_train/loss": mean_loss / (step + 1), "rm_eval/reward score": reward_score, "rm_eval/acc": acc})
                rm_model.train()
                save_model(rm_model, tokenizer, args)

        rm_model.tput_timer.update_epoch_count()

    save_model(rm_model, tokenizer, args)


if __name__ == "__main__":
    main()
