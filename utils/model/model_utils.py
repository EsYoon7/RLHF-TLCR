# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig

from .reward_model import RewardModel
from .token_reward_model import TokenRewardModel
from .discriminator_reward_model import DiscriminatorRewardModel
from .discriminator_holistic_reward_model import DiscriminatorHolisticRewardModel
from .discriminator_uni_head_reward_model import DiscriminatorUniHeadRewardModel
from .discriminator_holistic_uni_head_reward_model import DiscriminatorHolisticUniHeadRewardModel
from .figa_model import FigaModel
from ..utils import load_state_dict_into_model

from transformers import LlamaForCausalLM

def create_hf_model(
    model_class,
    model_name_or_path,
    tokenizer,
    ds_config=None,
    rlhf_training=False,
    disable_dropout=False,
):
    model_config = AutoConfig.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None
    
    if rlhf_training:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config) 
    else:
        model = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            trust_remote_code=True,
            # use_flash_attention_2=True if "llama" in model_name_or_path else False,
            config=model_config,
            offload_folder='offload', # added by esyoon 2024-03-21-21:22:26
        )

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(
        int(8 * math.ceil(len(tokenizer) / 8.0))
    )  # make the vocab size multiple of 8

    return model


def create_critic_model(
    model_name_or_path,
    tokenizer,
    ds_config,
    num_padding_at_beginning=0,
    rlhf_training=False,
    disable_dropout=False,
    zero_stage=0,
    eval_mode=False,
    bidir_attn=False,
    bidir_layer_num=1
):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = RewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning, bidir_attn=bidir_attn, bidir_layer_num=bidir_layer_num
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model

# added by esyoon 2024-01-13-15:45:09
def create_token_critic_model(
    model_name_or_path,
    tokenizer,
    ds_config,
    num_padding_at_beginning=0,
    rlhf_training=False,
    disable_dropout=False,
    zero_stage=0,
    eval_mode=False,
    bidir_attn=False,
    bidir_layer_num=1,
    top_k=1.0
):
    # OPT model family always put a padding token at the beginning of the sequence,
    # we did not see this in other models but not sure if it is a general rule

    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = TokenRewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning, bidir_attn=bidir_attn, bidir_layer_num=bidir_layer_num,
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model


# added by esyoon 2024-01-27-16:22:43
def create_discriminator_critic_model(
        model_name_or_path,
        tokenizer,
        ds_config,
        num_padding_at_beginning=0,
        rlhf_training=False,
        disable_dropout=False,
        zero_stage=0,
        eval_mode=False,
        bidir_attn=False,
        bidir_layer_num=1,
    ):
    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = DiscriminatorRewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning, bidir_attn=bidir_attn, bidir_layer_num=bidir_layer_num,
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model


def create_discriminator_w_holistc_critic_model(
        model_name_or_path,
        tokenizer,
        ds_config,
        num_padding_at_beginning=0,
        rlhf_training=False,
        disable_dropout=False,
        zero_stage=0,
        eval_mode=False,
        bidir_attn=False,
        bidir_layer_num=1,
    ):
    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = DiscriminatorHolisticRewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning, bidir_attn=bidir_attn, bidir_layer_num=bidir_layer_num,
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model


def create_discriminator_uni_critic_model(
        model_name_or_path,
        tokenizer,
        ds_config,
        num_padding_at_beginning=0,
        rlhf_training=False,
        disable_dropout=False,
        zero_stage=0,
        eval_mode=False,
        bidir_attn=False,
        bidir_layer_num=1,
    ):
    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = DiscriminatorUniHeadRewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning, bidir_attn=bidir_attn, bidir_layer_num=bidir_layer_num,
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model

def create_discriminator_holistic_uni_critic_model(
        model_name_or_path,
        tokenizer,
        ds_config,
        num_padding_at_beginning=0,
        rlhf_training=False,
        disable_dropout=False,
        zero_stage=0,
        eval_mode=False,
        bidir_attn=False,
        bidir_layer_num=1,
    ):
    import time

    start = time.time()
    critic_model = create_hf_model(
        AutoModel,
        model_name_or_path,
        tokenizer,
        ds_config,
        rlhf_training,
        disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = DiscriminatorHolisticUniHeadRewardModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning, bidir_attn=bidir_attn, bidir_layer_num=bidir_layer_num,
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model

def create_figa_model(
        model_name_or_path,
        tokenizer,
        ds_config,
        num_padding_at_beginning=0,
        rlhf_training=False,
        disable_dropout=False,
        zero_stage=0,
        eval_mode=False,
    ):
    import time

    start = time.time()
    critic_model = create_hf_model(
        model_class=AutoModelForCausalLM,
        model_name_or_path=model_name_or_path,
        tokenizer=tokenizer,
        ds_config=ds_config,
        disable_dropout=disable_dropout,
    )
    end = time.time()
    if eval_mode or torch.distributed.get_rank() == 0:
        print(f"> Creating model from_config took {end - start} seconds")

    critic_model = FigaModel(
        critic_model, tokenizer, num_padding_at_beginning=num_padding_at_beginning,
    )

    if rlhf_training:
        # load critic model from checkpoint

        if not os.path.isdir(model_name_or_path):
            model_name_or_path = snapshot_download(model_name_or_path)
        model_ckpt_path = os.path.join(model_name_or_path, "pytorch_model.bin")
        assert os.path.exists(
            model_ckpt_path
        ), f"Cannot find model checkpoint at {model_ckpt_path}"

        start = time.time()
        model_ckpt_state_dict = torch.load(model_ckpt_path, map_location="cpu")
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> torch.load took {end - start} seconds")

        # load critic model from checkpoint with zero-stage 3 compatibility
        # this functionality may be moved to DS checkpoint load API in future
        start = time.time()
        load_state_dict_into_model(
            critic_model, model_ckpt_state_dict, "", zero_stage=zero_stage
        )
        end = time.time()
        if eval_mode or torch.distributed.get_rank() == 0:
            print(f"> Loading model state dict took {end - start} seconds")

    return critic_model