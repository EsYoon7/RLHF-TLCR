# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
from torch import nn
import ipdb

def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')

## Note that the following code is modified from
## https://github.com/CarperAI/trlx/blob/main/examples/summarize_rlhf/reward_model/reward_model.py
class DiscriminatorUniHeadRewardModel(nn.Module):
    def __init__(self, base_model, tokenizer, num_padding_at_beginning=0, bidir_attn=None, bidir_layer_num=1):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            # self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=False)
            self.discriminate_head = nn.Linear(self.config.word_embed_proj_dim, 2, bias=False)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = (
                self.config.hidden_size
                if hasattr(self.config, "hidden_size")
                else self.config.n_embd
            )
            # self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self.discriminate_head = nn.Linear(self.config.n_embd, 2, bias=False)

        self.rwtranrsformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.use_template = False
        self.hard_label = False
        self.random_dense_reward = False
        self.fixed_reward = False
        self.sum_to_one_reward = False
        self.positive_reward_only = False
        self.negative_reward_only = False

        self.positive_reward = None
        self.negative_reward = None

        self.custom_template = False
        

        if bidir_attn:
            from transformers import LlamaModel

            if type(base_model) == LlamaModel:
                bound_method = new_Llama_forward.__get__(self.rwtranrsformer, self.rwtranrsformer.__class__)
                setattr(self.rwtranrsformer, 'forward', bound_method)

                self.rwtranrsformer.bidir_layer_num = bidir_layer_num
                for i in range(bidir_layer_num):
                    layer_num = -(i + 1)
                    bound_method = new_forward.__get__(self.rwtranrsformer.layers[layer_num].self_attn, self.rwtranrsformer.layers[layer_num].self_attn.__class__)
                    setattr(self.rwtranrsformer.layers[layer_num].self_attn, 'forward', bound_method)
            else:
                import ipdb; ipdb.set_trace() #not implemented yet
         
        assert tokenizer.eos_token == tokenizer.pad_token

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        reject_edit_label=None,
        modification_edit_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prompt_attn_mask=None, # added by esyoon 2024-01-13-18:36:51
        use_cache=False,
    ):
        loss = None
        positive_loss = None
        negative_loss = None

        if self.config.model_type == "llama" or self.config.model_type == "mistral":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        # rewards = self.v_head(hidden_states).squeeze(-1)
        if reject_edit_label is not None and modification_edit_label is not None:
            discriminator_token_pred = self.discriminate_head(hidden_states).squeeze(-1)

        chosen_mean_scores = []
        rejected_mean_scores = []

        # NOTE: mean score tracking for discriminator 어떻게?

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        
        # input_ids and attention_mask contains that for chosen, reject, and modification
        bs = input_ids.shape[0] // 3
        seq_len = input_ids.shape[1]
        prompt_len = attention_mask.sum(1)

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:bs*2]
        modification_ids = input_ids[bs*2:]

        negative_token_pred = discriminator_token_pred[bs:bs*2]
        positive_token_pred = discriminator_token_pred[bs*2:]
        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0
        holistic_loss = 0
        positive_loss = 0
        negative_loss = 0

        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            if not self.custom_template:
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                c_ind = (
                    c_inds[self.num_padding_at_beginning].item()
                    if len(c_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            else:
                chosen_id_temp = chosen_id.clone()
                inds_temp = (chosen_id[:prompt_len[i]-1] == self.PAD_ID).nonzero()
                chosen_id_temp[inds_temp] = -1000
                c_inds_temp = (chosen_id_temp == self.PAD_ID).nonzero()
            
                c_ind = (
                    c_inds_temp[self.num_padding_at_beginning].item()
                    if len(c_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )

            # # added by esyoon 2024-01-13-17:43:54
            # # for new template for llama this should be used becuase there are pad token inside the template
            # if len(c_inds) > self.num_padding_at_beginning:
            #     c_ind = seq_len - 1
            # else:
            #     for idx in c_inds:
            #         if chosen_id[idx.item()+1] == self.PAD_ID:
            #             c_ind = idx.item()
            #             break
            #         else:
            #             continue
                
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_id.size(-1) # added by esyoon 2024-02-01-15:27:17
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                if not self.custom_template:
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = (
                        r_inds[self.num_padding_at_beginning].item()
                        if len(r_inds) > self.num_padding_at_beginning
                        else seq_len - 1
                    )
                else:
                    # added by esyoon 2024-01-13-17:47:23
                    rejected_id_temp = rejected_id.clone()
                    inds_temp = (rejected_id[:prompt_len[bs+i] - 1] == self.PAD_ID).nonzero()
                    rejected_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                    r_inds_temp = (rejected_id_temp == self.PAD_ID).nonzero()
                
                    r_ind = (
                        r_inds_temp[self.num_padding_at_beginning].item()
                        if len(r_inds_temp) > self.num_padding_at_beginning
                        else seq_len
                    )

                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            # token level positive and negative cross entropy loss
            if self.hard_label:
                if reject_edit_label is not None and modification_edit_label is not None:
                    modification_false_idx = (modification_edit_label[i] == 0).nonzero().flatten()
                    rejection_true_idx = (reject_edit_label[i] == 1).nonzero().flatten()
                    rejection_false_idx = (reject_edit_label[i] == 0).nonzero().flatten()

                    reject_edit_label_ = reject_edit_label.clone()
                    modification_edit_label_ = modification_edit_label.clone()

                    reject_edit_label_[i, rejection_true_idx] = 0
                    reject_edit_label_[i, rejection_false_idx] = -100
                    modification_edit_label_[i, modification_false_idx] = -100

                    positive_loss += torch.nn.functional.cross_entropy(positive_token_pred[i], modification_edit_label_[i], reduction='mean')
                    negative_loss += torch.nn.functional.cross_entropy(negative_token_pred[i], reject_edit_label_[i], reduction='mean')


            else:
                if reject_edit_label is not None and modification_edit_label is not None:
                    reject_edit_idx_label = (reject_edit_label[i] != -100).nonzero().flatten()
                    reject_edit_label_ = reject_edit_label.clone()
                    reject_edit_label_ = reject_edit_label_[i, reject_edit_idx_label]
                    

                    modification_edit_idx_label = (modification_edit_label[i] != -100).nonzero().flatten()
                    modification_edit_label_ = modification_edit_label.clone()
                    modification_edit_label_ = modification_edit_label_[i, modification_edit_idx_label]

                    modification_false_idx = (modification_edit_label_ == 0).nonzero().flatten()
                    rejection_true_idx = (reject_edit_label_ == 1).nonzero().flatten()
                    rejection_false_idx = (reject_edit_label_ == 0).nonzero().flatten()

                    reject_edit_label_ = torch.nn.functional.one_hot(reject_edit_label_, 2).to(torch.float32)
                    modification_edit_label_ = torch.nn.functional.one_hot(modification_edit_label_, 2).to(torch.float32)

                    modification_edit_label_[modification_false_idx, :] = torch.tensor([0.5, 0.5]).to(modification_edit_label_.device)
                    reject_edit_label_[rejection_false_idx, :] = torch.tensor([0.5, 0.5]).to(reject_edit_label_.device)     
                    reject_edit_label_[rejection_true_idx, :] = torch.FloatTensor([1, 0]).to(reject_edit_label_.device) 
                    # positive loss
                    positive_loss += softmax_cross_entropy_with_softtarget(positive_token_pred[i, modification_edit_idx_label], modification_edit_label_, reduction='mean')
                    # positive_loss += torch.nn.functional.cross_entropy(positive_token_pred[i], modification_edit_label[i], reduction='mean')
                    # negative loss
                    negative_loss += softmax_cross_entropy_with_softtarget(negative_token_pred[i, reject_edit_idx_label], reject_edit_label_, reduction='mean')

        # TODO: 필요하다면 chosen mean score, rejected mean score를 prediction을 통해서 같이 return 할 수 있도록 코드를 짜야됨 # added by esyoon 2024-02-01-15:37:30
                
        positive_loss = positive_loss / bs
        negative_loss = negative_loss / bs
        loss = positive_loss + negative_loss
        # chosen_mean_scores = torch.stack(chosen_mean_scores)
        # rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "positive_loss": positive_loss,
            "negative_loss": negative_loss,
            # "chosen_mean_scores": chosen_mean_scores,
            # "rejected_mean_scores": rejected_mean_scores,
        } 
     # we don't use this forward value for critic model instead use discriminator_holisct verseion since it contains value head already  # added by esyoon 2024-02-01-15:40:58
    def forward_value(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        return_value_only=False,
        prompt_length=0,
        use_cache=False,
    ):
        if self.config.model_type == "llama" or self.config.model_type == "mistral":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        discriminator_token_pred = self.discriminate_head(hidden_states).squeeze(-1)

        discriminator_token_prob = discriminator_token_pred.softmax(dim=-1)

        if return_value_only:
            ipdb.set_trace()
            return discriminator_token_pred
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            assert (
                prompt_length > 1
            ), "prompt_length must be greater than 1 to help select the end score"
            bs = input_ids.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = (
                []
            )  # we use this name for consistency with the original forward function
            reward_scores = (
                []
            )
            token_reward_scores = (
                []
            )
            
            for i in range(bs):
                input_id = input_ids[i]
                # value = values[i]
                # holistic_reward = torch.zeros_like(value)
                

                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = (
                    c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len - 1
                )   
                output_reward = torch.zeros_like(input_id, dtype=torch.float32)
                token_prob = discriminator_token_prob[i][prompt_length:c_ind].softmax(dim=1)
                token_reward_decision_prob, token_reward_decision = token_prob.max(dim=-1)
                reward_sign = (-1) * (-1) ** token_reward_decision
                token_reward = torch.where(((0.4 < token_reward_decision_prob) & (token_reward_decision_prob < 0.6)), 0 , token_reward_decision_prob)
                # reward = torch.mul(reward_sign, token_reward)
                if self.random_dense_reward:
                    reward_sign = torch.rand(reward_sign.size())
                    reward_sign = torch.where(reward_sign <= 0.4, -1, reward_sign)
                    reward_sign = torch.where(reward_sign >= 0.6, 1, reward_sign)
                    reward_sign = torch.where(((reward_sign < 0.6) & (reward_sign > 0.4)),  0, reward_sign)
                    token_reward = (1 - 0.55) * torch.rand(token_reward.size()) + 0.55
                
                token_reward = torch.mul(reward_sign, token_reward)

                if self.fixed_reward:
                    positive_token_reward_idx = (token_reward > 0).nonzero().flatten()
                    negative_token_reward_idx = (token_reward < 0).nonzero().flatten()
                    token_reward[positive_token_reward_idx] = self.positive_reward
                    token_reward[negative_token_reward_idx] = self.negative_reward

                if self.sum_to_one_reward:
                    token_reward = 2 * token_prob[: ,1] - 1
                    
                    if self.positive_reward_only:
                        token_reward = torch.where(token_reward < 0, 0, token_reward)
                    if self.negative_reward_only:
                        token_reward = torch.where(token_reward > 0, 0, token_reward)

                    if self.fixed_reward:
                        positive_token_reward_idx = (token_reward > 0).nonzero().flatten()
                        negative_token_reward_idx = (token_reward < 0).nonzero().flatten()
                        token_reward[positive_token_reward_idx] = self.positive_reward
                        token_reward[negative_token_reward_idx] = self.negative_reward
                    
                output_reward[prompt_length:c_ind] = token_reward
                # holistic_reward = selective_postivie_token_reward[i] - selective_negative_token_reward[i]
                # # reward = holistic_reward + selective_postivie_token_reward[i] - selective_negative_token_reward[i]
                # reward = selective_postivie_token_reward[i] - selective_negative_token_reward[i]
                # token_reward = selective_postivie_token_reward[i] - selective_negative_token_reward[i]
                # chosen_end_scores.append(value[c_ind])
                reward_scores.append(output_reward)
                token_reward_scores.append(output_reward)
            return {
                "values": torch.stack(reward_scores), #not used
                # "chosen_end_scores": torch.stack(chosen_end_scores),
                "reward_scores": torch.stack(reward_scores),
                "token_reward_scores": torch.stack(token_reward_scores),
            }
        

    def forward_discriminator_preference_eval(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        prompt_length=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prompt_attn_mask=None, # added by esyoon 2024-01-13-18:36:51
        use_cache=False,
    ):
        loss = None
        holistic_loss = None
        positive_loss = None
        negative_loss = None

        if self.config.model_type == "llama" or self.config.model_type == "mistral":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        discriminator_token_pred = self.discriminate_head(hidden_states).squeeze(-1)

        chosen_mean_token_scores = []
        rejected_mean_token_scores = []

        # NOTE: mean score tracking for discriminator 어떻게?

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        
        # input_ids and attention_mask contains that for chosen, reject
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]
        prompt_len = attention_mask.sum(1)


        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]

        chosen_token_pred = discriminator_token_pred[:bs]
        rejected_token_pred = discriminator_token_pred[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding

        for i in range(bs):
            input_prompt_length = prompt_length[i].item()
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            if not self.custom_template:

                c_inds = (chosen_id == self.PAD_ID).nonzero()
                c_ind = (
                    c_inds[self.num_padding_at_beginning].item()
                    if len(c_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            else:
                chosen_id_temp = chosen_id.clone()
                inds_temp = (chosen_id[:prompt_len[i]-1] == self.PAD_ID).nonzero()
                chosen_id_temp[inds_temp] = -1000
                c_inds_temp = (chosen_id_temp == self.PAD_ID).nonzero()
            
                c_ind = (
                    c_inds_temp[self.num_padding_at_beginning].item()
                    if len(c_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )
            # c_ind = (
            #     prompt_len[i].item()
            #     if prompt_len[i].item() < seq_len
            #     else seq_len - 1
            # )

            # # added by esyoon 2024-01-13-17:43:54
            # # for new template for llama this should be used becuase there are pad token inside the template
            # if len(c_inds) > self.num_padding_at_beginning:
            #     c_ind = seq_len - 1
            # else:
            #     for idx in c_inds:
            #         if chosen_id[idx.item()+1] == self.PAD_ID:
            #             c_ind = idx.item()
            #             break
            #         else:
            #             continue
                
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_id.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                if not self.custom_template:
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = (
                        r_inds[self.num_padding_at_beginning].item()
                        if len(r_inds) > self.num_padding_at_beginning
                        else seq_len - 1
                    )
                else:
                    # added by esyoon 2024-01-13-17:47:23
                    rejected_id_temp = rejected_id.clone()
                    inds_temp = (rejected_id[:prompt_len[bs+i] - 1] == self.PAD_ID).nonzero()
                    rejected_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                    r_inds_temp = (rejected_id_temp == self.PAD_ID).nonzero()
                
                    r_ind = (
                        r_inds_temp[self.num_padding_at_beginning].item()
                        if len(r_inds_temp) > self.num_padding_at_beginning
                        else seq_len 
                    )

                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0

            if chosen_id[c_ind] == self.PAD_ID:
                chosen_token_prob = chosen_token_pred[i][input_prompt_length:c_ind-1].softmax(dim=1)
            else:
                chosen_token_prob = chosen_token_pred[i][input_prompt_length:c_ind].softmax(dim=1)

            if rejected_id[r_ind] == self.PAD_ID:
                rejected_token_prob = rejected_token_pred[i][input_prompt_length:r_ind-1].softmax(dim=1)
            else:
                rejected_token_prob = rejected_token_pred[i][input_prompt_length:r_ind].softmax(dim=1)


            # chosen_positive_token_reward = chosen_postivie_token_entropy * chosen_positive_token_prob[:, 1]
            # chosen_negative_token_reward = chosen_negative_token_entropy * chosen_negative_token_prob[:, 1]
            # rejected_positive_token_reward = rejected_postivie_token_entropy * rejected_positive_token_prob[:, 1]
            # rejected_negative_token_reward = rejected_negative_token_entropy * rejected_negative_token_prob[:, 1]
                
            chosen_token_decision = chosen_token_prob.argmax(dim=1)
            rejected_token_decision = rejected_token_prob.argmax(dim=1)

            chonse_token_decision_prob, chosen_token_decision = chosen_token_prob.max(dim=-1)
            rejected_token_decision_prob, rejected_token_decision = rejected_token_prob.max(dim=-1)
            if self.hard_label:
                chosen_token_positive_reward_idx = (chosen_token_decision == 1).nonzero().flatten()
                chosen_token_negative_reward_idx = (chosen_token_decision == 0).nonzero().flatten()

                rejected_token_positive_reward_idx = (rejected_token_decision == 1).nonzero().flatten()
                rejected_token_negative_reward_idx = (rejected_token_decision == 0).nonzero().flatten()

                chosen_sum_token_reward = chonse_token_decision_prob[chosen_token_positive_reward_idx].sum() - chonse_token_decision_prob[chosen_token_negative_reward_idx].sum()
                reject_sum_token_reward = rejected_token_decision_prob[rejected_token_positive_reward_idx].sum() - rejected_token_decision_prob[rejected_token_negative_reward_idx].sum()

            else:
                chosen_reward_sign = (-1) * (-1) ** chosen_token_decision
                rejected_reward_sign = (-1) * (-1) ** rejected_token_decision

                chosen_token_reward = torch.where(((0.45 < chonse_token_decision_prob) & (chonse_token_decision_prob < 0.55)), 0 , chonse_token_decision_prob)
                rejected_token_reward = torch.where(((0.45 < rejected_token_decision_prob) & (rejected_token_decision_prob < 0.55)), 0 , rejected_token_decision_prob)

                chosen_token_reward = torch.mul(chosen_reward_sign, chosen_token_reward)
                rejected_token_reward = torch.mul(rejected_reward_sign, rejected_token_reward)

                chosen_sum_token_reward = chosen_token_reward.sum()
                reject_sum_token_reward = rejected_token_reward.sum()

            # selective_chosen_positive_token_reward = torch.where(chosen_positive_token_reward > chosen_negative_token_reward, chosen_positive_token_reward, 0)
            # selective_chosen_negative_token_reward = torch.where(chosen_positive_token_reward < chosen_negative_token_reward, chosen_negative_token_reward, 0)  
            # selective_rejected_positive_token_reward = torch.where(rejected_positive_token_reward > rejected_negative_token_reward, rejected_positive_token_reward, 0)  
            # selective_rejected_negative_token_reward = torch.where(rejected_positive_token_reward < rejected_negative_token_reward, rejected_negative_token_reward, 0)  

            # # chosen_token_reward = chosen_positive_token_reward.mean() - chosen_negative_token_reward.mean() + chosen_reward[c_ind]
            # # rejected_token_reward = rejected_positive_token_reward.mean() - rejected_negative_token_reward.mean() + rejected_reward[r_ind]

            # # chosen_token_reward = selective_chosen_positive_token_reward.mean() - selective_chosen_negative_token_reward.mean() + chosen_reward[c_ind]
            # chosen_token_reward = selective_chosen_positive_token_reward.sum() - selective_chosen_negative_token_reward.sum()
            # # rejected_token_reward = selective_rejected_positive_token_reward.mean() - selective_rejected_negative_token_reward.mean() + rejected_reward[r_ind]
            # rejected_token_reward = selective_rejected_positive_token_reward.sum() - selective_rejected_negative_token_reward.sum()

            chosen_mean_token_scores.append(chosen_sum_token_reward)
            rejected_mean_token_scores.append(reject_sum_token_reward)

        chosen_mean_token_scores = torch.stack(chosen_mean_token_scores)
        rejected_mean_token_scores = torch.stack(rejected_mean_token_scores)
        return {
            "chosen_mean_token_scores": chosen_mean_token_scores,
            "rejected_mean_token_scores": rejected_mean_token_scores,
        }

    def forward_discriminator_eval(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        reject_edit_label=None,
        modification_edit_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prompt_length=None,
        prompt_attn_mask=None, # added by esyoon 2024-01-13-18:36:51
        use_cache=False,
    ):
        loss = None
        positive_loss = 0
        negative_loss = 0

        if self.config.model_type == "llama" or self.config.model_type == "mistral":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        if reject_edit_label is not None and modification_edit_label is not None:
            # positive_token_pred = self.positive_discriminate_head(hidden_states).squeeze(-1)
            # negative_token_pred = self.negative_disciriminate_head(hidden_states).squeeze(-1)
            discriminator_token_pred = self.discriminate_head(hidden_states).squeeze(-1)

        # NOTE: mean score tracking for discriminator 어떻게?

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        
        # input_ids and attention_mask contains that for chosen, reject, and modification
        bs = input_ids.shape[0] // 3
        seq_len = input_ids.shape[1]
        prompt_len = attention_mask.sum(1)

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:bs*2]
        modification_ids = input_ids[bs*2:]

        negative_token_pred = discriminator_token_pred[bs:bs*2]
        positive_token_pred = discriminator_token_pred[bs*2:]
        # Compute pairwise loss. Only backprop on the different tokens before padding

        positive_accuracy = 0
        negative_accuracy = 0

        num_reject_pred_both_ratio = 0
        num_modification_pred_both_ratio = 0

        for i in range(bs):
            input_prompt_length = prompt_length[i].item()
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            modification_id = modification_ids[i]
            if not self.custom_template:
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                
                c_ind = (
                    c_inds[self.num_padding_at_beginning].item()
                    if len(c_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            else:
                chosen_id_temp = chosen_id.clone()
                inds_temp = (chosen_id[:prompt_len[i]-1] == self.PAD_ID).nonzero()
                chosen_id_temp[inds_temp] = -1000
                c_inds_temp = (chosen_id_temp == self.PAD_ID).nonzero()
            
                c_ind = (
                    c_inds_temp[self.num_padding_at_beginning].item()
                    if len(c_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )

            # # added by esyoon 2024-01-13-17:43:54
            # # for new template for llama this should be used becuase there are pad token inside the template
            # if len(c_inds) > self.num_padding_at_beginning:
            #     c_ind = seq_len - 1
            # else:
            #     for idx in c_inds:
            #         if chosen_id[idx.item()+1] == self.PAD_ID:
            #             c_ind = idx.item()
            #             break
            #         else:
            #             continue
                
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_id.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                if not self.custom_template:
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = (
                        r_inds[self.num_padding_at_beginning].item()
                        if len(r_inds) > self.num_padding_at_beginning
                        else seq_len - 1
                    )
                else: 
                    # added by esyoon 2024-01-13-17:47:23
                    rejected_id_temp = rejected_id.clone()
                    inds_temp = (rejected_id[:prompt_len[bs+i] - 1] == self.PAD_ID).nonzero()
                    rejected_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                    r_inds_temp = (rejected_id_temp == self.PAD_ID).nonzero()
                
                    r_ind = (
                        r_inds_temp[self.num_padding_at_beginning].item()
                        if len(r_inds_temp) > self.num_padding_at_beginning
                        else seq_len 
                    )

                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            if not self.custom_template:
                m_inds = (modification_id == self.PAD_ID).nonzero()
                m_ind = (
                    m_inds[self.num_padding_at_beginning].item()
                    if len(m_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )
            else:
                modification_id_temp = modification_id.clone()
                inds_temp = (modification_id[:prompt_len[2*bs+i] - 1] == self.PAD_ID).nonzero()
                modification_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                m_inds_temp = (modification_id_temp == self.PAD_ID).nonzero()
            
                m_ind = (
                    m_inds_temp[self.num_padding_at_beginning].item()
                    if len(m_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )

            # token level positive and negative cross entropy loss
            if reject_edit_label is not None and modification_edit_label is not None:
                reject_edit_label_ = reject_edit_label.clone()
                modification_edit_label_ = modification_edit_label.clone()
                modification_false_idx = (modification_edit_label[i] == 0).nonzero().flatten()
                rejection_true_idx = (reject_edit_label[i] == 1).nonzero().flatten()
                rejection_false_idx = (reject_edit_label[i] == 0).nonzero().flatten()
                modification_edit_label_[i, modification_false_idx] = -100
                # reject_edit_label_[i, rejection_false_idx] = 0.5
                reject_edit_label_[i, rejection_false_idx] = -100
                reject_edit_label_[i, rejection_true_idx] = 0
                positive_token_decision = positive_token_pred[i][input_prompt_length:m_ind].argmax(dim=-1)
                negative_token_decision = negative_token_pred[i][input_prompt_length:r_ind].argmax(dim=-1)
                try:
                    positive_correct = (positive_token_decision == modification_edit_label_[i][input_prompt_length:m_ind]).sum()
                    negative_correct = (negative_token_decision == reject_edit_label_[i][input_prompt_length:r_ind]).sum()
                except:
                    ipdb.set_trace()
                positive_accuracy += (positive_correct / (m_ind - input_prompt_length))
                negative_accuracy += (negative_correct / (r_ind - input_prompt_length))

                positive_loss += torch.nn.functional.cross_entropy(positive_token_pred[i], modification_edit_label_[i], reduction='mean')
                # negative loss
                negative_loss += torch.nn.functional.cross_entropy(negative_token_pred[i], reject_edit_label_[i], reduction='mean')
                
            if positive_accuracy.isnan() or negative_accuracy.isnan():
                import ipdb; ipdb.set_trace()        

        positive_loss = positive_loss / bs
        negative_loss = negative_loss / bs
        positive_accuracy = positive_accuracy / bs
        negative_accuracy = negative_accuracy / bs

        num_reject_pred_both_ratio = num_reject_pred_both_ratio / bs
        num_modification_pred_both_ratio = num_modification_pred_both_ratio / bs

        

        return {
            "positive_acc" : positive_accuracy,
            "negative_acc" : negative_accuracy, 
            "positive_loss": positive_loss,
            "negative_loss": negative_loss,
        }
    def forward_discriminator_eval_f1(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        reject_edit_label=None,
        modification_edit_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prompt_length=None,
        prompt_attn_mask=None, # added by esyoon 2024-01-13-18:36:51
        use_cache=False,
    ):
        loss = None
        positive_loss = 0
        negative_loss = 0

        if self.config.model_type == "llama" or self.config.model_type == "mistral":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        if reject_edit_label is not None and modification_edit_label is not None:
            # positive_token_pred = self.positive_discriminate_head(hidden_states).squeeze(-1)
            # negative_token_pred = self.negative_disciriminate_head(hidden_states).squeeze(-1)
            discriminator_token_pred = self.discriminate_head(hidden_states).squeeze(-1)

        # NOTE: mean score tracking for discriminator 어떻게?

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        
        # input_ids and attention_mask contains that for chosen, reject, and modification
        bs = input_ids.shape[0] // 3
        seq_len = input_ids.shape[1]
        prompt_len = attention_mask.sum(1)

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:bs*2]
        modification_ids = input_ids[bs*2:]

        negative_token_pred = discriminator_token_pred[bs:bs*2]
        positive_token_pred = discriminator_token_pred[bs*2:]
        # Compute pairwise loss. Only backprop on the different tokens before padding

        positive_f1 = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
        negative_f1 = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

        for i in range(bs):
            input_prompt_length = prompt_length[i].item()
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            modification_id = modification_ids[i]
            if not self.custom_template:
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                
                c_ind = (
                    c_inds[self.num_padding_at_beginning].item()
                    if len(c_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            else:
                chosen_id_temp = chosen_id.clone()
                inds_temp = (chosen_id[:prompt_len[i]-1] == self.PAD_ID).nonzero()
                chosen_id_temp[inds_temp] = -1000
                c_inds_temp = (chosen_id_temp == self.PAD_ID).nonzero()
            
                c_ind = (
                    c_inds_temp[self.num_padding_at_beginning].item()
                    if len(c_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )

            # # added by esyoon 2024-01-13-17:43:54
            # # for new template for llama this should be used becuase there are pad token inside the template
            # if len(c_inds) > self.num_padding_at_beginning:
            #     c_ind = seq_len - 1
            # else:
            #     for idx in c_inds:
            #         if chosen_id[idx.item()+1] == self.PAD_ID:
            #             c_ind = idx.item()
            #             break
            #         else:
            #             continue
                
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_id.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                if not self.custom_template:
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = (
                        r_inds[self.num_padding_at_beginning].item()
                        if len(r_inds) > self.num_padding_at_beginning
                        else seq_len - 1
                    )
                else: 
                    # added by esyoon 2024-01-13-17:47:23
                    rejected_id_temp = rejected_id.clone()
                    inds_temp = (rejected_id[:prompt_len[bs+i] - 1] == self.PAD_ID).nonzero()
                    rejected_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                    r_inds_temp = (rejected_id_temp == self.PAD_ID).nonzero()
                
                    r_ind = (
                        r_inds_temp[self.num_padding_at_beginning].item()
                        if len(r_inds_temp) > self.num_padding_at_beginning
                        else seq_len 
                    )

                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            if not self.custom_template:
                m_inds = (modification_id == self.PAD_ID).nonzero()
                m_ind = (
                    m_inds[self.num_padding_at_beginning].item()
                    if len(m_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )
            else:
                modification_id_temp = modification_id.clone()
                inds_temp = (modification_id[:prompt_len[2*bs+i] - 1] == self.PAD_ID).nonzero()
                modification_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                m_inds_temp = (modification_id_temp == self.PAD_ID).nonzero()
            
                m_ind = (
                    m_inds_temp[self.num_padding_at_beginning].item()
                    if len(m_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )


            # token level positive and negative cross entropy loss
            if reject_edit_label is not None and modification_edit_label is not None:
                reject_edit_label_ = reject_edit_label.clone()
                modification_edit_label_ = modification_edit_label.clone()
                modification_false_idx = (modification_edit_label[i] == 0).nonzero().flatten()
                rejection_true_idx = (reject_edit_label[i] == 1).nonzero().flatten()
                rejection_false_idx = (reject_edit_label[i] == 0).nonzero().flatten()
                modification_edit_label_[i, modification_false_idx] = -100
                # reject_edit_label_[i, rejection_false_idx] = 0.5
                reject_edit_label_[i, rejection_false_idx] = -100
                reject_edit_label_[i, rejection_true_idx] = 0
                positive_token_decision = positive_token_pred[i][input_prompt_length:m_ind].argmax(dim=-1)
                negative_token_decision = negative_token_pred[i][input_prompt_length:r_ind].argmax(dim=-1)
                
                positive_TP = [1 if positive_token_decision[j] == modification_edit_label_[i][input_prompt_length:m_ind][j] and modification_edit_label_[i][input_prompt_length:m_ind][j] == 1 else 0 for j in range(len(positive_token_decision))]
                # positive_TN = [1 if positive_token_decision[j] == modification_edit_label_[i][input_prompt_length:m_ind][j] and modification_edit_label_[i][input_prompt_length:m_ind][j] != 1 else 0 for j in range(len(positive_token_decision))]
                positive_FP = [1 if positive_token_decision[j] != modification_edit_label_[i][input_prompt_length:m_ind][j] and modification_edit_label_[i][input_prompt_length:m_ind][j] != 1 else 0 for j in range(len(positive_token_decision))]
                positive_FN = [1 if positive_token_decision[j] != modification_edit_label_[i][input_prompt_length:m_ind][j] and modification_edit_label_[i][input_prompt_length:m_ind][j] == 1 else 0 for j in range(len(positive_token_decision))]
                
                negative_TP = [1 if negative_token_decision[j] == reject_edit_label_[i][input_prompt_length:r_ind][j] and reject_edit_label_[i][input_prompt_length:r_ind][j] == 0 else 0 for j in range(len(negative_token_decision))]
                # negative_TN = [1 if negative_token_decision[j] == reject_edit_label_[i][input_prompt_length:r_ind][j] and reject_edit_label_[i][input_prompt_length:r_ind][j] != 0 else 0 for j in range(len(negative_token_decision))]
                negative_FP = [1 if negative_token_decision[j] != reject_edit_label_[i][input_prompt_length:r_ind][j] and reject_edit_label_[i][input_prompt_length:r_ind][j] != 0 else 0 for j in range(len(negative_token_decision))]
                negative_FN = [1 if negative_token_decision[j] != reject_edit_label_[i][input_prompt_length:r_ind][j] and reject_edit_label_[i][input_prompt_length:r_ind][j] == 0 else 0 for j in range(len(negative_token_decision))]

                positive_f1["TP"] += sum(positive_TP)
                positive_f1["FP"] += sum(positive_FP)
                positive_f1["FN"] += sum(positive_FN)
                negative_f1["TP"] += sum(negative_TP)
                negative_f1["FP"] += sum(negative_FP)
                negative_f1["FN"] += sum(negative_FN)



        return {
            "positive_f1_dict": positive_f1,
            "negative_f1_dict": negative_f1,
        }
        
    def forward_discriminator_eval_ece(self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        reject_edit_label=None,
        modification_edit_label=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prompt_length=None,
        prompt_attn_mask=None, # added by esyoon 2024-01-13-18:36:51
        use_cache=False,
    ):
        loss = None
        positive_loss = 0
        negative_loss = 0

        if self.config.model_type == "llama" or self.config.model_type == "mistral":
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)
        transformer_outputs = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        if reject_edit_label is not None and modification_edit_label is not None:
            # positive_token_pred = self.positive_discriminate_head(hidden_states).squeeze(-1)
            # negative_token_pred = self.negative_disciriminate_head(hidden_states).squeeze(-1)
            discriminator_token_pred = self.discriminate_head(hidden_states).squeeze(-1)

        # NOTE: mean score tracking for discriminator 어떻게?

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        
        # input_ids and attention_mask contains that for chosen, reject, and modification
        bs = input_ids.shape[0] // 3
        seq_len = input_ids.shape[1]
        prompt_len = attention_mask.sum(1)

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:bs*2]
        modification_ids = input_ids[bs*2:]

        negative_token_pred = discriminator_token_pred[bs:bs*2]
        positive_token_pred = discriminator_token_pred[bs*2:]
        # Compute pairwise loss. Only backprop on the different tokens before padding

        positive_accuracy = 0
        negative_accuracy = 0

        num_reject_pred_both_ratio = 0
        num_modification_pred_both_ratio = 0

        positive_head_probs = []
        negative_head_probs = []
        positivie_head_preds = []
        negative_head_preds = []
        positive_labels = []
        negative_labels = []

        for i in range(bs):
            input_prompt_length = prompt_length[i].item()
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            modification_id = modification_ids[i]
            if not self.custom_template:
                c_inds = (chosen_id == self.PAD_ID).nonzero()
                
                c_ind = (
                    c_inds[self.num_padding_at_beginning].item()
                    if len(c_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            else:
                chosen_id_temp = chosen_id.clone()
                inds_temp = (chosen_id[:prompt_len[i]-1] == self.PAD_ID).nonzero()
                chosen_id_temp[inds_temp] = -1000
                c_inds_temp = (chosen_id_temp == self.PAD_ID).nonzero()
            
                c_ind = (
                    c_inds_temp[self.num_padding_at_beginning].item()
                    if len(c_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )

            # # added by esyoon 2024-01-13-17:43:54
            # # for new template for llama this should be used becuase there are pad token inside the template
            # if len(c_inds) > self.num_padding_at_beginning:
            #     c_ind = seq_len - 1
            # else:
            #     for idx in c_inds:
            #         if chosen_id[idx.item()+1] == self.PAD_ID:
            #             c_ind = idx.item()
            #             break
            #         else:
            #             continue
                
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_id.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                if not self.custom_template:
                    r_inds = (rejected_id == self.PAD_ID).nonzero()
                    r_ind = (
                        r_inds[self.num_padding_at_beginning].item()
                        if len(r_inds) > self.num_padding_at_beginning
                        else seq_len - 1
                    )
                else: 
                    # added by esyoon 2024-01-13-17:47:23
                    rejected_id_temp = rejected_id.clone()
                    inds_temp = (rejected_id[:prompt_len[bs+i] - 1] == self.PAD_ID).nonzero()
                    rejected_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                    r_inds_temp = (rejected_id_temp == self.PAD_ID).nonzero()
                
                    r_ind = (
                        r_inds_temp[self.num_padding_at_beginning].item()
                        if len(r_inds_temp) > self.num_padding_at_beginning
                        else seq_len 
                    )

                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            if not self.custom_template:
                m_inds = (modification_id == self.PAD_ID).nonzero()
                m_ind = (
                    m_inds[self.num_padding_at_beginning].item()
                    if len(m_inds) > self.num_padding_at_beginning
                    else seq_len - 1
                )
            else:
                modification_id_temp = modification_id.clone()
                inds_temp = (modification_id[:prompt_len[2*bs+i] - 1] == self.PAD_ID).nonzero()
                modification_id_temp[inds_temp] = -1000 # random value not in the vocab list 
                m_inds_temp = (modification_id_temp == self.PAD_ID).nonzero()
            
                m_ind = (
                    m_inds_temp[self.num_padding_at_beginning].item()
                    if len(m_inds_temp) > self.num_padding_at_beginning
                    else seq_len 
                )

            # token level positive and negative cross entropy loss
            if reject_edit_label is not None and modification_edit_label is not None:
                reject_edit_label_ = reject_edit_label.clone()
                modification_edit_label_ = modification_edit_label.clone()
                modification_false_idx = (modification_edit_label[i] == 0).nonzero().flatten()
                rejection_true_idx = (reject_edit_label[i] == 1).nonzero().flatten()
                rejection_false_idx = (reject_edit_label[i] == 0).nonzero().flatten()
                modification_edit_label_[i, modification_false_idx] = -100
                # reject_edit_label_[i, rejection_false_idx] = 0.5
                reject_edit_label_[i, rejection_false_idx] = -100
                reject_edit_label_[i, rejection_true_idx] = 0

                positivie_token_prob = positive_token_pred[i][input_prompt_length:m_ind].softmax(dim=1)
                negative_token_prob = negative_token_pred[i][input_prompt_length:r_ind].softmax(dim=1)

                positive_token_decision_prob, positive_token_decision = positivie_token_prob.max(dim=-1)
                negative_token_decision_prob, negative_token_decision = negative_token_prob.max(dim=-1)
            
                positive_head_probs+= positive_token_decision_prob.tolist()
                negative_head_probs += negative_token_decision_prob.tolist()
                positivie_head_preds += positive_token_decision.tolist()
                negative_head_preds += negative_token_decision.tolist()
                positive_labels += modification_edit_label[i][input_prompt_length:m_ind].tolist()
                negative_labels += reject_edit_label[i][input_prompt_length:r_ind].tolist()

                
                

        return {
            "positive_head_probs": positive_head_probs,
            "negative_head_probs": negative_head_probs,
            "positive_head_preds": positivie_head_preds,
            "negative_head_preds": negative_head_preds,
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
    
        }    


from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
import torch
from torch import nn
import math 
from transformers import LlamaForCausalLM
def new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        
        #attention_mask.zero_() #added by HSY 1/16/24 remove causal mask
        #zeroed_attention_mask = torch.zeros_like(attention_mask) #added by HSY 1/17/24
        attn_weights = attn_weights + self.non_causal_attention_mask #modified by HSY 1/17/24

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


from transformers.modeling_outputs import BaseModelOutputWithPast
def new_Llama_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)


    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    #added by HSY 1/17/24 ----
    # 4d mask is passed through the layers
    non_causal_attention_mask = ours_prepare_4d_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    for i in range(self.bidir_layer_num):
        layer_num = -(i + 1)
        self.layers[layer_num].self_attn.non_causal_attention_mask = non_causal_attention_mask
    # for idx, decoder_layer in enumerate(self.layers):
    #     decoder_layer.self_attn.non_causal_attention_mask = non_causal_attention_mask
    # -----------------------

    if getattr(self.config, "_flash_attn_2_enabled", False):
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
    

        
    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

from transformers.modeling_attn_mask_utils import AttentionMaskConverter
def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask

def ours_prepare_4d_attention_mask(
    attention_mask: Optional[torch.Tensor],
    input_shape: Union[torch.Size, Tuple, List],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`torch.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`torch.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    attn_mask_converter = AttentionMaskConverter(is_causal=False, sliding_window=sliding_window) # is_causal = True -> False 로 바꿈.

    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None:
        attention_mask = attn_mask_converter.to_4d(
            attention_mask, input_shape[-1], key_value_length, dtype=inputs_embeds.dtype
        )
    else:
        attention_mask = attn_mask_converter.to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )

    return attention_mask