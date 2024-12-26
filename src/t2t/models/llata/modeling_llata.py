from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import random
from transformers import logging
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
)

from .configuration_llata import LlataConfig, LlataForTrajectoryGenerationConfig
from t2t.utils.nn import create_embedding_layer, create_placeholder_tensor
from t2t.utils.metrics import default_sl_loss
from t2t.utils import (
    current_state_id, action_id, next_state_id, 
    pad_id, end_id
    )


logger = logging.get_logger(__name__)


@dataclass
class TrajectoryGenerationModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_loss: Optional[torch.FloatTensor] = None
    action_loss: Optional[torch.FloatTensor] = None
    observation_logits: torch.FloatTensor = None
    action_logits: torch.FloatTensor = None
    # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    traj_loss: Optional[torch.FloatTensor] = None
    token_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None  # token logits


class LlataModel(LlamaModel):
    config_class = LlataConfig

    def __init__(self, config: LlataConfig):
        super().__init__(config)

        self.observation_type = config.observation_type
        self.observation_dim = config.observation_dim
        self.action_type = config.action_type
        self.action_dim = config.action_dim

        self.embed_observation = create_embedding_layer(self.observation_type, self.observation_dim, config.hidden_size)
        self.embed_action = create_embedding_layer(self.action_type, self.action_dim, config.hidden_size)

        # lky waste: load pretrained observation/action embedding if specified
        # if config.do_load_embedding:
            # self.embed_observation.load_state_dict(torch.load(config.pretrain_dir + f'/embed_observation.pt'))
            # self.embed_action.load_state_dict(torch.load(config.pretrain_dir + f'/embed_action.pt'))
        
        self.traj_start_layer = config.traj_start_layer  # lky: add this to set where traj start

        # trajectory layer norm
        if config.traj_ln:
            self.traj_ln = torch.nn.LayerNorm(config.hidden_size)
        else:
            self.traj_ln = None

        # init weights and apply final processing
        self.post_init()

    # TODO: add past key value and cache
    def forward(
        self,
        inst_tokens: torch.LongTensor = None,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        inst_masks: Optional[torch.Tensor] = None,
        observation_masks: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # check inputs and infer sequence length
        assert inst_tokens is not None, "You have to specify inst_tokens"

        batch_sizes = [inst_tokens.shape[0]]
        if observations is not None:
            batch_sizes.append(observations.shape[0])
        else:
            observations = create_placeholder_tensor(
                self.observation_type, batch_sizes[0], 0, self.observation_dim
            ).to(inst_tokens.device)
        if actions is not None:
            batch_sizes.append(actions.shape[0])
        else:
            actions = create_placeholder_tensor(
                self.action_type, batch_sizes[0], 0, self.action_dim
            ).to(inst_tokens.device)

        text_length, obss_length, acts_length = inst_tokens.shape[1], observations.shape[1], actions.shape[1]
        assert len(set(batch_sizes)) == 1, "inst_tokens, observations and actions should have the same batch size"
        assert obss_length == acts_length or obss_length == acts_length + 1, \
            "The length of observations should be equal to the length of actions or one more"

        seq_length = text_length + obss_length + acts_length

        # get instruction and trajectory embeddings
        inst_embeds = self.embed_tokens(inst_tokens)
        obss_embeds = self.embed_observation(observations)
        action_embeds = self.embed_action(actions)

        # alternate observations and actions embeddings
        action_embeds = action_embeds.to(obss_embeds.device)
        traj_embeds = torch.stack(
            [obss_embeds[:, :acts_length], action_embeds], dim=1
        ).permute(0, 2, 1, 3).reshape(batch_sizes[0], -1, self.config.hidden_size)

        # concatenate possible last observation
        traj_embeds = torch.cat([traj_embeds, obss_embeds[:, acts_length:]], dim=1)

        if self.traj_ln is not None:
            # print('test\n\n\n\n\n')
            traj_embeds = self.traj_ln(traj_embeds)

        # concatenate instruction and trajectory embeddings
        traj_embeds = traj_embeds.to(inst_embeds.device)
        inputs_embeds = torch.cat([inst_embeds, traj_embeds], dim=1)

        # prepare attention mask
        if inst_masks is None:
            inst_masks = torch.ones((batch_sizes[0], text_length), dtype=torch.int, device=inputs_embeds.device)
        if observation_masks is None:
            observation_masks = torch.ones((batch_sizes[0], obss_length), dtype=torch.int, device=inputs_embeds.device)
        if action_masks is None:
            action_masks = torch.ones((batch_sizes[0], acts_length), dtype=torch.int, device=inputs_embeds.device)

        traj_mask = torch.stack(
            [observation_masks[:, :acts_length], action_masks], dim=1
        ).permute(0, 2, 1).reshape(batch_sizes[0], -1)

        # print(observation_masks.shape)
        # print(action_masks.shape)
        # print(traj_mask)

        traj_mask = torch.cat([traj_mask, observation_masks[:, acts_length:]], dim=1)

        # print(traj_mask)

        # print('inst\n', inst_masks, '\n')
        # print('traj\n', traj_mask, '\n')
        # print('traj_shape\n', traj_mask.shape, '\n')

        attention_mask = torch.cat([inst_masks, traj_mask], dim=1)

        # print(attention_mask)
        # print(attention_mask.shape)
        # exit(114)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_sizes[0], seq_length), inputs_embeds, 0)

        # prepare positional encoding
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inst_embeds.device)
        position_ids = position_ids.unsqueeze(0)

        # embed positions
        hidden_states = inputs_embeds

        # print(hidden_states.shape)
        # print(attention_mask.shape)
        # print(attention_mask)
        # print(position_ids)
        # exit(114)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # lky: inst, traj seperate
        for decoder_layer in self.layers[:self.traj_start_layer]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # lky: inst, traj together
        for decoder_layer in self.layers[self.traj_start_layer:]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), traj_embeds


class LlataForTrajectoryGeneration(LlataModel):
    def __init__(self, config: LlataForTrajectoryGenerationConfig):
        super().__init__(config)

        self.action_loss_ratio = config.action_loss_ratio
        self.scale_loss_ratio = config.scale_loss_ratio  # add this to control scale_loss

        # trajectory prediction heads
        self.observation_head = torch.nn.Linear(config.hidden_size, config.observation_dim)
        self.action_head = torch.nn.Linear(config.hidden_size, config.action_dim)

        # init weights and apply final processing
        self.post_init()

    def forward(
        self,
        inst_tokens: torch.LongTensor = None,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,  # add this to correct action mask in collator
        inst_masks: Optional[torch.Tensor] = None,
        observation_masks: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        observation_labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        pattern: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs, traj_embeds = super().forward(
            inst_tokens=inst_tokens,
            observations=observations,
            actions=actions,
            inst_masks=inst_masks,
            observation_masks=observation_masks,
            action_masks=action_masks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        batch_size, inst_length = inst_tokens.shape[:2]

        # retrieve hidden states for observations and actions and make prediction
        hidden_states = outputs[0]
        traj_states = hidden_states[:, inst_length-1:]
        obss_logits = self.observation_head(traj_states[:, ::2])
        action_logits = self.action_head(traj_states[:, 1::2])

        # calculate losses (only default supervzised learning loss yet)
        obss_loss = None
        action_loss = None
        loss = None
        if observation_labels is not None:
            logits = obss_logits[:, :observation_labels.shape[1]].contiguous()
            labels = observation_labels.contiguous()
            masks = observation_masks.contiguous()
            # enable model parallelism
            labels = labels.to(logits.device)
            masks = masks.to(logits.device)

            obss_loss = default_sl_loss(self.observation_type, logits, labels, masks)
        if action_labels is not None:
            logits = action_logits[:, :action_labels.shape[1]].contiguous()
            labels = action_labels.contiguous()
            masks = action_masks.contiguous()
            # enable model parallelism
            labels = labels.to(logits.device)
            masks = masks.to(logits.device)

            action_loss = default_sl_loss(self.action_type, logits, labels, masks)
        if obss_loss is not None and action_loss is not None:
            loss = (1 - self.action_loss_ratio) * obss_loss + self.action_loss_ratio * action_loss
        
        # lky: change loss here to control the order of magnitude of state/action embeddings
        # llama_token_scale = 1.1783247  # already pre calcuated
        # current_scale = torch.mean(torch.sum(torch.pow(traj_embeds, 2), dim=-1), dim=-1)  # traj_embed: [batch_size, length, 4096]
        # scale_diff = current_scale - llama_token_scale
        # scale_loss = torch.pow(scale_diff, 2)

        # temp_zero = torch.zeros(traj_embeds.shape).to(traj_embeds.device)
        # temp_loss_func = torch.nn.MSELoss()
        # temp_loss = temp_loss_func(traj_embeds, temp_zero)
        # temp_loss = torch.nn.functional.mse_loss(traj_embeds, temp_zero)

        # temp_loss = torch.mean(torch.mean(torch.mean(torch.pow(traj_embeds, 2), dim=-1), dim=-1))

        # print(traj_embeds[:2, :30])
        # print(temp_loss)
        # loss += temp_loss
        
        # print('\n\n-----\n\n')
        # print(self.scale_loss_ratio)
        # print(scale_loss)
        # print(torch.mean(scale_loss))
        # exit(114514)
        # print(loss)
        # print('\n\n-----\n\n')
        
        # loss += self.scale_loss_ratio * scale_loss
        # loss += self.scale_loss_ratio * torch.mean(scale_loss)

        if not return_dict:
            output = (obss_logits, action_logits) + outputs[1:]
            if all([loss is None, obss_loss is None, action_loss is None]):
                return output
            else:
                losses = []
                if loss is not None:
                    losses.append(loss)
                if obss_loss is not None:
                    losses.append(obss_loss)
                if action_loss is not None:
                    losses.append(action_loss)
                return tuple(losses) + output

        return TrajectoryGenerationModelOutput(
            loss=loss,
            observation_loss=obss_loss,
            action_loss=action_loss,
            observation_logits=obss_logits,
            action_logits=action_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class Llata2Model(LlamaModel):
    config_class = LlataConfig

    def __init__(self, config: LlataConfig):
        super().__init__(config)

        self.observation_type = config.observation_type
        self.observation_dim = config.observation_dim
        self.action_type = config.action_type
        self.action_dim = config.action_dim

        self.embed_observation = create_embedding_layer(self.observation_type, self.observation_dim, config.hidden_size)
        self.embed_action = create_embedding_layer(self.action_type, self.action_dim, config.hidden_size)

        # trajectory layer norm
        if config.traj_ln:
            self.traj_ln = torch.nn.LayerNorm(config.hidden_size)
        else:
            self.traj_ln = None

        # init weights and apply final processing
        self.post_init()

    def forward(
        self,
        inst_tokens: torch.LongTensor = None,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        inst_masks: Optional[torch.Tensor] = None,
        observation_masks: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        pattern: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        states_sample: Optional[list] = None
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # check inputs and infer sequence length
        assert inst_tokens is not None, "You have to specify inst_tokens"

        batch_sizes = [inst_tokens.shape[0]]
        if observations is not None:
            batch_sizes.append(observations.shape[0])
        else:
            observations = create_placeholder_tensor(
                self.observation_type, batch_sizes[0], 0, self.observation_dim
            ).to(inst_tokens.device)
        if actions is not None:
            batch_sizes.append(actions.shape[0])
        else:
            actions = create_placeholder_tensor(
                self.action_type, batch_sizes[0], 0, self.action_dim
            ).to(inst_tokens.device)

        text_length, obss_length, acts_length = inst_tokens.shape[1], observations.shape[1], actions.shape[1]
        assert len(set(batch_sizes)) == 1, "inst_tokens, observations and actions should have the same batch size"
        assert obss_length == acts_length or obss_length == acts_length + 1, \
            "The length of observations should be equal to the length of actions or one more"
        
        # check attention mask
        if inst_masks is None:
            raise NotImplementedError('should give instruction mask')
        if observation_masks is None:
            raise NotImplementedError('should give observation mask')
        if action_masks is None:
            raise NotImplementedError('should give action mask')

        seq_length = text_length  # change for new prompt
        traj_length = obss_length + acts_length  
        
        # get instruction and trajectory embeddings, wait for traj embeddings to replace part of it
        inst_embeds = self.embed_tokens(inst_tokens)
        obss_embeds = self.embed_observation(observations)
        action_embeds = self.embed_action(actions)

        # alternate observations and actions embeddings
        action_embeds = action_embeds.to(obss_embeds.device)
        traj_embeds = torch.stack(
            [obss_embeds[:, :acts_length], action_embeds], dim=1
        ).permute(0, 2, 1, 3).reshape(batch_sizes[0], -1, self.config.hidden_size)

        # concatenate possible last observation
        traj_embeds = torch.cat([traj_embeds, obss_embeds[:, acts_length:]], dim=1)

        if self.traj_ln is not None:
            traj_embeds = self.traj_ln(traj_embeds)
        
        traj_embeds = traj_embeds.to(inst_embeds.device)

        # prepare embedding
        for i in range(batch_sizes[0]):
            if pattern[i] == 0 or pattern[i] == 3:
                # inject trajectory embeddings to instruction embeddings
                traj_start_idx = torch.where(inst_tokens[i]==current_state_id)[0][0]
                true_traj_length = torch.sum(observation_masks[i]) + torch.sum(action_masks[i])
                inst_embeds[i, traj_start_idx:traj_start_idx+true_traj_length] = traj_embeds[i, :true_traj_length]
            elif pattern[i] == 1:
                current_state_idx = torch.where(inst_tokens[i]==current_state_id)[0][0]
                inst_embeds[i, current_state_idx] = obss_embeds[i, states_sample[i]]

                action_idx = torch.where(inst_tokens[i]==action_id)[0][0]
                inst_embeds[i, action_idx] = action_embeds[i, states_sample[i]]

                next_state_idx = torch.where(inst_tokens[i]==next_state_id)[0][0]
                inst_embeds[i, next_state_idx] = obss_embeds[i, states_sample[i]+1]
            elif pattern[i] == 2:
                current_state_idx = torch.where(inst_tokens[i]==current_state_id)[0][0]
                inst_embeds[i, current_state_idx] = obss_embeds[i, 0]

                next_state_idx = torch.where(inst_tokens[i]==next_state_id)[0][0]
                try:
                    last_observation_idx = torch.where(observation_masks[i]==0)[0][0] - 1
                except:
                    last_observation_idx = obss_embeds.shape[1] - 1  # can't find means last state is the last state of the whole traj
                inst_embeds[i, next_state_idx] = obss_embeds[i, last_observation_idx]
            elif pattern[i] == -1:
                pass  # input instruction direcyly
            else:
                raise NotImplementedError(f'unsupported pattern :{pattern[i]}')
        
        inputs_embeds = inst_embeds

        # prepare attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            inst_masks, (batch_sizes[0], seq_length), inputs_embeds, 0)

        # prepare positional encoding
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inst_embeds.device)
        position_ids = position_ids.unsqueeze(0)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Llata2ForTrajectoryGeneration(LlamaPreTrainedModel):
    def __init__(self, config: LlataForTrajectoryGenerationConfig):
        super().__init__(config)
        self.model = Llata2Model(config=config)

        self.action_loss_ratio = config.action_loss_ratio
        self.scale_loss_ratio = config.scale_loss_ratio  # add this to control scale_loss

        # trajectory prediction heads
        self.observation_head = torch.nn.Linear(config.hidden_size, config.observation_dim)
        self.action_head = torch.nn.Linear(config.hidden_size, config.action_dim)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # init weights and apply final processing
        self.post_init()

    def forward(
        self,
        inst_tokens: torch.LongTensor = None,
        observations: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,  # add this to correct action mask in collator
        inst_masks: Optional[torch.Tensor] = None,
        observation_masks: Optional[torch.Tensor] = None,
        action_masks: Optional[torch.Tensor] = None,
        observation_labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
        pattern: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        inst_tokens_b0: Optional[torch.Tensor] = None,  # add this to avoid a batch don't contain only pattern 0
        inst_masks_b0: Optional[torch.Tensor] = None,  # add this to avoid a batch don't contain only pattern 0
        inst_tokens_b3: Optional[torch.Tensor] = None,  # add this to avoid a batch don't contain only pattern 3
        inst_masks_b3: Optional[torch.Tensor] = None,  # add this to avoid a batch don't contain only pattern 3
        pattern_num: int = 0,
        available_pattern_list: list = [0],
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = inst_tokens.shape[0]
        obss_length, acts_length = observations.shape[1], actions.shape[1]
        traj_length = obss_length + acts_length
        
        # sample a s-a-s pair for each traj
        states_sample = [random.randint(0, torch.sum(observation_masks[i])-2) if pattern[i] == 1 else 0 for i in range(batch_size)]

        outputs = self.model(
            inst_tokens=inst_tokens,
            observations=observations,
            actions=actions,
            inst_masks=inst_masks,
            observation_masks=observation_masks,
            action_masks=action_masks,
            pattern=pattern,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            states_sample=states_sample
        )

        # retrieve hidden states for observations and actions and make prediction
        hidden_states = outputs[0]

        obss_logits_all = self.observation_head(hidden_states)
        action_logits_all = self.action_head(hidden_states)
        token_logits = self.lm_head(hidden_states)
        
        # calculate losses
        # compute traj_loss for each data point 
        # and get token_mask to avoid compute traj loss when compute token loss 
        # token_masks = inst_masks
        # traj_loss_list, obss_loss_list, action_loss_list, token_loss_list = [], [], [], []
        obss_loss_list, action_loss_list, token_loss_list = [], [], []
        for i in range(batch_size):
            if pattern[i] == 0:
                traj_start_idx = torch.where(inst_tokens[i, :]==current_state_id)[0][0]
                # true_traj_length = torch.sum(observation_masks[i])+torch.sum(action_masks[i])
                # token_masks[i, traj_start_idx-1:traj_start_idx-1+true_traj_length] = 0

                obss_logits = obss_logits_all[i, traj_start_idx-1:traj_start_idx-1+traj_length][::2]
                action_logits = action_logits_all[i, traj_start_idx-1:traj_start_idx-1+traj_length][1::2]

                if observation_labels is not None:
                    logits_o = obss_logits[:observation_labels.shape[1]].unsqueeze(0).contiguous()
                    labels_o = observation_labels[i].unsqueeze(0).contiguous()
                    masks_o = observation_masks[i].unsqueeze(0).contiguous()
                else:
                    raise NotImplementedError('observation_labels should not be None')

                if action_labels is not None:
                    logits_a = action_logits[:action_labels.shape[1]].unsqueeze(0).contiguous()
                    labels_a = action_labels[i].unsqueeze(0).contiguous()
                    masks_a = action_masks[i].unsqueeze(0).contiguous()
                else:
                    raise NotImplementedError('action_labels should not be None')
                    
            elif pattern[i] == 1:
                # current_state_idx = torch.where(inst_tokens[i, :]==current_state_id)[0][0]
                # current_obss_logits = obss_logits_all[i, current_state_idx-1]
                next_state_idx = torch.where(inst_tokens[i, :]==next_state_id)[0][0]
                # next_obss_logits = obss_logits_all[i, next_state_idx-1]
                # logits_o = torch.stack([current_obss_logits, next_obss_logits], dim=0).contiguous()
                logits_o = obss_logits_all[i, next_state_idx-1].unsqueeze(0).contiguous()
                # labels_o = torch.stack(
                #     [observation_labels[i, states_sample[i]], observation_labels[i, states_sample[i]+1]], dim=0
                #     ).contiguous()
                labels_o = observation_labels[i, states_sample[i]+1].unsqueeze(0).contiguous()
                masks_o = torch.ones(labels_o.shape, device=labels_o.device).contiguous()

                # action_idx = torch.where(inst_tokens[i, :]==action_id)[0][0]
                # logits_a = action_logits_all[i, action_idx-1].unsqueeze(0).contiguous()
                # labels_a = action_labels[i, states_sample[i]].unsqueeze(0).contiguous()
                # masks_a = torch.ones(labels_a.shape, device=logits_a.device).contiguous()
            
            elif pattern[i] == 2:
                # current_state_idx = torch.where(inst_tokens[i, :]==current_state_id)[0][0]
                # current_obss_logits = obss_logits_all[i, current_state_idx-1]
                next_state_idx = torch.where(inst_tokens[i, :]==next_state_id)[0][0]
                # next_obss_logits = obss_logits_all[i, next_state_idx-1]
                # logits_o = torch.stack([current_obss_logits, next_obss_logits], dim=0).contiguous()
                logits_o = obss_logits_all[i, next_state_idx-1].unsqueeze(0).contiguous()
                try:
                    last_observation_idx = torch.where(observation_masks[i]==0)[0][0] - 1
                except:
                    last_observation_idx = observation_labels.shape[1] - 1  # can't find means last state is the last state of the whole traj
                # labels_o = torch.stack(
                #     [observation_labels[i, 0], observation_labels[i, last_observation_idx]], dim=0
                #     ).contiguous()
                labels_o = observation_labels[i, last_observation_idx].unsqueeze(0).contiguous()
                masks_o = torch.ones(labels_o.shape, device=logits_o.device).contiguous()
            
            elif pattern[i] == 3:
                last_obss_idx = torch.where(inst_tokens[i, :]==current_state_id)[0][-1]
                last_action_idx = torch.where(inst_tokens[i, :]==action_id)[0][-1]
                instr_start_idx = max(last_obss_idx, last_action_idx) + 4

                shift_logits = token_logits[i, instr_start_idx:-2, :].unsqueeze(0).contiguous()
                shift_labels = inst_tokens[i, 1+instr_start_idx:-1].unsqueeze(0).contiguous()
                token_masks = inst_masks[i, 1+instr_start_idx:-1].clone()

                last_one_idx = torch.where(token_masks==1)[0][-1]  # add this to avoid predict '.' and '</s>'
                token_masks[last_one_idx] = 0
                # token_masks[last_one_idx-1] = 0  # remove this to predict '.'
                token_masks.unsqueeze(0).contiguous()
            
            obss_loss = None
            action_loss = None
            traj_loss = None
            token_loss = None
            
            if pattern[i] != 3:
                # enable model parallelism
                labels_o = labels_o.to(logits_o.device)
                masks_o = masks_o.to(logits_o.device)
                obss_loss = default_sl_loss(self.model.observation_type, logits_o, labels_o, masks_o)
                obss_loss_list.append((1 - self.action_loss_ratio) * obss_loss)
                # obss_loss_list.append((1 - self.action_loss_ratio) * ((obss_logits_all[i]**2).sum()))  # for debug
            # else:
            #     obss_loss = 0
            # obss_loss_list.append((1 - self.action_loss_ratio) * obss_loss)

            if pattern[i] == 0:
                labels_a = labels_a.to(logits_a.device)
                masks_a = masks_a.to(logits_a.device)
                action_loss = default_sl_loss(self.model.action_type, logits_a, labels_a, masks_a)
                action_loss_list.append(self.action_loss_ratio * action_loss)
            # else:
            #     action_loss = 0
            # action_loss_list.append(self.action_loss_ratio * action_loss)

            # if obss_loss is not None and action_loss is not None:
            #     traj_loss = (1 - self.action_loss_ratio) * obss_loss + self.action_loss_ratio * action_loss
            #     traj_loss_list.append(traj_loss)
            # else:
            #     raise NotImplementedError('should compute obss_loss and action_loss')
            
            if pattern[i] == 3:
                shift_labels = shift_labels.to(shift_logits.device)
                token_masks = token_masks.to(shift_logits.device)
                token_loss = default_sl_loss("discrete", shift_logits, shift_labels, token_masks)
                token_loss_list.append(token_loss)
            # else:
            #     token_loss = 0
            # token_loss_list.append(token_loss)

        # compute traj_loss
        # traj_batch_loss = torch.mean(torch.tensor(traj_loss_list, requires_grad=True))
        # obss_batch_loss = torch.mean(torch.tensor(obss_loss_list, requires_grad=True))
        obss_batch_loss = sum(obss_loss_list) / len(obss_loss_list) if len(obss_loss_list) > 0 else torch.tensor(0.0, requires_grad=True)
        # action_batch_loss = torch.mean(torch.tensor(action_loss_list, requires_grad=True))
        action_batch_loss = sum(action_loss_list) / len(action_loss_list) if len(action_loss_list) > 0 else torch.tensor(0.0, requires_grad=True)
        traj_batch_loss = obss_batch_loss + action_batch_loss
        # token_batch_loss = torch.mean(torch.tensor(token_loss_list, requires_grad=True))
        token_batch_loss = sum(token_loss_list) / len(token_loss_list) if len(token_loss_list) > 0 else torch.tensor(0.0, requires_grad=True, device=traj_batch_loss.device)
        
        # token_logits = self.lm_head(hidden_states)  # mask is edited above to avoid compute traj loss when compute token loss 
        # shift_logits = token_logits[..., :-1, :].contiguous()
        # shift_labels = inst_tokens[..., 1:].contiguous()
        # token_masks = token_masks[..., 1:].contiguous()
        
        # shift_labels = shift_labels.to(shift_logits.device)
        # token_masks = token_masks.to(shift_logits.device)

        # token_loss = default_sl_loss("discrete", shift_logits, shift_labels, token_masks)

        # loss = traj_batch_loss + token_loss
        loss = traj_batch_loss + token_batch_loss
        # loss = obss_batch_loss

        if not return_dict:
            raise NotImplementedError('shoule set return_dict to True')

        return TrajectoryGenerationModelOutput(
            loss=loss,
            logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            traj_loss=traj_batch_loss,
            token_loss=token_batch_loss,
            observation_loss=obss_batch_loss,
            action_loss=action_batch_loss
        )
