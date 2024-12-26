from itertools import chain
from collections.abc import Mapping
from typing import Any, Dict, List
from dataclasses import dataclass

import torch
import torch.utils.data
import numpy as np
from transformers import (
    DataCollatorWithPadding,
)

from t2t.utils import (
    pad_id,
    action_id,
    current_state_id
)


@dataclass
class DataCollatorForSingleInstructedTrajectory(DataCollatorWithPadding):
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert self.return_tensors == "pt", "Should be torch tensors"

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}
        batch_size = len(features)

        assert "observations" in first and "actions" in first, "Observations and actions should be provided"

        # process pattern
        batch['pattern'] = torch.tensor(np.stack([f['pattern'] for f in features]))

        # Tokenizing or padding language instruction
        if "inst_tokens" in first:
            # tokenized data
            if (not (batch['pattern'] == 0).any()) and (0 in first['available_pattern_list']):
                batch['pattern'][0] = 0  # if pattern 0 available, at least one 0 in each batch
            if 0 not in first['available_pattern_list']:
                if (1 in first['available_pattern_list']) or (2 in first['available_pattern_list']):
                    if (not (batch['pattern'] == 1).any()) and (not (batch['pattern'] == 2).any()):
                        batch['pattern'][0] = 1  # 0 not available, 1 or 2 is available -> at least one 1 or 2 in each batch
            if (not (batch['pattern'] == 3).any()) and (3 in first['available_pattern_list']):
                # if pattern 3 available, at least one 3 in each batch
                if 0 in first['available_pattern_list']:
                    # avoid remove the only 0
                    if batch['pattern'][-1] != 0:
                        batch['pattern'][-1] = 3
                    else:
                        batch['pattern'][0] = 3
                else:
                    batch['pattern'][0] = 3
            
            '''
            for f in features:
                if f['pattern']!=1:
                    samples = np.random.randint(0, f["inst_tokens"].shape[0], size=batch_size)
                    break
            '''
            samples = np.random.randint(0, first["inst_tokens_b0"].shape[0], size=batch_size)
            
            temp_token_list = []
            temp_mask_list = []
            for idx, f in enumerate(features):
                if batch['pattern'][idx] == 0:
                    temp_token_list.append(f['inst_tokens_b0'][samples[idx]])
                    temp_mask_list.append(f['inst_masks_b0'][samples[idx]])
                elif batch['pattern'][idx] == 1:
                    temp_token_list.append(f['inst_tokens'])
                    temp_mask_list.append(f['inst_masks'])
                elif batch['pattern'][idx] == 2:
                    temp_token_list.append(f['inst_tokens'][samples[idx]])
                    temp_mask_list.append(f['inst_masks'][samples[idx]])
                elif batch['pattern'][idx] == 3:
                    temp_token_list.append(f['inst_tokens_b3'][samples[idx]])
                    temp_mask_list.append(f['inst_masks_b3'][samples[idx]])
            batch["inst_tokens"] = torch.tensor(np.stack(temp_token_list))
            batch["inst_masks"] = torch.tensor(np.stack(temp_mask_list))
            
            '''
            if not (batch['pattern'] == 2).all():
                batch["inst_tokens"] = torch.tensor(  # move sample from below to here to keep data in same shape
                    np.stack([f["inst_tokens"][samples[idx]] if f['pattern']!=1 else f["inst_tokens"] for idx, f in enumerate(features)])
                    )
                batch["inst_masks"] = torch.tensor(
                    np.stack([f["inst_masks"][samples[idx]] if f['pattern']!=1 else f["inst_masks"] for idx, f in enumerate(features)])
                    )
            else:  # add this to avoid a batch contain only pattern 2; if all 2, change the last data to 1
                batch["inst_tokens"] = torch.tensor(
                    np.stack([f["inst_tokens"][samples[idx]] if idx != batch_size -1 else f["inst_tokens_b"] for idx, f in enumerate(features)])
                    )
                batch["inst_masks"] = torch.tensor(
                    np.stack([f["inst_masks"][samples[idx]] if idx != batch_size -1 else f["inst_masks_b"] for idx, f in enumerate(features)])
                    )
                batch['pattern'][-1] = 1
            '''

        elif "instructions" in first:
            # tokenize dynamically
            if hasattr(self.tokenizer, "add_eos_token"):
                add_eos_token_flag = getattr(self.tokenizer, "add_eos_token")
                setattr(self.tokenizer, "add_eos_token", True)
            instructions = [f["instruction"] for f in features]
            if not isinstance(first["instructions"], str):
                instructions = list(chain(*instructions))

            tokenized_inst = self.tokenizer(
                instructions,
                add_special_tokens=True,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=self.return_tensors,
            )
            if hasattr(self.tokenizer, "add_eos_token"):
                setattr(self.tokenizer, "add_eos_token", add_eos_token_flag)

            batch["inst_tokens"] = tokenized_inst["input_ids"]
            batch["inst_masks"] = tokenized_inst["attention_mask"]
            if batch["inst_tokens"].shape[0] != batch_size:
                batch["inst_tokens"] = batch["inst_tokens"].reshape(
                    batch_size, batch["inst_tokens"].shape[0] // batch_size, -1)
                batch["inst_masks"] = batch["inst_masks"].reshape(
                    batch_size, batch["inst_masks"].shape[0] // batch_size, -1)
        else:
            raise ValueError("Either inst_tokens or instructions should be provided")

        # sample from diverse instructions
        if len(batch["inst_tokens"].shape) > 2:
            sample_indices = torch.randint(0, batch["inst_tokens"].shape[1], (batch_size,))
            batch["inst_tokens"] = batch["inst_tokens"][torch.arange(batch_size), sample_indices]
            batch["inst_masks"] = batch["inst_masks"][torch.arange(batch_size), sample_indices]

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("inst_tokens", "inst_masks", "label", "label_ids", "instructions", 'pattern') \
                and v is not None and not isinstance(v, str):
                if isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        # Handling of masks
        if "observation_masks" not in first:
            if "masks" in first:
                batch["observation_masks"] = batch["masks"].clone()
            else:
                batch["observation_masks"] = torch.ones(*batch["observations"].shape[:2])
        if "action_masks" not in first:
            if "masks" in first:
                batch["action_masks"] = batch["masks"].clone()
            else:
                batch["action_masks"] = torch.ones(batch["actions"].shape[:2])

        # Handling of labels
        if "observation_labels" not in first:
            batch["observation_labels"] = batch["observations"].clone()
        if "action_labels" not in first:
            batch["action_labels"] = batch["actions"].clone()
        
        # prepare inst mask and pad for pattern 0 and 3
        obss_length, acts_length = batch['observations'].shape[1], batch['actions'].shape[1]
        traj_length = obss_length + acts_length
        traj_mask = torch.stack(
            [batch["observation_masks"][:, :acts_length], batch["action_masks"]], dim=1
        ).permute(0, 2, 1).reshape(batch_size, -1)
        traj_mask = torch.cat([traj_mask, batch["observation_masks"][:, acts_length:]], dim=1)

        for i in range(batch_size):
            if batch['pattern'][i] == 0 or batch['pattern'][i] == 3:
                traj_start_idx = torch.where(batch["inst_tokens"][i, :]==current_state_id)[0][0]  # find first traj_mark
                traj_end_idx = torch.where(batch["inst_tokens"][i, :]==action_id)[0][-1]  # find last traj_mark
                batch["inst_masks"][i, traj_start_idx:traj_start_idx+traj_length] = traj_mask[i]

                try:
                    zero_start_idx = torch.where(batch["inst_masks"][i, :]==0)[0][0]
                    if zero_start_idx < traj_start_idx+traj_length:
                        batch["inst_masks"][i, zero_start_idx:zero_start_idx+batch["inst_masks"][i, traj_end_idx+1:].shape[0]] = batch["inst_masks"][i, traj_end_idx+1:].clone()
                        batch["inst_tokens"][i, zero_start_idx:zero_start_idx+batch["inst_masks"][i, traj_end_idx+1:].shape[0]] = batch["inst_tokens"][i, traj_end_idx+1:].clone()
                        batch["inst_masks"][i, zero_start_idx+batch["inst_masks"][i, traj_end_idx+1:].shape[0]:] = 0
                        batch["inst_tokens"][i, zero_start_idx+batch["inst_masks"][i, traj_end_idx+1:].shape[0]:] = pad_id
                except:
                    pass  # if can't find proper zero start, it means traj_mask is all 1 and shift is not needed

        return batch

