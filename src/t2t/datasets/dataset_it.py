from abc import ABC
from typing import Dict

import numpy as np
import torch.utils.data


class InstructedTrajectoryDataset(torch.utils.data.Dataset, ABC):
    """Dataset of trajectories following specified instruction(s)"""

    def __init__(
        self,
        observations,
        actions,
        next_observations,
        rewards,
        terminals,
        masks,
        instructions,
        inst_tokens=None,
        inst_masks=None,
        observation_type=None,
        observation_dim=-1,
        action_type=None,
        action_dim=-1,
        available_pattern_list=[0],
    ) -> None:
        # assertions
        assert len(observations) >= 3, "Data shape should be [dataset_size, traj_len, ...]"
        assert all([inst_tokens is not None, inst_masks is not None]) or all([inst_tokens is None, inst_masks is None]), \
            "Both inst_tokens and inst_masks should be provided or neither should be provided"

        super().__init__()

        # data
        self.observations = observations
        self.actions = actions
        self.next_observations = next_observations
        self.rewards = rewards
        self.terminals = terminals
        self.masks = masks
        self.instructions = instructions
        self.inst_tokens = inst_tokens
        self.inst_masks = inst_masks
        self.pattern_num = len(available_pattern_list)  # num of available patterns
        self.available_pattern_list = available_pattern_list
        self.possible_pattern = 4
        assert max(self.available_pattern_list) < self.possible_pattern, f'pattern bigger than {self.possible_pattern-1}, available patterns: {self.available_pattern_list}'

        # attrs
        self.observation_type = observation_type
        self.observation_dim = observation_dim
        self.action_type = action_type
        self.action_dim = action_dim

        if not self.observation_type:
            dtype = self.observations.dtype
            if np.issubdtype(dtype, np.floating):
                self.observation_type = "continuous"
            elif np.issubdtype(dtype, np.integer):
                self.observation_type = "discrete"
            else:
                raise ValueError(f"Unsupported dtype {dtype} for observations type inference.")
        if not self.action_type:
            dtype = self.actions.dtype
            if np.issubdtype(dtype, np.floating):
                self.action_type = "continuous"
            elif np.issubdtype(dtype, np.integer):
                self.action_type = "discrete"
            else:
                raise ValueError(f"Unsupported dtype {dtype} for actions type inference.")

        if self.observation_dim < 0:
            assert self.observation_type == "continuous", \
                "Only support dimension inference for continuous observations"
            self.observation_dim = self.observations.shape[-1]
        if self.action_dim < 0:
            assert self.action_type == "continuous", \
                "Only support dimension inference for continuous actions"
            self.action_dim = self.actions.shape[-1]
        self.max_traj_len = self.observations.shape[1]

        for i in range(len(self.instructions)):  # add this to remove '.' in instruction
            for j in range(len(self.instructions[i])):
                if self.instructions[i][j][-1] == '.':
                    self.instructions[i][j] = self.instructions[i][j][:-1]
                if '.' in self.instructions[i][j][-1]:
                    raise NotImplementedError(f"data . error, inst: {self.instructions[i][j]}, position:{(i, j)}")

    def __len__(self):
        return self.observations.shape[0] * self.pattern_num

    def __getitem__(self, index) -> Dict:
        raise NotImplementedError


class SingleInstructedTrajectoryDataset(InstructedTrajectoryDataset):
    """
    Dataset of trajectories following single specified instruction
    (each instruction corresponds to one full trajectory)
    """

    def __init__(
        self,
        observations,
        actions,
        next_observations,
        rewards,
        terminals,
        masks,
        instructions,
        inst_tokens=None,
        inst_masks=None,
        observation_type=None,
        observation_dim=-1,
        action_type=None,
        action_dim=-1,
        include_last_observation=True,
        available_pattern_list=[0],
    ) -> None:
        if include_last_observation:
            # extend the trajectory length of all attrs. to include the last observation
            new_traj_len = observations.shape[1] + 1
            # extend observations
            new_obss = np.zeros(
                (observations.shape[0], new_traj_len, *observations.shape[2:]),
                dtype=observations.dtype)
            new_obss[:, 0] = observations[:, 0]
            new_obss[:, 1:] = next_observations[:]
            observations = new_obss

            # extend masks (and save different masks for the observations)
            new_masks = np.zeros(
                (masks.shape[0], new_traj_len, *masks.shape[2:]),
                dtype=masks.dtype)
            obss_masks = new_masks.copy()
            obss_masks[:, 0] = masks[:, 0]
            obss_masks[:, 1:] = masks[:]
            new_masks[:, :-1] = masks[:]
            self.observation_masks = obss_masks # special masks for observation
            masks = new_masks # general masks for other attrs.

            # extend other attrs. (except instructions and inst_tokens)
            new_actions = np.zeros(
                (actions.shape[0], new_traj_len, *actions.shape[2:]),
                dtype=actions.dtype)
            new_actions[:, :-1] = actions[:]
            actions = new_actions

            new_next_obss = np.zeros(
                (next_observations.shape[0], new_traj_len, *next_observations.shape[2:]),
                dtype=next_observations.dtype)
            new_next_obss[:, :-1] = next_observations[:]
            next_observations = new_next_obss

            new_rewards = np.zeros(
                (rewards.shape[0], new_traj_len, *rewards.shape[2:]),
                dtype=rewards.dtype)
            new_rewards[:, :-1] = rewards[:]
            rewards = new_rewards

            new_terminals = np.zeros(
                (terminals.shape[0], new_traj_len, *terminals.shape[2:]),
                dtype=terminals.dtype)
            new_terminals[:, :-1] = terminals[:]
            terminals = new_terminals

        super().__init__(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminals=terminals,
            masks=masks,
            instructions=instructions,
            inst_tokens=inst_tokens,
            inst_masks=inst_masks,
            observation_type=observation_type,
            observation_dim=observation_dim,
            action_type=action_type,
            action_dim=action_dim,
            available_pattern_list=available_pattern_list,
        )

    def __getitem__(self, index) -> Dict:
        # pattern = index // (self.__len__() // self.pattern_num)
        pattern = self.available_pattern_list[index // (self.__len__() // self.pattern_num)]
        index = index % (self.__len__() // self.pattern_num)

        datapoint = {
            "observations": self.observations[index],
            "actions": self.actions[index],
            "next_observations": self.next_observations[index],
            "rewards": self.rewards[index],
            "terminals": self.terminals[index],
            "masks": self.masks[index],
            "instructions": self.instructions[index],
        }

        if (self.inst_tokens is not None) or (self.inst_tokens_lky is not None):
            # datapoint["inst_tokens"] = self.inst_tokens[index]
            # datapoint["inst_masks"] = self.inst_masks[index]
            if pattern == 1:
                datapoint["inst_tokens"] = self.inst_tokens_lky[pattern]
                datapoint["inst_masks"] = self.inst_masks_lky[pattern]
            else:
                if self.env_nums == None:
                    datapoint["inst_tokens"] = self.inst_tokens_lky[pattern][self.goal_nums[index]]
                    datapoint["inst_masks"] = self.inst_masks_lky[pattern][self.goal_nums[index]]
                else:  # lky: for meta data
                    datapoint["inst_tokens"] = self.inst_tokens_lky[pattern][self.env_nums[index]]
                    datapoint["inst_masks"] = self.inst_masks_lky[pattern][self.env_nums[index]]
            datapoint['pattern'] = pattern

            # add backup to avoid a batch don't contain only pattern 0 or don't contain pattern 3
            if self.env_nums == None:
                datapoint["inst_tokens_b0"] = self.inst_tokens_lky[0][self.goal_nums[index]]
                datapoint["inst_masks_b0"] = self.inst_masks_lky[0][self.goal_nums[index]]
            else:
                datapoint["inst_tokens_b0"] = self.inst_tokens_lky[0][self.env_nums[index]]
                datapoint["inst_masks_b0"] = self.inst_masks_lky[0][self.env_nums[index]]
            # if self.pattern_num >= 4:
            if 3 in self.available_pattern_list:
                if self.env_nums == None:
                    datapoint["inst_tokens_b3"] = self.inst_tokens_lky[3][self.goal_nums[index]]
                    datapoint["inst_masks_b3"] = self.inst_masks_lky[3][self.goal_nums[index]]
                else:
                    datapoint["inst_tokens_b3"] = self.inst_tokens_lky[3][self.env_nums[index]]
                    datapoint["inst_masks_b3"] = self.inst_masks_lky[3][self.env_nums[index]]
                
        if hasattr(self, "observation_masks"):
            datapoint["observation_masks"] = self.observation_masks[index]

        return datapoint


class ClevrDataset(SingleInstructedTrajectoryDataset):
    def __init__(
        self,
        observations,
        actions,
        next_observations,
        rewards,
        terminals,
        masks,
        instructions,
        goals,
        inst_tokens=None,
        inst_masks=None,
        observation_type=None,
        observation_dim=-1,
        action_type=None,
        action_dim=-1,
        include_last_observation=True,
        num_objects=None,
        available_pattern_list=[0],
        env_names=None,
    ) -> None:
        super().__init__(
            observations,
            actions,
            next_observations,
            rewards,
            terminals,
            masks,
            instructions,
            inst_tokens,
            inst_masks,
            observation_type,
            observation_dim,
            action_type,
            action_dim,
            include_last_observation,
            available_pattern_list,
        )

        self.goals = goals.astype(int)
        self.goals_list = self.goals[:, 0].tolist()  # lky: change tokenization

        if num_objects is None:
            # each goal contains flags for each object and a flag for the direction
            num_objects = self.goals.shape[-1] - 1
        self.num_objects = num_objects

        # lky: change tokenization
        self.all_goals_list = []
        for i in range(4):
            self.all_goals_list.extend([
                [1, 2, 0, 0, 0, i], [1, 0, 2, 0, 0, i], [1, 0, 0, 2, 0, i], [1, 0, 0, 0, 2, i], [0, 1, 2, 0, 0, i], 
                [0, 1, 0, 2, 0, i], [0, 1, 0, 0, 2, i], [0, 0, 1, 2, 0, i], [0, 0, 1, 0, 2, i], [0, 0, 0, 1, 2, i], 
                [2, 1, 0, 0, 0, i], [2, 0, 1, 0, 0, i], [2, 0, 0, 1, 0, i], [2, 0, 0, 0, 1, i], [0, 2, 1, 0, 0, i], 
                [0, 2, 0, 1, 0, i], [0, 2, 0, 0, 1, i], [0, 0, 2, 1, 0, i], [0, 0, 2, 0, 1, i], [0, 0, 0, 2, 1, i]
                ])
        if env_names is None:
            self.goal_nums = [self.all_goals_list.index(i) for i in self.goals_list]

        # lky: for meta data
        if env_names is not None:
            self.env_names = env_names
            self.all_env_names_list = list(set(name for name in self.env_names))
            self.env_nums = [self.all_env_names_list.index(i) for i in self.env_names]
        else:
            self.env_names = None
            self.all_env_names_list = None
            self.env_nums = None

        self.inst_tokens_lky = [None for _ in range(self.possible_pattern)]
        self.inst_masks_lky = [None for _ in range(self.possible_pattern)]

    def __getitem__(self, index) -> Dict:
        datapoint = super().__getitem__(index)
        # the goal of one trajectory is the same for all timesteps
        datapoint["goals"] = self.goals[index % (self.__len__() // self.pattern_num)][0]
        datapoint['pattern_num'] = self.pattern_num
        datapoint['available_pattern_list'] = self.available_pattern_list
        return datapoint

