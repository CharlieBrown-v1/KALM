import torch
import h5py
import random
import d3rlpy
import argparse
import dataclasses
import numpy as np
import torch.nn as nn

from typing import Any, Dict, List, Union, Sequence
from tqdm import tqdm
from d3rlpy.dataset.components import Episode
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.dataset import ReplayBuffer, InfiniteBuffer
from d3rlpy.dataset.types import Observation, ObservationSequence
from d3rlpy.dataset.mini_batch import TransitionMiniBatch, stack_observations, cast_recursively
from d3rlpy.dataset.transition_pickers import Transition, TransitionPickerProtocol, _validate_index, retrieve_observation, create_zero_observation


gd_max_tau_len = 50
gd_batch_size = 1024
num_obj = 5
goal_src = 1
goal_dst = 2
ds_type_list = [
    'baseline',
    'step_level',
    'tau_level',
    'task_level',
]


def convert_dataset(ds_type: str, dataset_path: str) -> List:
    assert ds_type in ds_type_list
    with h5py.File(dataset_path, "r") as dataset_fr:
        valid_len_arr = np.array(dataset_fr["masks"]).sum(axis=1).astype(int)
        observations = np.array(dataset_fr['observations'])
        actions = np.array(dataset_fr['actions'])
        rewards = np.array(dataset_fr['rewards'])

    llata_episodes = []
    for tau_idx in tqdm(range(valid_len_arr.shape[0])):
        tau_len = valid_len_arr[tau_idx].item()
        tau_obs_arr = observations[tau_idx][:tau_len]
        tau_action_arr = actions[tau_idx][:tau_len]
        tau_reward_arr = rewards[tau_idx][:tau_len]

        llata_episode = LlataEpisode(observations=tau_obs_arr, actions=tau_action_arr, rewards=tau_reward_arr, terminated=True, tau_idx=None)
        llata_episodes.append(llata_episode)

    return llata_episodes


class LlataEncoder(nn.Module):
    def __init__(self, observation_shape, feature_size, hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.output_dim = feature_size
        
        self.fc1 = nn.Linear(observation_shape[0], self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.feature_size)
    
    def get_feature_size(self):
        return self.feature_size

    def forward(self, x):
        assert len(x.shape) == 2, 'x must be 2-dim tensor. (batch_size, observation_size + tau_idx)'

        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))

        return h


# your own encoder factory
@dataclasses.dataclass()
class LlataEncoderFactory(EncoderFactory):
    feature_size: int
    hidden_size: int

    def create(self, observation_shape):
        return LlataEncoder(observation_shape, self.feature_size, self.hidden_size)

    @staticmethod
    def get_type() -> str:
        return "Llata"


@dataclasses.dataclass(frozen=True)
class LlataEpisode(Episode):
    observations: ObservationSequence
    actions: np.ndarray
    rewards: np.ndarray
    terminated: bool
    tau_idx: int

    def serialize(self) -> Dict[str, Any]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "terminated": self.terminated,
            "tau_idx": self.tau_idx,
        }

    @classmethod
    def deserialize(cls, serializedData: Dict[str, Any]) -> "LlataEpisode":
        return cls(
            observations=serializedData["observations"],
            actions=serializedData["actions"],
            rewards=serializedData["rewards"],
            terminated=serializedData["terminated"],
            tau_idx=serializedData["tau_idx"],
        )


@dataclasses.dataclass(frozen=True)
class LlataTransition(Transition):
    observation: Observation  # (...)
    action: np.ndarray  # (...)
    reward: np.ndarray  # (1,)
    next_observation: Observation  # (...)
    terminal: float
    interval: int

    tau_idx: int


class LlataTransitionPicker(TransitionPickerProtocol):
    def __call__(self, episode: LlataEpisode, index: int) -> LlataTransition:
        _validate_index(episode, index)

        observation = retrieve_observation(episode.observations, index)
        is_terminal = episode.terminated and index == episode.size() - 1
        if is_terminal:
            next_observation = create_zero_observation(observation)
        else:
            next_observation = retrieve_observation(
                episode.observations, index + 1
            )
        return LlataTransition(
            observation=observation,
            action=episode.actions[index],
            reward=episode.rewards[index],
            next_observation=next_observation,
            terminal=float(is_terminal),
            interval=1,

            tau_idx=episode.tau_idx,
        )


@dataclasses.dataclass(frozen=True)
class LlataTransitionMiniBatch(TransitionMiniBatch):
    observations: Union[np.ndarray, Sequence[np.ndarray], Dict]  # (B, ...)
    actions: np.ndarray  # (B, ...)
    rewards: np.ndarray  # (B, 1)
    next_observations: Union[np.ndarray, Sequence[np.ndarray], Dict]  # (B, ...)
    terminals: np.ndarray  # (B, 1)
    intervals: np.ndarray  # (B, 1)

    @classmethod
    def from_transitions(
        cls, transitions: Sequence[Transition]
    ) -> "LlataTransitionMiniBatch":
        observations = stack_observations(
            [transition.observation for transition in transitions]
        )
        actions = np.stack(
            [transition.action for transition in transitions], axis=0
        )
        rewards = np.stack(
            [transition.reward for transition in transitions], axis=0
        )
        next_observations = stack_observations(
            [transition.next_observation for transition in transitions]
        )
        terminals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        intervals = np.reshape(
            np.array([transition.terminal for transition in transitions]),
            [-1, 1],
        )
        tau_idxes = np.reshape(
            np.array([transition.tau_idx for transition in transitions]),
            [-1, 1],
        )

        # obs_with_tau_idx = np.c_[cast_recursively(observations, np.float32), cast_recursively(tau_idxes, np.float32)]
        # next_obs_with_tau_idx = np.c_[cast_recursively(next_observations, np.float32), cast_recursively(tau_idxes, np.float32)]

        return LlataTransitionMiniBatch(
            observations=cast_recursively(observations, np.float32),
            actions=cast_recursively(actions, np.float32),
            rewards=cast_recursively(rewards, np.float32),
            next_observations=cast_recursively(next_observations, np.float32),
            terminals=cast_recursively(terminals, np.float32),
            intervals=cast_recursively(intervals, np.float32),
        )


class LlataReplayBuffer(ReplayBuffer):
    def sample_transition_batch(self, batch_size: int) -> LlataTransitionMiniBatch:
        r"""Samples a mini-batch of transitions.

        Args:
            batch_size: Mini-batch size.

        Returns:
            Mini-batch.
        """
        return LlataTransitionMiniBatch.from_transitions(
            [self.sample_transition() for _ in range(batch_size)]
        )


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ds_type", type=str, default="rephrase_level", choices=['baseline', 'rephrase_level', 'easy_level', 'hard_level'], help="The type of offlineRL dataset.")
    parser.add_argument("--agent_name", type=str, default="bc", help="The name of offlineRL agent.")
    parser.add_argument("--dataset_path", type=str, default="./data/clevr_robot.hdf5", help="The path of offlineRL dataset.")
    parser.add_argument("--device", type=str, default="cuda:0", help="The device for offlineRL training.")
    # offlineRL algorithm hyperparameters
    parser.add_argument("--seed", type=int, default=7, help="Seed.")
    # CQL
    parser.add_argument("--cql_alpha", type=float, default=10.0, help="Weight of conservative loss in CQL.")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    transfer2old = {
        'baseline': 'baseline',
        'rephrase_level': 'tau_level',
        'easy_level': 'step_level',
        'hard_level': 'task_level',
    }
    args.ds_type = transfer2old[args.ds_type]
    kwargs = vars(args)

    llata_episodes = convert_dataset(ds_type=args.ds_type, dataset_path=args.dataset_path)

    dataset = LlataReplayBuffer(
        InfiniteBuffer(),
        transition_picker=LlataTransitionPicker(),
        episodes=llata_episodes,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.agent_name == 'bc':
        alg_hyper_list = [
        ]
        agent = d3rlpy.algos.DiscreteBCConfig(
            encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
        ).create(device=args.device)
    elif args.agent_name == 'cql':
        alg_hyper_list = [
            'cql_alpha',
        ]
        agent = d3rlpy.algos.DiscreteCQLConfig(
            encoder_factory=LlataEncoderFactory(feature_size=256, hidden_size=256),
            alpha=args.cql_alpha,
        ).create(device=args.device)
    else:
        raise NotImplementedError

    record_key_list = [
        'ds_type',
        'seed',
        'agent_name',
    ] + alg_hyper_list
    exp_name = 'offlineCLEVR'
    for key in record_key_list:
        exp_name = f'{exp_name}&{key}={kwargs[key]}'

    # offline training
    agent.fit(
        dataset=dataset,
        n_steps=500000,
        experiment_name=exp_name,
        with_timestamp=False,
    )
