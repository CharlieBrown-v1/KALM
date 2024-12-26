import numpy as np

from t2t.args.data_args import (
    InstructedTrajectoryDataArguments,
    ClevrDataArguments,
)
from t2t.datasets import (
    InstructedTrajectoryDataset,
    SingleInstructedTrajectoryDataset,
    ClevrDataset,
)


def load_it_dataset(data_args: InstructedTrajectoryDataArguments) -> InstructedTrajectoryDataset:
    """load instructed trajectory datset (May set some data_args attrs)"""
    np_data = np.load(data_args.dataset_path, allow_pickle=True).item()

    try:
        # only single instructed trajectory dataset is supported for now
        dataset = SingleInstructedTrajectoryDataset(
            observations=np_data["observations"],
            actions=np_data["actions"],
            next_observations=np_data["next_observations"],
            rewards=np_data["rewards"],
            terminals=np_data["terminals"],
            masks=np_data["masks"],
            instructions=np_data["instructions"],
            observation_type=np_data["observation_type"] if "observation_type" in np_data else None,
            observation_dim=np_data["observation_dim"] if "observation_dim" in np_data else -1,
            action_type=np_data["action_type"] if "action_type" in np_data else None,
            action_dim=np_data["action_dim"] if "action_dim" in np_data else -1,
            include_last_observation=data_args.include_last_observation,
        )
    except KeyError as e:
        raise KeyError(f"Key {e} not found in dataset.")

    # infer dataset attrs
    if data_args.max_traj_len < 0:
        data_args.max_traj_len = dataset.max_traj_len
    if data_args.observation_dim < 0:
        data_args.observation_dim = dataset.observation_dim
    if data_args.action_dim < 0:
        data_args.action_dim = dataset.action_dim

    if not data_args.observation_type:
        dtype = dataset.observations.dtype
        if np.issubdtype(dtype, np.floating):
            data_args.observation_type = "continuous"
        elif np.issubdtype(dtype, np.integer):
            data_args.observation_type = "discrete"
        else:
            raise ValueError(f"Unsupported dtype {dtype} for observations type inference.")
    if not data_args.action_type:
        dtype = dataset.actions.dtype
        if np.issubdtype(dtype, np.floating):
            data_args.action_type = "continuous"
        elif np.issubdtype(dtype, np.integer):
            data_args.action_type = "discrete"
        else:
            raise ValueError(f"Unsupported dtype {dtype} for actions type inference.")

    return dataset


def load_clevr_dataset(data_args: ClevrDataArguments, available_pattern_list: list) -> ClevrDataset:
    """load clevr datset (May set some data_args attrs)"""
    np_data = np.load(data_args.dataset_path, allow_pickle=True).item()

    try:
        # only single instructed trajectory dataset is supported for now
        dataset = ClevrDataset(
            observations=np_data["observations"],
            actions=np_data["actions"],
            next_observations=np_data["next_observations"],
            rewards=np_data["rewards"],
            terminals=np_data["terminals"],
            masks=np_data["masks"] if "env_names" not in np_data else np.squeeze(np_data["masks"]).astype(np.int32),  # lky: for meta data
            instructions=np_data["instructions"],
            goals=np_data["goals"],
            observation_type=np_data["observation_type"] if "observation_type" in np_data else None,
            observation_dim=np_data["observation_dim"] if "observation_dim" in np_data else -1,
            action_type=np_data["action_type"] if "action_type" in np_data else None,
            action_dim=np_data["action_dim"] if "action_dim" in np_data else -1,
            include_last_observation=data_args.include_last_observation,
            num_objects=np_data["number_of_objects"] if "number_of_objects" in np_data else None,
            available_pattern_list=available_pattern_list,
            env_names=np_data['env_names'] if "env_names" in np_data else None  # lky: for meta data
        )
    except KeyError as e:
        raise KeyError(f"Key {e} not found in dataset.")

    # infer dataset attrs
    if data_args.max_traj_len < 0:
        data_args.max_traj_len = dataset.max_traj_len
    if data_args.observation_dim < 0:
        data_args.observation_dim = dataset.observation_dim
    if data_args.action_dim < 0:
        data_args.action_dim = dataset.action_dim

    if not data_args.observation_type:
        dtype = dataset.observations.dtype
        if np.issubdtype(dtype, np.floating):
            data_args.observation_type = "continuous"
        elif np.issubdtype(dtype, np.integer):
            data_args.observation_type = "discrete"
        else:
            raise ValueError(f"Unsupported dtype {dtype} for observations type inference.")
    if not data_args.action_type:
        dtype = dataset.actions.dtype
        if np.issubdtype(dtype, np.floating):
            data_args.action_type = "continuous"
        elif np.issubdtype(dtype, np.integer):
            data_args.action_type = "discrete"
        else:
            raise ValueError(f"Unsupported dtype {dtype} for actions type inference.")

    return dataset

