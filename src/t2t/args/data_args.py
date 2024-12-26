from enum import Enum
from dataclasses import dataclass, field, fields


@dataclass
class InstructedTrajectoryDataArguments:
    dataset_path: str = field(
        default="",
        metadata={"help": "The path of the instructed trajectory dataset."},
    )
    max_traj_len: int = field(
        default=-1,
        metadata={"help": "The maximum length of the trajectory (obtained from dataset if set negative)."},
    )
    observation_type: str = field(
        default="",
        metadata={"help": "The type of the observation (obtained from dataset if set empty)."},
    )
    observation_dim: int = field(
        default=-1,
        metadata={"help": "The dimension of the observation (obtained from dataset if set negative)."},
    )
    action_type: str = field(
        default="",
        metadata={"help": "The type of the action (obtained from dataset if set empty)."},
    )
    action_dim: int = field(
        default=-1,
        metadata={"help": "The dimension of the action (obtained from dataset if set negative)."},
    )
    validation_ratio: float = field(
        default=0.1,
        metadata={"help": "The ratio of validation set."},
    )
    include_last_observation: bool = field(
        default=True,
        metadata={"help": "Whether to include the last (next) observation in the trajectory."},
    )
    inst_token_trunc: bool = field(
        default=False,
        metadata={"help": "Whether to truncate the instruction tokens to the maximum length."},
    )
    inst_token_max_len: int = field(
        default=None,
        metadata={"help": "The maximum length of the instruction tokens (obtained from dataset if set None)."},
    )
    preprocess_inst_token: bool = field(
        default=True,
        metadata={"help": "Whether to preprocess the instruction to tokens."},
    )
    pad_token_id: int = field(
        default=0,
        metadata={"help": "The id of the padding token."},
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]

        return d


@dataclass
class ClevrDataArguments(InstructedTrajectoryDataArguments):
    split_by: str = field(
        default="index",
        metadata={"help": "The way to randomly split the dataset ('index', 'goal', 'direction', 'source', 'target')."},
    )

