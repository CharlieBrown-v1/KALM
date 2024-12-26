from enum import Enum
from dataclasses import dataclass, field, fields


@dataclass
class Text2TrajectoryModelArguments:
    pretrained_path: str = field(
        default="",
        metadata={"help": "The path of the pretrained model."},
    )
    load_full: bool = field(
        default=False,
        metadata={"help": "Whether to load the full model and configurations."},
    )
    traj_ln: bool = field(
        default=True,
        metadata={"help": "Whether to use layer normalization for trajectory data."},
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

