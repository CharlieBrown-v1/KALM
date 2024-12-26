from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class Text2TrajectoryTrainingArguments(TrainingArguments):
    plot_loss: bool = field(
        default=True,
        metadata={"help": "Whether to plot the loss curve after training."},
    )

