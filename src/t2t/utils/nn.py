from typing import Optional

import torch
from torch import nn


def create_embedding_layer(embed_type: str, ndim: int, hidden_size: int):
    if embed_type == "continuous":
        mid_size = 512
        mlp = nn.Sequential(
            nn.Linear(ndim, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, hidden_size),
        )
        return torch.nn.Linear(ndim, hidden_size)
        # return mlp  # replace 1 layer projection to 2 layers mlp
    elif embed_type == "discrete":
        return nn.Embedding(ndim, hidden_size)
    else:
        raise NotImplementedError(f"Embedding type {embed_type} not implemented.")


def create_placeholder_tensor(data_type: str, batch_size: int, seq_len: int, ndim: Optional[int] = None):
    if data_type == "continuous":
        assert ndim is not None, "ndim must be specified for continuous placeholder tensor."
        return torch.zeros(batch_size, seq_len, ndim)
    elif data_type == "discrete":
        return torch.zeros(batch_size, seq_len, dtype=torch.long)
    else:
        raise NotImplementedError(f"Placeholder tensor type {data_type} not implemented.")

