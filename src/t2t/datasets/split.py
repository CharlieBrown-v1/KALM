import numpy as np
import torch
import torch.utils.data


def fixed_split(dataset, split_indices, shuffle=True):
    assert sum([len(indices) for indices in split_indices]) == len(dataset), \
        "Sum of split indices should be equal to dataset length."

    # check if indices are valid and unique
    indices = np.concatenate(split_indices)
    assert len(indices) == len(set(indices)), "Indices should be unique."

    # shuffle indices
    if shuffle:
        for indices in split_indices:
            np.random.shuffle(indices)

    return split_indices, [torch.utils.data.Subset(dataset, indices) for indices in split_indices]


def random_split(dataset, ratios):
    assert abs(sum(ratios) - 1) < 1e-6, "Sum of ratios should be 1."

    lengths = [int(len(dataset) * ratio) for ratio in ratios[:-1]]
    lengths.append(len(dataset) - sum(lengths))
    indices = torch.randperm(len(dataset)).tolist()
    indices = [
        indices[offset - length:offset] for offset, length in zip(torch.cumsum(torch.tensor(lengths), dim=0), lengths)
    ]

    return indices, [torch.utils.data.Subset(dataset, index) for index in indices]


def random_split_by_label(dataset, labels, ratios):
    """Group datapoint with the same label and split the groups."""
    assert abs(sum(ratios) - 1) < 1e-6, "Sum of ratios should be 1."

    label_set = torch.unique(labels, dim=0)
    label_set = label_set[torch.randperm(len(label_set))]
    label_indices = [torch.where(torch.all(labels == label, dim=1))[0] for label in label_set]
    label_lengths = [int(len(label_indices) * ratio) for ratio in ratios[:-1]]
    label_lengths.append(len(label_indices) - sum(label_lengths))
    label_indices = [
        label_indices[offset - length:offset] for offset, length in zip(torch.cumsum(torch.tensor(label_lengths), dim=0), label_lengths)
    ]
    indices = [torch.concat(index_list, dim=0).flatten().tolist() for index_list in label_indices]

    # shuffle indices inside
    for index_list in indices:
        np.random.shuffle(index_list)

    return indices, [torch.utils.data.Subset(dataset, index) for index in indices]

