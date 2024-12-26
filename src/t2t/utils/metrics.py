from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from operator import itemgetter

from envs.clevr_robot_env import ClevrEnv
from envs.clevr_robot_env import Lang2Env


def default_sl_loss(
    data_type: str,
    pred: torch.Tensor,
    targ: torch.Tensor,
    mask: Optional[torch.Tensor] = None
):
    if mask is None:
        mask = torch.ones_like(targ)

    if data_type == "continuous":
        if len(mask.shape) < len(pred.shape):
            num_dims_to_add = len(pred.shape) - len(mask.shape)
            dim_add = pred.shape[-1]
            mask = mask.unsqueeze(-1).expand(*mask.shape, *([1] * num_dims_to_add))
        else:
            dim_add = 1

        # return (((pred - targ) ** 2) * mask).sum() / mask.sum()
        return (((pred - targ) ** 2) * mask).sum() / (mask.sum() * dim_add)
        # mask should be the same shape of pred and targ!!! otherwise broadcast will make mask.sum() dim_add times smaller

    elif data_type == "discrete":
        # predictions are assumed to be logits
        mask = mask.view(*targ.shape)
        loss = F.cross_entropy(pred.flatten(0, -2), targ.long().flatten(), reduce=False).view(*targ.shape)
        return (loss * mask).sum() / mask.sum()
    else:
        raise NotImplementedError


def dynamics_loss(
    Observations,
    Actions,
    States,
    loss_fn : str = "MSE",
):
    true_next_obs = []
    # sim_env = ClevrEnv(maximum_episode_steps=50,
    #             action_type='perfect',
    #             obs_type='order_invariant',
    #             use_subset_instruction=True,
    #             num_object=5,
    #             direct_obs=False,
    #             use_camera=True)
    sim_env = Lang2Env(maximum_episode_steps=50,
                     action_type="perfect",
                     obs_type='order_invariant',
                     direct_obs=True,
                     use_subset_instruction=True,
                     num_object= 5)
    for state, obs, act in zip(States[:len(Actions)], Observations[:len(Actions)], Actions):
        sim_env.set_state( state[0:37] , state[37:] )
        sim_env.step(act)
        true_next_obs.append( sim_env.get_direct_obs() )

    if loss_fn == "MSE":
        true_next_obs_tensor = torch.tensor(true_next_obs, dtype=torch.float32)
        observations_tensor = torch.tensor(Observations[1:], dtype=torch.float32)
        if true_next_obs_tensor.shape[0] > observations_tensor.shape[0]:
            true_next_obs_tensor = true_next_obs_tensor[:-1]
        true_next_obs_tensor = true_next_obs_tensor.to('cuda')
        observations_tensor = observations_tensor.to('cuda')

        mse_loss = torch.nn.functional.mse_loss(true_next_obs_tensor, observations_tensor)
        mse_loss_value = mse_loss.item()
        loss_value = mse_loss_value

    return loss_value, true_next_obs


# TODO: refactor this function
# def dynamics_loss(
#     simulator,
#     observations,
#     actions,
#     pred_next_observations,
#     dynamics_forward_fn,
#     loss_fn : str = "MSE",
# ):
#     loss = 0
#     true_next_obs = []
#     sim_env = ClevrEnv(maximum_episode_steps=50,
#                 action_type='perfect',
#                 obs_type='order_invariant',
#                 use_subset_instruction=True,
#                 num_object=5,
#                 direct_obs=False,
#                 use_camera=True)
#     for state, obs, act in zip(States, Observations, Actions):
#         sim_env.set_state( state[0:37] , state[37:] )
#         sim_env.step(act)
#         true_next_obs.append( sim_env.get_direct_obs() )

#     if loss_fn == "MSE":
#         true_next_obs_tensor = torch.tensor(true_next_obs[:-1], dtype=torch.float32)
#         observations_tensor = torch.tensor(Observations[1:], dtype=torch.float32)
#         true_next_obs_tensor = true_next_obs_tensor.to('cuda')
#         observations_tensor = observations_tensor.to('cuda')

#         mse_loss = torch.nn.functional.mse_loss(true_next_obs_tensor, observations_tensor)
#         mse_loss_value = mse_loss.item()
#         loss_value = mse_loss_value

#     return loss_value

