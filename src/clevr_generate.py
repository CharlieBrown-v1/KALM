import os
import argparse
import numpy as np
import torch
import transformers

from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Optional, Callable, Tuple, Dict, Any
from envs.utils.clevr_utils import DIRECTIONS_CNT, BEHIND, LEFT, FRONT, RIGHT, TAU_2, TAU_3, ORDER, SORT, color_list, _success
from t2t.models import LlataConfig
from t2t.models import LlataForTrajectoryGeneration


from envs.clevr_robot_env import LangEnv
from envs.utils.clevr_utils import terminal_fn_with_level as clevr_terminal_fn
num_obj = 5
env = LangEnv(
    maximum_episode_steps=50,
    action_type='perfect',
    obs_type='order_invariant',
    use_subset_instruction=True,
    num_object=num_obj,
    direct_obs=True,
    use_camera=False,
)
dir_value_to_desc_list = {
    BEHIND: ['behind', 'in behind of', 'to the behind of'],
    LEFT: ['to the left of', 'left', 'in left of'],
    FRONT: ['in front of', 'front', 'to the front of'],
    RIGHT: ['to the right of', 'right', 'in right of'],
}

goal_src = 1
goal_dst = 2
gd_max_tau_len = 50


def compute_dist(obs_arr: np.ndarray, goal_arr: np.ndarray) -> float:
    reshaped_obs = obs_arr.reshape((num_obj, -1))
    
    goal_src_idx = np.where(goal_arr[:num_obj] == goal_src)[0].item()
    goal_dst_idx = np.where(goal_arr[:num_obj] == goal_dst)[0].item()
    
    goal_src_xy = reshaped_obs[goal_src_idx]
    goal_dst_xy = reshaped_obs[goal_dst_idx]
    
    dist = np.linalg.norm(goal_src_xy - goal_dst_xy).item()

    return dist


def compute_tau_reward_arr(tau_obs_arr: np.ndarray, tau_next_obs_arr: np.ndarray, goal_arr: np.ndarray, is_success: bool) -> np.ndarray:
    success_reward = 100
    dist_scale = 10
    tau_reward_list = []
    for step_idx in range(tau_obs_arr.shape[0]):
        prev_dist = compute_dist(tau_obs_arr[step_idx], goal_arr)
        curr_dist = compute_dist(tau_next_obs_arr[step_idx], goal_arr)
        reward = (prev_dist - curr_dist) * dist_scale
        tau_reward_list.append(reward)
    # Hard code the reward of final step
    if is_success:
        # tau_reward_list[-1] = success_reward
        tau_reward_list.append(success_reward)
    else:
        tau_reward_list.append(0.0)

    tau_reward_arr = np.array(tau_reward_list).reshape(-1, 1)

    return tau_reward_arr


def compute_task_reward_arr(tau_obs_arr: np.ndarray, tau_next_obs_arr: np.ndarray, goal_arr: np.ndarray, is_success: bool) -> np.ndarray:
    success_reward = 100
    subgoal_reward = 10

    dist = 0.13 * 2.5
    order_target_diff_arr = np.array([dist, 0.0])
    norm_order_target_diff_arr = order_target_diff_arr / np.linalg.norm(order_target_diff_arr)
    order_max_error_angle = np.pi / 6
    sort_cycle_r = dist

    tau_reward_list = []
    goal_indicate = int(goal_arr[num_obj])
    curr_completed_goal_cnt = 0
    for step_idx in range(tau_obs_arr.shape[0]):
        curr_env_obs = tau_obs_arr[step_idx][:2 * num_obj]
        prev_completed_goal_cnt = curr_completed_goal_cnt
        if goal_indicate in [TAU_2, TAU_3]:
            if int(goal_indicate) == TAU_2:
                all_goal_arr = goal_arr[num_obj + 1: (num_obj + 1) * (2 + 1)].reshape(-1, num_obj + 1)
            elif int(goal_indicate) == TAU_3:
                all_goal_arr = goal_arr[num_obj + 1: (num_obj + 1) * (3 + 1)].reshape(-1, num_obj + 1)
            else:
                raise NotImplementedError
            curr_completed_goal_cnt = np.sum([_success(num_obj, curr_env_obs.reshape(1, -1), single_goal_arr.reshape(1, -1)).item() for single_goal_arr in all_goal_arr])
        elif goal_indicate == ORDER:
            diff_arr = np.diff(curr_env_obs.reshape((num_obj, -1)), axis=0)
            norm_diff_arr = diff_arr / np.linalg.norm(diff_arr, axis=1).reshape(-1, 1)
            diff_dot_arr = np.dot(norm_diff_arr, norm_order_target_diff_arr)
            completed_flag_arr = diff_dot_arr >= np.cos(order_max_error_angle)
            curr_completed_goal_cnt = completed_flag_arr.sum()
        elif goal_indicate == SORT:
            obs_xy_arr = curr_env_obs.reshape((num_obj, -1))
            cycle_center = obs_xy_arr[2]
            dist_list = []
            other_idx_list = [0, 1, 3, 4]
            for other_idx in other_idx_list:
                dist = np.linalg.norm(cycle_center - obs_xy_arr[other_idx])
                dist_list.append(dist)
            curr_completed_goal_cnt = (np.array(dist_list) < sort_cycle_r).sum()
        else:
            raise NotImplementedError
        step_reward = subgoal_reward * (curr_completed_goal_cnt - prev_completed_goal_cnt)
        tau_reward_list.append(step_reward)

    if is_success:
        # tau_reward_list[-1] = success_reward
        tau_reward_list.append(success_reward)
    else:
        tau_reward_list.append(0.0)

    tau_reward_arr = np.array(tau_reward_list).reshape(-1, 1)

    return tau_reward_arr


def default_terminal_fn(
    insts: List[str],
    observations: np.ndarray,
    kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return np.zeros(len(insts), dtype=bool)


def prepare_sequences(batch_size: int, seq_len: int, type: str, dim: int):
    if type == "continuous":
        return torch.zeros(batch_size, seq_len, dim)
    elif type == "discrete":
        return torch.zeros(batch_size, seq_len, dtype=torch.long)
    else:
        raise NotImplementedError


def logits_to_outputs(logits: torch.Tensor, type: str, dim: Optional[int] = None):
    if type == "continuous":
        return logits
    elif type == "discrete":
        return logits.argmax(dim=dim)
    else:
        raise NotImplementedError


@torch.no_grad()
def greedy_generate(
    model_tg: LlataForTrajectoryGeneration,
    tokenizer: PreTrainedTokenizer,
    instructions: List[str],
    init_observations: Optional[torch.Tensor] = None,
    init_actions: Optional[torch.Tensor] = None,
    max_inst_length: Optional[int] = None,
    max_trajectory_length: Optional[int] = None,
    terminal_fn: Optional[Callable[[List[str], np.ndarray], np.ndarray]] = None,
    early_stopping: bool = False,
    level: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if terminal_fn is None:
        terminal_fn = default_terminal_fn

    prompt_tokens = tokenizer(
        instructions,
        add_special_tokens=True,
        padding=True,
        max_length=max_inst_length,
        return_tensors="pt",
    )
    prompt_ids = prompt_tokens["input_ids"].to(model_tg.device)
    prompt_attention_masks = prompt_tokens["attention_mask"].to(model_tg.device)

    max_traj_len = max_trajectory_length or model_tg.config.max_trajectory_length
    observation_type = model_tg.model.observation_type
    action_type = model_tg.model.action_type
    observation_dim = model_tg.model.observation_dim
    action_dim = model_tg.model.action_dim
    obss_len = init_observations.shape[1] if init_observations is not None else 0
    acts_len = init_actions.shape[1] if init_actions is not None else 0

    assert obss_len == acts_len or obss_len == acts_len + 1, \
        "The length of initial observations should be equal to the length of initial actions or one more"

    # prepare initial observations and actions
    if init_observations is None:
        init_observations = prepare_sequences(
            len(instructions), 0, observation_type, observation_dim,
        )
    if init_actions is None:
        init_actions = prepare_sequences(
            len(instructions), 0, action_type, action_dim,
        )
    init_observations = init_observations.to(model_tg.device).to(model_tg.dtype)
    init_actions = init_actions.to(model_tg.device)
    observation_masks = torch.ones(len(instructions), obss_len, dtype=torch.bool).to(model_tg.device)
    action_masks = torch.ones(len(instructions), acts_len, dtype=torch.bool).to(model_tg.device)

    max_steps = max_traj_len - acts_len
    # the prompt part is considered as not done
    dones = torch.zeros(len(instructions), obss_len, dtype=bool).to(model_tg.device)
    success = torch.zeros(len(instructions), obss_len, dtype=bool).to(model_tg.device)
    for step in range(max_steps):
        # prepare inputs
        inputs = {
            "inst_tokens": prompt_ids,
            "inst_masks": prompt_attention_masks,
            "observations": init_observations,
            "actions": init_actions,
            "observation_masks": observation_masks,
            "action_masks": action_masks,
            "observation_labels": torch.zeros(size=init_observations.shape, dtype=init_observations.dtype),
            "action_labels": torch.zeros(size=init_actions.shape, dtype=init_actions.dtype),
            "pattern": torch.tensor([0 for _ in range(init_observations.shape[0])], dtype=torch.int32),
        }

        output = model_tg(**inputs, return_dict=True)
        # generate observation and check termination
        if obss_len == acts_len:
            # print(f'=' * 64)
            # print(f'Observation')
            # print(f'=' * 64)
            new_obs = logits_to_outputs(output.observation_logits[:, -1:], observation_type)

            # update observations
            init_observations = torch.cat([init_observations, new_obs], dim=1)
            inputs["observations"] = init_observations
            obss_len += 1

            # check termination
            # NOTE: do not check the initial observation due to data's fault
            if obss_len > 1:
                if level == 'step_level':
                    terminals = terminal_fn(insts=instructions, observations=new_obs[:, 0].cpu().numpy(), level='step_level_a', hist_observations=init_observations.cpu().numpy(), actions=init_actions[:, -1].cpu().numpy())
                else:
                    terminals = terminal_fn(insts=instructions, observations=new_obs[:, 0].cpu().numpy(), level=level)
                done, succ = terminals["done"], terminals["success"]
            else:
                done, succ = torch.zeros(len(instructions), dtype=bool), torch.zeros(len(instructions), dtype=bool)

            # update flags and masks
            dones = torch.cat([dones, torch.as_tensor(done).to(dones).unsqueeze(-1)], dim=1)
            success = torch.cat([success, torch.as_tensor(succ).to(success).unsqueeze(-1)], dim=1)
            observation_masks = torch.cat([observation_masks, ~(dones.any(dim=-1, keepdim=True))], dim=1)
            inputs["observation_masks"] = observation_masks

            if early_stopping and dones.any(dim=-1).all().item():
                break

        # generate action
        if obss_len != acts_len:
            # print(f'=' * 64)
            # print(f'Action')
            # print(f'=' * 64)
            new_act = logits_to_outputs(output.action_logits[:, -1:], action_type, dim=-1)
            init_actions = torch.cat([init_actions, new_act], dim=1)
            action_masks = torch.cat([action_masks, ~(dones.any(dim=-1, keepdim=True))], dim=1)

            # update actions and masks
            inputs["actions"] = init_actions
            inputs["action_masks"] = action_masks
            acts_len += 1

    return {
        "observations": init_observations,
        "actions": init_actions,
        "observation_masks": observation_masks,
        "action_masks": action_masks,
        "dones": dones,
        "success": success,
    }


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="The path of the finetuned model.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="",
        help="The path of the instruction prompts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="The path to save the generated trajectories.",
    )
    parser.add_argument(
        "--max_trajectory_length",
        type=int,
        default=50,
        help="The maximum length of generated trajectories.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="The batch size of model inference.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to stop early when terminal function satisfied.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=0,
        help="The id of the padding token.",
    )
    parser.add_argument(
        "--level",
        type=str,
        # default="rephrase_level",
        # choices=['rephrase_level', 'easy_level', 'hard_level'],
        help="The evaluation level.",
    )

    args = parser.parse_args()

    return args


def generate_one_level(args, model_tg: LlataForTrajectoryGeneration):
    from functools import partial
    # load prompts
    level = args.level
    prompts_data = np.load(args.prompt_path, allow_pickle=True).item()

    if model_tg is None:
        return -1

    # NOTE: special arguments for CLEVR Robot environment termination function
    num_obj = prompts_data["number_of_objects"]
    goals = prompts_data["goals"]

    # Dataset preprocessing
    llm_encoding_list = [tau_obs_arr[0, 2 * num_obj:] for tau_obs_arr in prompts_data["observations"]]
    prompts_data["observations"] = [tau_obs_arr[:1, :2 * num_obj] for tau_obs_arr in prompts_data["observations"]]
    if 'actions' in prompts_data.keys():
        prompts_data.pop("actions")

    instructions = prompts_data["instructions"]
    init_observations = torch.as_tensor(np.array(prompts_data["observations"])) \
        if "observations" in prompts_data \
        else None
    init_actions = torch.as_tensor(prompts_data["actions"]) \
        if "actions" in prompts_data \
        else None

    # generate
    # NOTE: terminal function is set to CLEVR Robot environment termination function
    observations, actions, observation_masks, action_masks, dones, success = [], [], [], [], [], []
    rewards = []
    for i in tqdm(range(0, len(instructions), args.batch_size)):
        generation = greedy_generate(
            model_tg=model_tg,
            tokenizer=tokenizer,
            instructions=instructions[i:i+args.batch_size],
            init_observations=init_observations[i:i+args.batch_size] if init_observations is not None else None,
            init_actions=init_actions[i:i+args.batch_size] if init_actions is not None else None,
            max_trajectory_length=args.max_trajectory_length,
            terminal_fn=partial(
                clevr_terminal_fn,
                number_of_objects=num_obj,
                goals=goals[i:i+args.batch_size],
            ),
            early_stopping=args.early_stopping,
            level=level,
        )
        
        observations.extend(list(generation["observations"].cpu().numpy()))
        actions.extend(list(generation["actions"].cpu().numpy()))
        observation_masks.extend(list(generation["observation_masks"].cpu().numpy()))
        action_masks.extend(list(generation["action_masks"].cpu().numpy()))
        dones.extend(list(generation["dones"].cpu().numpy()))
        success.extend(list(generation["success"].cpu().numpy()))

        for j in range(generation["observations"].cpu().numpy().shape[0]):
            tau_len = np.sum(generation["observation_masks"].cpu().numpy()[j])
            tau_len -= int(tau_len == gd_max_tau_len)
            is_success = generation["success"].cpu().numpy()[j][tau_len]
            tau_obs_arr = generation["observations"].cpu().numpy()[j][:tau_len]
            tau_next_obs_arr = generation["observations"].cpu().numpy()[j][1 :tau_len + 1]
            goal_arr = goals[i:i+args.batch_size][j]
            if level == 'step_level':
                tau_reward_arr = np.array([100 * is_success]).reshape(-1, 1)
            elif level == 'tau_level':
                tau_reward_arr = compute_tau_reward_arr(tau_obs_arr[:, :2 * num_obj], tau_next_obs_arr[:, :2 * num_obj], goal_arr, is_success)
            elif level == 'task_level':
                tau_reward_arr = compute_task_reward_arr(tau_obs_arr, tau_next_obs_arr, goal_arr, is_success)
            else:
                raise NotImplementedError
            rewards.append(tau_reward_arr)

    # save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    
    # append llm encoding
    full_observations = []
    for tau_idx in range(len(observations)):
        tau_obs = observations[tau_idx]
        tau_len = len(tau_obs)
        llm_encoding = llm_encoding_list[tau_idx]
        tau_llm_encoding = llm_encoding.reshape(1, -1).repeat(tau_len, axis=0)
        full_tau_obs = np.c_[tau_obs, tau_llm_encoding]
        full_observations.append(full_tau_obs)
    
    np.save(
        args.output_path,
        {
            "instructions": instructions,
            "observations": full_observations,
            "actions": actions,
            "observation_masks": observation_masks,
            "action_masks": action_masks,
            "terminals": dones,
            "success": success,
            "goals": goals,
            "number_of_objects": num_obj,
            "rewards": rewards,
        }
    )


if __name__ == "__main__":
    args = get_args()
    transfer2old = {
        'rephrase_level': 'tau_level',
        'easy_level': 'step_level',
        'hard_level': 'task_level',
    }
    # args.level = transfer2old[args.level]

    transformers.set_seed(args.seed)
    # load tokenizer
    config_kwargs = {
        "trust_remote_code": True,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        **config_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = args.pad_token_id
    # load model
    config = LlataConfig.from_pretrained(args.model_path, **config_kwargs)
    config.use_mlp = False
    model_tg = LlataForTrajectoryGeneration.from_pretrained(
        args.model_path, config=config,
        device_map='auto',
    )
    model_tg.eval()
    args.max_trajectory_length = 50
    args.early_stopping = True
    print("===> arguments:", args)
    generate_one_level(args, model_tg=model_tg)
