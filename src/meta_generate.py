import os
import argparse
import numpy as np
import torch
import transformers

from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
from typing import List, Optional, Callable, Tuple, Dict, Any
from t2t.models import LlataConfig
from t2t.models import LlataForTrajectoryGeneration


from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from meta_utils import MetaWrapper
from meta_utils import rephrase_level_env_name_list, easy_level_env_name_list, hard_level_env_name_list, en2nl, TAU_LEN, num_tau
from meta_utils import FakeEnv, obs_online2offline


wrap_info = {
    'reward_shaping': True
}


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
    fake_env_list: List[FakeEnv],
    tokenizer: PreTrainedTokenizer,
    instructions: List[str],
    init_observations: Optional[torch.Tensor] = None,
    init_actions: Optional[torch.Tensor] = None,
    max_inst_length: Optional[int] = None,
    max_trajectory_length: Optional[int] = None,
    early_stopping: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    rewards = torch.zeros(len(instructions), obss_len, dtype=float).to(model_tg.device)
    dones = torch.zeros(len(instructions), obss_len, dtype=bool).to(model_tg.device)
    successes = torch.zeros(len(instructions), obss_len, dtype=bool).to(model_tg.device)
    for step in range(max_steps):
        # prepare inputs
        inputs = {
            "inst_tokens": prompt_ids,
            "inst_masks": prompt_attention_masks,
            "observations": init_observations,
            "actions": init_actions,
            "observation_masks": observation_masks,
            "action_masks": action_masks,
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
            reward_list = []
            done_list = []
            succ_list = []
            if obss_len > 1:
                for fake_env in fake_env_list:
                    single_next_obs, single_reward, single_done, single_info = fake_env.fake_step(step_idx=step, obs=new_obs[:, 0].cpu().numpy(), action=init_actions[:, -1].cpu().numpy())
                    single_succ = single_info['success']
                    reward_list.append(single_reward)
                    done_list.append(single_done)
                    succ_list.append(single_succ)
                reward = np.array(reward_list)
                done = np.array(done_list)
                succ = np.array(succ_list)
            else:
                done, succ = torch.zeros(len(instructions), dtype=bool), torch.zeros(len(instructions), dtype=bool)

            # update rewards, flags and masks
            rewards = torch.cat([rewards, torch.as_tensor(reward).to(rewards).reshape(args.batch_size, 1)], dim=1)
            dones = torch.cat([dones, torch.as_tensor(done).to(dones).reshape(args.batch_size, 1)], dim=1)
            successes = torch.cat([successes, torch.as_tensor(succ).to(successes).reshape(args.batch_size, 1)], dim=1)
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
        "rewards": rewards,
        "dones": dones,
        "successes": successes,
    }


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="The path of the pretrained model_tg.",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="",
        help="The path of the prompts.",
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
        default=TAU_LEN,
        help="The maximum length of generated trajectories.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="The batch size.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Whether to stop early when terminal function satisfied.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=47,
    )
    parser.add_argument(
        "--pad_token_id",
        type=int,
        default=0,
        help="The id of the padding token.",
    )
    parser.add_argument('--num_tau', type=int, default=num_tau)
    parser.add_argument(
        "--level",
        type=str,
        default="rephrase_level",
        choices=['rephrase_level', 'easy_level', 'hard_level'],
        help="The evaluation level.",
    )

    args = parser.parse_args()

    return args


def generate_one_env(args, env_name: str, model_tg: LlataForTrajectoryGeneration):
    if model_tg is None:
        return -1
    
    env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
    env._freeze_rand_vec = False  # random reset
    env.max_path_length = TAU_LEN  # Try to avoid returns of successful tau less than failed tau
    env = MetaWrapper(env, wrap_info=wrap_info)

    # Dataset preprocessing
    np.random.seed(args.seed)
    random_idx_arr = np.random.randint(0, len(en2nl[env_name]), size=args.num_tau)
    original_instructions = [en2nl[env_name][idx] for idx in random_idx_arr]
    instructions = [f'Translate the textual instruction to state/action trajectory.\nInstruction: {inst}\nTrajectory:' for inst in original_instructions]
    init_online_observations = [env.reset()[0] for _ in range(args.num_tau)]
    init_offline_observations = [obs_online2offline(env_name=env_name, online_obs=online_obs) for online_obs in init_online_observations]
    init_observations = torch.as_tensor(np.array(init_offline_observations))
    init_actions = None

    # generate
    fake_env_list = [FakeEnv(env_name=env_name, wrap_info=wrap_info) for _ in range(args.batch_size)]
    observations, actions, observation_masks, action_masks, rewards, dones, successes = [], [], [], [], [], [], []
    for i in tqdm(range(0, len(instructions), args.batch_size)):
        for env_idx in range(args.batch_size):
            fake_env_list[env_idx].reset()

        generation = greedy_generate(
            model_tg=model_tg,
            fake_env_list=fake_env_list,
            tokenizer=tokenizer,
            instructions=instructions[i:i+args.batch_size],
            init_observations=init_observations[i:i+args.batch_size].reshape(args.batch_size, 1, -1) if init_observations is not None else None,
            init_actions=init_actions[i:i+args.batch_size] if init_actions is not None else None,
            max_trajectory_length=args.max_trajectory_length,
            early_stopping=args.early_stopping,
        )
        
        observations.extend(list(generation["observations"].cpu().numpy()))
        actions.extend(list(generation["actions"].cpu().numpy()))
        observation_masks.extend(list(generation["observation_masks"].cpu().numpy()))
        action_masks.extend(list(generation["action_masks"].cpu().numpy()))
        rewards.extend(list(generation["rewards"].cpu().numpy()))
        dones.extend(list(generation["dones"].cpu().numpy()))
        successes.extend(list(generation["successes"].cpu().numpy()))

    # save
    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))
    
    np.save(
        args.output_path,
        {
            "instructions": instructions,
            "observations": observations,
            "actions": actions,
            "observation_masks": observation_masks,
            "action_masks": action_masks,
            "rewards": rewards,
            "terminals": dones,
            "successes": successes,
        }
    )


if __name__ == "__main__":
    args = get_args()
    level = args.level

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
    # load model_tg
    config = LlataConfig.from_pretrained(args.model_path, **config_kwargs)
    model_tg = LlataForTrajectoryGeneration.from_pretrained(
        args.model_path,
        config=config,
        device_map='auto',
    )
    model_tg.eval()
    if level == 'rephrase_level':
        env_name_list = rephrase_level_env_name_list.copy()
    elif level == 'easy_level':
        env_name_list = easy_level_env_name_list.copy()
    elif level == 'hard_level':
        env_name_list = hard_level_env_name_list.copy()
    else:
        raise NotImplementedError
    
    for env_name in env_name_list:
        env_prefix = env_name[:env_name.find('-v2')]
        args.max_trajectory_length = TAU_LEN
        args.early_stopping = True
        print("===> arguments:", args)
        generate_one_env(args, env_name=env_name, model_tg=model_tg)
