import os
import sys
import d3rlpy
import gymnasium as gym
from d3rlpy.models.encoders import register_encoder_factory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from tqdm import tqdm
import numpy as np
from meta_offline_train import LlataEncoderFactory
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from meta_utils import MetaWrapper, baseline_env_name_list, rephrase_level_env_name_list, easy_level_env_name_list, hard_level_env_name_list
from meta_utils import TAU_LEN, obs_online2offline, get_noisy_entity_list, obs_online2noisy_offline, num_noisy_entity, data_dir


succ2flag = {
    True: '[Success]',
    False: '[Failed]',
}


def eval_given_level(
    args: argparse.Namespace,
    env: gym.Env,
    env_name: str,
    model_path: str,
    inst_list: list,
):
    policy = d3rlpy.load_learnable(model_path, device=args.device)

    eval_result_list = []
    success_mean = 0.0

    env_prefix = env_name[:env_name.find('-v2')]
    eval_sample_num = 11
    np.random.seed(args.seed)
    range_tqdm = tqdm(range(eval_sample_num))
    range_tqdm.set_description(f"Evaluating {args.level}-{env_prefix} using {args.agent_name.upper()}")
    for i in range_tqdm:
        online_obs = env.reset()[0]
        inst_encoding_idx = np.random.randint(low=0, high=len(inst_list))  # sample a inst uniformly
        inst_encoding = inst_list[inst_encoding_idx].flatten()
        eval_result = {
            'done': np.array([False]),
            'success': np.array([False]),
            'failure': np.array([False]),
        }
        noisy_entity_list = get_noisy_entity_list(env_name=env_name)
        tau_noisy_entity_list = np.random.choice(noisy_entity_list, size=num_noisy_entity)
        for step in range(env.max_path_length):

            # offline_obs = obs_online2offline(env_name, online_obs)
            offline_obs = obs_online2noisy_offline(env_name, online_obs, tau_noisy_entity_list=tau_noisy_entity_list)

            env_obs = offline_obs.copy()
            policy_obs = np.r_[env_obs, inst_encoding]
            action = policy.predict(policy_obs.reshape(1, -1)).flatten()
            next_obs, reward, terminated, truncated, info = env.step(action)
            terminated = terminated or bool(info['success'])
            done = terminated or truncated

            eval_result['done'] = np.array([done])
            eval_result['success'] = np.array([info['is_success']])
            eval_result['failure'] = np.array([done and not info['is_success']])

            online_obs = next_obs

            if done:
                break

        eval_result_list.append(eval_result)
        success_mean = (success_mean * i + eval_result['success'].item()) / (i + 1)
        range_tqdm.set_postfix_str(f"Avg SR: {success_mean:.5f}")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to sample evaluate cases.",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="baseline",
        help="The path to save the generated trajectories.",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="cql",
        help="Name of offlineRL algorithm.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="The path to the trained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    inst_encoding_dict = np.load(data_dir.joinpath('meta_instructions_encoding.npy'), allow_pickle=True).item()
    register_encoder_factory(LlataEncoderFactory)

    wrap_info = {
        'reward_shaping': True
    }
    env_name_list = []
    if args.level == 'baseline':
        env_name_list = baseline_env_name_list.copy()
    elif args.level == 'rephrase_level':
        env_name_list = rephrase_level_env_name_list.copy()
    elif args.level == 'easy_level':
        env_name_list = easy_level_env_name_list.copy()
    elif args.level == 'hard_level':
        env_name_list = hard_level_env_name_list.copy()
    else:
        raise NotImplementedError
    
    model_path = args.model_path
    for env_name in env_name_list:
        eval_env = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]()
        eval_env._freeze_rand_vec = False  # random reset
        eval_env.max_path_length = TAU_LEN  # Try to avoid returns of successful tau less than failed tau
        eval_env = MetaWrapper(eval_env, wrap_info=wrap_info)
        inst_list = inst_encoding_dict[env_name]
        eval_given_level(
            args=args,
            env=eval_env,
            env_name=env_name,
            model_path=model_path,
            inst_list=inst_list,
        )

    print(f'Evaluating finished!')
