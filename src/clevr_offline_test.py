import os
import sys
import d3rlpy
from d3rlpy.models.encoders import register_encoder_factory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import argparse
from tqdm import tqdm
from typing import List, Optional
from operator import itemgetter
import numpy as np
from envs.clevr_robot_env import LlataEnv
from envs.utils.clevr_utils import data_dir, terminal_fn_with_level, CLEVR_QPOS_OBS_INDICES
from clevr_offline_train import LlataEncoderFactory


succ2flag = {
    True: '[Success]',
    False: '[Failed]',
}


def eval_given_level(
    args: argparse.Namespace,
    env: LlataEnv,
    num_obj: int,
    max_traj_len: int,
    model_path: str,
    instructions: List[str],
    observations: List[np.ndarray],
    terminals: Optional[List[np.ndarray]] = None,
    goals: Optional[List[np.ndarray]] = None,
    actions: Optional[List[np.ndarray]] = None,
    success: Optional[List[np.ndarray]] = None,
):
    assert len(observations) == len(instructions), "instructions, observations number mismatch"
    if goals is not None:
        assert len(observations) == len(goals), \
            "instructions, observations, goals number mismatch"
    if terminals is not None:
        assert len(observations) == len(terminals), \
            "observations, terminals number mismatch"
    if actions is not None:
        # for recording only
        assert len(observations) == len(actions) or len(observations) == len(actions) + 1, \
            "observations, actions number mismatch"
    if success is not None:
        # for recording only
        assert len(observations) == len(success), \
            "observations, success number mismatch"
        
    policy = d3rlpy.load_learnable(model_path, device=args.device)
    if 'step_level' in args.level:
        eval_sample_num = 666
    elif args.level in ['baseline', 'tau_level', 'task_level']:
        eval_sample_num = 111
    else:
        raise NotImplementedError

    assert eval_sample_num <= len(instructions), "eval_sample_num should be less than the number of instructions"
    np.random.seed(args.seed)
    sampled_indices = np.random.choice(len(instructions), eval_sample_num, replace=False)
    range_tqdm = tqdm(sampled_indices)
    range_tqdm.set_description(f"Evaluating {args.level} using {args.agent_name.upper()}")
    for i in range_tqdm:
        insts = instructions[i]
        obss = observations[i]
        if len(goals[i].shape) == 2:
            single_goal = goals[i][0]
        elif len(goals[i].shape) == 1:
            single_goal = goals[i]
        else:
            raise NotImplementedError
        single_inst = np.random.choice(insts)
        
        # Reset Env to dataset obs
        env.reset()
        init_env_obs = obss[0][:2 * num_obj]
        inst_encoding = obss[0][2 * num_obj:]
        qpos, qvel = env.physics.data.qpos.copy(), env.physics.data.qvel.copy()
        qpos[CLEVR_QPOS_OBS_INDICES(num_obj)] = init_env_obs
        env.set_state(qpos, qvel)

        hist_obs_list = [init_env_obs]
        eval_result = {
            'done': np.array([False]),
            'success': np.array([False]),
            'failure': np.array([False]),
        }
        for step in range(max_traj_len):
            obs = env.get_obs()
            env_obs = obs[:2 * num_obj]
            policy_obs = np.r_[env_obs, inst_encoding]
            action = policy.predict(policy_obs.reshape(1, -1)).flatten()
            next_obs, reward, done, info = env.step(action)
            next_env_obs = next_obs[:2 * num_obj]
            hist_obs_list.append(next_env_obs)
            
            terminal_kwargs = dict(
                insts=[single_inst],
                observations=np.array([next_env_obs]),
                number_of_objects=num_obj,
                goals=np.array([single_goal]),
                level=args.level,
            )
            if 'step_level' in args.level:
                terminal_kwargs['hist_observations'] = np.array(hist_obs_list).reshape(1, 2, 2 * num_obj)
                terminal_kwargs['actions'] = np.array([action]).reshape(1, -1)
            if args.level == 'baseline':
                terminal_kwargs['level'] = 'tau_level'

            eval_result = terminal_fn_with_level(**terminal_kwargs)
            done = eval_result['done']
            if done.item():
                break
   

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_traj_len",
        type=int,
        default=50,
        help="The maximum length of generated trajectories.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
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
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="The path to the trained model.",
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    max_traj_len = args.max_traj_len
    data = np.load(data_dir.joinpath(f'clevr_test_{args.level}.npy'), allow_pickle=True).item()

    instructions, observations, num_obj = itemgetter(
        "instructions", "observations", "number_of_objects")(data)
    goals = data["goals"] if "goals" in data else None
    terminals = data["terminals"] if "terminals" in data else None
    actions = data["actions"] if "actions" in data else None
    success = data["success"] if "success" in data else None
    register_encoder_factory(LlataEncoderFactory)
    env = LlataEnv(
        maximum_episode_steps=max_traj_len,
        action_type='perfect',
        obs_type='order_invariant',
        use_subset_instruction=True,
        num_object=num_obj,
        direct_obs=not args.render,
        use_camera=args.render,
    )
    model_path = args.model_path
    eval_given_level(
        args=args,
        env=env,
        num_obj=num_obj,
        max_traj_len=max_traj_len,
        model_path=model_path,
        instructions=instructions,
        observations=observations,
        terminals=terminals,
        goals=goals,
        actions=actions,
        success=success,
    )

    print(f'Evaluating finished!')
