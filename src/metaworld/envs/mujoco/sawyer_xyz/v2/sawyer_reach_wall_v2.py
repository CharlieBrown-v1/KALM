import numpy as np
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerReachWallEnvV2(SawyerXYZEnv):
    """SawyerReachWallEnv.

    Motivation for V2:
        V1 was difficult to solve since the observations didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/17/20) Separated reach from reach-push-pick-place.
        - (6/17/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
    """

    def __init__(self, render_mode=None, camera_name=None, camera_id=None):
        goal_low = (-0.05, 0.85, 0.05)
        goal_high = (0.05, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.6, 0.015)
        obj_high = (0.05, 0.65, 0.015)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            render_mode=render_mode,
            camera_name=camera_name,
            camera_id=camera_id,
        )

        self.init_config = {
            "obj_init_angle": 0.3,
            "obj_init_pos": np.array([0, 0.6, 0.02]),
            "hand_init_pos": np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.05, 0.8, 0.2])

        self.obj_init_angle = self.init_config["obj_init_angle"]
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_reach_wall_v2.xml")

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        reward, tcp_to_object, in_place = self.compute_reward(action, obs)
        success = float(tcp_to_object <= 0.05)

        info = {
            "success": success,
            "near_object": 0.0,
            "grasp_success": 0.0,
            "grasp_reward": 0.0,
            "in_place_reward": in_place,
            "obj_to_target": tcp_to_object,
            "unscaled_reward": reward,
        }

        return reward, info

    def _get_pos_objects(self):
        obj_xpos = self.data.geom("objGeom").xpos
        wall_xpos = self.get_body_com("wall")
        return np.r_[obj_xpos, wall_xpos]

    def _get_quat_objects(self):
        obj_quat = Rotation.from_matrix(self.data.geom("objGeom").xmat.reshape(3, 3)).as_quat()
        wall_quat = self.data.body("wall").xquat
        return np.r_[obj_quat, wall_quat]

    def render_reset(self, obs_info: dict):
        hand_state = obs_info['hand_state'].copy()
        goal_pos = obs_info['goal_pos'].copy()
        obj_pos = obs_info['obj_pos'].copy()

        self.render_reset_hand(hand_state=hand_state)
        self.obj_init_angle = self.init_config["obj_init_angle"]
        self._target_pos = goal_pos
        self.obj_init_pos = obj_pos

        self._set_obj_xyz(self.obj_init_pos)
        import mujoco
        mujoco.mj_forward(self.model, self.data)

        for site in self._target_site_config:
            self._set_pos_site(*site)

        return self._get_obs()

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config["obj_init_angle"]

        goal_pos = self._get_state_rand_vec()
        self._target_pos = goal_pos[3:]
        while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
        self._target_pos = goal_pos[-3:]
        self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)

        return self._get_obs()

    def compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        # obj = obs[4:7]
        # tcp_opened = obs[3]
        target = self._target_pos

        tcp_to_target = np.linalg.norm(tcp - target)
        # obj_to_target = np.linalg.norm(obj - target)

        in_place_margin = np.linalg.norm(self.hand_init_pos - target)
        in_place = reward_utils.tolerance(
            tcp_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=in_place_margin,
            sigmoid="long_tail",
        )

        return [10 * in_place, tcp_to_target, in_place]
