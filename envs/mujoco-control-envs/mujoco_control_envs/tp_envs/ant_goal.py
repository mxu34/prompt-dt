import numpy as np

from src.tp_envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, **kwargs):
        super().__init__(task, n_tasks, **kwargs)
        
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        goal_marker_idx = self.sim.model.site_name2id('goal')
        self.data.site_xpos[goal_marker_idx,:2] = self._goal
        self.data.site_xpos[goal_marker_idx,-1] = 1
        
        
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) + 4.0 # make it happy, not suicidal
        ctrl_cost = 0.5 * 1e-2 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.05
        
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )
#        
#    def step(self, action):
#        self.forward_dynamics(action)
#        com = self.get_body_com("torso")
#        # ref_x = x + self._init_torso_x
#        goal_reward = -np.sum(np.abs(com[:2] - self._goal_pos)) + 4.0 # make it happy, not suicidal
#        lb, ub = self.action_bounds
#        scaling = (ub - lb) * 0.5
#        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
#        contact_cost = 0.5 * 1e-3 * np.sum(
#            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
#        survive_reward = 0.05
#        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
#        state = self._state
#        notdone = np.isfinite(state).all() \
#            and state[2] >= 0.2 and state[2] <= 1.0
#        done = not notdone
#        ob = self.get_current_obs()
#        return Step(ob, float(reward), done)

    def sample_tasks(self, num_tasks):
#        a = np.random.random(num_tasks) * 2 * np.pi
#        r = 3 * np.random.random(num_tasks) ** 0.5
#        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        goals = np.random.uniform(-3.0, 3.0, (num_tasks, 2, ))
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
