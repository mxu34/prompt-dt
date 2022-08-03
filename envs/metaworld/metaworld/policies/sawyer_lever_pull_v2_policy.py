import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerLeverPullV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lever_pos': obs[4:7],
            'unused_info': obs[7:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_lever = o_d['lever_pos'] + np.array([.0, -.055, .0])

        if np.linalg.norm(pos_curr[:2] - pos_lever[:2]) > 0.02:
            print('mode 1', pos_lever + np.array([0., 0., -0.1]))
            return pos_lever + np.array([0., 0., -0.1])
        elif abs(pos_curr[2] - pos_lever[2]) > 0.02:
            print('mode 2', pos_lever)
            return pos_lever
        else:
            print('mode 3', pos_lever)
            print('goal_pos', o_d['goal_pos'])
            return o_d['goal_pos'] + np.array([.0, -.04, .0]) # modification: change offset
            # return pos_lever + np.array([.0, .08, .02])