import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDialTurnV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_gripper_open': obs[3],
            'dial_pos': obs[4:7],
            'extra_info': obs[7:-3],
            'target_pos': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_pow'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        hand_pos = o_d['hand_pos']
        dial_pos = o_d['dial_pos'] + np.array([0.05, 0.02, 0.09])

        if np.linalg.norm(hand_pos[:2] - dial_pos[:2]) > 0.2:
            print('mode 1')
            return np.array([*dial_pos[:2], 0.2])
        elif abs(hand_pos[2] - dial_pos[2]) > 0.02:
            print('mode 2')
            return dial_pos
        elif hand_pos[1] < 0.75:
            print('mode 3', dial_pos + np.array([-.05, .005, .0]) )
            return dial_pos + np.array([-.05, .005, .0])
        else:
            print('mode 4')
            return dial_pos + np.array([-.02, .005, .0])
