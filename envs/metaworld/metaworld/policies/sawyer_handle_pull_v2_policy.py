import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerHandlePullV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'handle_pos': obs[4:7],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_handle = o_d['handle_pos'] + np.array([0, -0.04, 0])

        if np.linalg.norm(pos_curr[:2] - pos_handle[:2]) > 0.02:
            print('mode 1', pos_handle)
            return pos_handle
        if abs(pos_curr[2] - pos_handle[2]) > 0.02:
            print('mode 2', pos_handle[2])
            return pos_handle[2]
        print('mode 3', pos_handle + np.array([0., 0., 0.1]))
        return pos_handle + np.array([0., 0., 0.1])


        # pos_button = o_d['handle_pos'] + np.array([.0, -.02, .0])
        # if abs(pos_curr[0] - pos_button[0]) > 0.04:
        #     print('mode 1')
        #     return pos_button + np.array([0., 0., 0.2])
        # elif abs(pos_curr[2] - pos_button[2]) > 0.03:
        #     print('mode 2')
        #     return pos_button + np.array([.0, -.1, -.01])
        # elif abs(pos_curr[1] - pos_button[1]) > .01:
        #     print('mode 3')
        #     return np.array([pos_button[0], pos_button[1] + .04, pos_curr[2]])
        # else:
        #     print('mode 4')
        #     return np.array([pos_button[0], pos_button[1] + .04, pos_curr[2]]) 

    @staticmethod
    def _grab_effort(o_d):
        return 1.
