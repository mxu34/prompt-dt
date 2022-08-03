from xmlrpc.client import TRANSPORT_ERROR
import numpy as np
from jacopinpad.jacopinpad_tools_v0 import env_init, move_to_position, default_controller_init, \
    get_joint_adresses, reset_state
import pickle
from gym import Env
from .settings import TARGET_DEF
from typing import List



class jacopinpad(Env):

    def __init__(self, sketch_length=5, action_repeat=20, noise=0., state_signal='dist', VISUAL=False,
                 max_steps_per_sketch=300):
        self.max_step_per_sketch = max_steps_per_sketch
        self.model, self.sim, self.control_scheme, self.viewer = env_init(render=False)
        #self.RENDER = RENDER
        self.VISUAL = VISUAL
        #if args.RENDER:
        #    self.viewer.render()        
        
        self.img_dims = np.array([112, 112], dtype=np.int)
        self.capture_dims = np.array([202, 212], dtype=np.int)
        if VISUAL:
            self.frame = self.sim.render(self.capture_dims[0], self.capture_dims[1], camera_name="camera_main")
        self.joint_addr = get_joint_adresses(self.sim)
        self.controller = default_controller_init()
        
        self.targ_def = pickle.load(open(TARGET_DEF, 'rb'))
        self.nb_points = len(self.targ_def['joint_angles'])-1
        self.init_pos = self.targ_def['joint_angles'][-1]
        
        self.sketch_length = int(sketch_length)
        
        self.action_repeat = action_repeat
        
        
        #state signal
        #   0: state: [ja, dist]
        #   1: state: [ja, xyz]
        self.state_signal = state_signal        
        
        #noise
        self.noise = noise
        noise_var = np.abs(self.controller.c_low) * self.noise
        self.dart_noise = np.random.uniform(-noise_var, noise_var, size=9)

        self.local_reward = 0

    def seed(self, seed=None):
        self.random = np.random.RandomState(seed)

    def reset(self, sketch=None):
        self.model, self.sim, self.control_scheme, self.viewer = \
            env_init(render=False)
        reset_state(self.sim, self.init_pos)
        move_to_position(self.sim, self.controller, self.init_pos, render=False,viewer=self.viewer)
        self.sketch_idx = 0
        self.ctr = 0
        self.local_score = []
        self.target_list = self.generate_target() if sketch is None else sketch
        self.update()
        self.local_reward = 0
        return self._obs()
                
    def step(self, action):
        terminate = True
        action = np.min([action, self.controller.c_high], axis=0)
        action = np.max([action, np.array(self.controller.c_low)], axis=0)
        if self.VISUAL:
            ja = self.sim.get_state().qpos[self.joint_addr]
            targ_fin = np.zeros(9)
            targ_fin[-3:] = np.array([0.68,0.68,0.68])
            a_fingers = self.controller.act(ja, targ_fin)
            action[-3:] = a_fingers[-3:]

        for i in range(self.action_repeat):
            self.sim.data.ctrl[:] = action
            self.update()
            d_xyz = self.target_xyz - np.copy(self.sim.data.body_xpos[-1])
            
            reward, done, info = self.check(d_xyz, terminate)
            #reward = 1 if np.sum(self.local_score) == len(self.local_score) else 0
            #done = self.sketch_idx == sketch_length
            if done:
                break

        self.ctr += 1
        return self._obs(), reward, done, info
    
    def update(self):
        if self.sketch_idx < self.sketch_length:
            self.targ_num = self.target_list[self.sketch_idx]
        self.target = self.targ_def['joint_angles'][1::2][self.targ_num]
        self.target_xyz = self.targ_def['xyz'][1::2][self.targ_num]
        
        self.control_scheme.execute_step()
        self.sim.step()

    def _obs(self):
        ja = self.sim.get_state().qpos[self.joint_addr]
        if self.state_signal == 'dist':
            dist = np.ravel(self.sim.data.body_xpos[-1] - \
                            np.array(self.targ_def['xyz'])[range(1, self.nb_points + 1, 2)])
            state = np.concatenate((ja,dist))
        elif self.state_signal == 'xyz':
            state = np.concatenate((ja, np.copy(self.sim.data.body_xpos[-1])))
        else:
            raise ValueError('Invalid state signal {}'.format(self.state_signal))
            
        if self.VISUAL:
            frame = self.sim.render(self.capture_dims[1], self.capture_dims[0], camera_name="camera_main")
            state = frame
            
            rem = np.array((self.capture_dims - self.img_dims) / 2, dtype=np.int)
            state = state[rem[0]:-rem[0], 20:-80]
        
        obs = {
            'state': state,
            'sketch': self.target_list,
            'sketch_idx': self.sketch_idx
        }

        obs = state

        return obs
        
    def generate_target(self):
        L = [self.generate_one_target(-1)]
        for i in range(1, self.sketch_length):
            L.append(self.generate_one_target(L[-1]))
        return L
        
    def generate_one_target(self, pre):
        while True:
            targ_num = np.random.randint(0, int(self.nb_points / 2), 1)
            targ_num = targ_num[0]
            if targ_num != pre:
                break
        return targ_num
        
    def check(self, d_xyz, terminate):
        reward = 0
        info = {'log': 'None', 'sparse_r': 0, 'sketch_idx':self.sketch_idx}
        done = False
        if np.sum(np.abs(d_xyz)) < 0.1 and terminate:
            self.ctr = 0
            if self.sketch_idx < self.sketch_length:
                self.sketch_idx += 1
            self.local_score.append(1)
            
            if self.sketch_idx == self.sketch_length:
                reward = 1
                done = True
                info['sparse_r'] = 1
            else:            
                self.targ_num = self.target_list[self.sketch_idx]
                self.local_reward += 1
        
        use_local_reward = False
        if use_local_reward:
            reward = self.local_reward
            done = self.ctr >= self.max_step_per_sketch
        else:
            done = done or self.ctr >= self.max_step_per_sketch

        info['sketch_idx'] = self.sketch_idx
        return reward, done, info   
        
    def render(self):
        return None

class jacopinpad_multi(jacopinpad):
    def __init__(self, tasks: List[dict]):
        sketch_length = len(tasks[0]['goal'])
        super().__init__(sketch_length=sketch_length)
        if tasks is None:
            tasks = [{'goal': [4, 5]}]
        self.tasks = tasks
        self.target_list = tasks[0]['goal']
        self._max_episode_steps = 200
    
    def reset(self):
        self.model, self.sim, self.control_scheme, self.viewer = \
            env_init(render=False)
        reset_state(self.sim, self.init_pos)
        move_to_position(self.sim, self.controller, self.init_pos, render=False,viewer=self.viewer)
        self.sketch_idx = 0
        self.ctr = 0
        self.local_score = []
        self.local_reward = 0
        self.update()
        return self._obs()
