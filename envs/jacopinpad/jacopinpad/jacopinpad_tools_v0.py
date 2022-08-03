#!/usr/bin/env python3
"""
Example script for manual control of Jaco arm sim
"""
import os
import numpy as np
from mujoco_py import load_model_from_path, MjSim
import jacopinpad.jac as jac
from jacopinpad.jac.custom_viewer import CustomMjViewer, CustomControlScheme
from collections import deque

JACO_MODEL_PATH = os.path.dirname(jac.__file__)+'/model/jaco_pinpad.xml'


def get_joint_adresses(sim):
    joint_names = ['jaco_joint_1',
                   'jaco_joint_2',
                   'jaco_joint_3',
                   'jaco_joint_4',
                   'jaco_joint_5',
                   'jaco_joint_6',
                   'jaco_joint_finger_1',
                   'jaco_joint_finger_2',
                   'jaco_joint_finger_3']

    joint_addr = [sim.model.get_joint_qpos_addr(i) for i in joint_names]
    return joint_addr


def move_to_position(sim,controller,target,tol = 0.03,render=False,viewer=None):
    joint_addr = get_joint_adresses(sim)
    while True:
        sim.step()

        ja = sim.get_state().qpos[joint_addr]
        if render:
            viewer.render()
        # ice()
        a =controller.act(ja, target)
        sim.data.ctrl[:] = a
        e = ja- target
        if np.sum(np.abs(e))<tol:
            break
            # first collect like 25 of these The target point collection. Then assign 5 of these to each colour.

def env_init(render = False):
    camera_name = 'camera_main'
    model = load_model_from_path(JACO_MODEL_PATH)
    sim = MjSim(model)
    control_scheme = CustomControlScheme(sim.data.ctrl)
    if render:

        viewer = CustomMjViewer(sim,
                                     control_scheme,
                                     camera_name=camera_name)
    else:
        viewer= False
    return model,sim,control_scheme,viewer

def reset_state(sim,start_state = np.array([-0.51385716, -0.24882012, -0.52184923, -39.85733113 + 37.69911184, 26.983593 - 25.13274123 , -0.73750337, 0.69813948, 0.69306134, 0.69024486])):

    sim_state = sim.get_state()
    i=0
    joint_addr = [sim.model.get_joint_qpos_addr("jaco_joint_" + str(j)) for j in range(1, 7)]
    joint_addr.extend([sim.model.get_joint_qpos_addr("jaco_joint_finger_" + str(j)) for j in range(1, 4)])
    for addr in joint_addr:
        sim_state.qpos[addr] = start_state[i]
        i+=1
    sim.set_state(sim_state)
    sim.forward()
    return

class PIcontroller(object):
    def __init__(self,KP,KI,c_high,c_low = 0):
        self.KP = KP
        self.KI = KI
        self.c_high = c_high
        self.c_low = c_low
        self.accum_error  = deque([],200)
    def act(self,current,target):
        error=np.zeros(len(target))
        for i in range(len(target)):
            error[i] = (target[i] - current[i]) % (2*np.pi) if ((target[i] - current[i]) % (2*np.pi)) < np.pi \
                                                  else ((target[i] - current[i]) % (2*np.pi)) - (2*np.pi)
        action = self.KP*error +self.KI*np.sum(self.accum_error) if len(list(self.accum_error))>0 else self.KP*error
        self.accum_error.append(error)
        action = np.min([action, self.c_high], axis=0)
        action = np.max([action, np.array(self.c_low)], axis=0)
        return action



def p_controller(K,current,target,c_high):
    action = K*(target - current)
    action = np.min([action, c_high], axis=0)

    return action


def default_controller_init():
    KP = np.array([0.5, 2.5, 0.9, 0.9, 0.9, 1.,1.,1.,1.]) *8.5
    KI = np.array([0.001, 0.001, 0.001, 0.005, 0.005, 0.001,0.001,0.001,0.001])*0.1

    c_high = np.array([0.07, 0.6, 0.07, 0.05, 0.07, 0.065,0.07,0.07,0.07]) * 1.0
    c_low = [-c_high[0], -c_high[1] - 0.4, -c_high[2] - 0.064, -c_high[3], -c_high[4], -c_high[5],-c_high[6],-c_high[7],-c_high[8]]
    controller = PIcontroller(KP, KI, c_high, c_low)

    return controller