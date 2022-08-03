#!/usr/bin/env python3
"""
Example script for manual control of Jaco arm sim
"""

from tkinter.messagebox import NO
import numpy as np
from pygame import init
# from .jacopinpad_tools_old import env_init,PIcontroller,move_to_position,default_controller_init,get_joint_adresses,reset_state
from .jacopinpad_tools import env_init,PIDcontroller,move_to_position,default_controller_init,get_joint_adresses,reset_state,default_controller_init_PI
import pickle
import matplotlib.pyplot as plt
from .settings import TARGET_DEF
import time

def get_permuted_indexes(nb_points):
    idx = np.random.permutation(nb_points)
    while True:
        # idx = np.random.randint(len(PERMUTATIONS))
        # return PERMUTATIONS[idx]
        if np.sum(idx == np.arange(nb_points)) == nb_points:
            idx = np.random.permutation(nb_points)
        else:
            return idx
        
def collect_data(nb_episodes, len_sketch, given_sketch=None, img_collect=False, render=False,
                 use_dart=True, subsample=20, use_xyz=False, permute=True, 
                 step_per_act=1, tolerance=0.03):
    print('##### CONFIG ######')
    print('USE DART:\t', use_dart)
    print('SUBSAMPLE:\t', subsample)
    print('####################')
    model, sim, control_scheme, viewer = env_init(render=render)
    joint_addr = get_joint_adresses(sim)
    controller = default_controller_init_PI()
    controller_pi = default_controller_init_PI()
    dataset = {'states': [], 'actions': [], 'gt_onsets': [], 'tasks': [], 'images': [],
               'sketch': [], 'local_rewards':[], 'board_cfg':[]}
    targ_def = pickle.load(open(TARGET_DEF, 'rb'))

    init_pos = targ_def['joint_angles'][-1]
    transition_length = 30
    nb_points =10# (len(targ_def['joint_angles'])-1)/2
    lengths = []
    for i in range(nb_episodes):
        if permute:
            perm_idx = get_permuted_indexes(nb_points) # this
        else:
            perm_idx = np.arange(nb_points)
        targets_init = [targ_def['joint_angles'][::2][i] for i in perm_idx]
        targets_fin = [targ_def['joint_angles'][::2][i] for i in perm_idx]
        # targets_fin = [targ_def['joint_angles'][1::2][i] for i in perm_idx] # final is for pressing the pin
        targets_xyz_init = [targ_def['xyz'][::2][i] for i in perm_idx]
        targets_xyz_fin = [targ_def['xyz'][::2][i] for i in perm_idx]
        # targets_xyz_fin = [targ_def['xyz'][1::2][i] for i in perm_idx] # final is for pressing the pin

        init_pos_noise = init_pos + np.random.uniform(-0.1,0.1,9)
        init_pos_noise[0] = init_pos_noise[0] + np.random.uniform(0.0, 0.2)
        init_pos_noise[1] = init_pos_noise[1] + np.random.uniform(-0.2, 0.0)
        init_pos_noise = init_pos_noise if use_dart else init_pos
        reset_state(sim, init_pos_noise)
        start_time = time.time()
        init_sign = move_to_position(sim, controller_pi, init_pos, render=render, viewer=viewer)
        if not init_sign:
            print('Failed to move to position')
            break
        xyz = [];rpy=[];joint_angles = [] # basically a datapoint starts here.
        states =[]
        actions =[]
        onsets =[]
        sketch =[]
        images=[]
        local_rewards = []
        t = 0
        im = 0
        local_reward = 0
        dart_amount = np.random.choice([0, 0.2, 0.3])# if i < int(num_sketches/1.5) else 0.001
        dart_noise_curr = np.random.normal(0, 0.01, size=9)
        noise_presist = np.random.randint(5, 30)
        dart_ctr = noise_presist + 1

        img_dims = np.array([112,112],dtype=np.int)
        capture_dims = np.array([202,212],dtype=np.int)

        targ_num = np.random.randint(0, int(nb_points)) # generate target for the current episode
        if given_sketch is not None:
            targ_num = given_sketch[len(sketch)]

        # plt.ion()
        inter = True
        while True:
            target_int = targets_init[targ_num]
            target_fin = targets_fin[targ_num]
            target_xyz_int = targets_xyz_init[targ_num]
            target_xyz_fin = targets_xyz_fin[targ_num]
            control_scheme.execute_step()
            for _ in range(step_per_act): # subsample here
                sim.step()
            ja = sim.get_state().qpos[joint_addr]
            xyz.append(np.copy(sim.data.body_xpos[-1]))  # xyz of fingers
            rpy.append(np.copy(sim.data.body_xpos[-1]))  # roll_pitch_yaw 0f fingers (in quaternion)
            joint_angles.append(ja)
            if render:
                viewer.render()
                time.sleep(0.01)
            a_int = controller.act(ja, target_int)
            a_fin = controller.act(ja, target_fin)

            if t < transition_length and use_dart:
                a_int[:-3] = a_int[:-3]*(t/float(transition_length))*0.8
                a_fin[:-3] = a_fin[:-3]*(t/float(transition_length))*0.8
                a_int[1] = -1.3
                a_fin[1] = -1.3

            if np.sum(np.abs(xyz[-1] - target_xyz_int)) > 0.05 and inter:
                a = a_int
            else: # TODO: disable the pressing actions
                inter = False
                a = a_fin

            if dart_ctr > noise_presist:
                noise_var = np.abs(controller.c_low)*dart_amount
                noise_var[0] += np.random.uniform(0,0.1)
                dart_noise_curr = np.random.normal(0, noise_var, size=9)
                dart_ctr = 0
            if use_dart:
                sim.data.ctrl[:] = a + dart_noise_curr
            else:
                sim.data.ctrl[:] = a
            dart_ctr += 1

            if im % subsample == 0 and img_collect:
                frame = sim.render(512, 512, camera_name="camera_main")
                #rem = np.array((capture_dims - img_dims)/2,dtype=np.int)
                #frame = frame[rem[0]:-rem[0],20:-80]
                frame = np.flip(np.flip(frame), axis=1)[:400, 56:456, ::-1]
                images.append(frame)
            im += 1

            e = ja - target_fin
            e_xyz = xyz[-1] - target_xyz_fin

            # time to construct the state space.
            # you need the joint angles and the eucledian distance from all possible targets
            if use_xyz:
                dist = xyz[-1]
            else:
                dist = np.ravel(xyz[-1] - targets_xyz_fin)

            # input()
            states.append(np.concatenate((ja,dist)))
            actions.append(np.array(a))
            onsets.append(targ_num)
            local_rewards.append(local_reward)

            t += 1
            term_cond = np.sum(np.abs(e_xyz))
            if t> 1000:
                break

            if term_cond < tolerance:
                t=0
                inter = True
                sketch.append(targ_num)
                local_reward += 1

                # plt.plot(np.array(actions)[:,2])
                # plt.show()
                #
                # ce()
                if len(sketch) == len_sketch:
                    break
                else:
                    while True:
                        targ_num_sug =np.random.randint(0, nb_points)
                        if targ_num_sug != targ_num:
                            break
                    targ_num = targ_num_sug
                    if given_sketch is not None:
                        targ_num = given_sketch[len(sketch)]
                    t = 0

                # first collect like 25 of these The target point collection. Then assign 5 of these to each colour.

        # Collect one episodes
        epi_step_length = int(len(actions) / subsample)
        print("Episode: {}, sketch: {}, length: {}".format(i, sketch, epi_step_length))

        if len(sketch) == len_sketch :
            lengths.append(epi_step_length)
            data_to_save = list(range(1, nb_points + 1, 2)) # only record the press down configuration
            dataset['init_pos'] = init_pos
            dataset['targets_ja'] = [targ_def['joint_angles'][i] for i in data_to_save]
            dataset['targets_xyz'] = [targ_def['xyz'][i] for i in data_to_save]
            dataset['targets_rpy'] = [targ_def['rpy'][i] for i in data_to_save]
            dataset['states'].append(states[::subsample])
            dataset['actions'].append(actions[::subsample])
            dataset['local_rewards'].append(local_rewards[::subsample])
            dataset['gt_onsets'].append(onsets[::subsample])
            dataset['tasks'].append(sketch)
            dataset['images'].append(images)
            dataset['sketch'].append(sketch)
            dataset['board_cfg'].append(perm_idx)
        else:
            print('NOT APPENDED')


    print('avg lengths:', sum(lengths) / len(lengths))
    return dataset


def analyse_data(dataset):
    targ_def = dataset
    TIME_RES = 100
    dataset = {'mean_state': [], 'var_state': [], 'max_state': [], 'min_state': [], 'mean_action': [], 'var_action': [], 'max_action': [], 'min_action': [], 'num_eps': []}
    ep_lengths = [np.size(targ_def['states'][i], axis=0) for i in range(len(targ_def['states']))]
    max_length = np.max(ep_lengths)
    min_length = np.min(ep_lengths)
    mean_length = np.mean(ep_lengths)
    std_length = np.std(ep_lengths)
    num_episodes = len(targ_def['states'])
    num_eps = []

    # timeline plots
    for i in range(max_length):
        M = 0
        S = 0
        dataset['mean_state'].append(M)
        dataset['var_state'].append(S)
        dataset['max_state'].append(-float('inf')*np.ones(np.size(targ_def['states'][0],axis=1)))
        dataset['min_state'].append(float('inf')*np.ones(np.size(targ_def['states'][0],axis=1)))
        num_eps.append(1)
        for j in range(num_episodes):
            try:
                n = num_eps[-1]
                S += ((n * targ_def['states'][j][i] - M) ** 2) / (n * (n + 1))
                M += targ_def['states'][j][i]
                dataset['mean_state'][i] = M / n
                dataset['var_state'][i] = S / (n + 1)
                for k in range(np.size(targ_def['states'][0],axis=1)):
                    dataset['max_state'][i][k] = targ_def['states'][j][i][k] if targ_def['states'][j][i][k] > dataset['max_state'][i][k] else dataset['max_state'][i][k]
                    dataset['min_state'][i][k] = targ_def['states'][j][i][k] if targ_def['states'][j][i][k] < dataset['min_state'][i][k] else dataset['min_state'][i][k]
                num_eps[-1] += 1
            except IndexError:
                pass
            continue
    return dataset


def visualise_data(dataset):
    # plt.figure(1)
    _, ax_full = plt.subplots(9, sharex=True)
    for j in range(9):
        # ax_full = ['ax' + str(k) for k in range(9)]
        # _, ax_full = plt.subplots(9, sharex=True)
        ax_full[j].plot(range(len(dataset['mean_state'])),[dataset['mean_state'][i][j] for i in range(len(dataset['mean_state']))])
        # ax_full[j].fill_between(range(len(dataset['mean_state'])),[dataset['mean_state'][i][j] - (dataset['var_state'][i][j]) ** 0.5 for i in range(len(dataset['mean_state']))],[dataset['mean_state'][i][j] + (dataset['var_state'][i][j]) ** 0.5 for i in range(len(dataset['mean_state']))], alpha=0.4)
        ax_full[j].plot(range(len(dataset['max_state'])),[dataset['max_state'][i][j] for i in range(len(dataset['max_state']))])
        ax_full[j].plot(range(len(dataset['min_state'])),[dataset['min_state'][i][j] for i in range(len(dataset['min_state']))])
    plt.show()

