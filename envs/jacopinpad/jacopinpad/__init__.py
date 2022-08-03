try:
    import free_mjc
except ImportError:
    print('Use MOJUCO key')

from gym.envs.registration import register
from .jacopinpad_collect import *
from jacopinpad.jacopinpad_gym import jacopinpad

register(
    id='jacopinpad-v0',
    entry_point='jacopinpad.jacopinpad_gym:jacopinpad',
)

def load(config):
    cls_name = config.world.name
    try:
        cls = globals()[cls_name]
        return cls(config)
    except KeyError:
        raise Exception("No such world: {}".format(cls_name))

import gym
make = gym.make
