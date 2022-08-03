from rand_param_envs.base import MetaEnv
from rand_param_envs.gym.envs.registration import register

register(
    id='Walker2DRandParams-v0',
    entry_point='rand_param_envs.walker2d_rand_params:Walker2DRandParamsEnv',
)

register(
    id='HopperRandParams-v0',
    entry_point='rand_param_envs.hopper_rand_params:HopperRandParamsEnv',
)

register(
    id='PR2Env-v0',
    entry_point='rand_param_envs.pr2_env_reach:PR2Env',
)


