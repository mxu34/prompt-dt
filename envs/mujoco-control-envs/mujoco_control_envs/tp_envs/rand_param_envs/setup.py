from setuptools import setup, find_packages

setup(name="rand_param_envs",
      version='0.1',
      description='Environments with random model parameters, using gym 0.7.4 and mujoco-py 0.5.7',
      url='https://github.com/dennisl88/rand_param_envs',
      author='Dennis Lee, Ignasi Clavera, Jonas Rothfuss',
      author_email='dennisl88@berkeley.edu',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('rand_param_envs')],
      install_requires=[
        'numpy>=1.10.4',
        'requests>=2.0',
        'six',
        'pyglet>=1.2.0',
        'scipy',
        'PyOpenGL>=3.1.0',
        'nose>=1.3.7'
      ],
      package_data={'rand_param_envs.gym': [
        'envs/mujoco/assets/*.xml',
        'envs/mujoco/assets/meshes/*',
        'envs/classic_control/assets/*.png',
        'envs/robotics/assets/LICENSE.md',
        'envs/robotics/assets/fetch/*.xml',
        'envs/robotics/assets/hand/*.xml',
        'envs/robotics/assets/stls/fetch/*.stl',
        'envs/robotics/assets/stls/hand/*.stl',
        'envs/robotics/assets/textures/*.png']
      },
      zip_safe=False)
