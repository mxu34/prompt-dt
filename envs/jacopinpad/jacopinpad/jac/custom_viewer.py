#!/usr/bin/env python3
"""
Viewer for MuJoCo which supports taking keyboard input to control the simulation
"""
from concurrent.futures import ThreadPoolExecutor

from .screenshot_taker import ScreenshotTaker
from mujoco_py import MjViewer
from mujoco_py.generated import const

import glfw  # This line must come after importing MjViewer


class CustomMjViewer(MjViewer):

    def __init__(self,
                 sim,
                 control_scheme,
                 camera_name: str=None,
                 screenshot_dir: str='/tmp/mujoco_screens',
                 irl_model =None):
        super().__init__(sim)

        self._control_scheme = control_scheme
        self._control_active = False

        self._irl_model = irl_model
        self._record_images = False

        self._screenshot_taker = ScreenshotTaker(sim, camera_name, screenshot_dir)

        self._update_reward()

    def render(self):
        if self._record_images:
            self._screenshot_taker.take_screenshot()

        super().render()

    def key_callback(self, window, key, scancode, action, mods):
        self._set_control_flag(key, action)

        if key == glfw.KEY_SLASH and action == glfw.RELEASE:
            if self._record_images:
                self._screenshot_taker.save_screenshots()
            self._record_images = not self._record_images

        if self._control_active:
            self._send_input_to_control_scheme(key, action)
        else:
            super().key_callback(window, key, scancode, action, mods)

    def _set_control_flag(self, key, action):
        if key == glfw.KEY_LEFT_CONTROL:
            if action == glfw.PRESS:
                self._control_active = True
            if action == glfw.RELEASE:
                self._control_active = False
                self._control_scheme.deactivate()

    def _send_input_to_control_scheme(self, key, action):
        if action == glfw.PRESS:
            self._control_scheme.key_down(key)
        if action == glfw.RELEASE:
            self._control_scheme.key_up(key)

    def _create_full_overlay(self):
        super()._create_full_overlay()

        if self._record_images:
            self.add_overlay(const.GRID_TOPLEFT,
                             "[/] Stop image sequence capture", "")
        else:
            self.add_overlay(const.GRID_TOPLEFT, "[/] Start image sequence capture", "")

        if self._irl_model is not None:
            self.add_overlay(const.GRID_TOPLEFT, "Current reward: {:.2f}".format(self._current_reward), "")

    def _update_reward(self):
        if self._irl_model is not None:
            executor = ThreadPoolExecutor(1)
            try:
                image = self._screenshot_taker.take_screenshot()
                self._current_reward = self._irl_model.obtain_reward_from_image(image)
                executor.submit(self._update_reward)
            finally:
                executor.shutdown(wait=False)  # Don't block


class ControlScheme:

    def __init__(self):
        self._active_keys = set()
        self._key_down_callbacks = {}
        self._key_up_callbacks = {}
        self._while_key_down_actions = {}

    def add_key_up_callback(self, key, callback):
        self._key_up_callbacks[key] = callback

    def add_key_down_callback(self, key, callback):
        self._key_down_callbacks[key] = callback

    def add_while_key_down_action(self, key, callback):
        self._while_key_down_actions[key] = callback

    def deactivate(self):
        self._active_keys.clear()
        for callback in self._key_up_callbacks.values():
            callback()

    def key_down(self, key):
        self._active_keys.add(key)
        if key in self._key_down_callbacks:
            self._key_down_callbacks[key]()

    def key_up(self, key):
        if key in self._active_keys:
            self._active_keys.remove(key)
        if key in self._key_up_callbacks:
            self._key_up_callbacks[key]()

    def execute_step(self):
        for key in self._active_keys & self._while_key_down_actions.keys():
            print(self._active_keys)
            self._while_key_down_actions[key]()


class CustomControlScheme(ControlScheme):

    _BASE = 0
    _SHOULDER = 1
    _ELBOW = 2
    _UPPER_WRIST = 3
    _LOWER_WRIST = 4
    _HAND = 5
    _FINGERS = [6, 7, 8]

    def __init__(self, controls):
        super().__init__()

        # Base rotation controls
        self._map_key_to_action(glfw.KEY_Q, self._BASE, 0.1)
        self._map_key_to_action(glfw.KEY_E, self._BASE, -0.1)

        # Shoulder controls
        self._map_key_to_action(glfw.KEY_W, self._SHOULDER, -1)
        self._map_key_to_action(glfw.KEY_S, self._SHOULDER, 0.75)

        # Elbow controls
        self._map_key_to_action(glfw.KEY_A, self._ELBOW, 0.1)
        self._map_key_to_action(glfw.KEY_D, self._ELBOW, -0.1)

        # Upper wrist controls
        self._map_key_to_action(glfw.KEY_Y, self._UPPER_WRIST, 0.1)
        self._map_key_to_action(glfw.KEY_G, self._UPPER_WRIST, -0.1)

        # Lower wrist controls
        self._map_key_to_action(glfw.KEY_H, self._LOWER_WRIST, 0.1)
        self._map_key_to_action(glfw.KEY_B, self._LOWER_WRIST, -0.1)

        # Hand rotation contols
        self._map_key_to_action(glfw.KEY_T, self._HAND, 0.1)
        self._map_key_to_action(glfw.KEY_F, self._HAND, -0.1)

        # Finger controls
        self.add_while_key_down_action(glfw.KEY_U, self._move_fingers_callback(0.1))
        self.add_key_up_callback(glfw.KEY_U, self._move_fingers_callback(0))
        self.add_while_key_down_action(glfw.KEY_I, self._move_fingers_callback(-0.1))
        self.add_key_up_callback(glfw.KEY_I, self._move_fingers_callback(0))

        self._controls = controls

    def _map_key_to_action(self, key, actuator, value):
        self.add_while_key_down_action(key, self._rotate(actuator, value))
        self.add_key_up_callback(key, self._rotate(actuator, 0))

    def _rotate(self, control_index, value):
        def callback():
            self._controls[control_index] = value
        return callback

    def _move_fingers_callback(self, value):
        def callback():
            for index in self._FINGERS:
                self._controls[index] = value
        return callback
