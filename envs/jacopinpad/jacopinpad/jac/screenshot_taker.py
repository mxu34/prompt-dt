import os
from concurrent.futures import ThreadPoolExecutor
from math import inf
from time import perf_counter

import imageio
from mujoco_py import MjSim


class ScreenshotTaker:

    def __init__(self,
                 sim: MjSim,
                 camera_name: str=None,
                 screenshot_dir: str='/tmp/mujoco_screens'):
        self._sim = sim
        self._camera_name = camera_name
        self._screenshot_dir = screenshot_dir

        self._screenshots = []
        self._image_name = "image_%04d.png"

        self._time_last_screen = -inf
        self._frames_per_second = 5

    def set_screenshot_dir(self, screenshot_dir: str):
        self._screenshot_dir = screenshot_dir

    def take_screenshot(self):
        if self._new_screenshot_needed():
            screenshot = self._take_screenshot_inner()
            self._screenshots.append(screenshot)
            self._time_last_screen = perf_counter()
        return self._screenshots[-1]

    def _take_screenshot_inner(self):
        return self._sim.render(width=640, height=480, camera_name=self._camera_name)

    def save_screenshots(self):
        os.makedirs(self._screenshot_dir, exist_ok=True)
        path_template = os.path.join(self._screenshot_dir, self._image_name)
        save_thread_pool = ThreadPoolExecutor(30)

        try:
            for index, image in enumerate(self._screenshots):
                image_path = path_template % index
                save_thread_pool.submit(self._save_screenshot, image, image_path)

            self._screenshots.clear()
        finally:
            save_thread_pool.shutdown(wait=False)  # Don't block while saving the images

    @staticmethod
    def _save_screenshot(image, path):
        imageio.imwrite(path, image)

    def _new_screenshot_needed(self):
        screenshots_empty = (len(self._screenshots) == 0)
        last_screenshot_stale = (perf_counter() - self._time_last_screen > 1 / self._frames_per_second)

        return screenshots_empty or last_screenshot_stale
