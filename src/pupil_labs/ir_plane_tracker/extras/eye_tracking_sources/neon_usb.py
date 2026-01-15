import contextlib
import importlib
import queue
import time
from threading import Event, Thread

import numpy as np

from pupil_labs.camera import Camera
from pupil_labs.neon_usb import (
    EyeCamera,
    Frame,
    SceneCamera,
)

from . import (
    EyeTrackingData,
    EyeTrackingSource,
)


def get_all_items(q: queue.Queue[Frame]) -> list[Frame]:
    """Retrieve all items from a queue and always at least one."""
    items = []
    # Need to get at least one item
    # Otherwise the queue might be spammed with requests
    items.append(q.get())
    while True:
        try:
            items.append(q.get_nowait())
        except queue.Empty:
            break
    return items


def image_receiver(
    CameraClass: type[SceneCamera | EyeCamera],
    intrinsics_q: queue.Queue[Camera] | None,
    output_q: queue.Queue[Frame],
    start_event: Event,
    stop_event: Event,
    wait_event: Event | None = None,
) -> None:
    cam = CameraClass()
    if intrinsics_q is not None:
        intrinsics_q.put(cam.get_intrinsics())
    start_event.set()
    if wait_event is not None:
        wait_event.wait()
    while True:
        if stop_event.is_set():
            cam.close()
            break
        image = cam.get_frame()
        with contextlib.suppress(queue.Full):
            output_q.put_nowait(image)


class LowExposureSceneCamera(SceneCamera):
    def __init__(self):
        super().__init__()
        self.exposure = 120  # Set low exposure to reduce motion blur


class NeonUSB(EyeTrackingSource):
    def __init__(self, compute_gaze: bool = True):
        super().__init__()

        self._init_scene_camera()
        self._init_eye_camera()

        if compute_gaze:
            self._pipeline = self._init_pipeline()
        else:
            self._pipeline = None

    def _init_scene_camera(self):
        print("Connecting to scene cam...", end="", flush=True)
        scene_start_event = Event()
        self.scene_stop_event = Event()
        scene_intrinsics_q = queue.Queue[Camera](maxsize=1)
        self.scene_q = queue.Queue[Frame](maxsize=10)
        scene_thread = Thread(
            target=image_receiver,
            args=(
                LowExposureSceneCamera,
                scene_intrinsics_q,
                self.scene_q,
                scene_start_event,
                self.scene_stop_event,
                None,
            ),
        )
        scene_thread.start()
        intrinsics = scene_intrinsics_q.get()
        self.scene_intrinsics = Camera(
            1600, 1200, intrinsics.camera_matrix, intrinsics.distortion_coefficients
        )
        scene_start_event.wait()
        print("Done.")

    def _init_eye_camera(self):
        print("Connecting to eye cam...", end="", flush=True)

        eye_start_event = Event()
        self.eye_stop_event = Event()

        self.eye_q = queue.Queue[Frame](maxsize=10)
        eye_thread = Thread(
            target=image_receiver,
            args=(
                EyeCamera,
                None,
                self.eye_q,
                eye_start_event,
                self.eye_stop_event,
                None,
            ),
        )
        eye_thread.start()
        eye_start_event.wait()
        print("Done.")

    def _init_pipeline(self):
        import os

        from dotenv import load_dotenv

        load_dotenv()

        if "NEON_PIPELINE_MODULE_NAME" not in os.environ:
            raise ValueError("Environment variable NEON_PIPELINE_MODULE_NAME not set.")

        neon_pipeline_module_name = os.environ["NEON_PIPELINE_MODULE_NAME"]
        neon_pipeline_class_name = os.environ["NEON_PIPELINE_CLASS_NAME"]
        neon_pipeline_version = os.environ["NEON_PIPELINE_VERSION"]
        neon_pipeline_module = importlib.import_module(neon_pipeline_module_name)
        neon_pipeline_class = getattr(neon_pipeline_module, neon_pipeline_class_name)

        print("Setting up pipeline...", end="", flush=True)
        pipeline = neon_pipeline_class(
            pipeline_version=neon_pipeline_version,
            camera_matrix=self.scene_intrinsics.camera_matrix,
            dist_coefs=self.scene_intrinsics.distortion_coefficients,
            batch_size=6,
        )
        print("Done.")
        return pipeline

    def get_sample(self) -> EyeTrackingData:
        ts = int(time.time() * 1e9)
        scene_frame = get_all_items(self.scene_q)[-1]
        eye_frames = get_all_items(self.eye_q)[-6:]

        if self._pipeline is None:
            gaze = np.array([0, 0], dtype=np.float64)
        else:
            gaze = self._pipeline(eye_frames)
            if gaze.ndim == 2:
                gaze = np.mean(gaze, axis=0)
        data = EyeTrackingData(
            time=ts,
            gaze_scene_distorted=gaze,
            scene_image_distorted=scene_frame.bgr,
            intrinsics=self.scene_intrinsics,
            eye_image=eye_frames[-1].gray,
        )
        return data

    def close(self):
        self.scene_stop_event.set()
        if self._pipeline is not None:
            self.eye_stop_event.set()
