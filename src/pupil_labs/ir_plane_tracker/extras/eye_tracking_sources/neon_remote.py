from functools import cached_property

import numpy as np

from pupil_labs.camera import Camera
from pupil_labs.realtime_api.simple import Device

from . import (
    EyeTrackingData,
    EyeTrackingSource,
)


class NeonRemote(EyeTrackingSource):
    def __init__(self, ip_address: str, port: int):
        super().__init__()
        try:
            device = Device(ip_address, port, start_streaming_by_default=True)
        except Exception:
            device = None

        if device is None:
            print(f"Failed to connect to device at {ip_address}:{port}.")
            raise RuntimeError("Could not connect to Neon Remote device.")

        data = None
        print(f"Attempting to receive data from device {device}...")
        counter = 0
        while data is None:
            counter += 1
            print(f"  Attempt {counter}...")
            data = device.receive_matched_scene_video_frame_and_gaze(
                timeout_seconds=1.0
            )
        print("  Success.")
        self._device = device

    @property
    def address(self) -> str:
        return self._device.address

    @property
    def port(self) -> int:
        return self._device.port

    @cached_property
    def scene_intrinsics(self) -> Camera:
        intrinsics = self._device.get_calibration()
        return Camera(
            pixel_width=1600,
            pixel_height=1200,
            camera_matrix=intrinsics.scene_camera_matrix,
            distortion_coefficients=intrinsics.scene_distortion_coefficients,
        )

    def get_sample(self) -> EyeTrackingData:
        scene_and_gaze = self._device.receive_matched_scene_video_frame_and_gaze(
            timeout_seconds=1 / 5
        )
        if scene_and_gaze is None:
            raise RuntimeError("No data received from Neon Remote device.")

        scene, gaze = scene_and_gaze
        gaze = np.array([gaze.x, gaze.y], dtype=np.float64)
        time = scene.timestamp_unix_ns
        return EyeTrackingData(
            time=time,
            gaze_scene_distorted=gaze,
            scene_image_distorted=scene.bgr_pixels,
            intrinsics=self.scene_intrinsics,
            eye_image=None,
        )

    def close(self):
        self._device.close()
