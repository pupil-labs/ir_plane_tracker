import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import time

import numpy as np
import numpy.typing as npt


@dataclass
class EyeTrackingData:
    timestamp: float
    gaze: tuple[float, float] | None
    scene: npt.NDArray[np.uint8]


@dataclass
class CameraIntrinsics:
    camera_matrix: npt.NDArray[np.float64]
    distortion_coeffs: npt.NDArray[np.float64]


class EyeTrackingSource(ABC):
    @property
    @abstractmethod
    def scene_intrinsics(self) -> CameraIntrinsics:
        pass

    @abstractmethod
    def get_sample(self) -> EyeTrackingData:
        pass

    @abstractmethod
    def close(self):
        pass


class RemoteSource(EyeTrackingSource):
    def __init__(self, auto_discover=False, ip=None, port=None):
        self._device = None
        self._connect(auto_discover, ip, port)

    @property
    def scene_intrinsics(
        self,
    ) -> CameraIntrinsics:
        if self._device is None:
            raise RuntimeError("Not connected to device")
        intrinsics = self._device.get_calibration()
        camera_matrix = intrinsics["scene_camera_matrix"]
        distortion_coeffs = intrinsics["scene_distortion_coefficients"]
        return CameraIntrinsics(camera_matrix, distortion_coeffs)

    def _connect(self, auto_discover=False, ip=None, port=None):
        from pupil_labs.realtime_api.simple import Device, discover_one_device

        assert auto_discover or (ip is not None and port is not None)

        if self._device is not None:
            self._device.close()

        try:
            if auto_discover:
                print("Connecting to device...")
                self._device = discover_one_device()
                print("\rdone")
            else:
                print(f"Connecting to device at {ip}:{port}...")
                self._device = Device(ip, port)
                print("\rdone")
        except Exception as exc:
            print(exc, file=sys.stderr)
            self._device = None

        if self._device is None:
            return None
        else:
            return self._device.phone_ip, self._device.port

    def get_sample(self) -> EyeTrackingData:
        if self._device is None:
            raise RuntimeError("Not connected to device")

        scene_and_gaze = self._device.receive_matched_scene_video_frame_and_gaze(
            timeout_seconds=1 / 10
        )
        if scene_and_gaze is None:
            raise RuntimeError("No scene and gaze data received")

        scene, gaze = scene_and_gaze
        timestamp = gaze.timestamp_unix_seconds
        return EyeTrackingData(timestamp, (gaze.x, gaze.y), scene.bgr_pixels)

    def close(self):
        if self._device is not None:
            self._device.close()


class RecordingSource(EyeTrackingSource):
    def __init__(self, recording_path):
        import pupil_labs.neon_recording as plr

        self._rec = plr.open(recording_path)
        self._rec_data = zip(
            self._rec.scene.ts,
            self._rec.scene.sample(self._rec.scene.ts),
            self._rec.gaze.sample(self._rec.scene.ts),
            strict=False,
        )

    @property
    def scene_intrinsics(
        self,
    ) -> CameraIntrinsics:
        intrinsics = self._rec.calibration
        assert intrinsics is not None
        camera_matrix = intrinsics.scene_camera_matrix
        distortion_coeffs = intrinsics.scene_distortion_coefficients
        return CameraIntrinsics(camera_matrix, distortion_coeffs)

    def get_sample(self) -> EyeTrackingData:
        ts, frame, gaze = next(self._rec_data)
        return EyeTrackingData(ts, (gaze.x, gaze.y), frame.bgr)

    def close(self):
        pass


class USBSource(EyeTrackingSource):
    def __init__(self):
        from common.camera import SceneCam

        self._cam = SceneCam()

    @property
    def scene_intrinsics(
        self,
    ) -> CameraIntrinsics:
        camera_matrix, dist_coeffs = self._cam.get_intrinsics()
        return CameraIntrinsics(camera_matrix, dist_coeffs)

    def get_sample(self) -> EyeTrackingData:
        frame = self._cam.get_frame()
        img = frame.bgr
        timestamp = time()
        gaze = None

        return EyeTrackingData(timestamp, gaze, img)

    def close(self):
        pass
