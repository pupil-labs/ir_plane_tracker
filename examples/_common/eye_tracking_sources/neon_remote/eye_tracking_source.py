from pupil_labs.realtime_api.simple import Device

from .. import (
    CameraIntrinsics,
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
        intrinsics = device.get_calibration()
        self._scene_intrinsics = CameraIntrinsics(
            intrinsics["scene_camera_matrix"],
            intrinsics["scene_distortion_coefficients"],
        )

    @property
    def address(self) -> str:
        return self._device.address

    @property
    def port(self) -> int:
        return self._device.port

    @property
    def scene_intrinsics(self) -> CameraIntrinsics:
        return self._scene_intrinsics

    def get_sample(self) -> EyeTrackingData:
        scene_and_gaze = self._device.receive_matched_scene_video_frame_and_gaze(
            timeout_seconds=1 / 5
        )
        if scene_and_gaze is None:
            return EyeTrackingData(time=0, gaze=None, scene=None)

        scene, gaze = scene_and_gaze
        time = gaze.timestamp_unix_seconds
        return EyeTrackingData(time, (gaze.x, gaze.y), scene.bgr_pixels)

    def close(self):
        self._device.close()
