import cv2
from common import eye_tracking_sources
from debug_app.app_window import AppWindow
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import (
    TrackerLineAndDots,
    TrackerLineAndDotsParams,
)


class DebugApp(QApplication):
    on_data_update = Signal(object, object, object)

    def __init__(self):
        super().__init__()
        self.setApplicationDisplayName("Debug App")
        self.eye_tracking_source = eye_tracking_sources.RecordingSource(
            recording_path="/home/marc/pupil_labs/IR_plane_tracker/examples/offline_recording/data/indoor4"
        )
        # self.eye_tracking_source = eye_tracking_sources.USBSource()

        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = self.eye_tracking_source.scene_intrinsics.distortion_coeffs
        self.params = TrackerLineAndDotsParams.from_json("resources/neon_monitor.json")
        self.tracker = TrackerLineAndDots(
            camera_matrix=self.camera_matrix,
            dist_coeffs=None,
            params=self.params,
        )

        self.main_window = AppWindow()
        self.on_data_update.connect(self.main_window.on_data_update)

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(int(1000 / 30))
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start()

        self.main_window.showMaximized()

    def poll(self):
        eye_tracking_data = self.eye_tracking_source.get_sample()
        eye_tracking_data.scene = cv2.undistort(
            eye_tracking_data.scene, self.camera_matrix, self.dist_coeffs
        )  # type: ignore
        plane_localization = self.tracker(eye_tracking_data.scene)
        self.on_data_update.emit(
            eye_tracking_data, plane_localization, self.tracker.debug
        )

    def exec(self):
        ret = super().exec()
        self.eye_tracking_source.close()
        return ret


def run():
    app = DebugApp()
    app.exec()


if __name__ == "__main__":
    run()
