import cv2
import qdarktheme
from common import eye_tracking_sources
from debug_app.app_window import AppWindow
from debug_app.tracker import Tracker
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import (
    TrackerLineAndDotsParams,
)


class DebugApp(QApplication):
    data_changed = Signal(object, object, object)

    def __init__(self):
        super().__init__()
        self.setApplicationDisplayName("Debug App")
        qdarktheme.setup_theme()
        self.eye_tracking_source = eye_tracking_sources.RecordingSource(
            # recording_path="/home/marc/pupil_labs/IR_plane_tracker/examples/offline_recording/data/indoor4",  # noqa: E501
            recording_path="/home/marc/Downloads/Native Recording Data (8)/2025-10-22_14-27-22-9a4d2c6d"  # noqa: E501
        )
        # self.eye_tracking_source = eye_tracking_sources.USBSource()

        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = self.eye_tracking_source.scene_intrinsics.distortion_coeffs
        self.tracker = Tracker(self.camera_matrix)

        self.main_window = AppWindow()

        # Connections
        self.data_changed.connect(self.main_window.set_data)
        self.main_window.make_connections(self.tracker)
        self.main_window.playback_toggled.connect(
            lambda: setattr(self, "playback", not self.playback)
        )
        self.main_window.next_frame_clicked.connect(self.update_data)

        # Data initialization
        self.params = TrackerLineAndDotsParams.from_json(
            "resources/neon_artificial.json"
        )
        self.tracker.set_params(self.params)
        self.playback = False
        self.last_data = None

        # Start
        self.poll_timer = QTimer()
        self.poll_timer.setInterval(int(1000 / 30))
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start()

        self.main_window.showMaximized()

    def update_data(self):
        eye_tracking_data = self.eye_tracking_source.get_sample()
        eye_tracking_data.scene = cv2.undistort(
            eye_tracking_data.scene, self.camera_matrix, self.dist_coeffs
        )  # type: ignore
        plane_localization = self.tracker(eye_tracking_data.scene)
        # self.tracker.debug.visualize()
        # cv2.waitKey(1)
        self.last_data = (eye_tracking_data, plane_localization, self.tracker.debug)

    def poll(self):
        if self.playback or self.last_data is None:
            self.update_data()

        assert self.last_data is not None
        self.data_changed.emit(*self.last_data)

    def exec(self):
        ret = super().exec()
        self.eye_tracking_source.close()
        return ret


def run():
    app = DebugApp()
    app.exec()


if __name__ == "__main__":
    run()
