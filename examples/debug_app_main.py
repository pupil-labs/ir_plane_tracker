import click
import cv2
import numpy as np
import qdarktheme
from common import eye_tracking_sources
from debug_app.app_window import AppWindow
from debug_app.tracker import Tracker
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import FeatureOverlay, TrackerParams


class DebugApp(QApplication):
    data_changed = Signal(object, object, object)

    def __init__(self, params_path, marker_config_path, feature_overlay=False):
        super().__init__()
        self.setApplicationDisplayName("Debug App")
        qdarktheme.setup_theme()
        # self.eye_tracking_source = eye_tracking_sources.RecordingSource(
        #     # recording_path="/home/marc/pupil_labs/IR_plane_tracker/examples/offline_recording/data/indoor4",  # noqa: E501
        #     recording_path="/home/marc/Downloads/Native Recording Data (8)/2025-10-22_14-27-22-9a4d2c6d"  # noqa: E501
        # )
        self.eye_tracking_source = eye_tracking_sources.USBSource()

        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = self.eye_tracking_source.scene_intrinsics.distortion_coeffs
        self.tracker = Tracker(self.camera_matrix)

        self.main_window = AppWindow()

        if feature_overlay:
            self.feature_overlay = FeatureOverlay(self.screens()[-1])
            self.feature_overlay.toggle_visibility()

        # Connections
        self.data_changed.connect(self.main_window.set_data)
        self.main_window.make_connections(self.tracker)
        self.main_window.playback_toggled.connect(
            lambda: setattr(self, "playback", not self.playback)
        )
        self.main_window.next_frame_clicked.connect(self.update_data)

        # Data initialization
        self.params = TrackerParams.from_json(params_path, marker_config_path)
        if feature_overlay:
            feature_values_px = self.feature_overlay.feature_values_px
            (
                self.params.top_pos,
                self.params.right_pos,
                self.params.bottom_pos,
                self.params.left_pos,
            ) = feature_values_px.reshape(-1, 2)[3::4, :]
            self.params.norm_line_points = np.concatenate([
                [0],
                np.cumsum(np.diff(feature_values_px[:4, 0])),
            ])
            self.params.plane_width = float(self.feature_overlay.screen_size_px[0])
            self.params.plane_height = float(self.feature_overlay.screen_size_px[1])

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
        self.last_data = eye_tracking_data

    def poll(self):
        if self.playback or self.last_data is None:
            self.update_data()

        plane_localization = self.tracker(self.last_data.scene)
        assert self.last_data is not None
        self.data_changed.emit(self.last_data, plane_localization, self.tracker.debug)

    def exec(self):
        ret = super().exec()
        self.eye_tracking_source.close()
        return ret


@click.command()
@click.option(
    "--params_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to tracker parameters JSON file.",
)
@click.option(
    "--marker_config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to marker configuration file.",
)
def main(params_path, marker_config_path):
    import sys

    sys.argv = [sys.argv[0]]
    app = DebugApp(
        params_path=params_path,
        marker_config_path=marker_config_path,
    )
    app.exec()


if __name__ == "__main__":
    main()
