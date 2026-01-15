import click
import qdarktheme
from debug_app.app_window import AppWindow
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import Tracker
from pupil_labs.ir_plane_tracker.feature_overlay import FeatureOverlay
from pupil_labs.ir_plane_tracker.tracker_params_wrapper import TrackerParamsWrapper
from pupil_labs.ir_plane_tracker.extras.eye_tracking_sources.neon_usb import NeonUSB


class DebugApp(QApplication):
    data_changed = Signal(object, object, object)

    def __init__(self, params_path, feature_overlay=True):
        super().__init__()
        self.setApplicationDisplayName("Debug App")
        qdarktheme.setup_theme()
        # self.eye_tracking_source = eye_tracking_sources.RecordingSource(
        #     # recording_path="/home/marc/pupil_labs/IR_plane_tracker/examples/offline_recording/data/indoor4",  # noqa: E501
        #     recording_path="/home/marc/Downloads/Native Recording Data (9)/2025-10-30_09-06-12-6ed1deef"  # noqa: E501
        # )
        self.eye_tracking_source = NeonUSB(compute_gaze=False)

        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = (
            self.eye_tracking_source.scene_intrinsics.distortion_coefficients
        )

        self.params = TrackerParamsWrapper.from_json(params_path)

        self.tracker = Tracker(
            camera_matrix=self.camera_matrix,
            dist_coeffs=None,
            params=self.params,
        )

        self.main_window = AppWindow()

        if feature_overlay:
            self.feature_overlay = FeatureOverlay(self.screens()[-1], self.params)
            self.feature_overlay.toggle_visibility()

        # Connections
        self.data_changed.connect(self.main_window.set_data)

        self.main_window.playback_toggled.connect(
            lambda: setattr(self, "playback", not self.playback)
        )
        self.main_window.next_frame_clicked.connect(self.update_data)
        self.main_window.param_updated.connect(
            lambda key, val: self.params.update_params({key: val})
        )
        self.params.changed.connect(self.main_window.set_tracker_params)

        if feature_overlay:
            self.params.changed.connect(self.feature_overlay.update_marker_positions)

        # Initialization
        self.params.changed.emit(self.params)
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
        self.last_data = eye_tracking_data

    def poll(self):
        if self.playback or self.last_data is None:
            self.update_data()

        assert self.last_data is not None
        plane_localization = self.tracker(self.last_data.scene_image_undistorted)
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
    "--feature_overlay",
    is_flag=True,
    default=False,
    help="Enable feature overlay display.",
)
def main(params_path, feature_overlay):
    import sys

    sys.argv = [sys.argv[0]]
    app = DebugApp(
        params_path=params_path,
        feature_overlay=feature_overlay,
    )
    app.exec()


if __name__ == "__main__":
    main()
