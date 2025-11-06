import click
import cv2
from common import eye_tracking_sources
from gaze_mapping_app.app_window import MainWindow
from gaze_mapping_app.gaze_overlay import GazeOverlay
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import Tracker
from pupil_labs.ir_plane_tracker.feature_overlay import FeatureOverlay
from pupil_labs.ir_plane_tracker.tracker_params_wrapper import TrackerParamsWrapper


class GazeMappingApp(QApplication):
    data_changed = Signal(object, object, object, object)

    def __init__(
        self,
        params_path: str,
        neon_ip: str | None = None,
        neon_port: int = 8080,
    ):
        super().__init__()
        self.setApplicationDisplayName("Gaze Mapping Demo")

        self.eye_tracking_source = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.params = TrackerParamsWrapper.from_json(params_path)
        self.tracker = Tracker(
            camera_matrix=self.camera_matrix,
            dist_coeffs=None,
            params=self.params,
        )
        self.original_marker_config = {
            "top_pos": self.tracker.params.top_pos,
            "right_pos": self.tracker.params.right_pos,
            "bottom_pos": self.tracker.params.bottom_pos,
            "left_pos": self.tracker.params.left_pos,
            "plane_width": self.tracker.params.plane_width,
            "plane_height": self.tracker.params.plane_height,
            "feature_point_positions_mm": self.tracker.params.feature_point_positions_mm,  # noqa: E501
        }

        if neon_ip is not None:
            device = eye_tracking_sources.RemoteSource(ip=neon_ip, port=neon_port)
            self.on_new_source_connected(device)

        screens = QGuiApplication.screens()
        target_screen = screens[-1]
        self.main_window = MainWindow(target_screen)
        self.main_window.setMinimumSize(1600, 600)

        self.feature_overlay = FeatureOverlay(target_screen, self.params)
        self.toggle_feature_overlay()
        self.gaze_overlay = GazeOverlay(target_screen)
        self.gaze_overlay.toggle_visibility()

        # Connections
        self.main_window.close_requested.connect(self.close_app)
        self.main_window.new_source_connected.connect(self.on_new_source_connected)
        self.main_window.source_disconnect_requested.connect(
            self.on_source_disconnect_requested
        )

        self.data_changed.connect(self.main_window.set_data)
        self.data_changed.connect(self.gaze_overlay.set_data)
        self.main_window.feature_overlay_toggled.connect(self.toggle_feature_overlay)
        self.main_window.gaze_overlay_toggled.connect(
            lambda: self.gaze_overlay.toggle_visibility()
        )

        self.main_window.show()

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(int(1000 / 30))
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start()

    def close_app(self):
        print("CLOSING APP")
        self.feature_overlay.hide()
        self.feature_overlay.close()
        self.gaze_overlay.hide()
        self.gaze_overlay.close()
        self.eye_tracking_source.close()
        self.main_window.source_widget.close()
        self.main_window.hide()
        self.main_window.close()
        self.quit()

    def on_new_source_connected(self, source: eye_tracking_sources.EyeTrackingSource):
        if self.eye_tracking_source is not None:
            self.eye_tracking_source.close()
        self.eye_tracking_source = source
        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = self.eye_tracking_source.scene_intrinsics.distortion_coeffs
        self.tracker.camera_matrix = self.camera_matrix

    def on_source_disconnect_requested(self):
        if self.eye_tracking_source is not None:
            self.eye_tracking_source.close()
            self.eye_tracking_source = None

    def toggle_feature_overlay(self):
        if self.feature_overlay.isVisible():
            self.feature_overlay.toggle_visibility()
            self.params.update_params({
                "top_pos": self.original_marker_config["top_pos"],
                "right_pos": self.original_marker_config["right_pos"],
                "bottom_pos": self.original_marker_config["bottom_pos"],
                "left_pos": self.original_marker_config["left_pos"],
                "plane_width": self.original_marker_config["plane_width"],
                "plane_height": self.original_marker_config["plane_height"],
                "feature_point_positions_mm": self.original_marker_config[
                    "feature_point_positions_mm"
                ],
            })

        else:
            self.feature_overlay.toggle_visibility()
            self.feature_overlay.update_marker_positions()

    def poll(self):
        if self.eye_tracking_source is not None:
            eye_tracking_data = self.eye_tracking_source.get_sample()
            eye_tracking_data.scene = cv2.undistort(
                eye_tracking_data.scene, self.camera_matrix, self.dist_coeffs
            )  # type: ignore
            plane_localization = self.tracker(eye_tracking_data.scene)
            gaze_mapped = None
            if plane_localization is not None:
                gaze = eye_tracking_data.gaze
                if gaze is not None:
                    gaze_mapped = plane_localization.img2plane @ [*gaze, 1]
                    gaze_mapped = gaze_mapped / gaze_mapped[2]
                    gaze_mapped = gaze_mapped[:2]

            self.data_changed.emit(
                eye_tracking_data, plane_localization, self.tracker.debug, gaze_mapped
            )

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
    "--neon_ip", type=str, default=None, help="IP address of the Neon device."
)
@click.option("--neon_port", type=int, default=8080, help="Port of the Neon device.")
def main(params_path, neon_ip, neon_port):
    import sys

    sys.argv = [sys.argv[0]]
    app = GazeMappingApp(
        params_path=params_path,
        neon_ip=neon_ip,
        neon_port=neon_port,
    )
    app.exec()


if __name__ == "__main__":
    main()
