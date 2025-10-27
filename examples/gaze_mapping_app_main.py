import click
import cv2
import numpy as np
from common import eye_tracking_sources
from gaze_mapping_app.app_window import MainWindow
from gaze_mapping_app.gaze_overlay import GazeOverlay
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import (
    Tracker,
    TrackerParams,
)
from pupil_labs.ir_plane_tracker.feature_overlay import FeatureOverlay


class GazeMappingApp(QApplication):
    on_data_update = Signal(object, object, object, object)

    def __init__(
        self,
        params_path: str,
        marker_config_path: str,
        neon_ip: str,
        neon_port: int = 8080,
    ):
        super().__init__()
        self.setApplicationDisplayName("Gaze Mapping Demo")
        self.eye_tracking_source = eye_tracking_sources.RemoteSource(
            ip=neon_ip, port=neon_port
        )
        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = self.eye_tracking_source.scene_intrinsics.distortion_coeffs
        self.params = TrackerParams.from_json(params_path, marker_config_path)
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
            "norm_line_points": self.tracker.params.norm_line_points,
        }

        screens = QGuiApplication.screens()
        target_screen = screens[-1]
        self.main_window = MainWindow(target_screen)
        self.main_window.setMinimumSize(1600, 600)

        self.feature_overlay = FeatureOverlay(target_screen)
        self.toggle_feature_overlay()
        self.gaze_overlay = GazeOverlay(target_screen)
        self.gaze_overlay.toggle_visibility()

        # Connections
        self.main_window.destroyed.connect(self.gaze_overlay.close)
        self.main_window.destroyed.connect(self.feature_overlay.close)

        self.on_data_update.connect(self.main_window.set_data)
        self.on_data_update.connect(self.gaze_overlay.set_data)
        self.main_window.on_feature_overlay_toggled.connect(self.toggle_feature_overlay)
        self.main_window.on_gaze_overlay_toggled.connect(
            lambda: self.gaze_overlay.toggle_visibility()
        )

        self.main_window.show()

        self.poll_timer = QTimer()
        self.poll_timer.setInterval(int(1000 / 30))
        self.poll_timer.timeout.connect(self.poll)
        self.poll_timer.start()

    def toggle_feature_overlay(self):
        if self.feature_overlay.isVisible():
            self.feature_overlay.toggle_visibility()
            self.tracker.params.top_pos = self.original_marker_config["top_pos"]
            self.tracker.params.right_pos = self.original_marker_config["right_pos"]
            self.tracker.params.bottom_pos = self.original_marker_config["bottom_pos"]
            self.tracker.params.left_pos = self.original_marker_config["left_pos"]
            self.tracker.params.plane_width = self.original_marker_config["plane_width"]
            self.tracker.params.plane_height = self.original_marker_config[
                "plane_height"
            ]
            self.tracker.params.norm_line_points = self.original_marker_config[
                "norm_line_points"
            ]
        else:
            self.feature_overlay.toggle_visibility()
            feature_values_px = self.feature_overlay.feature_values_px
            (
                self.tracker.params.top_pos,
                self.tracker.params.right_pos,
                self.tracker.params.bottom_pos,
                self.tracker.params.left_pos,
            ) = feature_values_px.reshape(-1, 2)[3::4, :]
            self.tracker.params.norm_line_points = np.concatenate([
                [0],
                np.cumsum(np.diff(feature_values_px[:4, 0])),
            ])
            self.tracker.params.plane_width = float(
                self.feature_overlay.screen_size_px[0]
            )
            self.tracker.params.plane_height = float(
                self.feature_overlay.screen_size_px[1]
            )

    def poll(self):
        eye_tracking_data = self.eye_tracking_source.get_sample()
        eye_tracking_data.scene = cv2.undistort(
            eye_tracking_data.scene, self.camera_matrix, self.dist_coeffs
        )  # type: ignore
        plane_localization = self.tracker(eye_tracking_data.scene)
        # self.tracker.debug.visualize()
        # cv2.waitKey(0)
        gaze_mapped = None
        if plane_localization is not None:
            gaze = eye_tracking_data.gaze
            if gaze is not None:
                gaze_mapped = plane_localization.img2plane @ [*gaze, 1]
                gaze_mapped = gaze_mapped / gaze_mapped[2]
                gaze_mapped = gaze_mapped[:2]

        self.on_data_update.emit(
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
    "--marker_config_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to marker configuration file.",
)
@click.option("--neon_ip", type=str, help="IP address of the Neon device.")
@click.option("--neon_port", type=int, default=8080, help="Port of the Neon device.")
def main(params_path, marker_config_path, neon_ip, neon_port):
    import sys

    sys.argv = [sys.argv[0]]
    app = GazeMappingApp(
        params_path=params_path,
        marker_config_path=marker_config_path,
        neon_ip=neon_ip,
        neon_port=neon_port,
    )
    app.exec()


if __name__ == "__main__":
    main()
