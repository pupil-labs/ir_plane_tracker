import cv2
import numpy as np
from common import eye_tracking_sources
from gaze_mapping_app.app_window import MainWindow
from gaze_mapping_app.gaze_overlay import GazeOverlay
from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker import (
    TrackerLineAndDots,
    TrackerLineAndDotsParams,
)
from pupil_labs.ir_plane_tracker.feature_overlay import FeatureOverlay
from pupil_labs.ir_plane_tracker.tracker_line_and_dots import (
    LinePositions,
)


class GazeMappingApp(QApplication):
    on_data_update = Signal(object, object, object, object)

    def __init__(self, neon_ip: str, neon_port: int = 8080, params_path: str = None):
        super().__init__()
        self.setApplicationDisplayName("Gaze Mapping Demo")
        self.eye_tracking_source = eye_tracking_sources.RemoteSource(
            ip=neon_ip, port=neon_port
        )
        self.camera_matrix = self.eye_tracking_source.scene_intrinsics.camera_matrix
        self.dist_coeffs = self.eye_tracking_source.scene_intrinsics.distortion_coeffs
        self.params = TrackerLineAndDotsParams.from_json(
            params_path or "resources/neon_artificial.json"
        )
        self.tracker = TrackerLineAndDots(
            camera_matrix=self.camera_matrix,
            dist_coeffs=None,
            params=self.params,
        )
        self.original_obj_point_map = self.tracker.obj_point_map.copy()
        self.original_plane_size = (
            self.tracker.params.plane_width,
            self.tracker.params.plane_height,
        )

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
            self.tracker.obj_point_map = self.original_obj_point_map.copy()
            self.tracker.params.plane_width = self.original_plane_size[0]
            self.tracker.params.plane_height = self.original_plane_size[1]
        else:
            self.feature_overlay.toggle_visibility()
            feature_values_px = self.feature_overlay.feature_values_px
            feature_values_px = np.hstack(
                [
                    feature_values_px,
                    np.zeros((feature_values_px.shape[0], 1)),
                ]
            )
            feature_values_px = feature_values_px.reshape(-1, 4, 3)
            self.tracker.obj_point_map = {
                LinePositions.TOP: feature_values_px[0],
                LinePositions.RIGHT: feature_values_px[1],
                LinePositions.BOTTOM: feature_values_px[2],
                LinePositions.LEFT: feature_values_px[3],
            }
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


def run():
    import sys

    params_path = sys.argv[3] if len(sys.argv) > 3 else None
    app = GazeMappingApp(
        neon_ip=sys.argv[1],
        neon_port=int(sys.argv[2]) if len(sys.argv) > 2 else 8080,
        params_path=params_path,
    )
    app.exec()


if __name__ == "__main__":
    run()
