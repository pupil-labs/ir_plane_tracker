import cv2
import numpy as np
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.tracker import TrackerParams
from pupil_labs.mar_common.eye_tracking_sources import EyeTrackingData
from pupil_labs.mar_common.ui.scaled_image_view import ScaledImageView

from .view import View


class OptimizationView(View):
    name = "Optimization"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.optimization_error_threshold = LabeledSlider(
            "optimization_error_threshold",
            0,
            80,
            25,
        )
        sidebar_layout.addWidget(self.optimization_error_threshold)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

        self.make_connections(tracker_param_changed)

    def set_tracker_params(self, params: TrackerParams) -> None:
        self.optimization_error_threshold.set_value(params.optimization_error_threshold)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        if debug.optimization_final_combination is not None:
            for position, line in debug.optimization_final_combination._map.items():
                if line is not None:
                    cv2.polylines(
                        vis,
                        [line.points.astype(int)],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    orientation = line.orientation.name
                    p = np.mean(line.points, axis=0).astype(int)
                    cv2.putText(
                        vis,
                        f"{position.name} {orientation}",
                        (int(p[0]), int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            cv2.putText(
                vis,
                f"Final Error: {debug.optimization_errors[-1]:.2f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        else:
            best_error = (
                np.min(debug.optimization_errors)
                if len(debug.optimization_errors) > 0
                else np.inf
            )
            cv2.putText(
                vis,
                f"No valid combination found! Best Error: {best_error:.2f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        self.image_view.set_image(vis)
