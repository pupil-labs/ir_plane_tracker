import cv2
import numpy as np
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.views import View
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.tracker_line_and_dots import TrackerLineAndDotsParams


class FeatureLinesView(View):
    name = "Feature Lines"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.max_cr_error = LabeledSlider(
            "max_cr_error",
            0,
            20,
            16,
        )
        sidebar_layout.addWidget(self.max_cr_error)
        self.max_feature_line_length = LabeledSlider(
            "max_feature_line_length",
            0,
            500,
            150,
        )
        sidebar_layout.addWidget(self.max_feature_line_length)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

    def set_tracker_params(self, params: TrackerLineAndDotsParams) -> None:
        self.max_cr_error.set_value(int(params.max_cr_error * 100))
        self.max_feature_line_length.set_value(int(params.max_feature_line_length))

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        if (
            debug.feature_lines_candidates is not None
            and debug.cr_values is not None
            and debug.feature_lines_lengths is not None
        ):
            for line, cr, length in zip(
                debug.feature_lines_candidates,
                debug.cr_values,
                debug.feature_lines_lengths,
                strict=False,
            ):
                cv2.polylines(
                    vis,
                    [line.points.astype(int)],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )

                for p in line.points:
                    cv2.circle(
                        vis,
                        (int(p[0]), int(p[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )

                p = np.mean(line.points, axis=0).astype(int)
                cv2.putText(
                    vis,
                    f"{line.orientation}",
                    (int(p[0]), int(p[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    vis,
                    f"{cr:.3f}",
                    (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    vis,
                    f"{length:.1f}",
                    (int(p[0]), int(p[1] + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        if debug.feature_lines_filtered is not None:
            for line in debug.feature_lines_filtered:
                cv2.polylines(
                    vis,
                    [line.points.astype(int)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )

                for p in line.points:
                    cv2.circle(
                        vis,
                        (int(p[0]), int(p[1])),
                        5,
                        (0, 255, 0),
                        -1,
                    )
        self.image_view.set_image(vis)
