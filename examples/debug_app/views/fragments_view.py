import cv2
import numpy as np
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.mar_common.eye_tracking_sources import EyeTrackingData
from pupil_labs.mar_common.ui.scaled_image_view import ScaledImageView

from .view import View


class FragmentsView(View):
    name = "Fragments"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)

        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.fragments_max_projection_error = LabeledSlider(
            "fragments_max_projection_error", 0, 50, 8
        )
        sidebar_layout.addWidget(self.fragments_max_projection_error)
        self.fragments_min_length = LabeledSlider("fragments_min_length", 0, 200, 60)
        sidebar_layout.addWidget(self.fragments_min_length)
        self.fragments_max_length = LabeledSlider("fragments_max_length", 0, 500, 200)
        sidebar_layout.addWidget(self.fragments_max_length)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

        self.make_connections(tracker_param_changed)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        if debug.fragments_raw is not None:
            for fragment in debug.fragments_raw:
                cv2.polylines(
                    vis,
                    [np.array([fragment.start_pt, fragment.end_pt], dtype=int)],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )
        if debug.fragments_filtered is not None:
            for fragment in debug.fragments_filtered:
                cv2.polylines(
                    vis,
                    [np.array([fragment.start_pt, fragment.end_pt], dtype=int)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )
        if debug.fragments_raw is not None and debug.fragments_length is not None:
            for fragment in debug.fragments_raw:
                cv2.circle(
                    vis,
                    (int(fragment.start_pt[0]), int(fragment.start_pt[1])),
                    3,
                    (255, 0, 255),
                    -1,
                )
                cv2.circle(
                    vis,
                    (int(fragment.end_pt[0]), int(fragment.end_pt[1])),
                    3,
                    (255, 0, 0),
                    -1,
                )

            for fragment, length in zip(
                debug.fragments_raw, debug.fragments_length, strict=True
            ):
                p = np.mean([fragment.start_pt, fragment.end_pt], axis=0).astype(int)

                cv2.putText(
                    vis,
                    f"{fragment.projection_error:<.1f}",
                    (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    vis,
                    f"{length:<.1f}",
                    (int(p[0]), int(p[1] + 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
        self.image_view.set_image(vis)
