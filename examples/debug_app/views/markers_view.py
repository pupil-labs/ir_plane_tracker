import cv2
from debug_app.views.view import View
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.mar_common.eye_tracking_sources import EyeTrackingData
from pupil_labs.mar_common.ui.scaled_image_view import ScaledImageView


class MarkersView(View):
    name = "Markers"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.padding_mm = LabeledSlider("padding_mm", 0, 30, 5)
        sidebar_layout.addWidget(self.padding_mm)

        self.circle_diameter_mm = LabeledSlider("circle_diameter_mm", 0, 30, 6)
        sidebar_layout.addWidget(self.circle_diameter_mm)

        self.line_thickness_mm = LabeledSlider("line_thickness_mm", 0, 10, 3)
        sidebar_layout.addWidget(self.line_thickness_mm)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=2)

        self.setLayout(layout)

        self.make_connections(tracker_param_changed)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()

        if plane_localization is not None:
            cv2.polylines(
                vis, [plane_localization.corners.astype(int)], True, (255, 0, 0), 3
            )

        self.image_view.set_image(vis)
