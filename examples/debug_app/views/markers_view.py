from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.feature_overlay import MarkerSpec

from .view import View


class MarkersView(View):
    name = "Markers"

    def __init__(self, parent=None):
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

    def set_tracker_params(self, spec: MarkerSpec) -> None:
        self.padding_mm.set_value(int(spec.padding_mm))
        self.circle_diameter_mm.set_value(int(spec.circle_diameter_mm))
        self.line_thickness_mm.set_value(int(spec.line_thickness_mm))

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()

        self.image_view.set_image(vis)
