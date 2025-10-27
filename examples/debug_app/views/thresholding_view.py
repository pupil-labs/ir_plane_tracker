from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.views import View
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.tracker import TrackerParams


class ThresholdingView(View):
    name = "Thresholding"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.thresh_c = LabeledSlider("c_thresh", 5, 80, 40)
        sidebar_layout.addWidget(self.thresh_c)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

    def set_tracker_params(self, params: TrackerParams) -> None:
        self.thresh_c.set_value(params.thresh_c)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        self.image_view.set_image(vis)
