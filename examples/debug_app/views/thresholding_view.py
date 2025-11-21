from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.mar_common.eye_tracking_sources import EyeTrackingData
from pupil_labs.mar_common.ui.scaled_image_view import ScaledImageView

from .view import View


class ThresholdingView(View):
    name = "Thresholding"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)

        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.thresh_c = LabeledSlider("thresh_c", 5, 80, 40)
        sidebar_layout.addWidget(self.thresh_c)

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
        self.image_view.set_image(vis)
