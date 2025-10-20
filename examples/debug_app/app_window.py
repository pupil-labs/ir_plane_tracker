from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.views.contours_view import ContoursView
from debug_app.views.raw_image_view import RawImageView
from debug_app.views.thresholding_view import ThresholdingView
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class AppWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.raw_img_view = RawImageView()
        self.tabs.addTab(self.raw_img_view, self.raw_img_view.name)

        self.thresholding_view = ThresholdingView()
        self.tabs.addTab(self.thresholding_view, self.thresholding_view.name)

        self.contours_view = ContoursView()
        self.tabs.addTab(self.contours_view, self.contours_view.name)

        layout.addWidget(self.tabs)

    def on_data_update(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        for view in self.tabs.findChildren(ScaledImageView):
            view.update_data(eye_tracking_data, plane_localization, debug)
