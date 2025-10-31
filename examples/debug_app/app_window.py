from common.eye_tracking_sources import EyeTrackingData
from debug_app import views
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.tracker import TrackerParams


class AppWindow(QWidget):
    playback_toggled = Signal()
    next_frame_clicked = Signal()
    param_updated = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        QShortcut(Qt.Key_Space, self, activated=self.playback_toggled.emit)  # type: ignore
        QShortcut(Qt.Key_Right, self, activated=self.next_frame_clicked.emit)  # type: ignore

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.raw_img_view = views.RawImageView(self.param_updated)
        self.tabs.addTab(self.raw_img_view, self.raw_img_view.name)

        self.thresholding_view = views.ThresholdingView(self.param_updated)
        self.tabs.addTab(self.thresholding_view, self.thresholding_view.name)

        self.contours_view = views.ContoursView(self.param_updated)
        self.tabs.addTab(self.contours_view, self.contours_view.name)

        self.fragment_view = views.FragmentsView(self.param_updated)
        self.tabs.addTab(self.fragment_view, self.fragment_view.name)

        self.ellipse_view = views.EllipseView(self.param_updated)
        self.tabs.addTab(self.ellipse_view, self.ellipse_view.name)

        self.feature_lines_view = views.FeatureLinesView(self.param_updated)
        self.tabs.addTab(self.feature_lines_view, self.feature_lines_view.name)

        self.optimization_view = views.OptimizationView(self.param_updated)
        self.tabs.addTab(self.optimization_view, self.optimization_view.name)

        self.markers_view = views.MarkersView(self.param_updated)
        self.tabs.addTab(self.markers_view, self.markers_view.name)

        layout.addWidget(self.tabs)

    def set_tracker_params(self, params: TrackerParams) -> None:
        for view in self.tabs.findChildren(views.View):
            view.set_tracker_params(params)

    def set_marker_spec(self, spec) -> None:
        self.markers_view.set_marker_spec(spec)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        for view in self.tabs.findChildren(views.View):
            view.set_data(eye_tracking_data, plane_localization, debug)

        self.markers_view.set_data(eye_tracking_data, plane_localization, debug)
