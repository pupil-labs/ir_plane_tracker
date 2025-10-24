from common.eye_tracking_sources import EyeTrackingData
from debug_app.tracker import Tracker
from debug_app.views import View
from debug_app.views.contours_view import ContoursView
from debug_app.views.ellipses_view import EllipseView
from debug_app.views.feature_lines_view import FeatureLinesView
from debug_app.views.fragments_view import FragmentsView
from debug_app.views.optimization_view import OptimizationView
from debug_app.views.raw_image_view import RawImageView
from debug_app.views.thresholding_view import ThresholdingView
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class AppWindow(QWidget):
    playback_toggled = Signal()
    next_frame_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        QShortcut(Qt.Key_Space, self, activated=self.playback_toggled.emit)  # type: ignore
        QShortcut(Qt.Key_Right, self, activated=self.next_frame_clicked.emit)  # type: ignore

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()

        self.raw_img_view = RawImageView()
        self.tabs.addTab(self.raw_img_view, self.raw_img_view.name)

        self.thresholding_view = ThresholdingView()
        self.tabs.addTab(self.thresholding_view, self.thresholding_view.name)

        self.contours_view = ContoursView()
        self.tabs.addTab(self.contours_view, self.contours_view.name)

        self.fragment_view = FragmentsView()
        self.tabs.addTab(self.fragment_view, self.fragment_view.name)

        self.ellipse_view = EllipseView()
        self.tabs.addTab(self.ellipse_view, self.ellipse_view.name)

        self.feature_lines_view = FeatureLinesView()
        self.tabs.addTab(self.feature_lines_view, self.feature_lines_view.name)

        self.optimization_view = OptimizationView()
        self.tabs.addTab(self.optimization_view, self.optimization_view.name)

        layout.addWidget(self.tabs)

    def make_connections(self, tracker: Tracker) -> None:
        tracker.params_changed.connect(self.thresholding_view.set_tracker_params)
        self.thresholding_view.thresh_c.valueChanged.connect(
            lambda val: tracker.update_params({"thresh_c": val})
        )

        tracker.params_changed.connect(self.contours_view.set_tracker_params)
        self.contours_view.min_contour_area_line.valueChanged.connect(
            lambda val: tracker.update_params({"min_contour_area_line": val})
        )
        self.contours_view.max_contour_area_line.valueChanged.connect(
            lambda val: tracker.update_params({"max_contour_area_line": val})
        )
        self.contours_view.min_contour_area_ellipse.valueChanged.connect(
            lambda val: tracker.update_params({"min_contour_area_ellipse": val})
        )
        self.contours_view.max_contour_area_ellipse.valueChanged.connect(
            lambda val: tracker.update_params({"max_contour_area_ellipse": val})
        )

        tracker.params_changed.connect(self.fragment_view.set_tracker_params)
        self.fragment_view.fragments_max_projection_error.valueChanged.connect(
            lambda val: tracker.update_params({"fragments_max_projection_error": val})
        )
        self.fragment_view.fragments_min_length.valueChanged.connect(
            lambda val: tracker.update_params({"fragments_min_length": val})
        )
        self.fragment_view.fragments_max_length.valueChanged.connect(
            lambda val: tracker.update_params({"fragments_max_length": val})
        )

        tracker.params_changed.connect(self.ellipse_view.set_tracker_params)
        self.ellipse_view.min_ellipse_size.valueChanged.connect(
            lambda val: tracker.update_params({"min_ellipse_size": val})
        )
        self.ellipse_view.max_ellipse_aspect_ratio.valueChanged.connect(
            lambda val: tracker.update_params({"max_ellipse_aspect_ratio": val})
        )

        tracker.params_changed.connect(self.feature_lines_view.set_tracker_params)
        self.feature_lines_view.max_cr_error.valueChanged.connect(
            lambda val: tracker.update_params({"max_cr_error": val / 100})
        )
        self.feature_lines_view.max_feature_line_length.valueChanged.connect(
            lambda val: tracker.update_params({"max_feature_line_length": float(val)})
        )

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        for view in self.tabs.findChildren(View):
            view.set_data(eye_tracking_data, plane_localization, debug)

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_Space:
    #         self.playback_toggled.emit()
    #     elif event.key() == Qt.ArrowRight:
    #         self.next_frame_clicked.emit()
    #     else:
    #         super().keyPressEvent(event)

    # def eventFilter(self, obj, event):
    #     if event.type() == event.KeyPress:
    #         if event.key() == Qt.Key_Space:
    #             self.playback_toggled.emit()
    #             return True  # Event handled
    #         elif event.key() == Qt.Key_Right:
    #             self.next_frame_clicked.emit()
    #             return True
    #     return super().eventFilter(obj, event)
