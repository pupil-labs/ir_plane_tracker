import numpy.typing as npt
from common.eye_tracking_sources import EyeTrackingData
from gaze_mapping_app.views.mapped_view import MappedView
from gaze_mapping_app.views.raw_image_view import RawImageView
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QMenuBar, QWidget

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class MainWindow(QWidget):
    on_feature_overlay_toggled = Signal()
    on_gaze_overlay_toggled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)

        menubar = QMenuBar(self)
        settings_menu = menubar.addMenu("Menu")

        gaze_toggle_action = QAction("Toggle Gaze Overlay", self)
        gaze_toggle_action.triggered.connect(self.on_gaze_overlay_toggled.emit)
        settings_menu.addAction(gaze_toggle_action)

        feature_overlay_toggle_action = QAction("Toggle Feature Overlay", self)
        feature_overlay_toggle_action.triggered.connect(
            self.on_feature_overlay_toggled.emit
        )
        settings_menu.addAction(feature_overlay_toggle_action)

        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close)
        settings_menu.addAction(close_action)

        layout.setMenuBar(menubar)

        self.raw_img_view = RawImageView()
        self.mapped_view = MappedView()

        layout.addWidget(self.raw_img_view)
        layout.addWidget(self.mapped_view)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization | None,
        debug: DebugData,
        gaze_mapped: npt.NDArray | None,
    ):
        self.raw_img_view.update_data(
            eye_tracking_data, plane_localization, debug, gaze_mapped
        )
        self.mapped_view.update_data(
            eye_tracking_data, plane_localization, debug, gaze_mapped
        )
