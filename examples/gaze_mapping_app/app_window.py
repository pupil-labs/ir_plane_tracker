import numpy.typing as npt
from common.eye_tracking_sources import EyeTrackingData
from common.source_widget import SourceWidget
from gaze_mapping_app.views.mapped_view import MappedView
from gaze_mapping_app.views.raw_image_view import RawImageView
from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QScreen
from PySide6.QtWidgets import QHBoxLayout, QMenuBar, QWidget

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class MainWindow(QWidget):
    feature_overlay_toggled = Signal()
    gaze_overlay_toggled = Signal()
    close_requested = Signal()
    new_source_connected = Signal(object)
    source_disconnect_requested = Signal()

    def __init__(self, target_screen: QScreen, parent=None):
        super().__init__(parent)

        self.source_widget = SourceWidget()
        self.source_widget.new_device_connected.connect(
            lambda device: self.new_source_connected.emit(device)
        )
        self.source_widget.disconnect_requested.connect(
            lambda: self.source_disconnect_requested.emit()
        )

        layout = QHBoxLayout(self)

        menubar = QMenuBar(self)
        settings_menu = menubar.addMenu("Menu")

        source_action = QAction("Source", self)
        source_action.triggered.connect(lambda: self.source_widget.show())
        settings_menu.addAction(source_action)

        gaze_toggle_action = QAction("Toggle Gaze Overlay", self)
        gaze_toggle_action.triggered.connect(self.gaze_overlay_toggled.emit)
        settings_menu.addAction(gaze_toggle_action)

        feature_overlay_toggle_action = QAction("Toggle Feature Overlay", self)
        feature_overlay_toggle_action.triggered.connect(
            self.feature_overlay_toggled.emit
        )
        settings_menu.addAction(feature_overlay_toggle_action)

        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close_requested.emit)
        settings_menu.addAction(close_action)

        layout.setMenuBar(menubar)

        self.raw_img_view = RawImageView()
        self.mapped_view = MappedView(target_screen)

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
