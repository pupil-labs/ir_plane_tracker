from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QWidget

from pupil_labs.ir_plane_tracker import (
    DebugData,
    PlaneLocalization,
    TrackerParams,
)
from pupil_labs.mar_common.eye_tracking_sources import EyeTrackingData


class View(QWidget):
    name = "Base View"

    def __init__(self, parent=None):
        super().__init__(parent)

    def make_connections(self, signal) -> None:
        for slider in self.findChildren(LabeledSlider):
            slider.valueChanged.connect(
                lambda val, s=slider: signal.emit(s.label.text(), val)
            )

    def set_tracker_params(self, params: TrackerParams) -> None:
        for slider in self.findChildren(LabeledSlider):
            name = slider.label.text()
            getattr(self, name).set_value(getattr(params, name))

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        pass
