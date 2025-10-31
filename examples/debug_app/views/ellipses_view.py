import cv2
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization
from pupil_labs.ir_plane_tracker.tracker import TrackerParams

from .view import View


class EllipseView(View):
    name = "Ellipses"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)
        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.min_ellipse_size = LabeledSlider("min_ellipse_size", 0, 50, 8)
        sidebar_layout.addWidget(self.min_ellipse_size)
        self.max_ellipse_aspect_ratio = LabeledSlider(
            "max_ellipse_aspect_ratio", 0, 20, 12
        )
        sidebar_layout.addWidget(self.max_ellipse_aspect_ratio)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=1)

        self.setLayout(layout)

        self.make_connections(tracker_param_changed)

    def make_connections(self, tracker_param_changed) -> None:
        for slider in self.findChildren(LabeledSlider):
            if slider.label.text() == "max_ellipse_aspect_ratio":
                slider.valueChanged.connect(
                    lambda val, s=slider: tracker_param_changed.emit(
                        s.label.text(), val / 10.0
                    )
                )
            else:
                slider.valueChanged.connect(
                    lambda val, s=slider: tracker_param_changed.emit(
                        s.label.text(), val
                    )
                )

    def set_tracker_params(self, params: TrackerParams) -> None:
        self.min_ellipse_size.set_value(params.min_ellipse_size)
        self.max_ellipse_aspect_ratio.set_value(
            int(params.max_ellipse_aspect_ratio * 10)
        )

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        if debug.ellipses_raw is not None:
            for ellipse in debug.ellipses_raw:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                size = (int(ellipse.size[0] / 2), int(ellipse.size[1] / 2))
                angle = ellipse.angle
                cv2.ellipse(vis, center, size, angle, 0, 360, (0, 0, 255), 2)

                cv2.putText(
                    vis,
                    (
                        f"{ellipse.minor_axis:.1f} "
                        f"{ellipse.major_axis / ellipse.minor_axis:.1f}"
                    ),
                    (center[0], center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )

        if debug.ellipses_filtered is not None:
            for ellipse in debug.ellipses_filtered:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                size = (int(ellipse.size[0] / 2), int(ellipse.size[1] / 2))
                angle = ellipse.angle
                cv2.ellipse(vis, center, size, angle, 0, 360, (0, 255, 0), 2)
        self.image_view.set_image(vis)
