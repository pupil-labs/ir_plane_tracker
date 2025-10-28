import cv2
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView
from debug_app.widgets.labeled_slider import LabeledSlider
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization

from .view import View


class ContoursView(View):
    name = "Contours"

    def __init__(self, tracker_param_changed, parent=None):
        super().__init__(parent)

        self.image_view = ScaledImageView(self)
        layout = QHBoxLayout()
        layout.addWidget(self.image_view, stretch=5)

        sidebar_layout = QVBoxLayout()

        self.min_contour_area_line = LabeledSlider(
            "min_contour_area_line", 100, 3000, 350
        )
        sidebar_layout.addWidget(self.min_contour_area_line)
        self.max_contour_area_line = LabeledSlider(
            "max_contour_area_line", 100, 6000, 2000
        )
        sidebar_layout.addWidget(self.max_contour_area_line)
        self.min_contour_area_ellipse = LabeledSlider(
            "min_contour_area_ellipse", 10, 800, 100
        )
        sidebar_layout.addWidget(self.min_contour_area_ellipse)
        self.max_contour_area_ellipse = LabeledSlider(
            "max_contour_area_ellipse", 10, 800, 500
        )
        sidebar_layout.addWidget(self.max_contour_area_ellipse)

        sidebar_layout.addStretch()
        layout.addLayout(sidebar_layout, stretch=2)

        self.setLayout(layout)

        self.make_connections(tracker_param_changed)

    def set_data(
        self,
        eye_tracking_data: EyeTrackingData,
        plane_localization: PlaneLocalization,
        debug: DebugData,
    ):
        vis = debug.img_thresholded.copy()
        assert debug.contours_raw is not None
        assert debug.contours_line is not None
        assert debug.contours_ellipse is not None
        assert debug.contour_areas is not None

        cv2.drawContours(vis, debug.contours_raw, -1, (0, 0, 255), 2)
        cv2.drawContours(vis, debug.contours_line, -1, (0, 255, 0), 2)
        cv2.drawContours(vis, debug.contours_ellipse, -1, (255, 0, 0), 2)

        for c, a in zip(debug.contours_raw, debug.contour_areas, strict=False):
            cv2.putText(
                vis,
                f"{a:.0f}",
                tuple(c[c[:, :, 1].argmin()][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        self.image_view.set_image(vis)
