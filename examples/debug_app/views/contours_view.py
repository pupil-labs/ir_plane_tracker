import cv2
from common.eye_tracking_sources import EyeTrackingData
from common.widgets.scaled_image_view import ScaledImageView

from pupil_labs.ir_plane_tracker import DebugData, PlaneLocalization


class ContoursView(ScaledImageView):
    name = "Contours"

    def update_data(
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
        # cv2.putText(
        #     vis,
        #     "Line",
        #     (50, 25),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     2,
        # )
        # cv2.putText(
        #     vis,
        #     f"Min Area: {self.params.min_contour_area_line}",
        #     (50, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     2,
        # )
        # cv2.putText(
        #     vis,
        #     f"Max Area: {self.params.max_contour_area_line}",
        #     (50, 75),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (0, 255, 0),
        #     2,
        # )
        # cv2.putText(
        #     vis,
        #     "Ellipse",
        #     (300, 25),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 0, 0),
        #     2,
        # )
        # cv2.putText(
        #     vis,
        #     f"Min Area: {self.params.min_contour_area_ellipse}",
        #     (300, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 0, 0),
        #     2,
        # )
        # cv2.putText(
        #     vis,
        #     f"Max Area: {self.params.max_contour_area_ellipse}",
        #     (300, 75),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.5,
        #     (255, 0, 0),
        #     2,
        # )

        self.set_image(vis)
