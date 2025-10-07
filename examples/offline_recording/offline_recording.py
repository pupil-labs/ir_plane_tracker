from pstats import Stats
from time import time

import cv2
import pupil_labs.neon_recording as plr

from pupil_labs.ir_plane_tracker import IRPlaneTracker, IRPlaneTrackerParams


def main():
    rec = plr.open("offline_recording/data/outdoor_hard")

    calibration = rec.calibration
    assert calibration is not None
    camera_matrix = calibration.scene_camera_matrix
    dist_coeffs = calibration.scene_distortion_coefficients

    params_json_path = "neon.json"

    params = IRPlaneTrackerParams.from_json(params_json_path)
    params.debug = True
    params.thresh_c = 35
    params.thresh_half_kernel_size = 70
    params.max_line_length = 500.0
    params.img_size_factor = 2.5
    params.optimization_error_threshold = 8.0

    tracker = IRPlaneTracker(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, params=params
    )

    deltas = []
    for frame in rec.scene:
        img = frame.bgr

        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        start_ts = time()
        screen_corners = tracker(img)  # noqa: F841
        end_ts = time()
        delta = end_ts - start_ts
        deltas.append(delta)
        if len(deltas) > 30:
            deltas = deltas[-30:]
        avg_delta = sum(deltas) / len(deltas)
        fps = 1.0 / avg_delta
        print(f"FPS: {fps:.2f}", end="\r")
        tracker.debug.visualize()
        key = cv2.waitKey(1)
        if key == ord("q"):
            break


if __name__ == "__main__":
    import cProfile

    with cProfile.Profile() as pr:
        main()
    # pr.sort_stats("time").print_stats("ir_plane_tracker")
    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(r"\((?!\_).*\)$")  # Exclude private and magic callables.
