from pstats import Stats
from time import time

import cv2

import pupil_labs.neon_recording as plr
from pupil_labs.ir_plane_tracker import IRPlaneTracker, IRPlaneTrackerParams


def main():
    rec = plr.open("offline_recording/data/indoor_medium")

    calibration = rec.calibration
    assert calibration is not None
    camera_matrix = calibration.scene_camera_matrix
    dist_coeffs = calibration.scene_distortion_coefficients

    params_json_path = "neon_small.json"

    params = IRPlaneTrackerParams.from_json(params_json_path)
    params.debug = True

    tracker = IRPlaneTracker(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, params=params
    )

    deltas = []
    for frame_idx, frame in enumerate(rec.scene):
        img = frame.bgr

        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        img = cv2.resize(
            img, None, fx=params.img_size_factor, fy=params.img_size_factor
        )
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
        # tracker.debug.visualize()
        # key = cv2.waitKey(0)
        # if key == ord("q"):
        #     break
        if frame_idx >= 200:
            break


if __name__ == "__main__":
    import cProfile

    with cProfile.Profile() as pr:
        main()

    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(r"\((?!\_).*\)$")
