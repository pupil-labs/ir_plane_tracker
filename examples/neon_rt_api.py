from time import time

import cv2

from pupil_labs.ir_plane_tracker import IRPlaneTracker, IRPlaneTrackerParams
from pupil_labs.realtime_api.simple import Device


def main():
    device = Device(address="192.168.178.28", port=8080)
    calibration = device.get_calibration()
    camera_matrix = calibration.scene_camera_matrix
    dist_coeffs = calibration.scene_distortion_coefficients
    params_json_path = "neon.json"

    params = IRPlaneTrackerParams.from_json(params_json_path)
    params.debug = True
    params.thresh_c = 35
    params.thresh_half_kernel_size = 70
    params.max_line_length = 500.0

    tracker = IRPlaneTracker(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, params=params
    )

    frame_counter = 700
    deltas = []
    while True:
        img, _ = device.receive_scene_video_frame()
        img = cv2.resize(img, (640, 480))

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
        if key == ord("s"):
            cv2.imwrite(
                f"/home/marc/pupil_labs/IR_plane_tracker/src/pupil_labs/ir_plane_tracker/data/dashed_images/frame_{frame_counter:04d}.png",
                img,
            )
            frame_counter += 1

            print(f"Saved frame_{frame_counter:04d}.png")


if __name__ == "__main__":
    import cProfile

    with cProfile.Profile() as pr:
        main()
        pr.print_stats()

    # main()
