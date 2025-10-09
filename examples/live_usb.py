from time import time

import cv2

from pupil_labs.ir_plane_tracker.tracker_line_and_dots import (
    TrackerLineAndDots,
    TrackerLineAndDotsParams,
)


def main():
    # cam = HDDigitalCam()
    # camera_matrix = np.load("camera_matrix.npy")
    # dist_coeffs = np.load("dist_coeffs.npy")
    # params_json_path = "hddigital.json"
    from camera import SceneCam

    cam = SceneCam()
    cam.exposure = 400  # in ms
    camera_matrix, dist_coeffs = cam.get_intrinsics()
    params_json_path = "neon.json"

    params = TrackerLineAndDotsParams.from_json(params_json_path)
    # params = TrackerLineAndDotsParams()

    tracker = TrackerLineAndDots(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, params=params
    )

    frame_counter = 1006
    deltas = []
    while True:
        frame = cam.get_frame()

        img = frame.bgr
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Raw Image", img)

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
        key = cv2.waitKey(0)

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
