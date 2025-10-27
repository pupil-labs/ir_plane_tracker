from time import time

import cv2

from pupil_labs.ir_plane_tracker.tracker import (
    Tracker,
    TrackerParams,
)


def main():
    # cam = HDDigitalCam()
    # camera_matrix = np.load("camera_matrix.npy")
    # dist_coeffs = np.load("dist_coeffs.npy")
    # params_json_path = "hddigital.json"
    from common.camera import SceneCam

    cam = SceneCam()
    camera_matrix, dist_coeffs = cam.get_intrinsics()
    params_json_path = "resources/neon_ipad.json"

    params = TrackerParams.from_json(params_json_path)

    tracker = Tracker(camera_matrix=camera_matrix, dist_coeffs=None, params=params)

    frame_counter = 1006
    deltas = []
    while True:
        frame = cam.get_frame()

        img = frame.bgr
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
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
