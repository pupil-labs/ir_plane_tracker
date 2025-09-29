from time import time

import cv2
import numpy as np
from camera import HDDigitalCam

from pupil_labs.ir_plane_tracker import IRPlaneTracker, IRPlaneTrackerParams


def main():
    cam = HDDigitalCam()
    camera_matrix = np.load("camera_matrix.npy")
    dist_coeffs = np.load("dist_coeffs.npy")
    params_json_path = "live_demo.json"

    params = IRPlaneTrackerParams.from_json(params_json_path)

    tracker = IRPlaneTracker(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, params=params
    )

    frame_counter = 600
    deltas = []
    while True:
        frame = cam.get_frame()

        img = frame.bgr
        img = cv2.undistort(img, camera_matrix, dist_coeffs)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Raw Image", img)

        start_ts = time()
        screen_corners = tracker(img)
        end_ts = time()
        delta = end_ts - start_ts
        deltas.append(delta)
        if len(deltas) > 30:
            deltas = deltas[-30:]
        avg_delta = sum(deltas) / len(deltas)
        fps = 1.0 / avg_delta
        vis = img.copy()
        cv2.putText(
            vis,
            f"FPS: {fps:.1f}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if screen_corners is None:
            cv2.putText(
                vis,
                "Screen not found",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        cv2.polylines(
            vis, [screen_corners], isClosed=True, color=(0, 255, 0), thickness=2
        )

        cv2.imshow("Tracked Screen", vis)

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
