from pstats import Stats
from time import time

import cv2

import pupil_labs.neon_recording as plr
from pupil_labs.ir_plane_tracker import TrackerLineAndDots, TrackerLineAndDotsParams


def main():
    rec = plr.open("offline_recording/data/indoor3")

    calibration = rec.calibration
    assert calibration is not None
    camera_matrix = calibration.scene_camera_matrix
    dist_coeffs = calibration.scene_distortion_coefficients

    params_json_path = "neon.json"

    params = TrackerLineAndDotsParams.from_json(params_json_path)

    tracker = TrackerLineAndDots(
        camera_matrix=camera_matrix, dist_coeffs=None, params=params
    )
    screenshot = cv2.imread("offline_recording/data/screenshot.png")

    deltas = []
    rec_data = zip(
        rec.scene.sample(rec.scene.ts), rec.gaze.sample(rec.scene.ts), strict=False
    )
    for frame, gaze in rec_data:
        img = frame.bgr

        img = cv2.undistort(img, camera_matrix, dist_coeffs)

        start_ts = time()
        localization = tracker(img)
        end_ts = time()
        delta = end_ts - start_ts
        deltas.append(delta)
        if len(deltas) > 30:
            deltas = deltas[-30:]
        avg_delta = sum(deltas) / len(deltas)
        fps = 1.0 / avg_delta
        print(f"FPS: {fps:.2f}", end="\r")

        tracker.debug.visualize()

        cv2.circle(img, (int(gaze.x), int(gaze.y)), 20, (0, 255, 0), 3)

        screen_vis = screenshot.copy()
        if localization is not None:
            gaze_mapped = localization.img2plane @ [gaze.x, gaze.y, 1]
            gaze_mapped = gaze_mapped / gaze_mapped[2]
            gaze_mapped = gaze_mapped[:2] * screenshot.shape[1::-1]
            cv2.circle(
                screen_vis,
                (int(gaze_mapped[0]), int(gaze_mapped[1])),
                40,
                (0, 255, 0),
                3,
            )

        screen_vis = cv2.resize(screen_vis, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("Scene Camera", img)
        cv2.imshow("Mapped Gaze", screen_vis)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(
                "/home/marc/pupil_labs/IR_plane_tracker/src/pupil_labs/ir_plane_tracker/data/dashed_images/test_img.png",
                img,
            )


if __name__ == "__main__":
    import cProfile

    with cProfile.Profile() as pr:
        main()
    # pr.sort_stats("time").print_stats("ir_plane_tracker")
    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(r"\((?!\_).*\)$")  # Exclude private and magic callables.
