from pstats import Stats
from time import time

import cv2

import pupil_labs.neon_recording as plr
from pupil_labs.ir_plane_tracker import TrackerLineAndDots, TrackerLineAndDotsParams


def main():
    rec = plr.open("offline_recording/data/ipad")

    video_writer = cv2.VideoWriter(
        "ipad.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        # (2240, 720),
        (960, 720),
    )
    # 1280x720 # Screen
    # 960x720 # Scene

    calibration = rec.calibration
    assert calibration is not None
    camera_matrix = calibration.scene_camera_matrix
    dist_coeffs = calibration.scene_distortion_coefficients

    params_json_path = "neon_ipad_small.json"
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
        scene_img = frame.bgr
        scene_img = cv2.undistort(scene_img, camera_matrix, dist_coeffs)
        scene_img = scene_img[150:1050, 200:1400]
        scene_img = cv2.resize(scene_img, (1600, 1200))
        scene_img = cv2.resize(
            scene_img, None, fx=params.img_size_factor, fy=params.img_size_factor
        )

        start_ts = time()
        localization = tracker(scene_img)
        end_ts = time()
        delta = end_ts - start_ts
        deltas.append(delta)
        if len(deltas) > 30:
            deltas = deltas[-30:]
        avg_delta = sum(deltas) / len(deltas)
        fps = 1.0 / avg_delta
        print(f"FPS: {fps:.2f}", end="\r")

        tracker.debug.visualize()

        cv2.circle(scene_img, (int(gaze.x), int(gaze.y)), 20, (0, 255, 0), 3)

        screen_vis = screenshot.copy()
        scene_vis = scene_img.copy()
        if localization is not None:
            cv2.polylines(
                scene_vis, [localization.corners.astype(int)], True, (255, 0, 0), 3
            )

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

        screen_vis = cv2.resize(screen_vis, (1280, 720))
        scene_vis = cv2.resize(scene_vis, (960, 720))
        video_img = scene_vis  # cv2.hconcat([scene_img, screen_img])
        video_writer.write(video_img)
        # cv2.imshow("Scene Camera", scene_img)
        # cv2.imshow("Mapped Gaze", screen_img)

        key = cv2.waitKey(0)
        if key == ord("q"):
            break
        elif key == ord("s"):
            cv2.imwrite(
                "/home/marc/pupil_labs/IR_plane_tracker/src/pupil_labs/ir_plane_tracker/data/dashed_images/test_img.png",
                scene_img,
            )
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import cProfile

    with cProfile.Profile() as pr:
        main()
    # pr.sort_stats("time").print_stats("ir_plane_tracker")
    stats = Stats(pr).sort_stats("cumtime")
    stats.print_stats(r"\((?!\_).*\)$")  # Exclude private and magic callables.
