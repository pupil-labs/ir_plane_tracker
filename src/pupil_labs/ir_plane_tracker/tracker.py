from enum import Enum
from functools import cached_property

import cv2
import numpy as np
import numpy.typing as npt


class Ellipse:
    def __init__(
        self,
        center: tuple[float, float],
        size: tuple[float, float],
        angle: float,
    ):
        self.center = np.array(center)
        self.size = np.array(size)
        self.angle = angle

    def __repr__(self):
        return f"Ellipse(center={self.center}, size={self.size}, angle={self.angle})"

    @cached_property
    def major_axis(self) -> float:
        return max(self.size)

    @cached_property
    def minor_axis(self) -> float:
        return min(self.size)


class Orientation(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4


class FeatureLine:
    def __init__(self, ellipses: list[Ellipse]):
        dir_vec = ellipses[-1].center - ellipses[0].center

        if abs(dir_vec[0]) > abs(dir_vec[1]):
            # HORIZONTAL
            ellipses = sorted(ellipses, key=lambda e: e.center[0])
            relative_positions = self._get_relative_positions(ellipses)

            if (
                relative_positions[1] - relative_positions[0]
                > relative_positions[3] - relative_positions[2]
            ):
                self.orientation = Orientation.RIGHT
            else:
                self.orientation = Orientation.LEFT
                ellipses = ellipses[::-1]

        else:
            # VERTICAL
            ellipses = sorted(ellipses, key=lambda e: e.center[1])
            relative_positions = self._get_relative_positions(ellipses)

            if (
                relative_positions[1] - relative_positions[0]
                > relative_positions[3] - relative_positions[2]
            ):
                self.orientation = Orientation.BOTTOM
            else:
                self.orientation = Orientation.TOP
                ellipses = ellipses[::-1]

        self.ellipses = ellipses

    def _get_relative_positions(
        self, ellipses: list[Ellipse]
    ) -> npt.NDArray[np.float64]:
        dir_vec = ellipses[-1].center - ellipses[0].center
        relative_positions = []
        for ellipse in ellipses:
            relative_pos = np.dot(
                ellipse.center - ellipses[0].center, dir_vec
            ) / np.dot(dir_vec, dir_vec)
            relative_positions.append(relative_pos)
        relative_positions = np.array(relative_positions)
        return relative_positions


class LinePositions(Enum):
    TOP_LEFT = 1
    TOP_RIGHT = 2
    BOTTOM_LEFT = 3
    BOTTOM_RIGHT = 4
    LEFT = 5
    RIGHT = 6


class FeatureLineCombination:
    def __init__(self) -> None:
        self._map = dict.fromkeys(LinePositions)

    def __getitem__(self, key: LinePositions) -> FeatureLine | None:
        return self._map[key]

    def __setitem__(self, key: LinePositions, value: FeatureLine) -> None:
        self._map[key] = value

    def copy(self) -> "FeatureLineCombination":
        new = FeatureLineCombination()
        new._map = self._map.copy()
        return new

    def __len__(self) -> int:
        return sum(1 for v in self._map.values() if v is not None)


class Combinations:
    def __init__(self) -> None:
        self._combinations: list[FeatureLineCombination] = [FeatureLineCombination()]

    def add_line(self, line: FeatureLine, positions: list[LinePositions]) -> None:
        new_combinations = []
        for combination in self._combinations:
            old_combination_length = len(new_combinations)

            dir1 = line.ellipses[-1].center - line.ellipses[0].center
            for position in positions:
                # Some lines need to be co-linear with other lines to be plausible
                if combination[position] is None:
                    if not self._co_linearity_requirements(position, combination, dir1):
                        continue

                    if not self._min_line_distances_requirements(
                        position, combination, dir1, line
                    ):
                        continue

                    if not self._left_right_order_requirements(
                        position, combination, dir1, line
                    ):
                        continue

                    if not self._top_bottom_order_requirements(
                        position, combination, dir1, line
                    ):
                        continue

                    c = combination.copy()
                    c[position] = line
                    new_combinations.append(c)

            if len(self._combinations) != old_combination_length:
                # Add a combination where this line is a false positive
                # If no new combination was added, the line can be ignored
                new_combinations.append(combination.copy())

        self._combinations = new_combinations

    def _co_linearity_requirements(
        self,
        position: LinePositions,
        combination: FeatureLineCombination,
        dir1: np.ndarray,
    ) -> bool:
        """Check if co-linearity requirements are met for this line."""
        if (
            (
                position == LinePositions.TOP_LEFT
                and combination[LinePositions.TOP_RIGHT] is not None
            )
            or (
                position == LinePositions.TOP_RIGHT
                and combination[LinePositions.TOP_LEFT] is not None
            )
            or (
                position == LinePositions.BOTTOM_LEFT
                and combination[LinePositions.BOTTOM_RIGHT] is not None
            )
            or (
                position == LinePositions.BOTTOM_RIGHT
                and combination[LinePositions.BOTTOM_LEFT] is not None
            )
        ):
            position_map = {
                LinePositions.TOP_LEFT: LinePositions.TOP_RIGHT,
                LinePositions.TOP_RIGHT: LinePositions.TOP_LEFT,
                LinePositions.BOTTOM_LEFT: LinePositions.BOTTOM_RIGHT,
                LinePositions.BOTTOM_RIGHT: LinePositions.BOTTOM_LEFT,
            }
            position2 = position_map[position]
            dir2 = (
                combination[position2].ellipses[-1].center
                - combination[position2].ellipses[0].center
            )
            angle = np.arccos(
                np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
            )
            if angle < np.deg2rad(5):
                return False
        return True

    def _min_line_distances_requirements(
        self,
        position: LinePositions,
        combination: FeatureLineCombination,
        dir1: np.ndarray,
        line: FeatureLine,
    ) -> bool:
        # Some lines must have a minimum distance from one another
        lines_with_min_distance = [
            (LinePositions.TOP_LEFT, LinePositions.BOTTOM_LEFT),
            (LinePositions.TOP_LEFT, LinePositions.BOTTOM_RIGHT),
            (LinePositions.TOP_RIGHT, LinePositions.BOTTOM_RIGHT),
            (LinePositions.TOP_RIGHT, LinePositions.BOTTOM_LEFT),
            (LinePositions.BOTTOM_LEFT, LinePositions.TOP_LEFT),
            (LinePositions.BOTTOM_LEFT, LinePositions.TOP_RIGHT),
            (LinePositions.BOTTOM_RIGHT, LinePositions.TOP_RIGHT),
            (LinePositions.BOTTOM_RIGHT, LinePositions.TOP_LEFT),
            (LinePositions.LEFT, LinePositions.RIGHT),
        ]
        for pos1, pos2 in lines_with_min_distance:
            if position == pos1 and combination[pos2] is not None:
                line2 = combination[pos2]
                distances = [
                    self._point_line_distance(
                        line.ellipses[0].center,
                        dir1,
                        line2.ellipses[0].center,
                    ),
                    self._point_line_distance(
                        line.ellipses[0].center,
                        dir1,
                        line2.ellipses[-1].center,
                    ),
                ]

                if min(distances) < 20:
                    return False
        return True

    def _left_right_order_requirements(  # noqa: C901
        self,
        position: LinePositions,
        combination: FeatureLineCombination,
        dir1: np.ndarray,
        line: FeatureLine,
    ) -> bool:
        # Some lines have a left/right order
        if (
            position == LinePositions.LEFT
            and combination[LinePositions.RIGHT] is not None
        ):
            p1 = line.ellipses[0].center
            p2 = combination[LinePositions.RIGHT].ellipses[0].center
            if p1[0] > p2[0]:
                return False
        elif (
            position == LinePositions.RIGHT
            and combination[LinePositions.LEFT] is not None
        ):
            p1 = combination[LinePositions.LEFT].ellipses[0].center
            p2 = line.ellipses[0].center
            if p1[0] > p2[0]:
                return False
        elif (
            position == LinePositions.TOP_LEFT
            and combination[LinePositions.TOP_RIGHT] is not None
        ):
            p1 = line.ellipses[0].center
            p2 = combination[LinePositions.TOP_RIGHT].ellipses[0].center
            if p1[0] > p2[0]:
                return False
        elif (
            position == LinePositions.TOP_RIGHT
            and combination[LinePositions.TOP_LEFT] is not None
        ):
            p1 = combination[LinePositions.TOP_LEFT].ellipses[0].center
            p2 = line.ellipses[0].center
            if p1[0] > p2[0]:
                return False
        elif (
            position == LinePositions.BOTTOM_LEFT
            and combination[LinePositions.BOTTOM_RIGHT] is not None
        ):
            p1 = line.ellipses[0].center
            p2 = combination[LinePositions.BOTTOM_RIGHT].ellipses[0].center
            if p1[0] > p2[0]:
                return False
        elif (
            position == LinePositions.BOTTOM_RIGHT
            and combination[LinePositions.BOTTOM_LEFT] is not None
        ):
            p1 = combination[LinePositions.BOTTOM_LEFT].ellipses[0].center
            p2 = line.ellipses[0].center
            if p1[0] > p2[0]:
                return False

        return True

    def _top_bottom_order_requirements(  # noqa: C901
        self,
        position: LinePositions,
        combination: FeatureLineCombination,
        dir1: np.ndarray,
        line: FeatureLine,
    ) -> bool:
        # Some lines have a top/bottom order
        if (
            position == LinePositions.TOP_LEFT
            and combination[LinePositions.BOTTOM_LEFT] is not None
        ):
            p1 = line.ellipses[0].center
            p2 = combination[LinePositions.BOTTOM_LEFT].ellipses[0].center
            if p1[1] > p2[1]:
                return False
        elif (
            position == LinePositions.TOP_LEFT
            and combination[LinePositions.BOTTOM_RIGHT] is not None
        ) or (
            position == LinePositions.TOP_RIGHT
            and combination[LinePositions.BOTTOM_RIGHT] is not None
        ):
            p1 = line.ellipses[0].center
            p2 = combination[LinePositions.BOTTOM_RIGHT].ellipses[0].center
            if p1[1] > p2[1]:
                return False
        elif (
            position == LinePositions.TOP_RIGHT
            and combination[LinePositions.BOTTOM_LEFT] is not None
        ):
            p1 = line.ellipses[0].center
            p2 = combination[LinePositions.BOTTOM_LEFT].ellipses[0].center
            if p1[1] > p2[1]:
                return False
        elif (
            position == LinePositions.BOTTOM_LEFT
            and combination[LinePositions.TOP_LEFT] is not None
        ):
            p1 = combination[LinePositions.TOP_LEFT].ellipses[0].center
            p2 = line.ellipses[0].center
            if p1[1] > p2[1]:
                return False
        elif (
            position == LinePositions.BOTTOM_LEFT
            and combination[LinePositions.TOP_RIGHT] is not None
        ) or (
            position == LinePositions.BOTTOM_RIGHT
            and combination[LinePositions.TOP_RIGHT] is not None
        ):
            p1 = combination[LinePositions.TOP_RIGHT].ellipses[0].center
            p2 = line.ellipses[0].center
            if p1[1] > p2[1]:
                return False

        return True

    def _point_line_distance(
        self, line_point: np.ndarray, line_dir: np.ndarray, target_point: np.ndarray
    ) -> float:
        line_dir = line_dir / np.linalg.norm(line_dir)
        vec = target_point - line_point
        proj_length = np.dot(vec, line_dir)
        proj_point = line_point + proj_length * line_dir
        distance = np.linalg.norm(target_point - proj_point)
        return distance

    def filter_and_sort(self) -> None:
        new_combinations = []
        for combination in self._combinations:
            # A single line does not suffice
            if len(combination) < 2:
                continue

            # Two co-linear lines do not suffice
            if len(combination) == 2:
                if (
                    combination[LinePositions.TOP_LEFT] is not None
                    and combination[LinePositions.TOP_RIGHT] is not None
                ):
                    continue
                if (
                    combination[LinePositions.BOTTOM_LEFT] is not None
                    and combination[LinePositions.BOTTOM_RIGHT] is not None
                ):
                    continue

            new_combinations.append(combination)
        new_combinations = sorted(new_combinations, key=len, reverse=True)
        self._combinations = new_combinations

    def __iter__(self):
        return iter(self._combinations)

    def __len__(self) -> int:
        return len(self._combinations)


class IRPlaneTracker:
    def __init__(
        self,
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        m_img_size_factor: float,
        thresh_c: int = 75,
        thresh_half_kernel_size: int = 20,
        min_contour_area: int = 20,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.img_size_factor = m_img_size_factor
        self.thresh_c = thresh_c
        self.thresh_kernel_size = (
            2 * int(thresh_half_kernel_size * m_img_size_factor) + 1
        )
        self.min_contour_area = min_contour_area
        self.debug = True
        self.vis = None

        total_width = 28.4
        total_height = 18.5
        top_left_margin = 1.57
        top_right_margin = 2.1
        bottom_left_margin = 1.55
        bottom_right_margin = 1.4
        right_top_margin = 4.6
        left_top_margin = 4.33
        norm_line_points = np.array([0, 6, 8, 10])
        self.obj_point_map = {
            LinePositions.TOP_LEFT: np.column_stack((
                ((10 - norm_line_points[::-1]) + top_left_margin)[::-1],
                np.zeros_like(norm_line_points),
                np.zeros_like(norm_line_points),
            )),
            LinePositions.TOP_RIGHT: np.column_stack((
                total_width + (norm_line_points - 10) - top_right_margin,
                np.zeros_like(norm_line_points),
                np.zeros_like(norm_line_points),
            )),
            LinePositions.BOTTOM_LEFT: np.column_stack((
                ((10 - norm_line_points[::-1]) + bottom_left_margin)[::-1],
                np.ones_like(norm_line_points) * total_height,
                np.zeros_like(norm_line_points),
            )),
            LinePositions.BOTTOM_RIGHT: np.column_stack((
                total_width + (norm_line_points - 10) - bottom_right_margin,
                np.ones_like(norm_line_points) * total_height,
                np.zeros_like(norm_line_points),
            )),
            LinePositions.LEFT: np.column_stack((
                np.zeros_like(norm_line_points),
                norm_line_points + left_top_margin,
                np.zeros_like(norm_line_points),
            )),
            LinePositions.RIGHT: np.column_stack((
                np.ones_like(norm_line_points) * total_width,
                (10 - norm_line_points[::-1] + right_top_margin)[::-1],
                np.zeros_like(norm_line_points),
            )),
        }

    def get_contours(self, img: np.ndarray) -> list[np.ndarray]:
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.thresh_kernel_size,
            self.thresh_c,
        )

        if self.debug:
            cv2.imshow("Thresholded Image", img)

        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )

        contours = [c for c in contours if cv2.contourArea(c) > self.min_contour_area]

        if self.debug:
            vis = self.vis.copy()
            for cnt in contours:
                cv2.drawContours(vis, [cnt], 0, (255, 165, 0), 2)
                cv2.putText(
                    vis,
                    f"{cv2.contourArea(cnt):.0f}",
                    tuple(cnt[cnt[:, :, 1].argmin()][0]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("Contours", vis)

        return contours

    def fit_ellipses_to_contours(  # noqa: C901
        self, contours: list[np.ndarray], img_shape: tuple[int, int]
    ) -> list[Ellipse]:
        # ------------ Ellipse extraction --------------------------------------
        min_ellipse_size = 3
        # int min_contour_points = int(1.5 * min_ellipse_size);
        max_ellipse_aspect_ratio = 2.0  # 7

        ellipses = []
        for i in range(len(contours)):
            count = contours[i].shape[0]
            if count < 6:
                continue

            contour_area = cv2.contourArea(contours[i])
            if contour_area < 20:
                continue

            pointsf = contours[i].astype(np.float32)
            ellipse = cv2.fitEllipse(pointsf)
            ellipse = Ellipse(*ellipse)  # type: ignore

            # Plausibility checks
            box_max = max(ellipse.size[0], ellipse.size[1])
            box_min = min(ellipse.size[0], ellipse.size[1])
            if box_max > box_min * max_ellipse_aspect_ratio:
                continue
            if box_max > min(img_shape[0], img_shape[1]) * 0.2:
                continue
            if box_min < 0.5 * min_ellipse_size:
                continue
            if box_max < min_ellipse_size:
                continue
            if (
                ellipse.center[0] < 0
                or ellipse.center[0] >= img_shape[1]
                or ellipse.center[1] < 0
                or ellipse.center[1] >= img_shape[0]
            ):
                continue

            add_ellipse = True

            # Check for double borders on circles and keep only larger ones
            for i in range(len(ellipses)):
                dist_thresh = box_min * 0.1
                dist = abs(ellipse.center[0] - ellipses[i].center[0]) + abs(
                    ellipse.center[1] - ellipses[i].center[1]
                )
                if dist < dist_thresh:
                    add_ellipse = False
                    ellipses[i] = ellipse
                    break

            if add_ellipse:
                ellipses.append(ellipse)

        if self.debug:
            vis = self.vis.copy()
            for ellipse in ellipses:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                size = (int(ellipse.size[0] / 2), int(ellipse.size[1] / 2))
                cv2.ellipse(vis, center, size, ellipse.angle, 0, 360, (255, 165, 0), 2)
                cv2.putText(
                    vis,
                    f"{max(ellipse.size):.1f}",
                    (int(ellipse.center[0]), int(ellipse.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("Ellipses - Max Axis", vis)

            vis = self.vis.copy()
            for ellipse in ellipses:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                size = (int(ellipse.size[0] / 2), int(ellipse.size[1] / 2))
                cv2.ellipse(vis, center, size, ellipse.angle, 0, 360, (255, 165, 0), 2)
                cv2.putText(
                    vis,
                    f"{max(ellipse.size) / min(ellipse.size):.1f}",
                    (int(ellipse.center[0]), int(ellipse.center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )
            cv2.imshow("Ellipses - Aspect Ratio", vis)
        return ellipses

    @staticmethod
    def cross_ratio(line: list[Ellipse]) -> float:
        AB = np.linalg.norm(line[1].center - line[0].center)
        BD = np.linalg.norm(line[3].center - line[1].center)
        AC = np.linalg.norm(line[2].center - line[0].center)
        CD = np.linalg.norm(line[3].center - line[2].center)
        cross_ratio = (AB / BD) / (AC / CD)
        return cross_ratio

    def find_feature_lines(self, ellipses: list[Ellipse]) -> list[list[Ellipse]]:  # noqa: C901
        max_line_length = 200 * self.img_size_factor
        max_point_inter_distance = max_line_length * 0.6

        feature_lines = []
        for i, ellipse_i in enumerate(ellipses):
            for j, ellipse_j in enumerate(ellipses[i + 1 :], start=i + 1):
                # Check point distance
                vec_IJ = ellipse_j.center - ellipse_i.center
                dist_IJ = np.linalg.norm(vec_IJ)
                if dist_IJ > max_line_length:
                    continue

                max_ellipse_difference = 0.5 * min(
                    ellipse_i.major_axis, ellipse_j.major_axis
                )
                if (
                    abs(ellipse_i.major_axis - ellipse_j.major_axis)
                    > max_ellipse_difference
                ):
                    continue

                # Compute line equation
                vec_IJ = ellipse_j.center - ellipse_i.center
                dot_IJ_IJ = np.dot(vec_IJ, vec_IJ)

                # Check all other ellipses if they fit to the line equation
                # Condition: Between two points are at most two other points
                # Not more and not less
                line_candidate = []
                nLine_Candidates = 0
                for k, ellipse_k in enumerate(ellipses):
                    if nLine_Candidates >= 2:
                        break

                    if k in (i, j):
                        continue

                    # Check point distance
                    vec_IK = ellipse_k.center - ellipse_i.center
                    dist_IK = np.linalg.norm(vec_IK)
                    if dist_IK > max_point_inter_distance:
                        continue

                    # Check area
                    max_ellipse_difference = 0.5 * min(
                        ellipse_j.major_axis, ellipse_k.major_axis
                    )
                    if (
                        abs(ellipse_j.major_axis - ellipse_k.major_axis)
                        > max_ellipse_difference
                    ):
                        continue

                    # Check if k lies on the line between i and j
                    t_k = np.dot(vec_IK, vec_IJ) / dot_IJ_IJ
                    if t_k < 0 or t_k > 1:
                        continue

                    # Check distance to line
                    proj_k = ellipse_i.center + vec_IJ * t_k
                    vec_KprojK = proj_k - ellipse_k.center
                    d_k_sqr = (vec_KprojK[0] ** 2) + (vec_KprojK[1] ** 2)

                    max_pixel_dist_to_line = np.max([4.0, ellipse_k.minor_axis])
                    if d_k_sqr > max_pixel_dist_to_line:
                        continue

                    for m, ellipse_m in enumerate(ellipses[k + 1 :], start=k + 1):
                        if nLine_Candidates >= 2:
                            break

                        if m in (i, j):
                            continue

                        # Check point distance
                        vec_IL = ellipse_m.center - ellipse_i.center
                        dist_IL = np.linalg.norm(vec_IL)
                        if dist_IL > max_point_inter_distance:
                            continue

                        # Check area
                        max_ellipse_difference = 0.5 * min(
                            ellipse_k.major_axis, ellipse_m.major_axis
                        )
                        if (
                            abs(ellipse_k.major_axis - ellipse_m.major_axis)
                            > max_ellipse_difference
                        ):
                            continue

                        # Check if l lies on the line between i and j
                        t_m = np.dot(vec_IL, vec_IJ) / dot_IJ_IJ
                        if t_m < 0 or t_m > 1:
                            continue

                        # Check distance to line
                        proj_m = ellipse_i.center + vec_IJ * t_m
                        vec_MprojM = proj_m - ellipse_m.center
                        d_m_sqr = (vec_MprojM[0] ** 2) + (vec_MprojM[1] ** 2)

                        max_pixel_dist_to_line = np.max([4.0, ellipse_m.minor_axis])
                        if d_m_sqr > max_pixel_dist_to_line:
                            continue

                        line_candidate.append(ellipse_i)
                        if t_k < t_m:
                            line_candidate.append(ellipse_k)
                            line_candidate.append(ellipse_m)
                        else:
                            line_candidate.append(ellipse_m)
                            line_candidate.append(ellipse_k)
                        line_candidate.append(ellipse_j)
                        nLine_Candidates += 1

                if nLine_Candidates == 1:
                    feature_lines.append(line_candidate)

        cross_ratio_max_dist = 0.03
        final_feature_lines = []
        cr_values_debug = []
        for line in feature_lines:
            cr = self.cross_ratio(line)

            # Check correctness of cross ratio
            if abs(cr - 0.375) < cross_ratio_max_dist:
                final_feature_lines.append(line)
                cr_values_debug.append(cr)

        if self.debug:
            vis = self.vis.copy()

            cv2.putText(
                vis,
                f"Lines: {len(final_feature_lines)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                vis,
                f"Ellipses: {len(ellipses)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            for line, cr in zip(final_feature_lines, cr_values_debug, strict=False):
                line_points = np.array(
                    [ellipse.center for ellipse in line], dtype=np.int32
                )
                cv2.polylines(
                    vis, [line_points], isClosed=False, color=(0, 255, 0), thickness=2
                )
                # cross ratio
                cv2.putText(
                    vis,
                    f"{cr:.3f}",
                    (int(line[0].center[0]), int(line[0].center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                for ellipse in line:
                    cv2.circle(
                        vis,
                        (int(ellipse.center[0]), int(ellipse.center[1])),
                        3,
                        (255, 0, 0),
                        -1,
                    )
            cv2.imshow("Feature Lines - CR", vis)

        return final_feature_lines

    def get_obj_and_img_points(
        self, combination: FeatureLineCombination
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        img_points = []
        obj_points = []
        for position in LinePositions:
            if combination[position] is not None:
                img_points.extend([
                    ellipse.center for ellipse in combination[position].ellipses
                ])

                obj_points.extend(self.obj_point_map[position])

        img_points = np.array(img_points, dtype=np.float64)
        obj_points = np.array(obj_points, dtype=np.float64)
        return obj_points, img_points

    def get_possible_combinations(
        self, feature_lines: list[FeatureLine]
    ) -> Combinations:
        combinations = Combinations()
        for line in feature_lines:
            if line.orientation == Orientation.LEFT:
                combinations.add_line(
                    line, [LinePositions.TOP_LEFT, LinePositions.BOTTOM_LEFT]
                )
            elif line.orientation == Orientation.RIGHT:
                combinations.add_line(
                    line, [LinePositions.TOP_RIGHT, LinePositions.BOTTOM_RIGHT]
                )
            elif line.orientation == Orientation.TOP:
                combinations.add_line(line, [LinePositions.RIGHT])
            elif line.orientation == Orientation.BOTTOM:
                combinations.add_line(line, [LinePositions.LEFT])
            else:
                raise ValueError("Unknown orientation")

        combinations.filter_and_sort()

        return combinations

    def fit_camera_pose(
        self, combinations: Combinations
    ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
        rvec = tvec = None
        mean_error = float("inf")
        num_optimizations = 0
        for combination in combinations:
            obj_points, img_points = self.get_obj_and_img_points(combination)

            ret, rvec, tvec = cv2.solvePnP(
                obj_points,
                img_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            num_optimizations += 1

            if not ret:
                rvec = tvec = None
                continue

            mean_error = self.reprojection_error(obj_points, img_points, rvec, tvec)

            if mean_error < 5.0:
                break

        if mean_error >= 5.0:
            rvec = tvec = None

        # if self.debug:
        #     vis = self.vis.copy()
        #     cv2.putText(
        #         vis,
        #         f"Optis: {num_optimizations}",
        #         (50, 100),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2,
        #     )
        #     cv2.putText(
        #         vis,
        #         f"Reproj Error: {mean_error:.2f}",
        #         (50, 150),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         1,
        #         (0, 0, 255),
        #         2,
        #     )
        #     for pos, line in combination._map.items():
        #         if line is not None:
        #             cv2.polylines(
        #                 vis,
        #                 [
        #                     np.array(
        #                         [e.center for e in line.ellipses],
        #                         dtype=np.int32,
        #                     )
        #                 ],
        #                 isClosed=False,
        #                 color=(0, 255, 0),
        #                 thickness=2,
        #             )
        #             p = np.mean(
        #                 np.array([e.center for e in line.ellipses]), axis=0
        #             ).astype(int)
        #             cv2.putText(
        #                 vis,
        #                 pos.name,
        #                 (int(p[0]), int(p[1])),
        #                 cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.5,
        #                 (0, 0, 255),
        #                 1,
        #             )

        #     cv2.imshow("Pose Estimation - Line Mapping", vis)
        return rvec, tvec

    def calculate_screen_corners(
        self, rvec: npt.NDArray[np.float64], tvec: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int32]:
        screen_corners = np.array([
            [0, 0, 0],
            [28.5, 0, 0],
            [28.5, 19.6, 0],
            [0, 19.6, 0],
        ])
        img_corners, _ = cv2.projectPoints(
            screen_corners,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        img_corners = img_corners.squeeze().astype(np.int32)

        return img_corners

    def __call__(self, image: npt.NDArray[np.uint8]):
        # image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

        if self.debug:
            self.vis = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = 255 - image

        contours = self.get_contours(image)
        if len(contours) < 4:
            return None

        ellipses = self.fit_ellipses_to_contours(contours, image.shape[:2])
        if len(ellipses) < 4:
            return None

        feature_lines = self.find_feature_lines(ellipses)
        if len(feature_lines) < 2:
            return None

        feature_lines = [FeatureLine(line) for line in feature_lines]

        if self.debug:
            vis = self.vis.copy()
            for line in feature_lines:
                line_points = np.array(
                    [ellipse.center for ellipse in line.ellipses], dtype=np.int32
                )
                cv2.polylines(
                    vis, [line_points], isClosed=False, color=(0, 255, 0), thickness=2
                )
                orientation = line.orientation.name
                p = np.mean(line_points, axis=0).astype(int)
                cv2.putText(
                    vis,
                    orientation,
                    (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
            cv2.imshow("Feature Lines - Orientation", vis)

            vis = self.vis.copy()
            for line in feature_lines:
                for idx, p in enumerate(line.ellipses):
                    cv2.circle(
                        vis,
                        (int(p.center[0]), int(p.center[1])),
                        3,
                        (255, 0, 0),
                        -1,
                    )

                    cv2.putText(
                        vis,
                        f"{idx}",
                        (int(p.center[0]), int(p.center[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (0, 255, 0),
                        1,
                    )

            cv2.imshow("Feature Lines - Point Order", vis)

        combinations = self.get_possible_combinations(feature_lines)

        rvec, tvec = self.fit_camera_pose(combinations)

        if rvec is None or tvec is None:
            return None

        screen_corners = self.calculate_screen_corners(rvec, tvec)

        return screen_corners

    def reprojection_error(self, obj_points, img_points, rvec, tvec) -> float:
        projected_points, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected_points = projected_points.squeeze()
        error = np.linalg.norm(img_points - projected_points, axis=1)
        mean_error = np.mean(error)
        return mean_error
