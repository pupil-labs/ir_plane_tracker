import itertools
from dataclasses import dataclass, field
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


@dataclass
class LineParams:
    vx: float
    vy: float
    x0: float
    y0: float

    def __post_init__(self):
        norm = np.sqrt(self.vx**2 + self.vy**2)
        if norm == 0:
            raise ValueError("Direction vector cannot be zero.")
        self.vx /= norm
        self.vy /= norm

    def __iter__(self):
        yield self.vx
        yield self.vy
        yield self.x0
        yield self.y0


def line_projection_error(
    points: npt.NDArray[np.float64 | np.int64],
    params: LineParams,
    reduce_result=True,
) -> float | npt.NDArray[np.float64]:
    points = np.squeeze(points)
    res = np.abs(
        (points[:, 0] - params.x0) * params.vy - (points[:, 1] - params.y0) * params.vx
    )
    if reduce_result:
        res = np.mean(res)

    return res


def project_to_line(
    params: LineParams, points: np.ndarray
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    line_origin = np.array([params.x0, params.y0])
    line_dir = np.array([params.vx, params.vy])
    vecs = points - line_origin
    t = np.dot(vecs, line_dir)
    projections = line_origin + np.outer(t, line_dir)
    return t, projections


class Fragment:
    def __init__(self, support: npt.NDArray[np.float64 | np.int64]):
        self.support = support
        self.params = self._fit_line(support)
        self._start_pt = None
        self._end_pt = None

    def _fit_line(
        self,
        points: npt.NDArray[np.float64 | np.int64],
    ) -> LineParams:
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        params = LineParams(vx.item(), vy.item(), x0.item(), y0.item())
        return params

    @cached_property
    def projection_error(self) -> float:
        return line_projection_error(self.support, self.params, reduce_result=True)

    @property
    def start_pt(self) -> npt.NDArray[np.float64]:
        if self._start_pt is None or self._end_pt is None:
            self._start_pt, self._end_pt = self._contour_line_endpoints(
                self.support, self.params
            )
        return self._start_pt

    @property
    def end_pt(self) -> npt.NDArray[np.float64]:
        if self._start_pt is None or self._end_pt is None:
            self._start_pt, self._end_pt = self._contour_line_endpoints(
                self.support, self.params
            )
        return self._end_pt

    def _contour_line_endpoints(
        self, support: npt.NDArray[np.float64 | np.int64], params: LineParams
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        support = support.squeeze()

        deltas = support - [params.x0, params.y0]
        # Project each delta onto the direction vector
        projections = np.dot(deltas, [params.vx, params.vy])
        proj_min = np.min(projections)
        proj_max = np.max(projections)

        start_pt = (params.x0 + proj_min * params.vx, params.y0 + proj_min * params.vy)
        end_pt = (params.x0 + proj_max * params.vx, params.y0 + proj_max * params.vy)

        start_pt = np.array([x.item() for x in start_pt])
        end_pt = np.array([x.item() for x in end_pt])
        return start_pt, end_pt


class Orientation(Enum):
    LEFT = 1
    RIGHT = 2
    TOP = 3
    BOTTOM = 4


class FeatureLine:
    def __init__(
        self, points: npt.NDArray[np.float64], projections: npt.NDArray[np.float64]
    ):
        dir_vec = points[-1] - points[0]

        if abs(dir_vec[0]) > abs(dir_vec[1]):
            # HORIZONTAL
            if projections[1] - projections[0] > projections[3] - projections[2]:
                self.orientation = Orientation.RIGHT
            else:
                self.orientation = Orientation.LEFT
                points = points[::-1]

        else:
            # VERTICAL
            if projections[1] - projections[0] > projections[3] - projections[2]:
                self.orientation = Orientation.BOTTOM
            else:
                self.orientation = Orientation.TOP
                points = points[::-1]

        self.points = points


# class LinePositions(Enum):
#     TOP_LEFT = 1
#     TOP_RIGHT = 2
#     BOTTOM_LEFT = 3
#     BOTTOM_RIGHT = 4
#     LEFT = 5
#     RIGHT = 6


# class FeatureLineCombination:
#     def __init__(self) -> None:
#         self._map = dict.fromkeys(LinePositions)

#     def __getitem__(self, key: LinePositions) -> FeatureLine | None:
#         return self._map[key]

#     def __setitem__(self, key: LinePositions, value: FeatureLine) -> None:
#         self._map[key] = value

#     def copy(self) -> "FeatureLineCombination":
#         new = FeatureLineCombination()
#         new._map = self._map.copy()
#         return new

#     def __len__(self) -> int:
#         return sum(1 for v in self._map.values() if v is not None)


# class Combinations:
#     def __init__(self) -> None:
#         self._combinations: list[FeatureLineCombination] = [FeatureLineCombination()]

#     def add_line(self, line: FeatureLine, positions: list[LinePositions]) -> None:
#         new_combinations = []
#         for combination in self._combinations:
#             old_combination_length = len(new_combinations)

#             dir1 = line.ellipses[-1].center - line.ellipses[0].center
#             for position in positions:
#                 # Some lines need to be co-linear with other lines to be plausible
#                 if combination[position] is None:
#                     if not self._co_linearity_requirements(position, combination, dir1):
#                         continue

#                     if not self._min_line_distances_requirements(
#                         position, combination, dir1, line
#                     ):
#                         continue

#                     if not self._left_right_order_requirements(
#                         position, combination, dir1, line
#                     ):
#                         continue

#                     if not self._top_bottom_order_requirements(
#                         position, combination, dir1, line
#                     ):
#                         continue

#                     c = combination.copy()
#                     c[position] = line
#                     new_combinations.append(c)

#             if len(self._combinations) != old_combination_length:
#                 # Add a combination where this line is a false positive
#                 # If no new combination was added, the line can be ignored
#                 new_combinations.append(combination.copy())

#         self._combinations = new_combinations

#     def _co_linearity_requirements(
#         self,
#         position: LinePositions,
#         combination: FeatureLineCombination,
#         dir1: np.ndarray,
#     ) -> bool:
#         """Check if co-linearity requirements are met for this line."""
#         if (
#             (
#                 position == LinePositions.TOP_LEFT
#                 and combination[LinePositions.TOP_RIGHT] is not None
#             )
#             or (
#                 position == LinePositions.TOP_RIGHT
#                 and combination[LinePositions.TOP_LEFT] is not None
#             )
#             or (
#                 position == LinePositions.BOTTOM_LEFT
#                 and combination[LinePositions.BOTTOM_RIGHT] is not None
#             )
#             or (
#                 position == LinePositions.BOTTOM_RIGHT
#                 and combination[LinePositions.BOTTOM_LEFT] is not None
#             )
#         ):
#             position_map = {
#                 LinePositions.TOP_LEFT: LinePositions.TOP_RIGHT,
#                 LinePositions.TOP_RIGHT: LinePositions.TOP_LEFT,
#                 LinePositions.BOTTOM_LEFT: LinePositions.BOTTOM_RIGHT,
#                 LinePositions.BOTTOM_RIGHT: LinePositions.BOTTOM_LEFT,
#             }
#             position2 = position_map[position]
#             dir2 = (
#                 combination[position2].ellipses[-1].center
#                 - combination[position2].ellipses[0].center
#             )
#             angle = np.arccos(
#                 np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2))
#             )
#             if angle < np.deg2rad(5):
#                 return False
#         return True

#     def _min_line_distances_requirements(
#         self,
#         position: LinePositions,
#         combination: FeatureLineCombination,
#         dir1: np.ndarray,
#         line: FeatureLine,
#     ) -> bool:
#         # Some lines must have a minimum distance from one another
#         lines_with_min_distance = [
#             (LinePositions.TOP_LEFT, LinePositions.BOTTOM_LEFT),
#             (LinePositions.TOP_LEFT, LinePositions.BOTTOM_RIGHT),
#             (LinePositions.TOP_RIGHT, LinePositions.BOTTOM_RIGHT),
#             (LinePositions.TOP_RIGHT, LinePositions.BOTTOM_LEFT),
#             (LinePositions.BOTTOM_LEFT, LinePositions.TOP_LEFT),
#             (LinePositions.BOTTOM_LEFT, LinePositions.TOP_RIGHT),
#             (LinePositions.BOTTOM_RIGHT, LinePositions.TOP_RIGHT),
#             (LinePositions.BOTTOM_RIGHT, LinePositions.TOP_LEFT),
#             (LinePositions.LEFT, LinePositions.RIGHT),
#         ]
#         for pos1, pos2 in lines_with_min_distance:
#             if position == pos1 and combination[pos2] is not None:
#                 line2 = combination[pos2]
#                 distances = [
#                     self._point_line_distance(
#                         line.ellipses[0].center,
#                         dir1,
#                         line2.ellipses[0].center,
#                     ),
#                     self._point_line_distance(
#                         line.ellipses[0].center,
#                         dir1,
#                         line2.ellipses[-1].center,
#                     ),
#                 ]

#                 if min(distances) < 20:
#                     return False
#         return True

#     def _left_right_order_requirements(
#         self,
#         position: LinePositions,
#         combination: FeatureLineCombination,
#         dir1: np.ndarray,
#         line: FeatureLine,
#     ) -> bool:
#         # Some lines have a left/right order
#         if (
#             position == LinePositions.LEFT
#             and combination[LinePositions.RIGHT] is not None
#         ):
#             p1 = line.ellipses[0].center
#             p2 = combination[LinePositions.RIGHT].ellipses[0].center
#             if p1[0] > p2[0]:
#                 return False
#         elif (
#             position == LinePositions.RIGHT
#             and combination[LinePositions.LEFT] is not None
#         ):
#             p1 = combination[LinePositions.LEFT].ellipses[0].center
#             p2 = line.ellipses[0].center
#             if p1[0] > p2[0]:
#                 return False
#         elif (
#             position == LinePositions.TOP_LEFT
#             and combination[LinePositions.TOP_RIGHT] is not None
#         ):
#             p1 = line.ellipses[0].center
#             p2 = combination[LinePositions.TOP_RIGHT].ellipses[0].center
#             if p1[0] > p2[0]:
#                 return False
#         elif (
#             position == LinePositions.TOP_RIGHT
#             and combination[LinePositions.TOP_LEFT] is not None
#         ):
#             p1 = combination[LinePositions.TOP_LEFT].ellipses[0].center
#             p2 = line.ellipses[0].center
#             if p1[0] > p2[0]:
#                 return False
#         elif (
#             position == LinePositions.BOTTOM_LEFT
#             and combination[LinePositions.BOTTOM_RIGHT] is not None
#         ):
#             p1 = line.ellipses[0].center
#             p2 = combination[LinePositions.BOTTOM_RIGHT].ellipses[0].center
#             if p1[0] > p2[0]:
#                 return False
#         elif (
#             position == LinePositions.BOTTOM_RIGHT
#             and combination[LinePositions.BOTTOM_LEFT] is not None
#         ):
#             p1 = combination[LinePositions.BOTTOM_LEFT].ellipses[0].center
#             p2 = line.ellipses[0].center
#             if p1[0] > p2[0]:
#                 return False

#         return True

#     def _top_bottom_order_requirements(
#         self,
#         position: LinePositions,
#         combination: FeatureLineCombination,
#         dir1: np.ndarray,
#         line: FeatureLine,
#     ) -> bool:
#         # Some lines have a top/bottom order
#         if (
#             position == LinePositions.TOP_LEFT
#             and combination[LinePositions.BOTTOM_LEFT] is not None
#         ):
#             p1 = line.ellipses[0].center
#             p2 = combination[LinePositions.BOTTOM_LEFT].ellipses[0].center
#             if p1[1] > p2[1]:
#                 return False
#         elif (
#             position == LinePositions.TOP_LEFT
#             and combination[LinePositions.BOTTOM_RIGHT] is not None
#         ) or (
#             position == LinePositions.TOP_RIGHT
#             and combination[LinePositions.BOTTOM_RIGHT] is not None
#         ):
#             p1 = line.ellipses[0].center
#             p2 = combination[LinePositions.BOTTOM_RIGHT].ellipses[0].center
#             if p1[1] > p2[1]:
#                 return False
#         elif (
#             position == LinePositions.TOP_RIGHT
#             and combination[LinePositions.BOTTOM_LEFT] is not None
#         ):
#             p1 = line.ellipses[0].center
#             p2 = combination[LinePositions.BOTTOM_LEFT].ellipses[0].center
#             if p1[1] > p2[1]:
#                 return False
#         elif (
#             position == LinePositions.BOTTOM_LEFT
#             and combination[LinePositions.TOP_LEFT] is not None
#         ):
#             p1 = combination[LinePositions.TOP_LEFT].ellipses[0].center
#             p2 = line.ellipses[0].center
#             if p1[1] > p2[1]:
#                 return False
#         elif (
#             position == LinePositions.BOTTOM_LEFT
#             and combination[LinePositions.TOP_RIGHT] is not None
#         ) or (
#             position == LinePositions.BOTTOM_RIGHT
#             and combination[LinePositions.TOP_RIGHT] is not None
#         ):
#             p1 = combination[LinePositions.TOP_RIGHT].ellipses[0].center
#             p2 = line.ellipses[0].center
#             if p1[1] > p2[1]:
#                 return False

#         return True

#     def _point_line_distance(
#         self, line_point: np.ndarray, line_dir: np.ndarray, target_point: np.ndarray
#     ) -> float:
#         line_dir = line_dir / np.linalg.norm(line_dir)
#         vec = target_point - line_point
#         proj_length = np.dot(vec, line_dir)
#         proj_point = line_point + proj_length * line_dir
#         distance = np.linalg.norm(target_point - proj_point)
#         return distance

#     def filter_and_sort(self) -> None:
#         new_combinations = []
#         for combination in self._combinations:
#             # A single line does not suffice
#             if len(combination) < 2:
#                 continue

#             # Two co-linear lines do not suffice
#             if len(combination) == 2:
#                 if (
#                     combination[LinePositions.TOP_LEFT] is not None
#                     and combination[LinePositions.TOP_RIGHT] is not None
#                 ):
#                     continue
#                 if (
#                     combination[LinePositions.BOTTOM_LEFT] is not None
#                     and combination[LinePositions.BOTTOM_RIGHT] is not None
#                 ):
#                     continue

#             new_combinations.append(combination)
#         new_combinations = sorted(new_combinations, key=len, reverse=True)
#         self._combinations = new_combinations

#     def __iter__(self):
#         return iter(self._combinations)

#     def __len__(self) -> int:
#         return len(self._combinations)


@dataclass
class IRPlaneTrackerParams:
    plane_width: float = 28.4
    plane_height: float = 18.5
    top_left_margin: float = 1.57
    top_right_margin: float = 2.1
    bottom_left_margin: float = 1.55
    bottom_right_margin: float = 1.4
    right_top_margin: float = 4.6
    left_top_margin: float = 4.33
    norm_line_points: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 6.0, 8.0, 10.0])
    )
    img_size_factor: float = 1.0
    thresh_c: int = 40
    thresh_half_kernel_size: int = 20
    min_contour_area_line: int = 200
    max_contour_area_line: int = 850
    min_contour_area_ellipse: int = 24
    max_contour_area_ellipse: int = 180
    min_contour_count: int = 8
    min_contour_support: int = 6
    fragments_max_projection_error: float = 5.0
    min_line_fragments_count: int = 8
    min_ellipse_size: int = 6
    max_ellipse_aspect_ratio: float = 2.0
    min_ellipse_count: int = 8
    max_cr_error: float = 0.03
    min_feature_line_count: int = 2
    max_line_length: float = 200.0
    optimization_error_threshold: float = 5.0
    debug: bool = False

    @staticmethod
    def from_json(json_path: str) -> "IRPlaneTrackerParams":
        import json

        with open(json_path) as f:
            data = json.load(f)

        params = IRPlaneTrackerParams(
            img_size_factor=data.get("img_size_factor", 1.0),
            thresh_c=data.get("thresh_c", 75),
            thresh_half_kernel_size=data.get("thresh_half_kernel_size", 20),
            min_contour_area=data.get("min_contour_area", 20),
            max_contour_area=data.get("max_contour_area", 120),
            min_contour_count=data.get("min_contour_count", 8),
            min_contour_support=data.get("min_contour_support", 6),
            min_line_fragments_count=data.get("min_ellipse_count", 8),
            max_cr_error=data.get("max_cr_error", 0.03),
            min_feature_line_count=data.get("min_feature_line_count", 2),
            plane_width=data.get("total_width", 28.4),
            plane_height=data.get("total_height", 18.5),
            top_left_margin=data.get("top_left_margin", 1.57),
            top_right_margin=data.get("top_right_margin", 2.1),
            bottom_left_margin=data.get("bottom_left_margin", 1.55),
            bottom_right_margin=data.get("bottom_right_margin", 1.4),
            right_top_margin=data.get("right_top_margin", 4.6),
            left_top_margin=data.get("left_top_margin", 4.33),
            norm_line_points=np.array(data.get("norm_line_points", [0, 6, 8, 10])),
        )
        return params


class DebugData:
    def __init__(self, params: IRPlaneTrackerParams) -> None:
        self.params = params
        self.img_raw: npt.NDArray[np.uint8] | None = None
        self.img_gray: npt.NDArray[np.uint8] | None = None
        self._img_thresholded: npt.NDArray[np.uint8] | None = None
        self.contours_raw: list[npt.NDArray[np.int32]] | None = None
        self.contour_areas: list[float] | None = None
        self.contours_line: list[npt.NDArray[np.int32]] | None = None
        self.contours_ellipse: list[npt.NDArray[np.int32]] | None = None
        self.fragments_raw: list[Fragment] | None = None
        self.fragments_filtered: list[Fragment] | None = None
        self.ellipses_raw: list[Ellipse] | None = None
        self.ellipses_filtered: list[Ellipse] | None = None
        self.feature_lines_raw: list[FeatureLine] | None = None
        self.feature_lines_filtered: list[FeatureLine] | None = None
        self.feature_lines_processed: list[FeatureLine] | None = None
        self.cr_values: list[float] | None = None
        self.optimization_errors: list[float] = []
        self.optimization_final_combination: FeatureLineCombination | None = None
        self.plane_corners: npt.NDArray[np.float64] | None = None

    @property
    def img_thresholded(self) -> npt.NDArray[np.uint8] | None:
        return self._img_thresholded

    @img_thresholded.setter
    def img_thresholded(self, value: npt.NDArray[np.uint8]) -> None:
        self._img_thresholded = cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)  # type: ignore

    def visualize(self):
        cv2.imshow("Raw Image", self.img_raw)
        cv2.imshow("Thresholded Image", self.img_thresholded)

        self._visualize_contours()
        self._visualize_fragments()
        self._visualize_ellipses()
        self._visualize_feature_lines()
        self._visualize_feature_lines_processed()
        self._visualize_optimization()
        self._visualize_plane()

    def _visualize_contours(self):
        vis = self.img_thresholded.copy()
        cv2.drawContours(vis, self.contours_raw, -1, (0, 0, 255), 2)
        cv2.drawContours(vis, self.contours_line, -1, (0, 255, 0), 2)
        cv2.drawContours(vis, self.contours_ellipse, -1, (255, 0, 0), 2)

        for c, a in zip(self.contours_raw, self.contour_areas, strict=False):
            cv2.putText(
                vis,
                f"{a:.0f}",
                tuple(c[c[:, :, 1].argmin()][0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        cv2.putText(
            vis,
            "Line",
            (50, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"Min Area: {self.params.min_contour_area_line}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            f"Max Area: {self.params.max_contour_area_line}",
            (50, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            vis,
            "Ellipse",
            (300, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            vis,
            f"Min Area: {self.params.min_contour_area_ellipse}",
            (300, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            vis,
            f"Max Area: {self.params.max_contour_area_ellipse}",
            (300, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Contours", vis)

    def _visualize_fragments(self):
        vis = self.img_raw.copy()
        if self.fragments_raw is not None:
            for fragment in self.fragments_raw:
                cv2.polylines(
                    vis,
                    [np.array([fragment.start_pt, fragment.end_pt], dtype=int)],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )
                cv2.circle(
                    vis,
                    (int(fragment.start_pt[0]), int(fragment.start_pt[1])),
                    1,
                    (255, 0, 255),
                    -1,
                )
                cv2.circle(
                    vis,
                    (int(fragment.end_pt[0]), int(fragment.end_pt[1])),
                    1,
                    (255, 0, 255),
                    -1,
                )
                p = np.mean([fragment.start_pt, fragment.end_pt], axis=0).astype(int)
                cv2.putText(
                    vis,
                    f"{fragment.projection_error:<.1f}",
                    (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

        if self.fragments_filtered is not None:
            for fragment in self.fragments_filtered:
                cv2.polylines(
                    vis,
                    [np.array([fragment.start_pt, fragment.end_pt], dtype=int)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )

        cv2.putText(
            vis,
            f"Max Error: {self.params.fragments_max_projection_error}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Fragements", vis)

    def _visualize_ellipses(self):
        vis = self.img_thresholded.copy()
        if self.ellipses_raw is not None:
            for ellipse in self.ellipses_raw:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                size = (int(ellipse.size[0] / 2), int(ellipse.size[1] / 2))
                angle = ellipse.angle
                cv2.ellipse(vis, center, size, angle, 0, 360, (0, 0, 255), 2)

                cv2.putText(
                    vis,
                    (
                        f"{ellipse.minor_axis:.1f} "
                        f"{ellipse.major_axis / ellipse.minor_axis:.1f}"
                    ),
                    (center[0] - 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )

        if self.ellipses_filtered is not None:
            for ellipse in self.ellipses_filtered:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                size = (int(ellipse.size[0] / 2), int(ellipse.size[1] / 2))
                angle = ellipse.angle
                cv2.ellipse(vis, center, size, angle, 0, 360, (0, 255, 0), 2)

        cv2.putText(
            vis,
            f"Min Size: {self.params.min_ellipse_size}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.putText(
            vis,
            f"Max Aspect Ratio: {self.params.max_ellipse_aspect_ratio}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        cv2.imshow("Ellipses", vis)

    def _visualize_feature_lines(self):
        vis = self.img_thresholded.copy()
        if self.feature_lines_raw is not None and self.cr_values is not None:
            for line, cr in zip(self.feature_lines_raw, self.cr_values, strict=False):
                cv2.polylines(
                    vis,
                    [line.points.astype(int)],
                    isClosed=False,
                    color=(0, 0, 255),
                    thickness=2,
                )

                p = np.mean(line.points, axis=0).astype(int)
                cv2.putText(
                    vis,
                    f"{line.orientation}",
                    (int(p[0]), int(p[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )
                cv2.putText(
                    vis,
                    f"{cr:.3f}",
                    (int(p[0]), int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

                for p in line.points:
                    cv2.circle(
                        vis,
                        (int(p[0]), int(p[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )

        # if self.feature_lines_filtered is not None:
        #     for line in self.feature_lines_filtered:
        #         line_points = np.array(
        #             [ellipse.center for ellipse in line], dtype=np.int32
        #         )
        #         cv2.polylines(
        #             vis, [line_points], isClosed=False, color=(0, 255, 0), thickness=2
        #         )
        #         for ellipse in line:
        #             cv2.circle(
        #                 vis,
        #                 (int(ellipse.center[0]), int(ellipse.center[1])),
        #                 5,
        #                 (0, 255, 0),
        #                 -1,
        #             )

        cv2.imshow("Feature Lines", vis)

    def _visualize_feature_lines_processed(self):
        vis = self.img_thresholded.copy()
        if self.feature_lines_processed is not None:
            for line in self.feature_lines_processed:
                line_points = np.array(
                    [ellipse.center for ellipse in line.points], dtype=np.int32
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

        cv2.imshow("Feature Lines Processed", vis)

    def _visualize_optimization(self):
        vis = self.img_thresholded.copy()
        if self.optimization_final_combination is not None:
            for position, line in self.optimization_final_combination._map.items():
                if line is not None:
                    line_points = np.array(
                        [ellipse.center for ellipse in line.ellipses], dtype=np.int32
                    )
                    cv2.polylines(
                        vis,
                        [line_points],
                        isClosed=False,
                        color=(0, 255, 0),
                        thickness=2,
                    )
                    orientation = line.orientation.name
                    p = np.mean(line_points, axis=0).astype(int)
                    cv2.putText(
                        vis,
                        f"{position.name} {orientation}",
                        (int(p[0]), int(p[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            cv2.putText(
                vis,
                f"Final Error: {self.optimization_errors[-1]:.2f}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        cv2.imshow("Optimization Result", vis)

    def _visualize_plane(self):
        vis = self.img_raw.copy()

        if self.plane_corners is not None:
            cv2.polylines(
                vis,
                [self.plane_corners.astype(np.int32)],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            for corner in self.plane_corners:
                cv2.circle(
                    vis,
                    (int(corner[0]), int(corner[1])),
                    10,
                    (0, 255, 0),
                    -1,
                )

        cv2.imshow("Tracked Plane", vis)


class IRPlaneTracker:
    def __init__(
        self,
        camera_matrix: npt.NDArray[np.float64],
        dist_coeffs: npt.NDArray[np.float64],
        params: IRPlaneTrackerParams | None = None,
    ):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        if params is None:
            self.params = IRPlaneTrackerParams()
        else:
            self.params = params

        # self.obj_point_map = self.init_object_point_map()
        self.debug = DebugData(params)

    # def init_object_point_map(self) -> dict[LinePositions, npt.NDArray[np.float64]]:
    #     obj_map = {
    #         LinePositions.TOP_LEFT: np.column_stack((
    #             (
    #                 (10 - self.params.norm_line_points[::-1])
    #                 + self.params.top_left_margin
    #             )[::-1],
    #             np.zeros_like(self.params.norm_line_points),
    #             np.zeros_like(self.params.norm_line_points),
    #         )),
    #         LinePositions.TOP_RIGHT: np.column_stack((
    #             self.params.plane_width
    #             + (self.params.norm_line_points - 10)
    #             - self.params.top_right_margin,
    #             np.zeros_like(self.params.norm_line_points),
    #             np.zeros_like(self.params.norm_line_points),
    #         )),
    #         LinePositions.BOTTOM_LEFT: np.column_stack((
    #             (
    #                 (10 - self.params.norm_line_points[::-1])
    #                 + self.params.bottom_left_margin
    #             )[::-1],
    #             np.ones_like(self.params.norm_line_points) * self.params.plane_height,
    #             np.zeros_like(self.params.norm_line_points),
    #         )),
    #         LinePositions.BOTTOM_RIGHT: np.column_stack((
    #             self.params.plane_width
    #             + (self.params.norm_line_points - 10)
    #             - self.params.bottom_right_margin,
    #             np.ones_like(self.params.norm_line_points) * self.params.plane_height,
    #             np.zeros_like(self.params.norm_line_points),
    #         )),
    #         LinePositions.LEFT: np.column_stack((
    #             np.zeros_like(self.params.norm_line_points),
    #             self.params.norm_line_points + self.params.left_top_margin,
    #             np.zeros_like(self.params.norm_line_points),
    #         )),
    #         LinePositions.RIGHT: np.column_stack((
    #             np.ones_like(self.params.norm_line_points) * self.params.plane_width,
    #             (
    #                 10
    #                 - self.params.norm_line_points[::-1]
    #                 + self.params.right_top_margin
    #             )[::-1],
    #             np.zeros_like(self.params.norm_line_points),
    #         )),
    #     }
    #     return obj_map

    def get_contours(
        self, img: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        img = 255 - img

        thresh_kernel_size = self.params.thresh_half_kernel_size * 2 + 1
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            thresh_kernel_size,
            self.params.thresh_c,
        )
        # img = cv2.morphologyEx(img, cv2.MORPH_DILATE, np.ones((1, 1), np.uint8))
        self.debug.img_thresholded = img.copy()

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        self.debug.contours_raw = contours

        contour_areas = [cv2.contourArea(c) for c in contours]
        self.debug.contour_areas = contour_areas
        line_contours = [
            c
            for c, a in zip(contours, contour_areas, strict=False)
            if a > self.params.min_contour_area_line
            and a < self.params.max_contour_area_line
            and len(c) >= self.params.min_contour_support
        ]
        ellipse_contours = [
            c
            for c, a in zip(contours, contour_areas, strict=False)
            if a > self.params.min_contour_area_ellipse
            and a < self.params.max_contour_area_ellipse
            and len(c) >= self.params.min_contour_support
        ]

        self.debug.contours_line = line_contours
        self.debug.contours_ellipse = ellipse_contours

        return line_contours, ellipse_contours

    def fit_line_fragments(self, contours: list[np.ndarray]) -> list[Fragment]:
        fragments = [Fragment(c) for c in contours]
        self.debug.fragments_raw = fragments.copy()
        fragments = [
            f
            for f in fragments
            if f.projection_error < self.params.fragments_max_projection_error
        ]
        self.debug.fragments_filtered = fragments.copy()

        return fragments

    def fit_ellipses_to_contours(  # noqa: C901
        self, contours: list[np.ndarray], img_shape: tuple[int, int]
    ) -> list[Ellipse]:
        # ------------ Ellipse extraction --------------------------------------

        ellipses = []
        for contour in contours:
            points = contour.astype(np.float32)
            ellipse = cv2.fitEllipse(points)
            ellipse = Ellipse(*ellipse)  # type: ignore
            ellipses.append(ellipse)

        self.debug.ellipses_raw = ellipses

        ellipses_filtered = []
        for ellipse in ellipses:
            if (
                ellipse.major_axis
                > ellipse.minor_axis * self.params.max_ellipse_aspect_ratio
            ):
                continue
            if ellipse.minor_axis > min(img_shape[0], img_shape[1]) * 0.2:
                continue
            if ellipse.minor_axis < 0.5 * self.params.min_ellipse_size:
                continue
            if ellipse.major_axis < self.params.min_ellipse_size:
                continue
            if (
                ellipse.center[0] < 0
                or ellipse.center[0] >= img_shape[1]
                or ellipse.center[1] < 0
                or ellipse.center[1] >= img_shape[0]
            ):
                continue

            ellipses_filtered.append(ellipse)

        ellipses_deduplicated = []
        for idx, ellipse in enumerate(ellipses_filtered):
            # Check for double borders on circles and keep only larger ones
            add_ellipse = True
            for i in range(idx, len(ellipses_filtered)):
                dist_thresh = ellipse.minor_axis * 0.1
                dist = abs(ellipse.center[0] - ellipses_filtered[i].center[0]) + abs(
                    ellipse.center[1] - ellipses_filtered[i].center[1]
                )
                if (
                    dist < dist_thresh
                    and ellipse.minor_axis < ellipses_filtered[i].minor_axis
                ):
                    add_ellipse = False
                    break

            if add_ellipse:
                ellipses_deduplicated.append(ellipse)
        self.debug.ellipses_filtered = ellipses_deduplicated

        return ellipses

    @staticmethod
    def cross_ratio(points) -> float:
        AB = points[1] - points[0]
        BD = points[3] - points[1]
        AC = points[2] - points[0]
        CD = points[3] - points[2]
        # AB = np.linalg.norm(points[1] - points[0])
        # BD = np.linalg.norm(points[3] - points[1])
        # AC = np.linalg.norm(points[2] - points[0])
        # CD = np.linalg.norm(points[3] - points[2])
        cross_ratio = (AB / BD) / (AC / CD)
        return float(cross_ratio)

    def find_feature_lines(
        self, fragments: list[Fragment], ellipses: list[Ellipse]
    ) -> list[FeatureLine]:
        if len(ellipses) < 2 or len(fragments) < 1:
            return []

        ellipse_points = np.array([e.center for e in ellipses])
        feature_lines = []
        cr_values = []
        for frag in fragments:
            t, line_points = project_to_line(frag.params, ellipse_points)
            distance = np.linalg.norm(line_points - ellipse_points, axis=1)
            mask = distance < self.params.fragments_max_projection_error

            ellipse_candidates = ellipse_points[mask]
            t_candidates = t[mask]
            line_points = line_points[mask]

            frag_t, _ = project_to_line(
                frag.params, np.array([frag.start_pt, frag.end_pt])
            )
            for i, j in itertools.combinations(np.arange(len(ellipse_candidates)), 2):
                # Both ellipses must be on the same side of the fragment
                if t_candidates[i] < frag_t[1] and t_candidates[j] > frag_t[1]:
                    continue

                t_values = np.array([
                    frag_t[0],
                    frag_t[1],
                    t_candidates[i],
                    t_candidates[j],
                ])
                ordered_indices = np.argsort(t_values)
                t_values = t_values[ordered_indices]

                cr = self.cross_ratio(t_values)
                target_cr = 0.375
                if abs(cr - target_cr) < self.params.max_cr_error:
                    feature_line_points = np.array([
                        frag.start_pt,
                        frag.end_pt,
                        ellipse_candidates[i],
                        ellipse_candidates[j],
                    ])
                    feature_line_points = feature_line_points[ordered_indices]
                    feature_lines.append(FeatureLine(feature_line_points, t_values))
                    cr_values.append(cr)

        self.debug.feature_lines_raw = feature_lines
        self.debug.cr_values = cr_values
        return feature_lines

        # max_line_length = self.params.max_line_length
        # max_point_inter_distance = max_line_length * 0.6

        # feature_lines = []
        # for i, ellipse_i in enumerate(fragments):
        #     for j, ellipse_j in enumerate(fragments[i + 1 :], start=i + 1):
        #         # Check point distance
        #         vec_IJ = ellipse_j.center - ellipse_i.center
        #         dist_IJ = np.linalg.norm(vec_IJ)
        #         if dist_IJ > max_line_length:
        #             continue

        #         max_ellipse_difference = 0.5 * min(
        #             ellipse_i.major_axis, ellipse_j.major_axis
        #         )
        #         if (
        #             abs(ellipse_i.major_axis - ellipse_j.major_axis)
        #             > max_ellipse_difference
        #         ):
        #             continue

        #         # Compute line equation
        #         vec_IJ = ellipse_j.center - ellipse_i.center
        #         dot_IJ_IJ = np.dot(vec_IJ, vec_IJ)

        #         # Check all other ellipses if they fit to the line equation
        #         # Condition: Between two points are at most two other points
        #         # Not more and not less
        #         line_candidate = []
        #         nLine_Candidates = 0
        #         for k, ellipse_k in enumerate(fragments):
        #             if nLine_Candidates >= 2:
        #                 break

        #             if k in (i, j):
        #                 continue

        #             # Check point distance
        #             vec_IK = ellipse_k.center - ellipse_i.center
        #             dist_IK = np.linalg.norm(vec_IK)
        #             if dist_IK > max_point_inter_distance:
        #                 continue

        #             # Check area
        #             max_ellipse_difference = 0.5 * min(
        #                 ellipse_j.major_axis, ellipse_k.major_axis
        #             )
        #             if (
        #                 abs(ellipse_j.major_axis - ellipse_k.major_axis)
        #                 > max_ellipse_difference
        #             ):
        #                 continue

        #             # Check if k lies on the line between i and j
        #             t_k = np.dot(vec_IK, vec_IJ) / dot_IJ_IJ
        #             if t_k < 0 or t_k > 1:
        #                 continue

        #             # Check distance to line
        #             proj_k = ellipse_i.center + vec_IJ * t_k
        #             vec_KprojK = proj_k - ellipse_k.center
        #             d_k_sqr = (vec_KprojK[0] ** 2) + (vec_KprojK[1] ** 2)

        #             max_pixel_dist_to_line = np.max([4.0, ellipse_k.minor_axis])
        #             if d_k_sqr > max_pixel_dist_to_line:
        #                 continue

        #             for m, ellipse_m in enumerate(fragments[k + 1 :], start=k + 1):
        #                 if nLine_Candidates >= 2:
        #                     break

        #                 if m in (i, j):
        #                     continue

        #                 # Check point distance
        #                 vec_IL = ellipse_m.center - ellipse_i.center
        #                 dist_IL = np.linalg.norm(vec_IL)
        #                 if dist_IL > max_point_inter_distance:
        #                     continue

        #                 # Check area
        #                 max_ellipse_difference = 0.5 * min(
        #                     ellipse_k.major_axis, ellipse_m.major_axis
        #                 )
        #                 if (
        #                     abs(ellipse_k.major_axis - ellipse_m.major_axis)
        #                     > max_ellipse_difference
        #                 ):
        #                     continue

        #                 # Check if l lies on the line between i and j
        #                 t_m = np.dot(vec_IL, vec_IJ) / dot_IJ_IJ
        #                 if t_m < 0 or t_m > 1:
        #                     continue

        #                 # Check distance to line
        #                 proj_m = ellipse_i.center + vec_IJ * t_m
        #                 vec_MprojM = proj_m - ellipse_m.center
        #                 d_m_sqr = (vec_MprojM[0] ** 2) + (vec_MprojM[1] ** 2)

        #                 max_pixel_dist_to_line = np.max([4.0, ellipse_m.minor_axis])
        #                 if d_m_sqr > max_pixel_dist_to_line:
        #                     continue

        #                 line_candidate.append(ellipse_i)
        #                 if t_k < t_m:
        #                     line_candidate.append(ellipse_k)
        #                     line_candidate.append(ellipse_m)
        #                 else:
        #                     line_candidate.append(ellipse_m)
        #                     line_candidate.append(ellipse_k)
        #                 line_candidate.append(ellipse_j)
        #                 nLine_Candidates += 1

        #         if nLine_Candidates == 1:
        #             feature_lines.append(line_candidate)
        # self.debug.feature_lines_raw = feature_lines

        # cr_values = [self.cross_ratio(line) for line in feature_lines]
        # target_cr_value = 0.375
        # self.debug.cr_values = cr_values
        # final_feature_lines = [
        #     line
        #     for line, cr in zip(feature_lines, cr_values, strict=False)
        #     if abs(cr - target_cr_value) < self.params.max_cr_error
        # ]
        # self.debug.feature_lines_filtered = final_feature_lines

        # return final_feature_lines

    # def get_obj_and_img_points(
    #     self, combination: FeatureLineCombination
    # ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    #     img_points = []
    #     obj_points = []
    #     for position in LinePositions:
    #         if combination[position] is not None:
    #             img_points.extend([
    #                 ellipse.center for ellipse in combination[position].ellipses
    #             ])

    #             obj_points.extend(self.obj_point_map[position])

    #     img_points = np.array(img_points, dtype=np.float64)
    #     obj_points = np.array(obj_points, dtype=np.float64)
    #     return obj_points, img_points

    # def get_possible_combinations(
    #     self, feature_lines: list[FeatureLine]
    # ) -> Combinations:
    #     combinations = Combinations()
    #     for line in feature_lines:
    #         if line.orientation == Orientation.LEFT:
    #             combinations.add_line(
    #                 line, [LinePositions.TOP_LEFT, LinePositions.BOTTOM_LEFT]
    #             )
    #         elif line.orientation == Orientation.RIGHT:
    #             combinations.add_line(
    #                 line, [LinePositions.TOP_RIGHT, LinePositions.BOTTOM_RIGHT]
    #             )
    #         elif line.orientation == Orientation.TOP:
    #             combinations.add_line(line, [LinePositions.RIGHT])
    #         elif line.orientation == Orientation.BOTTOM:
    #             combinations.add_line(line, [LinePositions.LEFT])
    #         else:
    #             raise ValueError("Unknown orientation")

    #     combinations.filter_and_sort()

    #     return combinations

    # def fit_camera_pose(
    #     self, combinations: Combinations
    # ) -> tuple[npt.NDArray[np.float64] | None, npt.NDArray[np.float64] | None]:
    #     optimization_error_threshold = (
    #         self.params.optimization_error_threshold * self.params.img_size_factor
    #     )
    #     rvec = tvec = None
    #     mean_error = float("inf")
    #     num_optimizations = 0
    #     self.debug.optimization_errors = []
    #     for combination in combinations:
    #         obj_points, img_points = self.get_obj_and_img_points(combination)

    #         ret, rvec, tvec = cv2.solvePnP(
    #             obj_points,
    #             img_points,
    #             self.camera_matrix,
    #             self.dist_coeffs,
    #             flags=cv2.SOLVEPNP_IPPE,
    #         )
    #         num_optimizations += 1

    #         if not ret:
    #             rvec = tvec = None
    #             continue

    #         mean_error = self.reprojection_error(obj_points, img_points, rvec, tvec)
    #         self.debug.optimization_errors.append(mean_error)

    #         # if self.params.debug:
    #         #     vis = self.vis.copy()
    #         #     for pos, line in combination._map.items():
    #         #         if line is not None:
    #         #             line_points = np.array(
    #         #                 [ellipse.center for ellipse in line.ellipses],
    #         #                 dtype=np.int32,
    #         #             )
    #         #             cv2.polylines(
    #         #                 vis,
    #         #                 [line_points],
    #         #                 isClosed=False,
    #         #                 color=(0, 255, 0),
    #         #                 thickness=2,
    #         #             )
    #         #             p = np.mean(line_points, axis=0).astype(int)
    #         #             cv2.putText(
    #         #                 vis,
    #         #                 f"{pos.name}",
    #         #                 (int(p[0]), int(p[1])),
    #         #                 cv2.FONT_HERSHEY_SIMPLEX,
    #         #                 0.5,
    #         #                 (0, 0, 255),
    #         #                 1,
    #         #             )
    #         #     cv2.putText(
    #         #         vis,
    #         #         f"Error: {mean_error:.2f}px",
    #         #         (10, 30),
    #         #         cv2.FONT_HERSHEY_SIMPLEX,
    #         #         1,
    #         #         (0, 255, 0),
    #         #         2,
    #         #     )
    #         #     cv2.imshow("Chosen Combination", vis)
    #         #     key = cv2.waitKey(0)
    #         #     if key == ord("q"):
    #         #         break
    #         if mean_error < optimization_error_threshold:
    #             break

    #     if mean_error >= optimization_error_threshold:
    #         rvec = tvec = None

    #     if rvec is not None and tvec is not None:
    #         self.debug.optimization_final_combination = combination

    #     return rvec, tvec

    # def calculate_plane_corners(
    #     self, rvec: npt.NDArray[np.float64], tvec: npt.NDArray[np.float64]
    # ) -> npt.NDArray[np.float64]:
    #     plane_corners = np.array([
    #         [0, 0, 0],
    #         [28.5, 0, 0],
    #         [28.5, 19.6, 0],
    #         [0, 19.6, 0],
    #     ])
    #     img_corners, _ = cv2.projectPoints(
    #         plane_corners,
    #         rvec,
    #         tvec,
    #         self.camera_matrix,
    #         self.dist_coeffs,
    #     )
    #     img_corners = img_corners.squeeze().astype(np.float64)
    #     self.debug.plane_corners = img_corners

    #     return img_corners

    def __call__(self, image: npt.NDArray[np.uint8]):
        self.debug = DebugData(self.params)
        self.debug.img_raw = image.copy()
        # image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs)

        if self.params.debug:
            self.vis = image.copy()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.debug.img_gray = image.copy()

        line_contours, ellipse_contours = self.get_contours(image)
        # if len(contours) < self.params.min_contour_count:
        #     return None

        fragments = self.fit_line_fragments(line_contours)
        # if len(line_fragments) < self.params.min_line_fragments_count:
        #     return None

        ellipses = self.fit_ellipses_to_contours(ellipse_contours, image.shape[:2])
        # if len(ellipses) < self.params.min_ellipse_count:
        #     return None

        feature_lines = self.find_feature_lines(fragments, ellipses)
        # if len(feature_lines) < self.params.min_feature_line_count:
        #     return None

        # feature_lines = [FeatureLine(line) for line in feature_lines]
        # self.debug.feature_lines_processed = feature_lines

        # combinations = self.get_possible_combinations(feature_lines)

        # rvec, tvec = self.fit_camera_pose(combinations)

        # if rvec is None or tvec is None:
        #     return None

        # screen_corners = self.calculate_plane_corners(rvec, tvec)

        # return screen_corners

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
