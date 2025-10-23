import numpy as np
import numpy.typing as npt
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QWidget


class FeatureOverlay(QWidget):
    feature_params_changed = Signal(object)

    def __init__(self, screen_size_px: tuple[int, int], parent=None):
        super().__init__(parent)

        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background:transparent;")
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowFlag(Qt.WindowTransparentForInput)

        self.screen_size_mm = 552, 310
        self.screen_size_px = screen_size_px
        self.mm2px = self.screen_size_px[0] / self.screen_size_mm[0]
        self.__padding_mm = 5
        self.__circle_diameter_mm = 7
        self.__line_thickness_mm = 7
        self.__norm_points_mm = np.array([0, 20, 40, 100])
        self.feature_color = Qt.white
        self.background_color = Qt.black

    @property
    def norm_points_px(self) -> npt.NDArray[np.int64]:
        return (self.__norm_points_mm * self.mm2px).astype(np.int64)

    @property
    def padding_px(self) -> int:
        return int(self.__padding_mm * self.mm2px)

    @property
    def circle_radius_px(self) -> int:
        return int(self.__circle_diameter_mm / 2 * self.mm2px)

    @property
    def line_thickness_px(self) -> int:
        return int(self.__line_thickness_mm * self.mm2px)

    @property
    def feature_values_px(self) -> np.ndarray:
        return np.concatenate([
            self._top_feature_px[::-1],
            self._right_feature_px[::-1],
            self._bottom_feature_px[::-1],
            self._left_feature_px[::-1],
        ])

    @property
    def feature_thickness_px(self):
        return max(self.circle_radius_px * 2, self.line_thickness_px)

    def _paint_circle(self, painter, center_x_px, center_y_px):
        painter.drawEllipse(
            int(center_x_px - self.circle_radius_px),
            int(center_y_px - self.circle_radius_px),
            int(self.circle_radius_px * 2),
            int(self.circle_radius_px * 2),
        )

    def _paint_line(self, painter, start_x_px, start_y_px, length_px, vertical=False):
        if vertical:
            x = start_x_px - self.line_thickness_px / 2
            y = start_y_px
            width = self.line_thickness_px
            height = length_px
        else:
            x = start_x_px
            y = start_y_px - self.line_thickness_px / 2
            width = length_px
            height = self.line_thickness_px
        painter.drawRect(x, y, width, height)

    @property
    def _top_feature_px(self) -> npt.NDArray[np.int64]:
        points = np.column_stack([
            self.norm_points_px.max() - self.norm_points_px[::-1],
            np.zeros(4),
        ])
        points[:, 0] -= points[:, 0].max() / 2
        points[:, 0] += self.screen_size_px[0] / 2
        points[:, 1] += self.feature_thickness_px / 2 + self.padding_px
        points = points[::-1].astype(np.int64)
        return points

    def _paint_top_feature(self, painter):
        p1, p2, p3, p4 = self._top_feature_px

        painter.setBrush(self.background_color)
        painter.setPen(Qt.NoPen)
        p = np.array(p4) - self.padding_px
        p[1] -= self.circle_radius_px
        width = max(self.norm_points_px) + 2 * self.padding_px + self.circle_radius_px
        height = self.feature_thickness_px + 2 * self.padding_px
        painter.drawRect(*p, width, height)

        painter.setBrush(self.feature_color)
        self._paint_circle(painter, *p1)
        self._paint_circle(painter, *p2)
        line_length = p3[0] - p4[0]
        self._paint_line(painter, *p4, line_length, vertical=False)

    @property
    def _bottom_feature_px(self) -> npt.NDArray[np.int64]:
        points = np.column_stack([
            self.norm_points_px,
            np.ones(4) * self.screen_size_px[1],
        ])
        points[:, 0] -= points[:, 0].max() / 2
        points[:, 0] += self.screen_size_px[0] / 2
        points[:, 1] -= self.feature_thickness_px / 2 + self.padding_px
        points = points.astype(np.int64)
        return points

    def _paint_bottom_feature(self, painter):
        p1, p2, p3, p4 = self._bottom_feature_px

        painter.setBrush(self.background_color)
        painter.setPen(Qt.NoPen)
        p = np.array(p1) - self.padding_px - self.circle_radius_px
        width = max(self.norm_points_px) + 2 * self.padding_px + self.circle_radius_px
        height = self.feature_thickness_px + 2 * self.padding_px
        painter.drawRect(*p, width, height)

        painter.setBrush(self.feature_color)
        self._paint_circle(painter, *p1)
        self._paint_circle(painter, *p2)
        line_length = p3[0] - p4[0]
        self._paint_line(painter, *p4, line_length, vertical=False)

    @property
    def _left_feature_px(self) -> npt.NDArray[np.int64]:
        points = np.column_stack([
            np.zeros(4),
            self.norm_points_px,
        ])
        points[:, 1] -= points[:, 1].max() / 2
        points[:, 1] += self.screen_size_px[1] / 2
        points[:, 0] += self.feature_thickness_px / 2 + self.padding_px
        points = points.astype(np.int64)
        return points

    def _paint_left_feature(self, painter):
        p1, p2, p3, p4 = self._left_feature_px

        painter.setBrush(self.background_color)
        painter.setPen(Qt.NoPen)
        p = np.array(p1) - self.padding_px - self.circle_radius_px
        width = self.feature_thickness_px + 2 * self.padding_px
        height = max(self.norm_points_px) + 2 * self.padding_px + self.circle_radius_px
        painter.drawRect(*p, width, height)

        painter.setBrush(self.feature_color)
        self._paint_circle(painter, *p1)
        self._paint_circle(painter, *p2)
        line_length = p3[1] - p4[1]
        self._paint_line(painter, *p4, line_length, vertical=True)

    @property
    def _right_feature_px(self) -> npt.NDArray[np.int64]:
        points = np.column_stack([
            np.ones(4) * self.screen_size_px[0],
            self.norm_points_px.max() - self.norm_points_px[::-1],
        ])
        points[:, 1] -= points[:, 1].max() / 2
        points[:, 1] += self.screen_size_px[1] / 2
        points[:, 0] -= self.feature_thickness_px / 2 + self.padding_px
        points = points[::-1].astype(np.int64)
        return points

    def _paint_right_feature(self, painter):
        p1, p2, p3, p4 = self._right_feature_px

        painter.setBrush(self.background_color)
        painter.setPen(Qt.NoPen)
        p = np.array(p4) - self.padding_px
        p[0] -= self.circle_radius_px
        width = self.feature_thickness_px + 2 * self.padding_px
        height = max(self.norm_points_px) + 2 * self.padding_px + self.circle_radius_px
        painter.drawRect(*p, width, height)

        painter.setBrush(self.feature_color)
        self._paint_circle(painter, *p1)
        self._paint_circle(painter, *p2)
        line_length = p3[1] - p4[1]
        self._paint_line(painter, *p4, line_length, vertical=True)

    def paintEvent(self, event):
        if self.isVisible():
            with QPainter(self) as painter:
                self._paint_top_feature(painter)
                self._paint_bottom_feature(painter)
                self._paint_right_feature(painter)
                self._paint_left_feature(painter)
