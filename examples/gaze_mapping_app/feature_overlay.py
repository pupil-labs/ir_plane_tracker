import numpy as np
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

        self.screen_size_mm = 552, 310
        self.screen_size_px = screen_size_px
        self.mm2px = self.screen_size_px[0] / self.screen_size_mm[0]
        self.padding_mm = 5
        self.circle_diameter_mm = 7
        self.line_thickness_mm = 3
        self.norm_points_mm = [0, 20, 40, 100]

    @property
    def feature_values_mm(self) -> np.ndarray:
        print(
            np.concatenate([
                self._top_obj_points_mm[::-1],
                self._right_obj_points_mm[::-1],
                self._bottom_obj_points_mm[::-1],
                self._left_obj_points_mm[::-1],
            ])
        )
        return np.concatenate([
            self._top_obj_points_mm[::-1],
            self._right_obj_points_mm[::-1],
            self._bottom_obj_points_mm[::-1],
            self._left_obj_points_mm[::-1],
        ])

    @property
    def feature_thickness_mm(self):
        return max(self.circle_diameter_mm, self.line_thickness_mm)

    @property
    def feature_width_mm(self):
        return (
            max(self.norm_points_mm) + 2 * self.padding_mm + self.circle_diameter_mm / 2
        )

    @property
    def feature_height_mm(self):
        return self.feature_thickness_mm + 2 * self.padding_mm

    def _paint_circle(self, painter, center_x_mm, center_y_mm):
        radius_mm = self.circle_diameter_mm / 2
        center_x_px = center_x_mm * self.mm2px
        center_y_px = center_y_mm * self.mm2px
        radius_px = radius_mm * self.mm2px
        painter.drawEllipse(
            int(center_x_px - radius_px),
            int(center_y_px - radius_px),
            int(radius_px * 2),
            int(radius_px * 2),
        )

    def _paint_line(self, painter, start_x_mm, start_y_mm, length_mm, vertical=False):
        thickness_mm = self.line_thickness_mm
        if vertical:
            x = start_x_mm - thickness_mm / 2
            y = start_y_mm
            width = thickness_mm
            height = length_mm
        else:
            x = start_x_mm
            y = start_y_mm - thickness_mm / 2
            width = length_mm
            height = thickness_mm
        painter.drawRect(
            int(x * self.mm2px),
            int(y * self.mm2px),
            int(width * self.mm2px),
            int(height * self.mm2px),
        )

    def _paint_feature_background(self, painter, target_pos, vertical=False):
        if vertical:
            x = target_pos[0] - self.padding_mm - self.feature_thickness_mm / 2
            y = target_pos[1] - self.padding_mm - self.circle_diameter_mm / 2
            width = self.feature_height_mm
            height = self.feature_width_mm + self.circle_diameter_mm

        else:
            x = target_pos[0] - self.padding_mm - self.circle_diameter_mm / 2
            y = target_pos[1] - self.padding_mm - self.circle_diameter_mm / 2
            width = self.feature_width_mm + self.circle_diameter_mm
            height = self.feature_height_mm
        painter.drawRect(
            int(x * self.mm2px),
            int(y * self.mm2px),
            int(width * self.mm2px),
            int(height * self.mm2px),
        )

    @property
    def _top_obj_points_mm(self):
        target_x = self.screen_size_mm[0] / 2 - self.feature_width_mm / 2
        target_y = self.feature_height_mm / 2
        p4 = (target_x, target_y)
        p3 = (target_x + self.norm_points_mm[3] - self.norm_points_mm[2], target_y)
        p2 = (target_x + self.norm_points_mm[3] - self.norm_points_mm[1], target_y)
        p1 = (target_x + self.norm_points_mm[3] - self.norm_points_mm[0], target_y)
        return np.array([p1, p2, p3, p4])

    def _paint_top_feature(self, painter):
        p1, p2, p3, p4 = self._top_obj_points_mm

        painter.setBrush(Qt.black)
        painter.setPen(Qt.NoPen)
        self._paint_feature_background(painter, (p4[0], p4[1]))

        painter.setBrush(Qt.white)
        self._paint_circle(painter, p1[0], p1[1])
        self._paint_circle(painter, p2[0], p2[1])
        line_length = p3[0] - p4[0]
        self._paint_line(painter, p4[0], p4[1], line_length, vertical=False)

    @property
    def _bottom_obj_points_mm(self):
        target_x = self.screen_size_mm[0] / 2 - self.feature_width_mm / 2
        target_y = self.screen_size_mm[1] - self.feature_height_mm / 2
        p1 = (target_x, target_y)
        p2 = (target_x + self.norm_points_mm[1], target_y)
        p3 = (target_x + self.norm_points_mm[2], target_y)
        p4 = (target_x + self.norm_points_mm[3], target_y)
        return np.array([p1, p2, p3, p4])

    def _paint_bottom_feature(self, painter):
        p1, p2, p3, p4 = self._bottom_obj_points_mm

        painter.setBrush(Qt.black)
        painter.setPen(Qt.NoPen)
        self._paint_feature_background(painter, (p1[0], p1[1]))

        painter.setBrush(Qt.white)
        self._paint_circle(painter, p1[0], p1[1])
        self._paint_circle(painter, p2[0], p2[1])
        line_length = p3[0] - p4[0]
        self._paint_line(painter, p4[0], p4[1], line_length, vertical=False)

    @property
    def _right_obj_points_mm(self):
        target_x = self.screen_size_mm[0] - self.feature_height_mm / 2
        target_y = self.screen_size_mm[1] / 2 - self.feature_width_mm / 2
        p4 = (target_x, target_y)
        p3 = (target_x, target_y + self.norm_points_mm[3] - self.norm_points_mm[2])
        p2 = (target_x, target_y + self.norm_points_mm[3] - self.norm_points_mm[1])
        p1 = (target_x, target_y + self.norm_points_mm[3] - self.norm_points_mm[0])
        return np.array([p1, p2, p3, p4])

    def _paint_right_feature(self, painter):
        p1, p2, p3, p4 = self._right_obj_points_mm

        painter.setBrush(Qt.black)
        painter.setPen(Qt.NoPen)
        self._paint_feature_background(painter, (p4[0], p4[1]), vertical=True)

        painter.setBrush(Qt.white)
        self._paint_circle(painter, p1[0], p1[1])
        self._paint_circle(painter, p2[0], p2[1])
        line_length = p3[1] - p4[1]
        self._paint_line(painter, p4[0], p4[1], line_length, vertical=True)

    @property
    def _left_obj_points_mm(self):
        target_x = self.feature_height_mm / 2
        target_y = self.screen_size_mm[1] / 2 - self.feature_width_mm / 2
        p1 = (target_x, target_y)
        p2 = (target_x, target_y + self.norm_points_mm[1])
        p3 = (target_x, target_y + self.norm_points_mm[2])
        p4 = (target_x, target_y + self.norm_points_mm[3])
        return np.array([p1, p2, p3, p4])

    def _paint_left_feature(self, painter):
        p1, p2, p3, p4 = self._left_obj_points_mm

        painter.setBrush(Qt.black)
        painter.setPen(Qt.NoPen)
        self._paint_feature_background(painter, (p1[0], p1[1]), vertical=True)

        painter.setBrush(Qt.white)
        self._paint_circle(painter, p1[0], p1[1])
        self._paint_circle(painter, p2[0], p2[1])
        line_length = p3[1] - p4[1]
        self._paint_line(painter, p4[0], p4[1], line_length, vertical=True)

        # target_x = self.feature_height_mm / 2
        # target_y = self.screen_size_mm[1] / 2 - self.feature_width_mm / 2

        # painter.setBrush(Qt.black)
        # painter.setPen(Qt.NoPen)
        # x = target_x
        # y = target_y
        # self._paint_feature_background(painter, (target_x, target_y), vertical=True)

        # painter.setBrush(Qt.white)
        # x = target_x
        # y = target_y
        # self._paint_circle(painter, x, y)
        # x = target_x
        # y = target_y + self.norm_points_mm[1]
        # self._paint_circle(painter, x, y)
        # x = target_x
        # y = target_y + self.norm_points_mm[2]
        # line_length = self.norm_points_mm[3] - self.norm_points_mm[2]
        # self._paint_line(painter, x, y, line_length, vertical=True)

    def paintEvent(self, event):
        if self.isVisible():
            with QPainter(self) as painter:
                self._paint_top_feature(painter)
                self._paint_bottom_feature(painter)
                self._paint_right_feature(painter)
                self._paint_left_feature(painter)
