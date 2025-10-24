from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker.feature_overlay import FeatureOverlay


class FeatureDisplayApp(QApplication):
    data_changed = Signal(object, object, object)

    def __init__(self):
        super().__init__()
        self.setApplicationDisplayName("Feature Display App")
        target_screen = self.screens()[0]  # Primary screen
        self.feature_overlay = FeatureOverlay(target_screen)


def run():
    app = FeatureDisplayApp()
    app.exec()


if __name__ == "__main__":
    run()
