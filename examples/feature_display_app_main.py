import click
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

from pupil_labs.ir_plane_tracker.feature_overlay import FeatureOverlay
from pupil_labs.ir_plane_tracker.tracker_params_wrapper import (
    TrackerParamsWrapper,
)


class FeatureDisplayApp(QApplication):
    data_changed = Signal(object, object, object)

    def __init__(self, params_path: str):
        super().__init__()
        self.setApplicationDisplayName("Feature Display App")
        self.params = TrackerParamsWrapper.from_json(params_path)
        target_screen = self.screens()[-1]
        self.feature_overlay = FeatureOverlay(target_screen, self.params)
        self.feature_overlay.toggle_visibility()


@click.command()
@click.option(
    "--params_path",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to tracker parameters JSON file.",
)
def main(params_path: str):
    app = FeatureDisplayApp(params_path)
    app.exec()


if __name__ == "__main__":
    main()
