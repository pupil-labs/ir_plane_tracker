import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QObject, Signal

from pupil_labs.ir_plane_tracker import Tracker, TrackerParams


class TrackerWrapper(QObject):
    params_changed = Signal(TrackerParams)

    def __init__(self, camera_matrix: npt.NDArray[np.float64]) -> None:
        super().__init__()
        self.tracker = Tracker(
            camera_matrix=camera_matrix,
            dist_coeffs=None,  # type: ignore
        )

    def set_params(self, params: TrackerParams) -> None:
        if self.tracker.params != params:
            self.tracker.params = params
            self.params_changed.emit(self.tracker.params)

    def update_params(self, params: dict) -> None:
        params_changed = False
        for key, val in params.items():
            if not hasattr(self.tracker.params, key):
                raise KeyError(f"Unknown parameter: {key}")

            if val != getattr(self.tracker.params, key):
                setattr(self.tracker.params, key, val)
                params_changed = True
        if params_changed:
            self.params_changed.emit(self.tracker.params)

    def __call__(self, scene_img: npt.NDArray[np.uint8]):
        return self.tracker(scene_img)

    @property
    def debug(self):
        return self.tracker.debug
