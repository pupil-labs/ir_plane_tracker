import numpy as np
from PySide6.QtCore import QObject, Signal

from pupil_labs.ir_plane_tracker import TrackerParams


class TrackerParamsWrapper(QObject, TrackerParams):
    changed = Signal(TrackerParams)

    def __init__(self) -> None:
        super().__init__()

    def set_params(self, params: TrackerParams) -> None:
        if self != params:
            for attr in vars(params):
                setattr(self, attr, getattr(params, attr))

            self.changed.emit(self)

    def update_params(self, params: dict) -> None:
        params_changed = False
        for key, val in params.items():
            if not hasattr(self, key):
                raise KeyError(f"Unknown parameter: {key}")

            old_val = getattr(self, key)
            if isinstance(val, np.ndarray) and isinstance(old_val, np.ndarray):
                is_equal = np.array_equal(val, old_val)
            else:
                is_equal = val == old_val

            if not is_equal:
                setattr(self, key, val)
                params_changed = True
        if params_changed:
            self.changed.emit(self)

    @staticmethod
    def from_json(params_path: str) -> "TrackerParamsWrapper":
        params = TrackerParams.from_json(params_path)
        wrapper = TrackerParamsWrapper()
        wrapper.set_params(params)
        return wrapper


# class TrackerParamsWrapper(QObject):
#     changed = Signal(TrackerParams)

#     def __init__(self, params: TrackerParams) -> None:
#         super().__init__()
#         self._params = params

#     def __getattribute__(self, name: str):
#         return self._params.__getattribute__(name)

#     def set_params(self, params: TrackerParams) -> None:
#         if self._params != params:
#             for attr in vars(params):
#                 setattr(self._params, attr, getattr(params, attr))

#             self.changed.emit(self._params)

#     def update_params(self, params: dict) -> None:
#         params_changed = False
#         for key, val in params.items():
#             if not hasattr(self._params, key):
#                 raise KeyError(f"Unknown parameter: {key}")

#             if val != getattr(self._params, key):
#                 setattr(self._params, key, val)
#                 params_changed = True
#         if params_changed:
#             self.changed.emit(self._params)
