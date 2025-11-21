from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class EyeTrackingData:
    time: float
    gaze: tuple[float, float] | None
    scene: npt.NDArray[np.uint8] | None


@dataclass
class CameraIntrinsics:
    camera_matrix: npt.NDArray[np.float64]
    distortion_coeffs: npt.NDArray[np.float64]


class EyeTrackingSource(ABC):
    @property
    @abstractmethod
    def scene_intrinsics(self) -> CameraIntrinsics:
        pass

    @abstractmethod
    def get_sample(self) -> EyeTrackingData:
        pass

    @abstractmethod
    def close(self):
        pass
