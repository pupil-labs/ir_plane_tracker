from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

from pupil_labs.camera import Camera


@dataclass
class EyeTrackingData:
    time: int
    """Timestamp of the data sample in nanoseconds."""

    scene_image_distorted: npt.NDArray[np.uint8]
    """Raw and distorted scene image."""

    gaze_scene_distorted: npt.NDArray[np.float64]
    """Gaze point in distorted scene image coordinates"""

    intrinsics: Camera
    """Intrinsics of the scene camera."""

    eye_image: npt.NDArray[np.uint8]
    """Raw eye image."""

    @cached_property
    def scene_image_undistorted(self) -> npt.NDArray[np.uint8]:
        """Undistorted scene image."""
        return self.intrinsics.undistort_image(self.scene_image_distorted)

    @cached_property
    def gaze_scene_undistorted(self) -> npt.NDArray[np.float64]:
        """Gaze point in undistorted scene image coordinates"""
        return self.intrinsics.undistort_points(self.gaze_scene_distorted)


class EyeTrackingSource(ABC):
    @cached_property
    @abstractmethod
    def scene_intrinsics(self) -> Camera:
        pass

    @abstractmethod
    def get_sample(self) -> EyeTrackingData:
        pass

    @abstractmethod
    def close(self):
        pass
