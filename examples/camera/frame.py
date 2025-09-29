from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt


class Frame(NamedTuple):
    data: npt.NDArray
    timestamp: float
    index: int

    @property
    def gray(self) -> npt.NDArray[np.uint8]:
        if len(self.data.shape) == 2:
            return self.data

        return cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)

    @property
    def bgr(self) -> npt.NDArray[np.uint8]:
        if len(self.data.shape) == 3 and self.data.shape[-1] == 3:
            return self.data

        return cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
