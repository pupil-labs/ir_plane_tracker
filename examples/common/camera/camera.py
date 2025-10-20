from typing import NamedTuple

from .frame import Frame


class CameraSpec(NamedTuple):
    name: str
    vendor_id: int
    product_id: int
    width: int
    height: int
    fps: int
    bandwidth_factor: float


class CameraNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        self.camera_name = name
        super().__init__(f"Camera {name} not found!")


class Camera:
    def __init__(self, spec: CameraSpec, backend_class) -> None:
        self.backend = backend_class(spec)

        self.spec = spec
        self.frame_counter = -1

    def get_frame(self) -> Frame:
        return self.backend.get_frame()

    def close(self) -> None:
        self.backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()
