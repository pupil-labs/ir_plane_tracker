import numpy as np
import usb.core
import usb.util

VC_READ_CALIRATION_DATA = 0xD4
VC_GET_VERSION = 0xC0
VC_SET_LEDS = 0xD2

CALIBRATION_DATA_LENGTH = 1024
CALIBRATION_DATA_CHUNK_SIZE = 64


def _read_calibration_data_chunk(dev, offset):
    """Read calibration data chunk from device"""
    data = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR,
        VC_READ_CALIRATION_DATA,
        wIndex=offset,
        data_or_wLength=CALIBRATION_DATA_CHUNK_SIZE,
    )
    if len(data) != CALIBRATION_DATA_CHUNK_SIZE:
        raise OSError("Reading 64 bytes from VC_READ_CALIRATION_DATA failed")

    return data


def _read_calibration_data(dev=None):
    """Read full calibration data block from device"""
    if dev is None:
        dev = _find_neon()

    data = []
    for offset in range(0, CALIBRATION_DATA_LENGTH, CALIBRATION_DATA_CHUNK_SIZE):
        data.extend(_read_calibration_data_chunk(dev, offset))

    return bytes(data)


def _parse_calibration_data(data):
    calib_data = _read_calibration_data()
    calib_descriptor = np.dtype([
        ("version", "u1"),
        ("serial", "6a"),
        ("scene_camera_matrix", "(3,3)d"),
        ("scene_distortion_coefficients", "8d"),
        ("scene_extrinsics_affine_matrix", "(4,4)d"),
        ("right_camera_matrix", "(3,3)d"),
        ("right_distortion_coefficients", "8d"),
        ("right_extrinsics_affine_matrix", "(4,4)d"),
        ("left_camera_matrix", "(3,3)d"),
        ("left_distortion_coefficients", "8d"),
        ("left_extrinsics_affine_matrix", "(4,4)d"),
        ("crc", "u4"),
    ])
    calibration_data = np.frombuffer(
        calib_data[: calib_descriptor.itemsize], calib_descriptor
    )

    return {
        field_name: calibration_data[field_name][0]
        for field_name in calib_descriptor.names
    }


def _find_neon():
    return usb.core.find(idVendor=0x16D0, idProduct=0x11D3)


def get_intrinsics_data():
    calib_data = _parse_calibration_data(_read_calibration_data())
    intrinsics = (
        calib_data["scene_camera_matrix"],
        calib_data["scene_distortion_coefficients"],
    )

    return intrinsics


def set_leds(mask, dev=None):
    """Write LED enables"""
    if dev is None:
        dev = _find_neon()

    data = mask.to_bytes(1, byteorder="little")
    n = dev.ctrl_transfer(usb.util.CTRL_TYPE_VENDOR, VC_SET_LEDS, 0, 0, data)
    if n != 1:
        raise OSError("Writing 1 bytes to VC_SET_LEDS failed")


def get_version(dev=None):
    """Read version information from device"""
    if dev is None:
        dev = _find_neon()

    data = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR, VC_GET_VERSION, 0, 0, 8
    )
    if len(data) != 8:
        raise OSError("Reading 8 bytes to VC_GET_VERSION failed")

    versions = {
        "fx2": int.from_bytes(data[:4], byteorder="little"),
        "fpga": int.from_bytes(data[4:], byteorder="little"),
    }

    return versions
