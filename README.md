# Pupil Labs IR Plane Tracker

[![ci](https://github.com/pupil-labs/pl-ir-plane-tracker/actions/workflows/main.yml/badge.svg)](https://github.com/pupil-labs/pl-ir-plane-tracker/actions/workflows/main.yml)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://pupil-labs.github.io/pl-ir-plane-tracker/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre_commit-black?logo=pre-commit&logoColor=FAB041)](https://github.com/pre-commit/pre-commit)
[![pypi version](https://img.shields.io/pypi/v/pupil-labs-ir-plane-tracker.svg)](https://pypi.org/project/pupil-labs-ir-plane-tracker/)
[![python version](https://img.shields.io/pypi/pyversions/pupil-labs-ir-plane-tracker)](https://pypi.org/project/pupil-labs-ir-plane-tracker/)

> [!IMPORTANT]
> This package is a work in progress. Use at your own risk!
> The API may change without warning, the computational performance is not optimized, and the tracking performance may lack in some cases.

This repository implements a marker-based plane tracking algorithm. The markers need to be co-planar with the plane of interest, ideally placed close to the boundary of the plane. The exact position of the markers in relation to the plane needs to be known.

Ideally, the camera is an IR camera with the markers made from a retroreflective material, but a regular RGB camera with black-and-white markers also works to some extent.

## Quick Start

To run the gaze mapping example application, install the dependencies using the following commands and finally run the example providing the IP of a Neon device on your network as an argument.

```bash
git clone git@github.com:pupil-labs/ir_plane_tracker.git
cd ir_plane_tracker/
uv sync  --extra examples
cd examples/
python gaze_mapping_app_main.py <NEON_IP_ADDRESS>
```

You can find the IP of your Neon device from the Neon Companion app when you open the network panel by clicking the icon in the top right corner of the home screen.

The instructions assume you have [uv](https://docs.astral.sh/uv/) installed.

## Marker Placement

A PNG image of the marker can be found [here](https://github.com/pupil-labs/ir_plane_tracker/blob/main/feature.png). A set of 4 markers in the right size on a printable DIN A4 sheet can be found [here](https://github.com/pupil-labs/ir_plane_tracker/blob/main/features-A4.pdf).

The markers have to be placed co-planar with the plane of interest and in parallel to the plane's boundary edges. One marker per edge is usesd. To maximize robustness against occlusions, it is recommended to place the markers close to the center of the plane. The markers do not have to be placed inside of the plane, but they should be close to it, such that they are visible in the camera view when the plane is in view.

The orientation of the markers matters and should follow what is visible in the example below. Also their exact position in relation to the plane needs to be known and specifeid in the `marker_config.json` file. A markers position is defined by the position of the outter circular feature. The origin of the plane coordinate system is the top left corner.

![Marker Placement](marker_placement.png)
