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
