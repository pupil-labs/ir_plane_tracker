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

This repository implements a plane tracking algorithm based on retroreflective IR markers. The markers need to be placed on the boundary of the plane of interest in a known pattern. The camera needs to be an IR camera with IR illuminators placed around the lense.

Any circular retroreflective marker with a diameter of around 10 mm should work. The ones we used for testing are [these](https://www.amazon.de/-/en/dp/B0DSJ54GRH).

The IR camera with IR illuminators we have used for testing is [this one](https://www.amazon.de/dp/B07C1JHB6K).

## Input Parameters

The IR markers need to be placed in groups of four in specific intervals to form a feature line. The intervals are defined by the `norm_line_points` parameter. By default they are placed on a 10 cm long line at steps `[0, 6, 8, 10]` cm. The intervals can be adjusted to fit the size of the plane better. It is recommended to keep the relative distances between the points the same and only scale them up or down linearly.

![Feature Line Definition](feature_line.png)

A total of 6 such feature lines should be placed along the boundary of the plane to be tracked as shown in the image below. Note that the orientation of the feature lines matters.

The exact position of the line on the plane boundary does not matter, but it has to be documented as an input parameter via the various `*_margin` parameters.

The width and height of the screen also need to be specified.

![Measurement Instructions](measurement_instructions.png)

Lastly, the resolution of the camera image needs to be specified via the `img_size_factor` parameter, which expresses the resolution as a multiple of the base resolution of 640x480 pixels. For example, if the camera image is 1280x960 pixels, the `img_size_factor` should be set to 2.0.

## Installation

```bash
pip install -e git+https://github.com/pupil-labs/pl-ir-plane-tracker.git
```
