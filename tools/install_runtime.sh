#!/usr/bin/env bash
set -euo pipefail
: "${EXCASCENE_PYTHON:=python3.11}"
"$EXCASCENE_PYTHON" -c 'import sys; assert sys.version_info[:3] == (3,11,14), sys.version'
"$EXCASCENE_PYTHON" -m venv .venv
.venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple -c requirements-fresh-lock.txt -r requirements-core.txt
.venv/bin/python -m pip --isolated install --index-url https://download.pytorch.org/whl/cu129 -c requirements-fresh-lock.txt 'torch==2.8.0+cu129' 'torchvision==0.23.0+cu129'
# Preserve the evaluated legacy runtime exactly. MMCV's optional unneeded
# opencv-python dependency is provided by headless OpenCV. This OpenCV wheel's
# NumPy>=2 metadata conflicts with the evaluated NumPy1.26.4; no-deps retains
# that recorded combination. The release fixture determines runtime viability.
.venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple --no-deps --no-build-isolation 'mmcv==1.4.0' 'opencv-python-headless==4.12.0.88'
.venv/bin/python -m pip freeze > environment.installed.txt
