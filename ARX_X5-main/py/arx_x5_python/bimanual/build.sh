#!/bin/bash
set -e

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
BUILD_DIR="build"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

if [ -f CMakeCache.txt ]; then
    CACHED_PYTHON="$(grep '^PYTHON_EXECUTABLE:FILEPATH=' CMakeCache.txt | cut -d= -f2- || true)"
    if [ -n "${CACHED_PYTHON}" ] && [ "${CACHED_PYTHON}" != "${PYTHON_BIN}" ]; then
        rm -f CMakeCache.txt
        rm -rf CMakeFiles
    fi
fi

cmake -DPYTHON_EXECUTABLE="${PYTHON_BIN}" -DPYBIND11_PYTHON_VERSION=3.10 ..
make
make install
