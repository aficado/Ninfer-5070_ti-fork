#!/usr/bin/env bash
# Uso: scripts/build-5070ti.sh /ruta/a/cuda-13.x   (requiere nvcc >= 13.x)
set -euo pipefail
N=${1:?ruta a CUDA 13.x}
export PATH=$N/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120a \
  -DCMAKE_CUDA_COMPILER=$N/bin/nvcc -DCUDAToolkit_ROOT=$N \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_C_COMPILER=gcc-14 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=ON -DNINFER_BUILD_BENCHMARKS=ON
cmake --build build --parallel "${JOBS:-6}"
