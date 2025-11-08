#!/usr/bin/env bash
set -e
# Build the OpenMP C++ binary into cpp/bin
mkdir -p cpp/bin
CXX=${CXX:-g++}
CXXFLAGS="-O3 -std=c++17 -march=native -fopenmp -pipe -Wall -Wextra"
SRC="cpp/parallel_update.cpp"
OUT="cpp/bin/parallel_update"

echo "Compiling with: ${CXX} ${CXXFLAGS}"
${CXX} ${CXXFLAGS} "${SRC}" -o "${OUT}" || { echo "Build failed"; exit 1; }
echo "Built ${OUT}"
