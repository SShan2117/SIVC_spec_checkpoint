#!/bin/bash
set -euo pipefail


: "${CONDA_PREFIX:?Conda env not active}"
export MKLROOT="${MKLROOT:-$CONDA_PREFIX}"

CXX=mpicxx
SRC=../CFMCut_SIVC.cpp
EXE=pin.x

CXXFLAGS="-O3 -DNDEBUG -march=x86-64-v3"
INCLUDES="-I${MKLROOT}/include"
LIBS="-L${MKLROOT}/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl"

echo "[build] using mpicxx: $(which ${CXX})"
echo "[build] using MKLROOT: ${MKLROOT}"

${CXX} ${CXXFLAGS} ${INCLUDES} -o ${EXE} ${SRC} ${LIBS}
echo "[build] done -> ${EXE}"