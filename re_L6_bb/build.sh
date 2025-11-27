#!/bin/bash
module load gcc/11.3
module load openmpi/4.1.4-gcc11.3
module load mkl/latest


mpic++ -O2 -o pin.x ../CFMCut_SIVC.cpp -I/path/to/mkl/include -L/path/to/mkl/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -lmpi
