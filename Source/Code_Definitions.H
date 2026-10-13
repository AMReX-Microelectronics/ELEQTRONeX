#ifndef CODE_DEFINITIONS_H_
#define CODE_DEFINITIONS_H_

#define Max_Ncell_Azim 500
#define Max_Ncell_Long 500
#define VFRAC_THREASHOLD 1e-5

#define NUM_MODES 1
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif
#define NUM_CONTACTS 2
#define NUM_ENERGY_PTS_REAL 10

#include <AMReX_BoxArray.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_REAL.H>

using ComplexType = amrex::GpuComplex<amrex::Real>;

template <typename T>
class TD;

#endif
