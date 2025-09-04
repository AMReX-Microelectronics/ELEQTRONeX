
![image](https://github.com/AMReX-Microelectronics/ELEQTRONeX/assets/42623728/994b2141-e79d-408f-89eb-19857f6e3396)

This is a framework for electrostatic-quantum transport modeling of nanomaterials at exascale, built using the AMReX library, currently supporting the modeling carbon nanotube field-effect transistors (CNTFETs) with multiple nanotubes. It is developed as part of a DOE-funded project called 'Codesign and Integration of Nanosensors on CMOS'. It builds upon the AMReX library, developed as part of DOE's exascale computing projects (ECP).

The framework comprises three major components: the electrostatic module, the quantum transport module, and the part that self-consistently couples the two modules. The electrostatic module computes the electrostatic potential induced by charges on the surface of carbon nanotubes, as well as by source, drain, and gate terminals, which can be modeled as embedded boundaries with intricate shapes. The quantum transport module uses the nonequilibrium Green's function (NEGF) method to model induced charge. Currently, it supports coherent (ballistic) transport, contacts modeled as semi-infinite leads, and Hamiltonian representation using the tight-binding approximation. The self-consistency between the two modules is achieved using Broyden's modified second algorithm, which is parallelized on both CPUs and GPUs. Preliminary studies have demonstrated that the electrostatic and quantum transport modules can compute the potential on billions of grid cells and compute the Green's function for a material with millions of site locations within a couple of seconds, respectively. 

![Summary_ELEQTRONeX](https://github.com/AMReX-Microelectronics/ELEQTRONeX/assets/42623728/bb489e73-8530-4a48-9992-0caf2b206588)

# Getting Help
Our community is here to help. Please report installation problems or general questions about the code in the github [Issues](https://github.com/AMReX-Microelectronics/ELEQTRONeX/issues) tab above.

# Installation

## Download AMReX and ELEQTRONeX Repositories
Make sure that AMReX and ELEQTRONeX are cloned at the same root location. \
``` >> git clone https://github.com/AMReX-Codes/amrex.git ``` \
``` >> git clone https://github.com/AMReX-Microelectronics/ELEQTRONeX.git ```

## Dependencies
By default, the code uses AMReX implementation of BiCGSTAB (Bi-Conjugate Gradient STABilized)
method for the multigrid solver for electrostatics. Alternatively, users have the flexibility to select from a range of methods or integrate with external libraries such as HYPRE (High-Performance Preconditioners) for enhanced robustness.

Installation instructions for HYPRE are provided here:
``` https://amrex-codes.github.io/amrex/docs_html/LinearSolvers.html#external-solvers ```

## Build

### GNU Make (Primary)
Navigate to ELEQTRONeX/Exec/ then edit the GNUmakefile accordingly and type:\
```make -j4```

For MPI/GPU installation with `USE_MPI=TRUE`, `USE_CUDA=TRUE` in the GNUmakefile.
To compile the code with HYPRE, ensure that `USE_HYPRE=TRUE` in the GNUmakefile.

### CMake (Alternative)
ELEQTRONeX also supports building with CMake, which automatically downloads and builds dependencies.

#### Basic CMake Build
```bash
# CPU build with default options (NOACC backend)
cmake -S . -B build
cmake --build build -j 4

# OpenMP build
cmake -S . -B build -DELEQTRONeX_COMPUTE=OMP
cmake --build build -j 4

# CUDA build
cmake -S . -B build -DELEQTRONeX_COMPUTE=CUDA
cmake --build build -j 4
```

#### Core Configuration Options
- **ELEQTRONeX_COMPUTE**: `NOACC` (default), `OMP`, `CUDA`, `HIP` - Computing backend
- **ELEQTRONeX_MPI**: `ON` (default), `OFF` - Multi-node support
- **ELEQTRONeX_EB**: `ON` (default), `OFF` - Embedded boundary support
- **ELEQTRONeX_TRANSPORT**: `ON` (default), `OFF` - Transport support
- **ELEQTRONeX_TIME_DEPENDENT**: `ON` (default), `OFF` - Time-dependent support
- **ELEQTRONeX_BROYDEN_PARALLEL**: `ON` (default), `OFF` - Broyden parallel support
- **ELEQTRONeX_HYPRE**: `OFF` (default), `ON` - HYPRE support

#### Print Debug Options (matching GNU Make)
- **ELEQTRONeX_PRINT_HIGH**: `OFF` (default), `ON` - High level debug printing
- **ELEQTRONeX_PRINT_MEDIUM**: `OFF` (default), `ON` - Medium level debug printing
- **ELEQTRONeX_PRINT_LOW**: `OFF` (default), `ON` - Low level debug printing
- **ELEQTRONeX_PRINT_NAME**: `OFF` (default), `ON` - Function name debug printing

#### External Dependencies

**AMReX Configuration:**
```bash
# Use external AMReX installation
cmake -S . -B build \
  -DELEQTRONeX_amrex_internal=OFF \
  -DAMReX_DIR=/path/to/amrex/lib/cmake/AMReX

# Use local AMReX source directory
cmake -S . -B build -DELEQTRONeX_amrex_src=/path/to/amrex/source

# Use custom AMReX repository/branch
cmake -S . -B build \
  -DELEQTRONeX_amrex_repo=https://github.com/user/amrex.git \
  -DELEQTRONeX_amrex_branch=my_branch

# Test with specific AMReX pull request (CI/testing)
cmake -S . -B build -DELEQTRONeX_amrex_pr=1234
```

#### Advanced Build Examples
```bash
# CPU build with embedded boundaries and transport
cmake -S . -B build \
  -DELEQTRONeX_COMPUTE=OMP \
  -DELEQTRONeX_EB=ON \
  -DELEQTRONeX_TRANSPORT=ON

# GPU build with HYPRE support
cmake -S . -B build \
  -DELEQTRONeX_COMPUTE=CUDA \
  -DELEQTRONeX_HYPRE=ON

# Debug build with all print options enabled
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DELEQTRONeX_PRINT_HIGH=ON

# Build with local AMReX source (recommended for development)
cmake -S . -B build -DELEQTRONeX_amrex_src=../amrex
cmake --build build -j 4

# Build with external AMReX using CMAKE_PREFIX_PATH
export CMAKE_PREFIX_PATH=/path/to/amrex/install:$CMAKE_PREFIX_PATH
cmake -S . -B build -DELEQTRONeX_amrex_internal=OFF
```

# Running ELEQTRONeX

Here we show how to simulate the band-alignment test for a carbon nanotube surrounded by a metal contact for the different voltages specified on the metal (0 to 1 V with a step size of 0.1 V).
Description of this case can be found in Reference,\
Leonard, F., & Stewart, D. A. (2006). Properties of short channel ballistic carbon nanotube transistors with ohmic contacts. Nanotechnology, 17(18), 4699.

## GNU Make builds (from Exec directory)
Run the test case as:
```bash
# For MPI+CPU build
mpirun -n 4 ./main3d.gnu.MPI.EB.TD.TRAN.BROYPRLL.SKIPGPU.ex ../input/negf/validation_studies/all_around_metal

# For MPI+CUDA build
mpirun -n 4 ./main3d.gnu.MPI.EB.TD.TRAN.BROYPRLL.ex ../input/negf/validation_studies/all_around_metal
```

## CMake builds (from project root directory)
```bash
# For CPU build (NOACC backend)
mpirun -n 4 ./build/main3d.gnu.MPI.EB.TD.TRAN.BROYPRLL.SKIPGPU.ex input/negf/validation_studies/all_around_metal

# For OpenMP build
mpirun -n 4 ./build/main3d.gnu.MPI.EB.TD.TRAN.BROYPRLL.SKIPGPU.OMP.ex input/negf/validation_studies/all_around_metal

# For CUDA build
mpirun -n 4 ./build/main3d.gnu.MPI.EB.TD.TRAN.BROYPRLL.CUDA.ex input/negf/validation_studies/all_around_metal

# Using the convenience symlink
mpirun -n 4 ./build/eleqtronex input/negf/validation_studies/all_around_metal
```

For both build types, a folder named `all_around_metal` will be generated in the execution directory where output will be written. The folder name is determined by the `plot.folder_name = all_around_metal` parameter in the input file and can be customized by modifying this parameter.

# Visualization and Data Analysis

The output data includes 3D plot files for each voltage step such as `plt0000/`, which can be visualized using Visualization tool such as Visit.

Refer to the following link for several visualization tools that can be used for AMReX plotfiles.
[Visualization](https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html)

This is a sample output visualized at `V = 0.1 V`. 

<img src="https://github.com/AMReX-Microelectronics/ELEQTRONeX/assets/42623728/fd43bd3c-79a9-4bfb-8a4c-2316877fb2a7" width="500" height="500">


The output specific to NEGF is written out to `Exec/all_around_metal/negf` folder for each material structure. 

For this test, the data is written out to `cnt` subfolder, as specified in the input file, for each converged step as,
`step<step_number>_<data_field>.dat` where data_field can be `Qout`: induced charge, `norm`: norm after convergence, `U`: electrostatic potential on the surface of the tube.
In addition, data for each iteration in a given step is outputted to `step<step_number>_iter/` folder.

This data can be visualized using a simple python script, `ELEQTRONeX/scripts/analysis/all_around_metal/bandstructure.ipynb`.
