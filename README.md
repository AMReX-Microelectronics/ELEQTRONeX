
![image](https://github.com/AMReX-Microelectronics/ELEQTRONeX/assets/42623728/994b2141-e79d-408f-89eb-19857f6e3396)

This is a framework for electrostatic-quantum transport modeling of nanomaterials at exascale, built using the AMReX library, currently supporting the modeling carbon nanotube field-effect transistors (CNTFETs) with multiple nanotubes. It is developed as part of a DOE-funded project called 'Codesign and Integration of Nanosensors on CMOS'. It builds upon the AMReX library, developed as part of DOE's exascale computing projects (ECP).

The framework comprises three major components: the electrostatic module, the quantum transport module, and the part that self-consistently couples the two modules. The electrostatic module computes the electrostatic potential induced by charges on the surface of carbon nanotubes, as well as by source, drain, and gate terminals, which can be modeled as embedded boundaries with intricate shapes. The quantum transport module uses the nonequilibrium Green's function (NEGF) method to model induced charge. Currently, it supports coherent (ballistic) transport, contacts modeled as semi-infinite leads, and Hamiltonian representation using the tight-binding approximation. The self-consistency between the two modules is achieved using Broyden's modified second algorithm, which is parallelized on both CPUs and GPUs. Preliminary studies have demonstrated that the electrostatic and quantum transport modules can compute the potential on billions of grid cells and compute the Green's function for a material with millions of site locations within a couple of seconds, respectively. 

![Summary_ELEQTRONeX](https://github.com/AMReX-Microelectronics/eXstatic/assets/42623728/bb489e73-8530-4a48-9992-0caf2b206588)

# Getting Help
Our community is here to help. Please report installation problems or general questions about the code in the github [Issues](https://github.com/AMReX-Microelectronics/ELEQTRONeX/issues) tab above.

# Installation
Below are instructions for MPI/GPU installation with `USE_MPI=TRUE`, `USE_GPU=TRUE` enabled.  

## Download AMReX and ELEQTRONeX Repositories
Make sure that AMReX and ELEQTRONeX are cloned at the same root location. \
``` >> git clone https://github.com/AMReX-Codes/amrex.git ``` \
``` >> git clone https://AMReX-Microelectronics/ELEQTRONeX.git ```

## Dependencies
By default, the code uses AMReX implementation of BiCGSTAB (Bi-Conjugate Gradient STABilized)
method for the multigrid solver for electrostatics. Alternatively, users have the flexibility to select from a range of methods or integrate with external libraries such as HYPRE (High-Performance Preconditioners) for enhanced robustness.

Installation instructions for HYPRE are provided here:
``` https://amrex-codes.github.io/amrex/tutorials_html/Hypre_Install.html ```

To compile the code with HYPRE, keep USE_HYPRE flag on.

## Build
 Navigate to ELEQTRONeX/Exec/ and run:\
```>> make -j4```

# Running ELEQTRONeX

Here we show how to simulate the band-alignment test for a carbon nanotube surrounded by a metal contact for the different voltages specified on the metal (0 to 1 V with a step size of 0.1 V).
Description of this case can be found in Reference,\
Leonard, F., & Stewart, D. A. (2006). Properties of short channel ballistic carbon nanotube transistors with ohmic contacts. Nanotechnology, 17(18), 4699.

Run the test case as,

```>> ./<compile_binary> ../input/negf/all_around_metal```

A folder named, `Exec/all_around_metal_test`, will be generated in which output will be written out. Location of this folder can be changed using `plot.folder_name` parameter in the input file.

# Visualization and Data Analysis

The output data includes 3D plot files for each voltage step such as `plt0000/`, which can be visualized using Visualization tool such as Visit.

Refer to the following link for several visualization tools that can be used for AMReX plotfiles.
[Visualization](https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html)

This is a sample output visualized at `V = 0.1 V`. 

<img src="https://github.com/AMReX-Microelectronics/ELEQTRONeX/assets/42623728/fd43bd3c-79a9-4bfb-8a4c-2316877fb2a7" width="500" height="500">


The output specific to NEGF is written out to `all_around_metal_test/negf` folder for each material structure. 

For this test, the data is written out to `cnt` subfolder, as specified in the input file, for each converged step as,
`step<step_number>_<data_field>.dat` where data_field can be `Qout`: induced charge, `norm`: norm after convergence, `U`: electrostatic potential on the surface of the tube.
In addition, data for each iteration in a given step is outputted to `step<step_number>_iter/` folder.

This data can be visualized using a simple python script, `scripts/all_around_metal/bandstructure.ipynb`.
