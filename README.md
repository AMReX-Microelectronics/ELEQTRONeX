This solver is a self-consistent, exascale electrostatic–quantum transport framework designed for modeling charge transport through a field-effect transistor (FET) using the nonequilibrium Green’s function (NEGF) method. It builds upon the AMReX library, developed as part of DOE's exascale computing projects (ECP), to implement dense matrix algebra routines fundamental to the NEGF method. The code offers a unique capability to model intricate shapes of metal contacts, incorporate external charges induced by quantum dots, and provides a robust infrastructure for accommodating different materials. The electrostatic solver enables computations of fields in domains with billions of computational cells and offers boundary conditions of type: homogeneous and inhomogeneous Dirichlet and Neumann, Robin, and Periodic, while the quantum transport framework can effectively handle millions of atoms.


![Summary_ELEQTRONeX](https://github.com/AMReX-Microelectronics/eXstatic/assets/42623728/bb489e73-8530-4a48-9992-0caf2b206588)
