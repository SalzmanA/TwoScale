# Distributed two-scale method library
This project contains a library that implements the distributed two-scale method introduced in Salzman and Moës [SMoes23] from seminal work of Duarte et al. [DKBabuvska07] and adapted to FEniCSx/PETSc usage.

The C++ portion of the library is intended to handle operations that are too resource-intensive for pure Python. The Python API is intended to provide the user with the various building blocks needed to use the “two-scale” method.

Full explanations on the library with examples can be found in the [documentation](https://salzmana.github.io/twoscale/twoscale.html). 

## License

Copyright (c) 2026 - Ecole Centrale de Nantes

Author: Alexis Salzman

All c++/python sources are under GNU LGPL license given in COPYING.LESSER file

Documentation (jupyter-book files) is under CC-BY-NC-ND <img src="docs/_static/cc-by-nc-nd.png" alt="CC-BY-NC-ND" width="50" height="18">

## Project status

An initial implementation using fenicsx 0.9 yielded some preliminary results.
It follows the directory structure of the libraries in this FEniCSx/PETSc ecosystem, with a `cpp` directory for C++ code and a `python` directory for Python code.
This is the starting point for this repository.
The idea is to keep pace with fenicsx releases.
As a result, two updates in quick succession made the library compatible with versions 0.10 and 0.11 without really taking into account the improvements introduced by those versions.
The current state of implementation is therefore somewhat incomplete (it is based on the features of version 0.9 but is compatible with version 0.11) and is likely to change.

## reference

[DKBabuvska07] C. A. Duarte, Dae-Jin Kim, and Ivo Babuška. A global-local approach for the construction of enrichment functions for the generalized fem and its application to three-dimensional cracks. In V. M. A. Leitão, C. J. S. Alves, and C. Armando Duarte, editors, Advances in Meshfree Techniques, 1–26. Dordrecht, 2007. Springer Netherlands.

[SMoes23] Alexis Salzman and Nicolas Moës. A two-scale solver for linear elasticity problems in the context of parallel message passing. Computer Methods in Applied Mechanics and Engineering, 407:115914, 2023. URL: https://www.sciencedirect.com/science/article/pii/S0045782523000373, doi:https://doi.org/10.1016/j.cma.2023.115914.

