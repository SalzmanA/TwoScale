# Distributed two-scale method library
This project contains a library that implements the distributed two-scale method introduced in Salzman and Moës [SMoes23] from seminal work of Duarte et al. [DKBabuvska07] and adapted to FEniCSx/PETSc usage.

The C++ portion of the library is intended to handle operations that are too resource-intensive for pure Python. The Python API is intended to provide the user with the various building blocks needed to use the “two-scale” method.


## License

Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes

All c++/python sources are under GNU LGPL license given in COPYING.LESSER file

Documentation (jupyter-book files) is under CC-BY-NC-ND <img src="docs/_static/cc-by-nc-nd.png" alt="CC-BY-NC-ND" width="50" height="18">

## Project status

An initial implementation using fenicsx 0.9 yielded some preliminary results.
It follows the directory structure of the libraries in this FEniCSx/PETSc ecosystem, with a `cpp` directory for C++ code and a `python` directory for Python code.
This is the starting point for this repository.
The idea is to keep pace with fenicsx releases.
As a result, two updates in quick succession made the library compatible with versions 0.10 and 0.11 without really taking into account the improvements introduced by those versions.
The current state of implementation is therefore somewhat incomplete (it is based on the features of version 0.9 but is compatible with version 0.11) and is likely to change.

## Ressources

[![Web Documentation](https://img.shields.io/badge/web-documentation-green?logo=github)](https://salzmana.github.io/TwoScale/intro.html#)


[![Docker Image](https://img.shields.io/badge/docker-ghcr.io%2FSalzmanA%2FTwoScale-blue?logo=docker)](https://github.com/SalzmanA/TwoScale/pkgs/container/twoscale)


[![DKBabuvska07](https://img.shields.io/badge/publication-DKBabuvska07-red)][publication_1]

[publication_1]: https://doi.org/10.1007/978-1-4020-6095-3_1

[![SMoes23](https://img.shields.io/badge/publication-SMoes23-red)][publication_2]

[publication_2]: https://doi.org/10.1016/j.cma.2023.115914

