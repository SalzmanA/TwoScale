This folder contains patches for PETSc and dolfinx, enabling the use of the twoscale 0.9.0 version.
The latest version of Twoscale is independent of them, since the latest versions of PETSc and dolfinx provide the required functionality.

* patch_petsc_3.22.2: patch to apply to PETSc<=3.22.2 by hand or with patch if mpiov.c compatible with 3.22.2 verison
* patch_dolfinx_0.9.0: patch to apply to dolfinx=0.9.0 by hand or with patch (for 3D test cases) 

These patches are not applied when creating the v0.9.0 Docker image. Thus, the Docker image for this version cannot run twoscale jobs in parallel or with 3D boundary conditions on faces. At least, Docker creation does confirm the correct compilation of the twoscale v0.9.0 library.
