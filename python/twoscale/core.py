# Copyright (C) 2026 - Ecole Centrale de Nantes
# Author: Alexis Salzman
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""core module that provides function to create/interogate two scale object."""
from twoscale import ts_cpp as _cpp

_ts_dolfinx_exist=False
try:
    import twoscale.ts_cpp.dolfinx as _ts
    _ts_dolfinx_exist=True
except ImportError:
    print ("Twoscale dolfinx implementation not available")
    print ("For now no other implementation exist")
    
if _ts_dolfinx_exist:
    print ("Use Twoscale dolfinx implementation")
    from dolfinx.fem import Function
    from dolfinx.fem import FunctionSpace
    from dolfinx.fem.bcs import DirichletBC
    from dolfinx.mesh import Mesh
    from dolfinx.mesh import MeshTags
#tmp ??
    from dolfinx import default_real_type
    from dolfinx.cpp.mesh import (CellType)
    from dolfinx_mpc.multipointconstraint import MultiPointConstraint
    import basix
    import basix.ufl
    import ufl
    from petsc4py import PETSc
    #tmp ?
    import numpy as np
    import numpy.typing as npt
    import typing

#=================================================================
    class scaleJump:
        """ A python scaleJump."""
        _scale_jump: typing.Union[_ts.scaleJump_float32, _ts.scaleJump_float64]
        _coarse_domain: ufl.Mesh
        _fine_domain: ufl.Mesh
        def __init__(self, scale_jump):
            """Initialize python scalejump object from a C++ object.

            Args:
                scale_jump: A C++ mesh object.

            Note:
                scaleJump objects should usually be constructed using
                :func:`topDown` and not using this class initializer.
            """
            self._scale_jump=scale_jump
            mesh=self._scale_jump.getCoarseMesh()
            self._coarse_domain = ufl.Mesh(basix.ufl.element("Lagrange", mesh.topology.cell_type.name, 1, shape=(mesh.topology.dim,), dtype=default_real_type))
            self._coarse_domain._ufl_cargo = mesh # TODO: clarify if this make sense. From Mesh constructor. Not clear what it makes and if it is correctly done
            mesh=self._scale_jump.getFineMesh()
            self._fine_domain = ufl.Mesh(basix.ufl.element("Lagrange", mesh.topology.cell_type.name, 1, shape=(mesh.topology.dim,), dtype=default_real_type))
            self._fine_domain._ufl_cargo = mesh # TODO: clarify if this make sense. From Mesh constructor. Not clear what it makes and if it is correctly done

        @property
        def needs_mpc(self):
            """
            tells if mpc are required or not
            """
            return self._scale_jump.needs_mpc

        @property
        def getFineMesh(self):
            """
            Return fine mesh 
            """
            return Mesh(self._scale_jump.getFineMesh(),self._fine_domain)

        @property
        def getCoarseMesh(self):
            """
            Return coarse mesh 
            """
            return Mesh(self._scale_jump.getCoarseMesh(),self._coarse_domain)

        def getChildren(self, coarse_idx):
            """
            Return list of fine mesh cells index embodied  in coarse cell of index 'coarse_idx'
            if no fine cell is associated to this coarse cell return an empty list
            """
            return self._scale_jump.getChildren(coarse_idx)

        def getFaceChilds(self, coarse_idx):
            """
            Return list of fine mesh faces index embodied  in coarse face of index 'coarse_idx'
            if no fine face is associated to this coarse face return an empty list
            Only coarse face of the support will provides childs
            """
            return self._scale_jump.getFaceChilds(coarse_idx)

        def getSurroundingFaceChilds(self, coarse_idx):
            """
            Return list of fine mesh faces index embodied  in coarse face of index 'coarse_idx'
            if no fine face is associated to this coarse face return an empty list
            Only coarse face of the surrounding element will provides childs
            """
            return self._scale_jump.getSurroundingFaceChilds(coarse_idx)
        
        @property
        def getEnriched(self):
            """
            Return coarse enriched node index (local and/or ghost))
            """
            return self._scale_jump.getEnriched()

        @property
        def getExtraEnriched(self):
            """
            Return coarse enriched node index (local and/or ghost))
            """
            return self._scale_jump.getExtraEnriched()

        @property
        def getSupport(self):
            """
            Return support of coarse enriched node (i.e. all index of
            cells connected localy to an enriched node (local or ghost))
            """
            return self._scale_jump.getSupport()

        @property
        def getSurroundingCells(self):
            """
            Return cell surronding support if any (i.e.  cell connected to
            node in Extra enriched group but not in support)
            """
            return self._scale_jump.getSurroundingCells()

        def isFineMasterNode(self,idx):
            """
            Return fine master node of mpc relation if any.
            """
            return self._scale_jump.isFineMasterNode(idx)

        @property
        def getCoarseMaster(self):
            """
            Return coarse face connected to node in Extra enriched group if
            any.
            """
            return self._scale_jump.getCoarseMaster()

    class enrichedFunction:
        """ A python enrichedFunction."""
        _enriched_function: typing.Union[_ts.enrichedFunction_float32,_ts.enrichedFunction_float64,_ts.enrichedFunction_complex32,_ts.enrichedFunction_complex64 ]
        def __init__(self, enriched_function):
            """Initialize python enrichedFunction object from a C++ object.

            Args:
                enriched_function: A C++ coarse manager object.

            Note:
                enriched_function objects should usually be constructed using
                :func:`generateEnrichedXXXFunction` and not using this class initializer.
            """
            self._enriched_function=enriched_function
#=================================================================
    class patchManager:
        """ A python patchManager."""
        _patch_manager: typing.Union[_ts.patchManager_MPIDirect]
        def __init__(self, patch_manager):
            """Initialize python patchManager object from a C++ object.

            :param patch_manager: The C++ object to warp.
            :type patchManager: A C++ coarse manager object.

            :note:
                patch_manager objects should usually be constructed using
                :func:`generatePatchManager` and not using this class initializer.

            """
            self._patch_manager=patch_manager
        def generateProblems(self,
                         Aff:PETSc.Mat,
                         bf:PETSc.Vec):
            """
            create patches system
            """
            self._patch_manager.generateProblems(Aff,bf)
        def solveProblems(self,
                         fine_field: Function):
            """
            solve patches system
            """
            xf=fine_field.x.petsc_vec
            self._patch_manager.solveProblems(xf)
        def grabPatchSolution(self,
                              seq: int,
                              fine_field: Function):
            """
            Grab patch solution for a specific sequence into a fine scale field

            :param seq: The sequence id for which the solution(s) is(are) collected
            :type seq: integer
            :param fine_field: The field where the patch(es) solution(s) is(are) stored
            :type fine_field: Function
            
            :danger: 
            Only for debbuging purpose. May be removed in the future and/or instable.

            """
            fine_field.x.array[:]=0.
            xf=fine_field.x.petsc_vec
            return self._patch_manager.grabPatchSolution(seq,xf)
        @property
        def numberOfSequence(self):
            """
             number of sequence to compute all patches
            """
            return self._patch_manager.numberOfSequence()
#=================================================================
    class coarseManager:
        """ A python coarseManager."""
        _coarse_manager: typing.Union[_ts.coarseManager_petsc]
        def __init__(self, coarse_manager):
            """Initialize python coarseManager object from a C++ object.

            :param coarse_manager: The C++ object to warp.
            :type coarse_manager: A C++ coarse manager object.

            :note:
            coarse_manager objects should usually be constructed using :py:func:`generateCoarseManager` and not using this class initializer.


            """
            self._coarse_manager=coarse_manager

        def setStdCoarse(self,
                             Aff:PETSc.Mat,
                             bf:PETSc.Vec,
                             use_imp_enriched:bool=False):
            """
            Create/update coarse system with new fine matrix and vector and
            compute standard part.

            :param Aff: System matrix assembled at fine level
            :type Aff: PETSc Mat

            :param bf: System rhs assembled at fine level
            :type bf: PETSc Vec

            :param use_imp_enriched: If true generate extra information to be able to use updateEImp and solveEImp
            :type use_imp_enriched: bool
            """
            self._coarse_manager.setStdCoarse(Aff,bf,use_imp_enriched)
        def resetCoarseToStd(self):
            """
            Reset coarse system with precomputed standard part
            """
            self._coarse_manager.resetCoarseToStd()
        def updateEnrichedOperator(self,
                                        pm: patchManager,
                                        func: enrichedFunction):
            """
            Compute the enriched part of the scale jump operator based on patches solution and the enrichment function generator
            """
            self._coarse_manager.updateEnrichedOperator(pm._patch_manager,func._enriched_function);
        def updateEnrichCoarse(self,
                         Aff:PETSc.Mat,
                         bf:PETSc.Vec):
            """
            update only enriched part of coarse system 
            """
            self._coarse_manager.updateEnrichCoarse(Aff,bf)
        def solve(self,
                  fine_field: Function):
            """
            solve coarse system and store project result in fine field 
            """
            xf=fine_field.x.petsc_vec
            self._coarse_manager.solve(xf)
        def updateEImp(self):
            """
            update imposed enriched contribution to rhs
            """
            self._coarse_manager.updateEImp()
        def solveEImp(self,
                  fine_field: Function):
            """
            solve coarse system and store project result in fine field considering all enriched dof imposed to one
            """
            xf=fine_field.x.petsc_vec
            self._coarse_manager.solveEImp(xf)
        def projectStdCoarse(self,
                            coarse_enriched_field: Function,
                            fine_field: Function):
            """
            Project standard part of coarse enriched field in fine field 
            """
            xc=coarse_enriched_field.x.petsc_vec
            xf=fine_field.x.petsc_vec
            self._coarse_manager.projectStdCoarse(xc,xf)
#=================================================================
    def topDown(mesh: Mesh,
                enriched: typing.Callable,
                crit: typing.Callable,
                level: int=1,
                clustering_dual_graph: bool=True,
                accurate_weight: bool=False,
                tags: typing.Optional[MeshTags]=None):
        """
           Function to create a scaleJump object containing a fine mesh
           generated from a coarse one, based on enriched nodes and refinement criteria

           :param mesh: The coarse mesh to start from
           :param enriched: The function that identify nodes to enrich
           :param crit: The function that identify element to refine for a given level of refinement
           :param level: The number of refinement pass to apply to the support of enriched nodes
           :param clustering_dual_graph: Force usage of clustering strategy for load balancing
           :param accurate_weight: Ask for use of accurate weight per element for graph node weighting used for load balancing
           :param tags: one days tags given by this argument will be promoted to generated fine scale mesh
           :return: The scaleJump object corresponding to the jump from coarse mesh to fine scale mesh
           :rtype: scaleJump

        """

        if tags:
            return scaleJump(_ts.topDown(mesh._cpp_object,enriched,crit,level,clustering_dual_graph,accurate_weight,tags._cpp_object))
        else:
            return scaleJump(_ts.topDown(mesh._cpp_object,enriched,crit,level,clustering_dual_graph,accurate_weight,None))

#=================================================================
    def generateMPC(sj: scaleJump,
                    space: FunctionSpace):

        """
           Function to create, from scaleJump object and studied space, all mpc (if any) needed to
           connect dof at support interface
        """
        mpc=MultiPointConstraint(space)
        mpc.add_constraint_from_mpc_data(mpc.V,_ts.generateMPC(sj._scale_jump,space._cpp_object))
        mpc.finalize()
        return mpc

#=================================================================
    def generateCoarseManager(sj: scaleJump,
                                   fine_field: Function,
                                   coarse_field: Function,
                                   mpc: typing.Optional[MultiPointConstraint]=None,
                                   bcs: list[DirichletBC] = []
                              ):

        """
           Function to create the coarseManager that operate on fine matrices and patches to construct coarse system and solve it.
           It mainely create the operators that transform field at fine scale to enriched field at coarse scale. 

           :param sj: scaleJump object holding mesh transition information
           :param fine_field: field at fine scale
           :param coarse_field:  field at coarse level 
           :param mpc:  if any, multi point constrain used to connect finefield dof at support interface
           :param bcs:  List of boundary condition to be impose at coarse scale on coarse_field. 

           :warning:
               The bcs must respect the following rules depending on implementation :

               With nested matrix strategy 

               * No constrain except that all BC space must be included in coarse_field space which is in this case a simple space representing blocked standard dofs.

               With the mixed space strategy

               * it must be constructed on the same space given by coarse_field argument which is in this case based on a mixed space with standard and enriched dofs
               * it must group all Dirichlet boundary condition in one dirichletBC object
               * Enriched dofs out of the support are automatically constrained by this function and should not be fixed by the user with bcs
             

        """
        _bcs = [bc._cpp_object for bc in bcs]
        return coarseManager(_ts.generateCoarseManager(sj._scale_jump,fine_field._cpp_object,coarse_field._cpp_object, None if mpc is None else mpc._cpp_object, _bcs))
#=================================================================
    def generatePatchManager(sj: scaleJump,
                             fine_field: Function):

        """
           Function to create, from scaleJump object , all patches and a manager of them
        """
        return patchManager(_ts.generatePatchManager(sj._scale_jump,fine_field._cpp_object))
#=================================================================
    def generateEnrichedShiftFunction(fine_field: Function):

        """
           Function to create, from fine field, a enrichedFunction object coresponding to a field shifted
           by value at enriched point

           :param fine_field: field at fine scale

        """
        return enrichedFunction(_ts.generateEnrichedShiftFunction(fine_field._cpp_object))
#=================================================================
    def useNest():

        """
           Function telling if librarie is using or not nested matrix strategy

           :return: True if nested matrix strategy is activated in the library

        """
        return _ts.useNest()
