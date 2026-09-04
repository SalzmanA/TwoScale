# Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""module that provides TwoScale linear solvers tools."""
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
    from twoscale import core
    from dolfinx import mesh
    from dolfinx.fem import Function
    from dolfinx.fem import functionspace
    from dolfinx.fem import FunctionSpace
    from dolfinx.fem import create_interpolation_data
    from dolfinx.fem.bcs import DirichletBC
    from dolfinx_mpc.multipointconstraint import MultiPointConstraint
    from petsc4py import PETSc
    from petsc4py import typing as petsc_typing
    import typing

    from twoscale import util
    import pyvista as pv
    import numpy as np
#=================================================================
    def residual( A: PETSc.Mat,
                  b: PETSc.Vec,
                  x: PETSc.Vec,
                  nb: petsc_typing.Scalar
                 ):
        """
           Function to compute the relative residual of the linear problem A.x=b provided as argument.


           :param A: The matrix of the system
           :type A: Petsc Mat
           :param b: The right hand side vector of the system
           :type b: Petsc Vec
           :param x: The solution vector of the system (i.e. :math:`x=A^{-1}.b`)
           :type x: Petsc Vec
           :param nb: The value to resize the norm and make it relative. In general it is expected to be the norm of b.
           :type nb: PetscScalar
           :return: The relative norm of the residual: :math:`\\frac{|A.x-b|}{nb}`
           :rtype: PetscScalar

        """
        R=b.duplicate()
        A.mult(x,R)
        #vi=PETSc.Viewer()
        #print("resi R0 vector")
        #vi(R)
        #print("resi b vector")
        #vi(b)
        #print("resi x vector")
        #vi(x)
        R.aypx(-1.,b)
        #print("resi R vector")
        #vi(R)
        return R.norm()/nb

#=================================================================
    def linearBasicLoop( sj: core.scaleJump,
                         fine_space: FunctionSpace,
                         Aff: PETSc.Mat,
                         ADff: PETSc.Mat,
                         Bf: PETSc.Vec,
                         BDf: PETSc.Vec,
                         efunc: core.enrichedFunction,
                         enriched_field: Function,
                         mpc: typing.Optional[MultiPointConstraint]=None,
                         bc: list[DirichletBC] = [],
                         itmx=10,
                         eps=1.e-3
                        ):

        """
           Function to solve a linear problem by looping in between scale with the twoscale approach


           :param sj: The scale jump describing the two scale
           :param fine_space: The space describing fine scale discretization related to fine scale linear problem
           :param Aff: The fine scale assembled matrix
           :param ADff: The fine scale assembled matrix with fine Dirichlet boundary condition eliminated
           :param Bf: The fine scale assembled rhs
           :param BDf: The fine scale assembled rhs with fine Dirichlet boundary condition eliminated treated
           :param efunc: The enriched function functor that transform patch solution into a enriched 
                function (i.g. shifting field to force enriched dof to be null)
           :param enriched_field: The field describing enriched field at coarse scale. The associate linear problem 
                is managed by coarseManager object
           :type enriched_field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param mpc: The multi point constraint, if any, used to create fine_space and created from a call to core.generateMPC with sj. 
           :type mpc: `MultiPointConstraint <https://jsdokken.com/dolfinx_mpc/docs/api.html#dolfinx_mpc.MultiPointConstraint>`_ or None
           :param bc: The list of boundary conditions applied to system at coarse level
           :type bc: list of `DirichletBC <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.DirichletBC>`_
           :param itmx: The maximum number of iteration to compute
           :type itmx: int
           :param eps: The target precision for  relative residual value to be considered as null
           :type eps: floating point value

           :return: the field describing fine scale function related to fine scale resolution by the twoScale solver,
                                the last residual computed
                                the number of iteration done in the loop
                                the history of residual evolution
           :rtype: tuple Function,float,int,list[float]
        """
        # check coherance with mpc
        do_mpc=sj.needs_mpc
        if do_mpc and mpc==None:
            raise RuntimeError("scale jump provided expect MPC creation and you provide no mpc !")

        # solution field at fine scale
        if do_mpc:
            solf=Function(mpc.function_space,name='sol_fine_ts')
        else:
            solf=Function(fine_space,name='sol_fine_ts')

        # generate coarseManager to deal with enriched problem at coarse scale
        stage = PETSc.Log.Stage("generateCM")
        stage.push()
        cm =core.generateCoarseManager(sj,fine_space,enriched_field.function_space,mpc,bc)
        stage.pop()
        # generate patchManager to deal with patches problem at fine scale
        stage = PETSc.Log.Stage("generatePM")
        stage.push()
        pm =core.generatePatchManager(sj,fine_space)
        stage.pop()
        
        # generate patches problem from fine linear assembled system with Dirichlet BC eliminated
        stage = PETSc.Log.Stage("genprbPM")
        stage.push()
        pm.generateProblems(ADff,BDf)
        stage.pop()

        # generate standard part of the coarse enriched problem from fine linear assembled system 
        stage = PETSc.Log.Stage("genstdCM")
        stage.push()
        cm.setStdCoarse(Aff,Bf)
        stage.pop()

        #obtaine PETSc solf
        xf=solf.x.petsc_vec

        # initialize twoscale loop by setting solf using standard part of the coarse enriched 
        # field guess given by the user in enriched_field
        cm.projectStdCoarse(enriched_field,solf)

        # compute initial relative residual : |res|/|b|
        nb=BDf.norm()
        ra=residual(ADff,BDf,xf,1.)
        hrb=[ra/nb]
        # compute initial relative residual : |res|/|res0|
        nr=ra
        hrr=[1.]
        # use max
        nm=max(nb,nr)
        epsr=nm*eps

        master=(enriched_field.function_space.mesh.comm.rank==0)

        # enter the loop
        stage1 = PETSc.Log.Stage("solvePM")
        stage2 = PETSc.Log.Stage("resetCM")
        stage3 = PETSc.Log.Stage("PEupdateCM")
        stage4 = PETSc.Log.Stage("updateCM")
        stage5 = PETSc.Log.Stage("solveCM")
        stage6 = PETSc.Log.Stage("residual")
        i=0
        while (i<itmx and ra>epsr):
            stage1.push()
            pm.solveProblems(solf)
            stage1.pop()
            stage2.push()
            cm.resetCoarseToStd()
            stage2.pop()
            stage3.push()
            cm.updateEnrichedOperator(pm,efunc)
            stage3.pop()
            stage4.push()
            cm.updateEnrichCoarse(Aff,Bf)
            stage4.pop()
            stage5.push()
            cm.solve(solf)
            stage5.pop()
            stage6.push()
            ra=residual(ADff,BDf,xf,1.)
            stage6.pop()
            hrb.append(ra/nb)
            hrr.append(ra/nr)
            i=i+1

        if do_mpc:
            solf.x.scatter_forward()
            mpc.backsubstitution(solf)

        return solf,ra,nm,i,hrb,hrr
#=================================================================
    def linearBasicLoopEimp( sj: core.scaleJump,
                         fine_space: FunctionSpace,
                         Aff: PETSc.Mat,
                         ADff: PETSc.Mat,
                         Bf: PETSc.Vec,
                         BDf: PETSc.Vec,
                         efunc: core.enrichedFunction,
                         enriched_field: Function,
                         mpc: typing.Optional[MultiPointConstraint]=None,
                         bc: list[DirichletBC] = [],
                         itswitch=4,
                         itmx=10,
                         eps=1.e-3
                        ):

        """
           Function to solve a linear problem by looping in between scale with the twoscale approach
           At a specific iteration the enriched dofs are all set to 1 considering that enrichment functions
           are sufficiently representative of the phenomena so that enriched dof fluctuation can be ignored 
           at coarse level. 

          :note: This function can only be used when enrichment function is shifted because its only in this case
          that enriched dofs converge to one.


           :param sj: The scale jump describing the two scale
           :param fine_space: The space describing fine scale discretization related to fine scale linear problem
           :param Aff: The fine scale assembled matrix
           :param ADff: The fine scale assembled matrix with fine Dirichlet boundary condition eliminated
           :param Bf: The fine scale assembled rhs
           :param BDf: The fine scale assembled rhs with fine Dirichlet boundary condition eliminated treated
           :param efunc: The enriched function functor that transform patch solution into a enriched 
                function (i.g. shifting field to force enriched dof to be null)
           :param enriched_field: The field describing enriched field at coarse scale. The associate linear problem 
                is managed by coarseManager object
           :type enriched_field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param mpc: The multi point constraint, if any, used to create fine_space and created from a call to core.generateMPC with sj. 
           :type mpc: `MultiPointConstraint <https://jsdokken.com/dolfinx_mpc/docs/api.html#dolfinx_mpc.MultiPointConstraint>`_ or None
           :param itswitch: The maximum number of iteration to compute without imposing enriched dofs to one
           :type itswitch: int
           :param itmx: The maximum number of iteration to compute
           :type itmx: int
           :param eps: The target precision for  relative residual value to be considered as null
           :type eps: floating point value

           :return: the field describing fine scale function related to fine scale resolution by the twoScale solver,
                                the last residual computed
                                the number of iteration done in the loop
                                the history of residual evolution
           :rtype: tuple Function,float,int,list[float]
        """
        # check coherance with mpc
        do_mpc=sj.needs_mpc
        if do_mpc and mpc==None:
            raise RuntimeError("scale jump provided expect MPC creation and you provide no mpc !")

        # solution field at fine scale
        if do_mpc:
            solf=Function(mpc.function_space,name='sol_fine_ts')
        else:
            solf=Function(fine_space,name='sol_fine_ts')

        # generate coarseManager to deal with enriched problem at coarse scale
        stage = PETSc.Log.Stage("generateCMI")
        stage.push()
        cm =core.generateCoarseManager(sj,fine_space,enriched_field.function_space,mpc,bc)
        stage.pop()
        # generate patchManager to deal with patches problem at fine scale
        stage = PETSc.Log.Stage("generatePMI")
        stage.push()
        pm =core.generatePatchManager(sj,fine_space)
        stage.pop()
        
        # generate patches problem from fine linear assembled system with Dirichlet BC eliminated
        stage = PETSc.Log.Stage("genprbPMI")
        stage.push()
        pm.generateProblems(ADff,BDf)
        stage.pop()

        # generate standard part of the coarse enriched problem from fine linear assembled system 
        stage = PETSc.Log.Stage("genstdCMI")
        stage.push()
        cm.setStdCoarse(Aff,Bf,True)
        stage.pop()

        #obtaine PETSc solf
        xf=solf.x.petsc_vec

        # initialize twoscale loop by setting solf using standard part of the coarse enriched 
        # field guess given by the user in enriched_field
        cm.projectStdCoarse(enriched_field,solf)

        # compute initial relative residual : |res|/|b|
        nb=BDf.norm()
        ra=residual(ADff,BDf,xf,1.)
        hrb=[ra/nb]
        # compute initial relative residual : |res|/|res0|
        nr=ra
        hrr=[1.]
        # use max
        nm=max(nb,nr)
        epsr=nm*eps

        master=(enriched_field.function_space.mesh.comm.rank==0)

        # enter the loop
        stage1 = PETSc.Log.Stage("solvePMI")
        stage2 = PETSc.Log.Stage("resetCMI")
        stage3 = PETSc.Log.Stage("PEupdateCMI")
        stage4 = PETSc.Log.Stage("updateCMI")
        stage5 = PETSc.Log.Stage("solveCMI")
        stage6 = PETSc.Log.Stage("residualI")
        stage7 = PETSc.Log.Stage("updateEImpCMI")
        stage8 = PETSc.Log.Stage("solveEImpCMI")
        i=0
        while (i<itmx and ra>epsr):
            stage1.push()
            pm.solveProblems(solf)
            stage1.pop()
            stage2.push()
            cm.resetCoarseToStd()
            stage2.pop()
            stage3.push()
            cm.updateEnrichedOperator(pm,efunc)
            stage3.pop()
            if i>itswitch:
                stage7.push()
                cm.updateEImp()
                stage7.pop()
                stage8.push()
                cm.solveEImp(solf)
                stage8.pop()
            else:
                stage4.push()
                cm.updateEnrichCoarse(Aff,Bf)
                stage4.pop()
                stage5.push()
                cm.solve(solf)
                stage5.pop()
            stage6.push()
            ra=residual(ADff,BDf,xf,1.)
            stage6.pop()
            hrb.append(ra/nb)
            hrr.append(ra/nr)
            i=i+1

        if do_mpc:
            solf.x.scatter_forward()
            mpc.backsubstitution(solf)

        return solf,ra,nm,i,hrb,hrr
#=================================================================
    def linearBasicLoopVideo( filename: str,
                              scale,
                              sj: core.scaleJump,
                              fine_space: FunctionSpace,
                              Aff: PETSc.Mat,
                              ADff: PETSc.Mat,
                              Bf: PETSc.Vec,
                              BDf: PETSc.Vec,
                              efunc: core.enrichedFunction,
                              enriched_field: Function,
                              mpc: typing.Optional[MultiPointConstraint]=None,
                              bc: list[DirichletBC] = [],
                              itmx=10,
                              eps=1.e-3,
                              camera=None,
                              patch=False
                            ):

        """
           Function to solve a linear problem by looping in between scale with the twoscale approach
           It works like linearBasicLoop but generate a gif movie file and thus should only be used 
           for debugging or ilustration but not for real life application.


           :param filename: The name of the gif file to create
           :param scale: scaling factor of the deformed mesh by the field
           :param sj: The scale jump describing the two scale
           :param fine_space: The space describing fine scale discretization related to fine scale linear problem
           :param Aff: The fine scale assembled matrix
           :param ADff: The fine scale assembled matrix with fine Dirichlet boundary condition eliminated
           :param Bf: The fine scale assembled rhs
           :param BDf: The fine scale assembled rhs with fine Dirichlet boundary condition eliminated treated
           :param efunc: The enriched function functor that transform patch solution into a enriched 
                function (i.g. shifting field to force enriched dof to be null)
           :param enriched_field: The field describing enriched field at coarse scale. The associate linear problem 
                is managed by coarseManager object
           :type enriched_field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param mpc: The multi point constraint, if any, used to create fine_space and created from a call to core.generateMPC with sj. 
           :type mpc: `MultiPointConstraint <https://jsdokken.com/dolfinx_mpc/docs/api.html#dolfinx_mpc.MultiPointConstraint>`_ or None
           :param itmx: The maximum number of iteration to compute
           :type itmx: int
           :param eps: The target precision for  relative residual value to be considered as null
           :type eps: floating point value

           :return: the field describing fine scale function related to fine scale resolution by the twoScale solver,
                                the last residual computed
                                the number of iteration done in the loop
           :rtype: tuple Function,float,int
        """
        # check coherance with mpc
        do_mpc=sj.needs_mpc
        if do_mpc and mpc==None:
            raise RuntimeError("scale jump provided expect MPC creation and you provide no mpc !")

        # solution field at fine scale
        if do_mpc:
            solf=Function(mpc.function_space,name='sol_fine_ts')
            ds=Function(mpc.function_space,name='delta_sol_fine_ts')
            if patch:
                patch_field=Function(mpc.function_space,name='patches')
        else:
            solf=Function(fine_space,name='sol_fine_ts')
            ds=Function(fine_space,name='delta_sol_fine_ts')
            if patch:
                patch_field=Function(fine_space,name='patches')


        # generate coarseManager to deal with enriched problem at coarse scale
        stage = PETSc.Log.Stage("generateCMV")
        stage.push()
        cm =core.generateCoarseManager(sj,fine_space,enriched_field.function_space,mpc,bc)
        stage.pop()
        # generate patchManager to deal with patches problem at fine scale
        stage = PETSc.Log.Stage("generatePMV")
        stage.push()
        pm =core.generatePatchManager(sj,fine_space)
        stage.pop()
        
        # generate patches problem from fine linear assembled system with Dirichlet BC eliminated
        pm.generateProblems(ADff,BDf)

        # for patches output
        if patch:
            list_seq=range(pm.numberOfSequence)
            domain=sj.getFineMesh
            dim=domain.topology.dim
            topoc=sj.getCoarseMesh.topology
            adj=topoc.connectivity(0,dim)
            

        # generate standard part of the coarse enriched problem from fine linear assembled system 
        cm.setStdCoarse(Aff,Bf)

        #obtaine PETSc solf
        xf=solf.x.petsc_vec
        dxf=ds.x.petsc_vec

        # initialize twoscale loop by setting solf using standard part of the coarse enriched 
        # field guess given by the user in enriched_field
        cm.projectStdCoarse(enriched_field,solf)

        # compute initial relative residual : |res|/|b|
        nb=BDf.norm()
        ra=residual(ADff,BDf,xf,1.)
        hrb=[ra/nb]
        # compute initial relative residual : |res|/|res0|
        nr=ra
        hrr=[1.]
        # compute initial relative variation : |du|/|u0|
        nd=np.sqrt(xf.dot(xf))
        rds=1.
        hds=[rds]
        ds.x.array[:]=-solf.x.array
        # use max
        nm=max(nb,nr)
        epsr=nm*eps

        master=(enriched_field.function_space.mesh.comm.rank==0)

        if do_mpc:
            solf.x.scatter_forward()
            mpc.backsubstitution(solf)
        warped=util.root_field(solf,scale)
        if do_mpc:
            mpc.homogenize(solf)
        if master:
            if patch:
                plt=pv.Plotter(shape=(1,3))
            else:
                plt=pv.Plotter(shape=(1,2))
            plt.open_gif(filename,fps=1)
            plt.subplot(0,1)
            chart=pv.Chart2D()
            chart.line(range(len(hrb)),hrb,label='|res|/|b|')
            chart.scatter(range(len(hrb)),hrb)
            chart.line(range(len(hrr)),hrr,color='#F5B027',label='|res|/|res_0|')
            chart.scatter(range(len(hrr)),hrr,color='#F5B027')
            chart.line(range(len(hds)),hds,color='#D41E0F',label='|ds|/|s|')
            chart.scatter(range(len(hds)),hds,color='#D41E0F')
            chart.x_label ="iterations"
            chart.y_label ="Relative residual"
            plt.add_chart(chart)
            plt.subplot(0,0)
            plt.add_mesh(warped,show_edges=True,show_scalar_bar=True)
            plt.add_title("iter 0: {}".format(ra/nm),font_size=8)
            if not camera==None:
                plt.camera_position=camera
            if patch:
                plt.subplot(0,2)
            else:
                plt.write_frame()
        if patch:
            pm.solveProblems(solf)
            for seq in list_seq:
                idp=pm.grabPatchSolution(seq,patch_field)
                patch_field.x.scatter_forward()
                if do_mpc:
                    mpc.backsubstitution(patch_field)
                # childs cells of the support of enriched nodes
                if idp>-1:
                    cells=[]
                    for el in adj.links(idp):
                        cells.append(sj.getChildren(el))
                    cells=np.sort(np.concatenate(cells))
                else:
                    cells = np.empty(0,dtype=int)
                # submesh of the support
                sdom=mesh.create_submesh(domain,dim,cells)
                space_patch=functionspace(sdom[0], sdom[0].ufl_domain().ufl_coordinate_element())
                field_patch=Function(space_patch)
                cell_over=np.arange(sdom[0].topology.index_map(dim).size_local,dtype=int)
                link=create_interpolation_data(space_patch,fine_space,cell_over)
                field_patch.interpolate_nonmatching(patch_field,cell_over,link)
                warpedf=util.root_field(field_patch,scale)
                if master:
                    plt.add_mesh(warpedf,show_edges=True,show_scalar_bar=True)
            if master:
                if not camera==None:
                    plt.camera_position=camera
                plt.write_frame()

        # enter the loop
        i=0
        stage1 = PETSc.Log.Stage("solvePMV")
        stage2 = PETSc.Log.Stage("resetCMV")
        stage3 = PETSc.Log.Stage("PEupdateCMV")
        stage4 = PETSc.Log.Stage("updateCMV")
        stage5 = PETSc.Log.Stage("solveCMV")
        stage6 = PETSc.Log.Stage("residualV")
        if master:
            print(f"At twoscale iteration {i} residual is {ra/nb}(b) 1.(r) 1.(ds)")
        while (i<itmx and ra>epsr and rds>eps/100):
            stage1.push()
            pm.solveProblems(solf)
            stage1.pop()
            stage2.push()
            cm.resetCoarseToStd()
            stage2.pop()
            stage3.push()
            cm.updateEnrichedOperator(pm,efunc)
            stage3.pop()
            stage4.push()
            cm.updateEnrichCoarse(Aff,Bf)
            stage4.pop()
            stage5.push()
            cm.solve(solf)
            stage5.pop()
            stage6.push()
            ra=residual(ADff,BDf,xf,1.)
            stage6.pop()
            ds.x.array[:]+=solf.x.array
            rds=np.sqrt(dxf.dot(dxf))/nd
            hrb.append(ra/nb)
            hrr.append(ra/nr)
            hds.append(rds)
            ds.x.array[:]=-solf.x.array
            i=i+1
            if do_mpc and True:
                solf.x.scatter_forward()
                mpc.backsubstitution(solf)
            warped=util.root_field(solf,scale)
            if do_mpc and True:
                mpc.homogenize(solf)
            if master:
                print(f"At twoscale iteration {i} residual is {ra/nb}(b) {ra/nr}(r) {rds}(ds)")
                plt.clear()
                plt.subplot(0,1)
                chart=pv.Chart2D()
                chart.line(range(len(hrb)),hrb,label='|res|/|b|')
                chart.scatter(range(len(hrb)),hrb)
                chart.line(range(len(hrr)),hrr,color='#F5B027',label='|res|/|res_0|')
                chart.scatter(range(len(hrr)),hrr,color='#F5B027')
                chart.line(range(len(hds)),hds,color='#D41E0F',label='|ds|/|s|')
                chart.scatter(range(len(hds)),hds,color='#D41E0F')
                chart.x_label ="iterations"
                chart.y_label ="Criterion"
                chart.y_axis.log_scale=True
                plt.add_chart(chart)
                plt.subplot(0,0)
                plt.add_mesh(warped,show_edges=True,show_scalar_bar=True)
                plt.add_title("iter {}: {}".format(i,ra/nm),font_size=8)
                if not camera==None:
                    plt.camera_position=camera
                if patch:
                    plt.subplot(0,2)
                else:
                    plt.write_frame()
            if patch:
                for seq in list_seq:
                    idp=pm.grabPatchSolution(seq,patch_field)
                    patch_field.x.scatter_forward()
                    if do_mpc:
                        mpc.backsubstitution(patch_field)
                    # childs cells of the support of enriched nodes
                    if idp>-1:
                        cells=[]
                        for el in adj.links(idp):
                            cells.append(sj.getChildren(el))
                        cells=np.sort(np.concatenate(cells))
                    else:
                        cells = np.empty(0,dtype=int)
                    # submesh of the support
                    sdom=mesh.create_submesh(domain,dim,cells)
                    space_patch=functionspace(sdom[0], sdom[0].ufl_domain().ufl_coordinate_element())
                    field_patch=Function(space_patch)
                    cell_over=np.arange(sdom[0].topology.index_map(dim).size_local,dtype=int)
                    link=create_interpolation_data(space_patch,fine_space,cell_over)
                    field_patch.interpolate_nonmatching(patch_field,cell_over,link)
                    warpedf=util.root_field(field_patch,scale)
                    if master:
                        plt.add_mesh(warpedf,show_edges=True,show_scalar_bar=True)
                if master:
                    if not camera==None:
                        plt.camera_position=camera
                    plt.write_frame()


        if master:
            plt.close()
        if do_mpc:
            #xf.view()
            solf.x.scatter_forward()
            mpc.backsubstitution(solf)
            #xf.view()

        return solf,ra/nm,i,pm,cm,hrb,hrr,hds
#=================================================================
    def initLinearBasicLoop( sj: core.scaleJump,
                             A: PETSc.Mat,
                             B: PETSc.Vec,
                             fine_space: FunctionSpace,
                             enriched_field: Function,
                             mpc: typing.Optional[MultiPointConstraint]=None,
                             bc: list[DirichletBC] = [],
                           ):

        """

           Function to initialize twoscale object used in runLinearBasicLoop
            
           :param sj: The scale jump describing the two scale
           :param A: The fine scale assembled matrix (it can be either with Dirichlet boundary condition eliminated or not)
           :param B: The fine scale assembled rhs (it can be either with Dirichlet boundary condition eliminated or not)
           :param fine_space: The space describing fine scale discretization related to fine scale linear problem
           :param enriched_field: The field describing enriched field at coarse scale. The associate linear problem 
                is managed by coarseManager object
           :type enriched_field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param mpc: The multi point constraint, if any, used to create fine_space and created from a call to core.generateMPC with sj. 
           :type mpc: `MultiPointConstraint <https://jsdokken.com/dolfinx_mpc/docs/api.html#dolfinx_mpc.MultiPointConstraint>`_ or None

           :return: The field describing fine scale function related to fine scale resolution by the twoScale solver,
                                the coarseManager object to use for the resolution
                                and the patchManager object to use for the resolution 
           :rtype: tuple Function,coarseManager,patchManager

           :warning: A and B must be both with Dirichlet boundary condition eliminated or not. Mixing eliminated and not eliminated is an error

        """

        # check coherance with mpc
        do_mpc=sj.needs_mpc
        if do_mpc and mpc==None:
            raise RuntimeError("scale jump provided expect MPC creation and you provide no mpc !")

        # solution field at fine scale
        if do_mpc:
            solf=Function(mpc.function_space,name='sol_fine_ts')
        else:
            solf=Function(fine_space,name='sol_fine_ts')

        # generate coarseManager to deal with enriched problem at coarse scale
        cm =core.generateCoarseManager(sj,fine_space,enriched_field.function_space,mpc,bc)
        # generate patchManager to deal with patches problem at fine scale
        pm =core.generatePatchManager(sj,fine_space)
        
        # generate patches problem from fine linear assembled system with Dirichlet BC eliminated
        pm.generateProblems(A,B)

        return solf,cm,pm
#=================================================================
    def runLinearBasicLoop( 
                            solf: Function,
                            cm: core.coarseManager,
                            pm: core.patchManager,
                            Aff: PETSc.Mat,
                            ADff: PETSc.Mat,
                            Bf: PETSc.Vec,
                            BDf: PETSc.Vec,
                            efunc: core.enrichedFunction,
                            enriched_field: Function,
                            mpc: typing.Optional[MultiPointConstraint]=None,
                            itmx=10,
                            eps=1.e-3,
                            reset_sol=True
                        ):

        """
           Function to solve a linear problem by looping in between scale with the twoscale approach


           :param solf: The field describing fine scale function related to fine scale resolution by the twoScale solver
           :param cm: coarseManager object initialized by initLinearBasicLoop function
           :type cm: twoscale.core.coarseManager object
           :param pm: patchManager object initialized by initLinearBasicLoop function
           :type pm: twoscale.core.patchManager object
           :param Aff: The fine scale assembled matrix
           :param ADff: The fine scale assembled matrix with fine Dirichlet boundary condition eliminated
           :param Bf: The fine scale assembled rhs
           :param BDf: The fine scale assembled rhs with fine Dirichlet boundary condition eliminated treated
           :param efunc: The enriched function functor that transform patch solution into a enriched 
                function (i.g. shifting field to force enriched dof to be null)
           :param enriched_field: The field describing enriched field at coarse scale. The associate linear problem 
                is managed by coarseManager object
           :type enriched_field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param mpc: The multi point constraint, if any, used to create fine_space and created from a call to core.generateMPC with sj. 
           :type mpc: `MultiPointConstraint <https://jsdokken.com/dolfinx_mpc/docs/api.html#dolfinx_mpc.MultiPointConstraint>`_ or None
           :param itmx: The maximum number of iteration to compute
           :type itmx: int
           :param eps: The target precision for  relative residual value to be considered as null
           :type eps: floating point value
           :param reset_sol: Reset the solution field using enrichied_fied by projection with standard operator

           :return:  the last min residual computed and the number of iteration done in the loop
           :rtype: tuple float,int
            
        """
        
        # generate standard part of the coarse enriched problem from fine linear assembled system 
        cm.setStdCoarse(Aff,Bf)

        #obtaine PETSc solf
        xf=solf.x.petsc_vec

        if reset_sol:
            # initialize twoscale loop by setting solf using standard part of the coarse enriched 
            # field guess given by the user in enriched_field
            cm.projectStdCoarse(enriched_field,solf)

        # compute initial relative residual : |res|/|b|
        nb=BDf.norm()
        ra=residual(ADff,BDf,xf,1.)
        hrb=[ra/nb]
        # compute initial relative residual : |res|/|res0|
        nr=ra
        hrr=[1.]
        # use max
        nm=max(nb,nr)
        epsr=nm*eps

        master=(enriched_field.function_space.mesh.comm.rank==0)

        # enter the loop
        i=0
        while (i<itmx and ra>epsr):
            pm.solveProblems(solf)
            cm.resetCoarseToStd()
            cm.updateEnrichedOperator(pm,efunc)
            cm.updateEnrichCoarse(Aff,Bf)
            cm.solve(solf)
            ra=residual(ADff,BDf,xf,1.)
            hrb.append(ra/nb)
            hrr.append(ra/nr)
            i=i+1

        if mpc is not None:
            solf.x.scatter_forward()
            mpc.backsubstitution(solf)

        return ra/nm,i,hrb,hrr
#=================================================================
    def runLinearBasicLoopEImp( 
                            solf: Function,
                            cm: core.coarseManager,
                            pm: core.patchManager,
                            Aff: PETSc.Mat,
                            ADff: PETSc.Mat,
                            Bf: PETSc.Vec,
                            BDf: PETSc.Vec,
                            efunc: core.enrichedFunction,
                            enriched_field: Function,
                            mpc: typing.Optional[MultiPointConstraint]=None,
                            itswitch=4,
                            itmx=10,
                            eps=1.e-3,
                            reset_sol=True
                        ):

        """
           Function to solve a linear problem by looping in between scale with the twoscale approach
           At a specific iteration the enriched dofs are all set to 1 considering that enrichment functions
           are sufficiently representative of the phenomena so that enriched dof fluctuation can be ignored 
           at coarse level. 

          :note: This function can only be used when enrichment function is shifted because its only in this case
          that enriched dofs converge to one.


           :param solf: The field describing fine scale function related to fine scale resolution by the twoScale solver
           :param cm: coarseManager object initialized by initLinearBasicLoop function
           :type cm: twoscale.core.coarseManager object
           :param pm: patchManager object initialized by initLinearBasicLoop function
           :type pm: twoscale.core.patchManager object
           :param Aff: The fine scale assembled matrix
           :param ADff: The fine scale assembled matrix with fine Dirichlet boundary condition eliminated
           :param Bf: The fine scale assembled rhs
           :param BDf: The fine scale assembled rhs with fine Dirichlet boundary condition eliminated treated
           :param efunc: The enriched function functor that transform patch solution into a enriched 
                function (i.g. shifting field to force enriched dof to be null)
           :param enriched_field: The field describing enriched field at coarse scale. The associate linear problem 
                is managed by coarseManager object
           :type enriched_field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param mpc: The multi point constraint, if any, used to create fine_space and created from a call to core.generateMPC with sj. 
           :type mpc: `MultiPointConstraint <https://jsdokken.com/dolfinx_mpc/docs/api.html#dolfinx_mpc.MultiPointConstraint>`_ or None
           :param itswitch: The maximum number of iteration to compute without imposing enriched dofs to one
           :type itswitch: int
           :param itmx: The maximum number of iteration to compute
           :type itmx: int
           :param eps: The target precision for  relative residual value to be considered as null
           :type eps: floating point value
           :param reset_sol: Reset the solution field using enrichied_fied by projection with standard operator

           :return:  the last min residual computed and the number of iteration done in the loop
           :rtype: tuple float,int
            
        """
        
        # generate standard part of the coarse enriched problem from fine linear assembled system 
        # and generate also information for imposed enriched dofs operations
        cm.setStdCoarse(Aff,Bf,True)

        #obtaine PETSc solf
        xf=solf.x.petsc_vec

        if reset_sol:
            # initialize twoscale loop by setting solf using standard part of the coarse enriched 
            # field guess given by the user in enriched_field
            cm.projectStdCoarse(enriched_field,solf)

        # compute initial relative residual : |res|/|b|
        nb=BDf.norm()
        ra=residual(ADff,BDf,xf,1.)
        hrb=[ra/nb]
        # compute initial relative residual : |res|/|res0|
        nr=ra
        hrr=[1.]
        # use max
        nm=max(nb,nr)
        epsr=nm*eps

        master=(enriched_field.function_space.mesh.comm.rank==0)

        # enter the loop
        i=0
        while (i<itmx and ra>epsr):
            pm.solveProblems(solf)
            cm.resetCoarseToStd()
            cm.updateEnrichedOperator(pm,efunc)
            if i>itswitch :
                cm.updateEImp()
                cm.solveEImp(solf)
            else:
                cm.updateEnrichCoarse(Aff,Bf)
                cm.solve(solf)
            ra=residual(ADff,BDf,xf,1.)
            hrb.append(ra/nb)
            hrr.append(ra/nr)
            i=i+1

        if mpc is not None:
            solf.x.scatter_forward()
            mpc.backsubstitution(solf)

        return ra/nm,i,hrb,hrr
