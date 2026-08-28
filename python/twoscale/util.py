# Copyright (C) 2026 - Ecole Centrale de Nantes
# Author: Alexis Salzman
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""util module that provides function/class used to simplify implementation."""
from twoscale import ts_cpp as _cpp

_ts_dolfinx_exist=False
try:
    import twoscale.ts_cpp.dolfinx as _ts
    _ts_dolfinx_exist=True
except ImportError:
    print ("Twoscale dolfinx implementation not available")
    print ("For now no other implementation exist")
    
if _ts_dolfinx_exist:
    import ufl
    import basix
    from twoscale import core
    from dolfinx import fem
    from dolfinx import la
    from dolfinx import cpp as dolfinx_cpp
    from dolfinx import mesh
    from dolfinx.plot import vtk_mesh
    from dolfinx.io import VTXWriter
    import dolfinx_mpc
    from petsc4py import PETSc
    import pyvista as pv
    import numpy as np
    import numpy.typing as npt
    import typing

#=================================================================
    def createFineScaleSytems(a: ufl.Form,
                              b: ufl.Form,
                              bcs: list[fem.bcs.DirichletBC] = [],
                              MPC: typing.Optional[dolfinx_mpc.MultiPointConstraint]=None):
        """

           Function to create fine scale systems based on given forms and boundary conditions. It return PETSc object
           :math:`A`, :math:`AD`, :math:`B`, :math:`BD`

           :param a: bilinear form of the fine scale system
           :type a: ufl expression
           :param b: linear form of the fine scale system
           :type b: ufl expression
           :param bcs: Dirichlet boundary condition to apply to the fine scale system
           :type bcs: list of DirichletBC object

           :return: The matrices :math:`A`, :math:`AD`, :math:`B`, :math:`BD`

           :note: This implementation is just for testing it assemble twice but petsc should be used instead

        """
        
        # generate form for assembly
        forma=fem.forms.form(a,dtype=PETSc.ScalarType) # compiler_option and jit not provided for now TODO
        formad=fem.forms.form(a,dtype=PETSc.ScalarType) # compiler_option and jit not provided for now TODO
        formb=fem.forms.form(b,dtype=PETSc.ScalarType) # compiler_option and jit not provided for now TODO
        formbd=fem.forms.form(b,dtype=PETSc.ScalarType) # compiler_option and jit not provided for now TODO

        if MPC!=None:
            # generate MPC matrix
            pattern = dolfinx_mpc.create_sparsity_pattern(forma, MPC)
            pattern.finalize()
            A = dolfinx_cpp.la.petsc.create_matrix(MPC.function_space.mesh.comm, pattern)
            AD = dolfinx_cpp.la.petsc.create_matrix(MPC.function_space.mesh.comm, pattern)
            # generate rhs
            B=la.petsc.create_vector([(MPC.function_space.dofmap.index_map,MPC.function_space.dofmap.index_map_bs)])
            BD=la.petsc.create_vector([(MPC.function_space.dofmap.index_map,MPC.function_space.dofmap.index_map_bs)])
            # Assemble matrix
            A.zeroEntries()
            dolfinx_mpc.assemble_matrix(forma,MPC,A=A)
            A.assemble()
            AD.zeroEntries()
            dolfinx_mpc.assemble_matrix(formad,MPC,bcs=bcs,A=AD)
            AD.assemble()
            # Assemble rhs
            with B.localForm() as b_loc:
                b_loc.set(0)
            dolfinx_mpc.assemble_vector(formb,MPC, b=B)
            B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            with BD.localForm() as b_loc:
                b_loc.set(0)
            dolfinx_mpc.assemble_vector(formbd,MPC, b=BD)
            # Apply boundary conditions to the rhs for D system only
            dolfinx_mpc.apply_lifting(BD, [formad], [bcs],MPC)
            BD.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(BD,bcs)
        else:
            #generate matrix
            A = fem.petsc.create_matrix(forma)
            AD = fem.petsc.create_matrix(formad)
            # generate rhs
            B=fem.petsc.create_vector(formb.function_spaces)
            BD=fem.petsc.create_vector(formbd.function_spaces)
            #  Set matrix and vector PETSc options. Needs to be set ??
            A.setFromOptions()
            AD.setFromOptions()
            B.setFromOptions()
            BD.setFromOptions()
            # Assemble matrix
            A.zeroEntries()
            fem.petsc.assemble_matrix(A,forma)
            A.assemble()
            AD.zeroEntries()
            fem.petsc.assemble_matrix(AD,formad, bcs=bcs)
            AD.assemble()
            # Assemble rhs
            with B.localForm() as b_loc:
                b_loc.set(0)
            fem.petsc.assemble_vector(B, formb)
            B.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            with BD.localForm() as b_loc:
                b_loc.set(0)
            fem.petsc.assemble_vector(BD, formbd)
            # Apply boundary conditions to the rhs for D system only
            fem.petsc.apply_lifting(BD, [formad], bcs=[bcs])
            BD.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            for bc in bcs:
                bc.set(BD.array_w)

        return (A,AD,B,BD)
#=================================================================
    def root_mesh(dom: mesh.Mesh):

        """

           Basic function to gather on root process (i.e. 0) the  ready to plot vtk grid from mesh given as argument.    
           Collective operation

           :param dom: the mesh distributed on a specific communicator. Somme process may hold no element of the mesh and thus will not participate to the visualization.
           :type dom: `Mesh <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/_modules/dolfinx/mesh.html#Mesh>`_  object
           :return: a vtk grid object


        """

        topox, cellx, geomx = vtk_mesh(dom)
        if len(topox)>0:
            num_dofsx_per_cell = topox[0]
            num_cellsx_local = len(topox) // (num_dofsx_per_cell+1)
            topology_dofsx = (np.arange(len(topox)) % (num_dofsx_per_cell+1)) != 0
            global_dofsx = dom.geometry.index_map().local_to_global(topox[topology_dofsx].copy())
            topox[topology_dofsx] = global_dofsx
            global_topologyx = dom.comm.gather(topox[:(num_dofsx_per_cell+1)*num_cellsx_local], root=0)
            global_geometryx = dom.comm.gather(geomx[:dom.geometry.index_map().size_local,:], root=0)
            global_ctx = dom.comm.gather(cellx[:num_cellsx_local])
        else:
            global_topologyx = dom.comm.gather(np.empty(0,dtype=int), root=0)
            global_geometryx = dom.comm.gather(np.empty((0,3),dtype=dom.geometry.x.dtype), root=0)
            global_ctx = dom.comm.gather(np.empty(0,dtype=int))
        if dom.comm.rank == 0:
            root_geom = np.vstack(global_geometryx)
            root_top = np.concatenate(global_topologyx)
            root_ct = np.concatenate(global_ctx)
            gridx = pv.UnstructuredGrid(root_top, root_ct, root_geom)
        else:
            gridx=None
        return gridx

#=================================================================
    def root_rank(dom: mesh.Mesh):

        """

           Basic function to gather on root process (i.e. 0) the ready to plot vtk grid given as argument and set a color per rank on each elements.    
           Collective operation

           :param dom: the mesh distributed on a specific communicator. Somme process may hold no element of the mesh and thus will not participate to the visualization.
           :type dom: `Mesh <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/_modules/dolfinx/mesh.html#Mesh>`_  object


        """

        spaceG = fem.functionspace(dom, basix.ufl.element("DG", dom.ufl_domain().ufl_coordinate_element().cell_type, 1))
        procfield= fem.Function(spaceG,name='proc')
        procfield.x.array[:]=dom.comm.rank
        topox, cellx, geomx = vtk_mesh(spaceG)
        num_cellsx_local = dom.topology.index_map(dom.topology.dim).size_local
        num_dofsx_local = spaceG.dofmap.index_map.size_local * spaceG.dofmap.index_map_bs
        if len(topox)>0:
            num_dofsx_per_cell = topox[0]
            topology_dofsx = (np.arange(len(topox)) % (num_dofsx_per_cell+1)) != 0
            global_dofsx = spaceG.dofmap.index_map.local_to_global(topox[topology_dofsx].copy())
            topox[topology_dofsx] = global_dofsx
            global_topologyx = dom.comm.gather(topox[:(num_dofsx_per_cell+1)*num_cellsx_local], root=0)
            global_geometryx = dom.comm.gather(geomx[:spaceG.dofmap.index_map.size_local,:], root=0)
            global_ctx = dom.comm.gather(cellx[:num_cellsx_local])
            global_valsx = dom.comm.gather(procfield.x.array[:num_dofsx_local])
        else:
            global_topologyx = dom.comm.gather(np.empty(0,dtype=int), root=0)
            global_geometryx = dom.comm.gather(np.empty((0,3),dtype=dom.geometry.x.dtype), root=0)
            global_ctx = dom.comm.gather(np.empty(0,dtype=int))
            global_valsx = dom.comm.gather(np.empty(0))
        if dom.comm.rank == 0:
            root_geom = np.vstack(global_geometryx)
            root_top = np.concatenate(global_topologyx)
            root_ct = np.concatenate(global_ctx)
            root_vals = np.concatenate(global_valsx)
            gridx = pv.UnstructuredGrid(root_top, root_ct, root_geom)
            gridx.point_data["proc"] = root_vals
        else:
            gridx=None
        return gridx
#=================================================================
    def par_rank(dom: mesh.Mesh):

        """

           Basic function to genarate the ready to plot vtk grid from mesh given as argument and set a color per rank on each elements.    
           Collective operation

           :param dom: the mesh distributed on a specific communicator. Somme process may hold no element of the mesh and thus will not participate to the visualization.
           :type dom: `Mesh <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/_modules/dolfinx/mesh.html#Mesh>`_  object


        """

        spaceG = fem.functionspace(dom, basix.ufl.element("DG", dom.ufl_domain().ufl_coordinate_element().cell_type, 1))
        procfield= fem.Function(spaceG,name='proc')
        procfield.x.array[:]=dom.comm.rank
        topox, cellx, geomx = vtk_mesh(spaceG)
        gridx = pv.UnstructuredGrid(topox, cellx, geomx)
        gridx.point_data["proc"] = procfield.x.array
        return gridx
#=================================================================
    def root_field( field: fem.Function,
                        scale,extrude=0):

        """

           Basic function to gather on root process (i.e. 0) the given field on the gathered mesh given as argument and deformed by that field.
           In 2D the field can be used as an extrusion value in the z direction.   
           Collective operation

           :param field: The field to plot
           :type field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param scale: The scaling value used to deform the mesh from field values
           :param extrude: In 2D if set to 1 the norm of the components at nodes is used as the z value. If set to 2 the square norm of the components is filtered by a Heaviside function to obtain a binary
            vision of the zone where the field is non null (1 extrusion) or null (0 extrusion)


        """

        space=field.function_space
        dom=space.mesh
        nbv=space.num_sub_spaces
        if nbv<1:
            nbv=1
        topox, cellx, geomx = vtk_mesh(space)
        num_cellsx_local = dom.topology.index_map(dom.topology.dim).size_local
        num_dofsx_local = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
        if len(topox)>0:
            num_dofsx_per_cell = topox[0]
            topology_dofsx = (np.arange(len(topox)) % (num_dofsx_per_cell+1)) != 0
            global_dofsx = space.dofmap.index_map.local_to_global(topox[topology_dofsx].copy())
            topox[topology_dofsx] = global_dofsx
            global_topologyx = dom.comm.gather(topox[:(num_dofsx_per_cell+1)*num_cellsx_local], root=0)
            global_geometryx = dom.comm.gather(geomx[:space.dofmap.index_map.size_local,:], root=0)
            global_ctx = dom.comm.gather(cellx[:num_cellsx_local],root=0)
            global_valsx = dom.comm.gather(field.x.array[:num_dofsx_local],root=0)
        else:
            global_topologyx = dom.comm.gather(np.empty(0,dtype=int), root=0)
            global_geometryx = dom.comm.gather(np.empty((0,3),dtype=dom.geometry.x.dtype), root=0)
            global_ctx = dom.comm.gather(np.empty(0,dtype=int),root=0)
            global_valsx = dom.comm.gather(np.empty(0,dtype=field.x.array.dtype),root=0)
        if dom.comm.rank == 0:
            root_geom = np.vstack(global_geometryx)
            root_top = np.concatenate(global_topologyx)
            root_ct = np.concatenate(global_ctx)
            root_vals = np.concatenate(global_valsx)
            gridx = pv.UnstructuredGrid(root_top, root_ct, root_geom)
            valuesx = np.zeros((gridx.GetNumberOfPoints(), 3))
            if extrude>0 and dom.topology.dim==2 :
                tmp= root_vals.reshape(gridx.GetNumberOfPoints(), nbv)
                if extrude>1 :
                    if nbv>1:
                        valuesx[:, dom.topology.dim] = np.heaviside(np.square(tmp[:,0])+np.square(tmp[:,1]),0) 
                    else:
                        valuesx[:, dom.topology.dim] = np.heaviside(np.square(tmp[:,0]),0) 
                else:
                    if nbv>1:
                        valuesx[:, dom.topology.dim] = np.sqrt(np.square(tmp[:,0])+np.square(tmp[:,1])) 
                    else:
                        valuesx[:, dom.topology.dim] = np.sqrt(np.square(tmp[:,0])) 
            else:
                #valuesx[:, :dom.topology.dim] = root_vals.reshape(gridx.GetNumberOfPoints(), dom.topology.dim)
                valuesx[:, :nbv] = root_vals.reshape(gridx.GetNumberOfPoints(), nbv)
            gridx.point_data[field.name] = valuesx
            warpedf = gridx.warp_by_vector(field.name, factor=scale)
            warpedf.set_active_vectors(field.name)
        else:
            warpedf=None
        return warpedf

#=================================================================
    def par_field( field: fem.Function,
                        scale,extrude=0):

        """

           Basic function to genrate from the given field the mesh deformed by that field.
           In 2D the field can be used as an extrusion value in the z direction.   
           Collective operation

           :param field: The field to plot
           :type field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_
           :param scale: The scaling value used to deform the mesh from field values
           :param extrude: In 2D if set to 1 the norm of the components at nodes is used as the z value. If set to 2 the square norm of the components is filtered by a Heaviside function to obtain a binary
            vision of the zone where the field is non null (1 extrusion) or null (0 extrusion)


        """

        space=field.function_space
        dom=space.mesh
        nbv=space.num_sub_spaces
        if nbv<1:
            nbv=1
        topox, cellx, geomx = vtk_mesh(space)
        gridx = pv.UnstructuredGrid(topox, cellx, geomx)
        valuesx = np.zeros((gridx.GetNumberOfPoints(), 3))
        if extrude>0 and dom.topology.dim==2 :
            tmp= field.x.array.reshape(gridx.GetNumberOfPoints(), nbv)
            if extrude>1 :
                if nbv>1:
                    valuesx[:, dom.topology.dim] = np.heaviside(np.square(tmp[:,0])+np.square(tmp[:,1]),0) 
                else:
                    valuesx[:, dom.topology.dim] = np.heaviside(np.square(tmp[:,0]),0) 
            else:
                if nbv>1:
                    valuesx[:, dom.topology.dim] = np.sqrt(np.square(tmp[:,0])+np.square(tmp[:,1])) 
                else:
                    valuesx[:, dom.topology.dim] = np.sqrt(np.square(tmp[:,0])) 
        else:
            #valuesx[:, :dom.topology.dim] = root_vals.reshape(gridx.GetNumberOfPoints(), dom.topology.dim)
            valuesx[:, :nbv] = field.x.array.reshape(gridx.GetNumberOfPoints(), nbv)
        gridx.point_data[field.name] = valuesx
        warpedf = gridx.warp_by_vector(field.name, factor=scale)
        warpedf.set_active_vectors(field.name)
        return warpedf


#=================================================================
    def all_sub(space: fem.FunctionSpace,
                   dom: mesh.Mesh,
                   subs,psid,l=0,k=0):
        if space.mesh==dom:
            if (space.num_sub_spaces>1):
                if l>0:
                    for i in range(space.num_sub_spaces):
                        all_sub(space.sub(i),dom,subs,psid,l+1,k)
                else:
                    for i in range(space.num_sub_spaces):
                        all_sub(space.sub(i),dom,subs,psid,l+1,i)
            else:
                subs.append(space)
                if l>1:
                    psid.append(k)
                else:
                    psid.append(0)

    def show_dofs_ids(space: fem.FunctionSpace,
                      dom: mesh.Mesh,
                      pv_plt: pv.Plotter,
                      pvopt: dict,
                      pvoptl: dict):
        """

           Function to plot  global(local)  indexes of the dofs related to a space on a domain.   
           It is relatively specific to TS library as space can only be:

           * related to fine scale
           * related to coarse scale without enrichment
           * related to coarse scale with enrichment

           :param space: space from which dofs are showed
           :type dom: `FunctionSpace <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.FunctionSpace>`_  object
           :param dom: mesh where dofs of space are located
           :type dom: `Mesh <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.mesh.html#dolfinx.mesh.Mesh>`_  object
           :param pv_plt: The Plotter (pyVista sens) to which this function add the dofs index
           :type pv_plt: An instance of `Plotter <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter#pyvista-plotter>`_ class
           :param pvopt: Option to set when adding mesh view with the `add_mesh <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh#pyvista.Plotter.add_mesh>`_ method
           :type pvopt: Python dictionary of options
           :param pvoptl: Option to set when adding label view with the `add_point_labels <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_point_labels#pyvista.Plotter.add_point_labels>`_ method
           :type pvoptl: Python dictionary of options

        """

        subs=[]
        psid=[]
        all_sub(space,dom,subs,psid)
        idxmaps=space.dofmap.index_map
        bs=space.dofmap.bs
        loc_label=np.array([i*bs+b for i in range(idxmaps.size_local+idxmaps.num_ghosts) for b in range(bs)])
        glob_label_loc=np.array([i for i in range(idxmaps.size_local)])
        glob_label_loc=idxmaps.local_to_global(glob_label_loc)
        glob_label=np.concatenate((np.array([i*bs+b for i in glob_label_loc for b in range(bs)],dtype=np.int64),np.array([i*bs+b for i in idxmaps.ghosts for b in range(bs)],dtype=np.int64)))
        labs=[]
        firstsub=True
        for s in subs:
            sc,mapping = s.collapse()
            if firstsub:
                xl=sc.tabulate_dof_coordinates()
            idxmap=sc.dofmap.index_map
            lab=[]
            for i in range(len(mapping)):
                j=mapping[i]
                if firstsub:
                    if i<idxmap.size_local:
                        lab.append("{}({}):{}({})".format(glob_label[j]//bs,loc_label[j]//bs,glob_label[j],loc_label[j]))
                    else:
                        lab.append("R:{}({}):{}({})".format(glob_label[j]//bs,loc_label[j]//bs,glob_label[j],loc_label[j]))
                else:
                    lab.append("{}({})".format(glob_label[j],loc_label[j]))
            labs.append(lab)
            firstsub=False

        labels=[]
        for i in range(xl.shape[0]):
            s="{}".format(labs[0][i])
            cp=psid[0]
            for j in range(1,len(labs)):
                if cp != psid[j]:
                    cp=psid[j]
                    s+="\n{}".format(labs[j][i])
                else:
                    s+=",{}".format(labs[j][i])
            labels.append(s)
        cells, types, fx = vtk_mesh(dom)
        grid = pv.UnstructuredGrid(cells, types, fx)
        pv_plt.add_mesh(grid,**pvopt)
        pv_plt.add_point_labels(xl, labels, **pvoptl)
#=================================================================
    def show_patch_sol(sj:core.scaleJump,
                       pm:core.patchManager,
                       field:fem.Function,
                       cdomain:mesh.Mesh,
                       pvopt: dict,
                       do_mpc: bool,
                       mpc,scale,seqb=-1,nbseq=1,merge=False,all_data=False,raw=False,extrude=0,
                       to_file: str = ""):
        """

           Function to plot  patche(s) solution(s) 

           :param sj: scale jump
           :param pm: Patch manager
           :param field: field to store  patche(s) solution(s)
           :type field: `Function <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.fem.Function>`_  object
           :param cdomain: mesh at coarse scale
           :type cdomain: `Mesh <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/generated/dolfinx.fem.html#dolfinx.mesh.Mesh>`_  object
           :param pvopt: Option to set when adding mesh view with the `add_mesh <https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh#pyvista.Plotter.add_mesh>`_ method
           :type pvopt: Python dictionary of options
           :param do_mpc: If true next argument must be used for mpc treatment
           :type do_mpc: Boolean
           :param mpc:  Multipoint constraint definition
           :param scale:  scaling for deformed mesh representation
           :param seqb:  first sequence to collect or -1 for all
           :param nbseq:  number of sequence from seqb to collect 
           :param merge: if true all selected sequence are plotted on the same plot. Othewise 1 sequence per subplot
           :type merge: Boolean
           :param all_data: if true one view per selected sequence with associated mesh (force merge to False)
           :type all_data: Boolean
           :param raw: if true show patches solution in a distributed way from fine field (force merge and all_data to False)
           :type raw: Boolean
           :param extrude:  In 2D, if 1 transform solution into z component. If 0 solution unchanged (2 meaning less in this context)
           :param to_file: Save patch(es) of a sequence in a file named to_file_xx with xx the sequence id. Only raw=False possible in this case
           :type merge: string

        """

        space=field.function_space
        domain=space.mesh
        dim=domain.topology.dim
        topoc=cdomain.topology
        adj=topoc.connectivity(0,dim)
        if extrude>1:
            scale=1
        if seqb<0:
            nbseq=pm.numberOfSequence
            list_seq=range(nbseq)
        else:
            list_seq=range(seqb,seqb+nbseq)
        r=0
        if nbseq%3>0:
            r=1
        if len(to_file)>0:
            save_to_file=True
        else:
            save_to_file=False
        if not raw and domain.comm.rank == 0:
            if not all_data:
                if merge:
                    plt=pv.Plotter()
                else:
                    plt=pv.Plotter(shape=(nbseq//3+r, 3))
        k=0
        for seq in list_seq:
            idp=pm.grabPatchSolution(seq,field)
            field.x.scatter_forward()
            if do_mpc:
                mpc.backsubstitution(field)
            if raw:
                warpedfr=root_field(field,scale,extrude)
                gridfr=root_rank(domain)
                if domain.comm.rank == 0:
                    plt=pv.Plotter(shape=(3, 1))
                    plt.subplot(2, 0)
                    plt.add_mesh(gridfr,**pvopt)
                    plt.subplot(0, 0)
                else:
                    plt=pv.Plotter()
                topo, cellt, geom = vtk_mesh(space)
                grid = pv.UnstructuredGrid(topo, cellt, geom)
                values = np.zeros((grid.GetNumberOfPoints(), 3))
                values[:, :dim] = field.x.array.reshape(grid.GetNumberOfPoints(), dim)
                grid.point_data["field"] = values
                warpedf = grid.warp_by_vector("field", factor=scale)
                warpedf.set_active_vectors("field")
                plt.add_mesh(warpedf,**pvopt)
                plt.add_title("seq {} (proc{})".format(seq,domain.comm.rank),font_size=7)
                plt.add_axes()
                if domain.comm.rank == 0:
                    plt.subplot(1, 0)
                    plt.add_mesh(warpedfr,**pvopt)
                    plt.link_views()
                plt.show()
            else:
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
                space_patch=fem.functionspace(sdom[0], sdom[0].ufl_domain().ufl_coordinate_element())
                field_patch=fem.Function(space_patch)
                cell_over=np.arange(sdom[0].topology.index_map(dim).size_local,dtype=int)
                link=fem.create_interpolation_data(space_patch,space,cell_over)
                field_patch.interpolate_nonmatching(field,cell_over,link)
                if save_to_file:
                    spaceG = fem.functionspace(sdom[0], basix.ufl.element("DG", sdom[0].ufl_domain().ufl_coordinate_element().cell_type, 1))
                    procfield= fem.Function(spaceG,name='proc')
                    procfield.x.array[:]=domain.comm.rank
                    with VTXWriter(field_patch.function_space.mesh.comm, f"{to_file}_{seq}_proc.bp", [procfield]) as ofile:
                        ofile.write(0.)
                    with VTXWriter(field_patch.function_space.mesh.comm, f"{to_file}_{seq}.bp", [field_patch]) as ofile:
                        ofile.write(0.)
                #field_patch.x.scatter_forward()
                #field_patch.x.scatter_reverse(la.InsertMode.insert)
                warpedf=root_field(field_patch,scale,extrude)
                if all_data:
                    gridf=root_mesh(sdom[0])
                    gridfr=root_rank(sdom[0])
                if domain.comm.rank == 0:
                    if all_data:
                        plt=pv.Plotter(shape=(2, 1))
                        plt.subplot(0, 0)
                        plt.add_mesh(warpedf,**pvopt)
                        plt.add_mesh(gridf,opacity=0.5,show_edges=True)
                        plt.add_title("seq {}".format(seq),font_size=7)
                        plt.subplot(1, 0)
                        plt.add_mesh(gridfr,**pvopt)
                        plt.add_axes()
                        plt.link_views()
                        plt.show()
                    else:
                        if merge:
                            plt.add_mesh(warpedf,**pvopt)
                        else:
                            plt.subplot(k//3, k%3)
                            plt.add_title("seq {}".format(seq),font_size=7)
                            plt.add_mesh(warpedf,**pvopt)
                            k+=1
        if domain.comm.rank == 0 and not all_data and not raw:
            plt.show()
#=================================================================
    def rootMasterFace(sj: core.scaleJump,
                       dom: mesh.Mesh):

        """

           Basic function to gather on root process (i.e. 0) the  ready to plot vtk grid of coarse mesh master faces.
           Collective operation

           :param sj: The scale jump describing the two scale
           :param dom: the mesh distributed on a specific communicator. Somme process may hold no element of the mesh and thus will not participate to the visualization.
           :type dom: `Mesh <https://docs.fenicsproject.org/dolfinx/v0.9.0/python/_modules/dolfinx/mesh.html#Mesh>`_  object
           :return: a vtk grid object


        """
        mf=np.sort(sj.getCoarseMaster)
        dimm1=dom.geometry.dim-1
        sub=mesh.create_submesh(dom,dimm1,mf[mf<dom.topology.index_map(dimm1).size_local])
        gridx=root_rank(sub[0])
        return gridx
#=================================================================
#=================================================================
#==== geom =======================================================
#=================================================================
#=================================================================
    def normalizev(x:npt.ArrayLike):
        """
        
        For a given 2D/3D vector compute and return its normalized version and its 2-norm

        :param x: The vector to normalize
        :return: Normalized version of x and its norm

        """
        n=np.linalg.norm(x)
        return x/n,n
#=================================================================
    def distanceFromSphere(p:npt.ArrayLike,
                  O:npt.ArrayLike,
                  r:float
                  ):
        """

        Give the distance of set of point p to the sphere defined by its center and radius

        :param p: The points to test (shape 3xnb points)
        :param O: The given center
        :param r: The radius
        :return: Distance of p to the sphere

        """
        # points to test
        pt=np.transpose(np.asarray(p))
        nbp=(pt.shape)[0]
        assert((pt.shape)[1]==3)
        # origin,point p vector
        d=pt-np.asarray(O,)
        # point p distance to sphere surface if not inside it
        nd=np.linalg.norm(d,axis=1)
        duc=np.zeros(nbp)
        duc[nd>r]=nd[nd>r]-r
        return duc
#=================================================================
    def distanceFromParallelogram(p:npt.ArrayLike,
                                  O:npt.ArrayLike,
                                  u:npt.ArrayLike,
                                  v:npt.ArrayLike):
        """

        Give the distance of a point p to the parallelogram defined by a corner and 2 vectors describing each non colinear edges

        :param p: The points to test (shape 3xnb points)
        :param O: The given corner
        :param u: The vector describing one pair of edges
        :param v: The vector describing the second pair of edges
        :return: Distance of p to the parallelogram

        """

        # points to test
        pt=np.transpose(np.asarray(p))
        nbp=(pt.shape)[0]
        assert((pt.shape)[1]==3)
        
        #plane bases
        [u,nu]=normalizev(u)
        [v,nv]=normalizev(v)
        #normal to plane
        [w,nw]=normalizev(np.cross(u,v))
        # origin,point p vector
        d=pt-np.asarray(O,)
        # point p coordinate in plane
        uc=np.dot(d,u)
        vc=np.dot(d,v)
        # point p distance to plane
        wc=np.dot(d,w)
        # point p distance to edges if not in parallelogram
        duc=np.zeros(nbp)
        duc[uc>nu]=uc[uc>nu]-nu
        duc[uc<0]=uc[uc<0]
        duc=duc*duc
        dvc=np.zeros(nbp)
        dvc[vc>nv]=vc[vc>nv]-nv
        dvc[vc<0]=vc[vc<0]
        dvc=dvc*dvc
        # finale distance to parallelogram
        return np.sqrt(duc+dvc+wc*wc)
#=================================================================
    def distanceFromEllipse(p:npt.ArrayLike,
                               p0:npt.ArrayLike,
                               p1:npt.ArrayLike,
                               p2:npt.ArrayLike):
        """

        Give the distance of a set of point p to the ellipse defined by 3 points (center, end axis 1, end axis 2)

        :param p: The points to test (shape 3xnb points)
        :param p0: The center
        :param p1: The end of the axis 1
        :param p2: The end of the axis 2
        :return: Distance of p to the ellipse

        """

        # points to test
        pt=np.transpose(np.asarray(p))
        nbp=(pt.shape)[0]
        assert((pt.shape)[1]==3)
        
        # origin
        O=np.asarray(p0)
        #plane bases
        [u,nu]=normalizev(np.asarray(p1)-O)
        [v,nv]=normalizev(np.asarray(p2)-O)
        nu2=nu*nu
        nv2=nv*nv
        #normal to plane
        [w,nw]=normalizev(np.cross(u,v))
        # origin,point p vector
        d=pt-O
        # point p coordinate in plane
        uc=np.dot(d,u)
        vc=np.dot(d,v)
        # ellipse equation
        eq=uc*uc/nu2+vc*vc/nv2
        # point p distance to plane
        wc=np.dot(d,w)
        # point p distance to edges if not in ellipse
        deq=np.zeros(nbp)
        deq[eq>1]=np.sqrt(eq[eq>1])
        # finale distance to ellipse
        return np.sqrt(deq*deq+wc*wc)
#=================================================================
    def insideParallelepiped(p:npt.ArrayLike,
                  O:npt.ArrayLike,
                  u:npt.ArrayLike,
                  v:npt.ArrayLike,
                  w:npt.ArrayLike,
                  ):
        """

        Mark as True the points of a given set that are inside a parallelepiped defined by a corner and 3 vectors describing each non colinear edges

        :param p: The points to test (shape 3xnb points)
        :param O: The given corner
        :param u: The vector describing one edges direction and size
        :param v: The vector describing the second edges direction and size
        :param w: The vector describing the third edges direction and size
        :return: array indicating if test points are in (True) our out (False) of the parallelepiped

        """
        #parallelepiped vector
        [u,nu]=normalizev(u)
        [v,nv]=normalizev(v)
        [w,nw]=normalizev(w)
        # points to test
        pt=np.transpose(np.asarray(p))
        assert((pt.shape)[1]==3)
        # origin,point p vector
        d=pt-np.asarray(O,)
        # point p coordinate in box
        uc=np.dot(d,u)
        vc=np.dot(d,v)
        wc=np.dot(d,w)
        return np.logical_and(np.logical_and(uc>=0,uc<=nu),np.logical_and(np.logical_and(vc>=0,vc<=nv),np.logical_and(wc>=0,wc<=nw)))
#=================================================================
    def insideSphere(p:npt.ArrayLike,
                  O:npt.ArrayLike,
                  r:float
                  ):
        """

        Mark as True the points of a given set that are inside a sphere defined by its center and radius

        :param p: The points to test (shape 3xnb points)
        :param O: The given center
        :param r: The radius
        :return: array indicating if test points are in (True) our out (False) of the sphere

        """
        # points to test
        pt=np.transpose(np.asarray(p))
        assert((pt.shape)[1]==3)
        # origin,point p vector
        d=pt-np.asarray(O,)
        return np.linalg.norm(d,axis=1)<r
#=================================================================
    def insideCylinder( p:npt.ArrayLike,
                        O:npt.ArrayLike,
                        axes:npt.ArrayLike,
                        r:float
                  ):
        """

        Mark as True the points of a given set that are inside a cylinder defined by an axis (that give also its lenght), a origine on that axis and a radius.

        :param p: The points to test (shape 3xnb points)
        :param O: The given cylindrical bases
        :param axes: The vector describing cylinder axes and its length
        :param r: The radius of the cylinder
        :return: array indicating if test points are in (True) or out (False) of the cylinder

        """
        [a,na]=normalizev(axes)
        # points to test
        pt=np.transpose(np.asarray(p))
        nbp=(pt.shape)[0]
        assert((pt.shape)[1]==3)
        # origin,point p vector
        d=pt-np.asarray(O,)
        # coordinate along axes
        ac=np.dot(d,a)
        # radial coordinate 
        rc=np.linalg.norm(d-np.dot(ac[:,np.newaxis],np.transpose(a[:,np.newaxis])),axis=1)
        return np.logical_and(rc<=r,np.logical_and(ac>=0,ac<=na))
#=================================================================
    def insideExtrudedEllipse(p:npt.ArrayLike,
                               p0:npt.ArrayLike,
                               p1:npt.ArrayLike,
                               p2:npt.ArrayLike,
                               h:float):
        """

        Mark as True the points of a given set that are inside an extruded ellipse volume.
        The ellipse is defined in space by 3 points (center, end axis 1, end axis 2) and an extrusion height

        :param p: The points to test (shape 3xnb points)
        :param p0: The center
        :param p1: The end of the axis 1
        :param p2: The end of the axis 2
        :param h: extrusion height along orthogonal vector to the plan formed by p0,p1 and p2
        :return: array indicating if test points are in (True) or out (False) of the elliptical volume

        """

        # points to test
        pt=np.transpose(np.asarray(p))
        nbp=(pt.shape)[0]
        assert((pt.shape)[1]==3)
        
        # origin
        O=np.asarray(p0)
        #plane bases
        [u,nu]=normalizev(np.asarray(p1)-O)
        [v,nv]=normalizev(np.asarray(p2)-O)
        nu2=nu*nu
        nv2=nv*nv
        #normal to plane
        [w,nw]=normalizev(np.cross(u,v))
        # origin,point p vector
        d=pt-O
        # point p coordinate in plane
        uc=np.dot(d,u)
        vc=np.dot(d,v)
        # ellipse equation
        eq=uc*uc/nu2+vc*vc/nv2
        # point p distance to plane
        wc=np.dot(d,w)
        return np.logical_and(np.logical_and(wc>=0,wc<=h),eq<=1.)
