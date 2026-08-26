from dolfinx import mesh
from dolfinx import fem
from dolfinx import la
from dolfinx import plot
from twoscale import util
import basix
import numpy as np
import pyvista as pv
from mpi4py import MPI

def root_plot(dom,pv_plt,pvopt):
    spaceG = fem.functionspace(dom, basix.ufl.element("DG", dom.ufl_domain().ufl_coordinate_element().cell_type, 1))
    procfield= fem.Function(spaceG,name='proc')
    procfield.x.array[:]=dom.comm.rank
    topox, cellx, geomx = plot.vtk_mesh(spaceG)
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
        pv_plt.add_mesh(gridx,**pvopt)

def root_plot_field(dom,pv_plt,pvopt,space,field,scale,extrude=0):
    topox, cellx, geomx = plot.vtk_mesh(space)
    num_cellsx_local = dom.topology.index_map(dom.topology.dim).size_local
    num_dofsx_local = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
    num_dofsx_per_cell = topox[0]
    topology_dofsx = (np.arange(len(topox)) % (num_dofsx_per_cell+1)) != 0
    global_dofsx = space.dofmap.index_map.local_to_global(topox[topology_dofsx].copy())
    topox[topology_dofsx] = global_dofsx
    global_topologyx = dom.comm.gather(topox[:(num_dofsx_per_cell+1)*num_cellsx_local], root=0)
    global_geometryx = dom.comm.gather(geomx[:space.dofmap.index_map.size_local,:], root=0)
    global_ctx = dom.comm.gather(cellx[:num_cellsx_local])
    global_valsx = dom.comm.gather(field.x.array[:num_dofsx_local])
    if dom.comm.rank == 0:
        root_geom = np.vstack(global_geometryx)
        root_top = np.concatenate(global_topologyx)
        root_ct = np.concatenate(global_ctx)
        root_vals = np.concatenate(global_valsx)
        gridx = pv.UnstructuredGrid(root_top, root_ct, root_geom)
        valuesx = np.zeros((gridx.GetNumberOfPoints(), 3))
        if extrude>0 and dom.topology.dim==2 :
            tmp= root_vals.reshape(gridx.GetNumberOfPoints(), dom.topology.dim)
            if extrude>1 :
                valuesx[:, dom.topology.dim] = np.heaviside(np.square(tmp[:,0])+np.square(tmp[:,1]),0) 
            else:
                valuesx[:, dom.topology.dim] = np.sqrt(np.square(tmp[:,0])+np.square(tmp[:,1])) 
        else:
            valuesx[:, :dom.topology.dim] = root_vals.reshape(gridx.GetNumberOfPoints(), dom.topology.dim)
        gridx.point_data["field"] = valuesx
        warpedf = gridx.warp_by_vector("field", factor=scale)
        warpedf.set_active_vectors("field")
        pv_plt.add_mesh(warpedf,**pvopt)

def get_face_ids(domain,dim):
    labels=[]
    idxmap=domain.topology.index_map(dim-1)
    loc=np.arange(0,idxmap.size_local+idxmap.num_ghosts,dtype=np.int32)
    glob=idxmap.local_to_global(loc)
    for i in range(0,idxmap.size_local+idxmap.num_ghosts):
        if i <idxmap.size_local:
            s=""
        else:
            s="R:"
        s+="{}({})".format(glob[i],loc[i])
        labels.append(s)
    xl=mesh.compute_midpoints(domain,dim-1,loc)
    return xl,labels

def show_dof_ids(space,domain,pvopt):
    label=[]
    labels=[]
    idxmap=space.dofmap.index_map
    for i in range(0,idxmap.size_local+idxmap.num_ghosts):
        label.append(i)
    loc=np.array(label,dtype=int)
    label=idxmap.local_to_global(loc)
    for i in range(0,idxmap.size_local+idxmap.num_ghosts):
        s="({}".format(label[i])
        if i <idxmap.size_local:
            s+=",l)"
        else:
            s+=",g)"
        labels.append(s)
    xl=space.tabulate_dof_coordinates()
    plt=pv.Plotter()
    cells, types, fx = plot.vtk_mesh(domain)
    grid = pv.UnstructuredGrid(cells, types, fx)
    plt.add_mesh(grid,**pvopt)
    plt.add_point_labels(xl, labels, point_size=2, font_size=12)
    plt.add_title("global index dofs on proc {}".format(MPI.COMM_WORLD.Get_rank()),font_size=10)
    plt.show()

def print_in_gmsh(dom):
    topo=dom.topology
    dim=topo.dim
    assert(dim==3 or dim==2) # only dev for now for 3D tet or 2D tria
    cells=topo.connectivity(dim,0)
    #ghost=dom.geometry.index_map().ghosts
    #nbl=dom.geometry.index_map().size_local
    con=cells.array.copy()
    # wrong ghost in global indexing ! con[con>=nbl]=ghost[con[con>=nbl]-nbl]
    if dim==3:
        assert(len(con)%4==0)
        con=con.reshape(len(con)//4,4)
    else:
        assert(len(con)%3==0)
        con=con.reshape(len(con)//3,3)
    k=1
    print("$NOD")
    print(f"{len(dom.geometry.x)}")
    for x in dom.geometry.x:
        print(f"{k} {x[0]} {x[1]} {x[2]}")
        k+=1
    k=1
    print("$ENDNOD")
    print("$ELM")
    print(f"{len(con)}")
    if dim==3:
        for x in con:
            print(f"{k} 4 121 26 4 {x[0]+1} {x[1]+1} {x[2]+1} {x[3]+1}")
            k+=1
    else:
        for x in con:
            print(f"{k} 2 121 26 3 {x[0]+1} {x[1]+1} {x[2]+1}")
            k+=1
    print("$ENDELM")

        


