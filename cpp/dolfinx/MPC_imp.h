/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_MPC_IMP
#define TS_DOLFINX_MPC_IMP
#ifndef TS_DOLFINX_MPC
#error "Should not be included by hand"
#endif
#include <basix/e-lagrange.h>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>

#include <limits>
#include <unordered_set>
#include <unordered_map>

#include "util.h"
// namespace twoscale
namespace twoscale_dolfinx
{
template <dolfinx::scalar T,
          std::floating_point U >
dolfinx_mpc::mpc_data<T> generateMPC(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &sj,
                                             std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> space)
{
   // In 3D some slaves on edges have to be transferred to other process where they can't be generated due to lack of
   // master 'face' connected to them. This data structure provides storage for slave (global id) and master components (global id + coefficient).
   // As embedded in this function it inherits coefficient type T
   class exchangedMPCData
   {
     public:
      exchangedMPCData(const std::int64_t &slave_ = -1, const std::array<T, 2> &coef_ = {static_cast<T>(0.), static_cast<T>(0.)},
                       const std::array<std::int64_t, 2> &master_ = {-1, -1},
                       const std::array<std::int32_t, 2> &owner_ = {-1, -1})
          : slave(slave_), coef(coef_), master(master_), owner(owner_)
      {
      }
      std::int64_t slave;
      std::array<T,2> coef;
      std::array<std::int64_t,2> master;
      std::array<std::int32_t,2> owner;
   };

   spdlog::info("Compute mpc relation if any by interpolation of 'face' and their child at support boundary");

   // =================================
   // container to store mpc definition
   // =================================
   dolfinx_mpc::mpc_data<T> data_mpc;
   std::vector<std::int32_t> &offsets = data_mpc.offsets;
   offsets.resize(1, 0);  // set to 0 first cell and fulfill assert requirements that offsets dimension must size of slaves +1

   // =============================
   // return empty if no mpc needed
   // =============================
   if (sj.needs_mpc == false)
   {
      return std::move(data_mpc);
   }

   // alias
   std::vector<std::int32_t> &slaves = data_mpc.slaves;
   std::vector<std::int64_t> &masters = data_mpc.masters;
   std::vector<T> &coeffs = data_mpc.coeffs;
   std::vector<std::int32_t> &owners = data_mpc.owners;

   // ===============
   // check coherance
   // ===============
   // TODO: mesh dim etc


   // ======================================
   // collect coarse topological information
   // ======================================
   auto cdomain = sj.getCoarseMesh();
   auto topoc = cdomain->topology();
   int dim = topoc->dim();
   int dimm1 = dim - 1;
   auto adjf2c = topoc->connectivity(dimm1, dim);

   // ======================================
   // collect coarse geometrical information
   // ======================================
   auto &geomc = cdomain->geometry();
   std::span<const U> xc_g = geomc.x();
   auto xc_dofmap = geomc.dofmap();
   const auto &cellc_cmaps = geomc.cmaps();
   const auto &cellc_cmap = cellc_cmaps.front();
   assert(cellc_cmap.is_affine());
   const std::size_t num_dofs_g = cellc_cmap.dim();
   const std::size_t gdim = geomc.dim();

   // ====================================
   // collect fine topological information
   // ====================================
   auto fdomain = sj.getFineMesh();
   auto topof = fdomain->topology();
   auto topofm = fdomain->topology_mutable();
   assert(dim == topof->dim());
   topofm->create_connectivity(dimm1,dim);
   topofm->create_connectivity(dimm1,0);
   auto adjf2v = topof->connectivity(dimm1, 0);
   //auto idxmap0f = topof->index_map(0);
   //std::span<const int> topo_ghost_owners = idxmap0f->owners();
 
   // =======================
   // collect dof information
   // =======================
   auto dofmap=space->dofmap();
   auto &idxmapd = dofmap->index_map;
   auto nbngl = idxmapd->size_local();
   auto bs_d=dofmap->index_map_bs();
   assert(bs_d > 0);
   auto xf_d=space->tabulate_dof_coordinates(false);
   std::span<const int> dof_ghost_owners = idxmapd->owners();
   std::span<const std::int64_t> global_dof_ghost = idxmapd->ghosts();
   
   // ================
   // collect mpi info
   // ================
   MPI_Comm comm = fdomain->comm();
   int pid = dolfinx::MPI::rank(comm);
   int nbproc = dolfinx::MPI::size(comm);

   // ==================
   // get master 'faces'
   // ==================
   auto facesc_master = sj.getCoarseMaster();
   std::int32_t nb_masterc=facesc_master.size();


   // ===============================================
   // preparing object for master 'faces' computation
   // ===============================================
   // container to store all child nodes
   std::unordered_set<std::int32_t> nodes;
   // container to store all child dof (master+slave)
   std::set<std::int32_t> child_dofs;
   // container to store slave child dof already locally treated (used only in 3D)
   std::set<std::int32_t> slaves_child_dofs;
   // container to store slave child nodes
   std::vector<std::int32_t> slaves_nodes;
   // container to store master child nodes
   std::vector<std::int32_t> master_nodes;
   //
   // container to store vertex coordinate of a cell connected to a master 'face' 
   std::vector<U> coord_dofs_b(num_dofs_g * gdim);
   mdspan_t<U, 2> coord_dofs(coord_dofs_b.data(), num_dofs_g, gdim);

   // origine of the cell
   std::array<U, 3> x0;

   // sizes
   const std::int32_t num_dofs_dimm1 = num_dofs_g - 1;

   // zero in U arithmetic
   const U zero=static_cast<U>(0.);

   // jacobian and its inverse
   std::vector<U> J_b(gdim * dim);
   mdspan_t<U, 2> J(J_b.data(), gdim, dim);
   std::vector<U> K_b(dim * gdim);
   mdspan_t<U, 2> K(K_b.data(), dim, gdim);

   // point shape
   std::array<std::size_t, 2> point_shape={1,dim};

   // cell interpolation function and its derivative for one point
   const std::array<std::size_t, 4> phi_master_shape = cellc_cmap.tabulate_shape(1, 1); 
   //PRINT("phi_master_shape",phi_master_shape);
   std::vector<U> phi_master_b(std::reduce(phi_master_shape.begin(), phi_master_shape.end(), 1, std::multiplies{}));
   mdspan_t<const U, 4> phi_master(phi_master_b.data(), phi_master_shape);
   cellc_cmap.tabulate(1, std::vector<U>(dim), point_shape, phi_master_b);// get for first node FF and derivative
   //PRINT("phi_master",phi_master);
   auto dphi_master = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
       phi_master, std::pair(1, dim + 1), 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);  // only derivative
   //PRINT("phi_master_b", phi_master_b);

   // reference coodinates
   std::array<U,3> X_child_b;
   mdspan_t<U, 2> X_child(X_child_b.data(), 1, dim);

   // epsilon in U arithmetic : use this limit is ok as normally
   // child nodes are note supposed to be that close to master nodes
   // 1000 time numerical limit is expected to be well suited to identify properly
   const U eps=  static_cast<U>(1000.)*std::numeric_limits<U>::epsilon();
   const U onemeps=  static_cast<U>(1.)-eps;

   // container to store temporarly master index
   std::vector<std::int32_t> elem_loc_idx(num_dofs_g, -1);
   std::vector<std::int64_t> elem_glob_idx(num_dofs_g, -1);

   // ================================================================
   // loop on master 'faces' to identify all fine dofs involved in mpc
   // ================================================================
   for (std::size_t mf = 0; mf < nb_masterc; ++mf)
   {
      //current face to treat
      const auto &f = facesc_master[mf];
      //PRINT("master face",f);

      // master face chids if any
      auto face_child = sj.getFaceChilds(f);

      // This proc do not holds childs, they are in remote. Simply pass to another face. Remote will treat this case
      if (face_child.size() < 1) continue;

      //PRINT("face_child", face_child);


      auto slaves_dofs_=dolfinx::fem::locate_dofs_topological(*topof,*dofmap,dimm1,face_child,false);

      child_dofs.insert(slaves_dofs_.begin(),slaves_dofs_.end());
      //PRINT("slaves_dofs", slaves_dofs_);
   }
   //PRINT("childs", child_dofs);
   if(dim>2) slaves_child_dofs=child_dofs;

   // =====================================
   // loop on master 'faces' to compute mpc
   // localy defined in this proc
   // =====================================
   for (std::size_t mf = 0; mf < nb_masterc; ++mf)
   {

      //current face to treat
      const auto &f = facesc_master[mf];
      //PRINT("master face",f);

      // master face chids if any
      auto face_child = sj.getFaceChilds(f);

      // This proc do not holds childs, they are in remote. Simply pass to another face. Remote will treat this case
      if (face_child.size() < 1) continue;

      //PRINT("face_child", face_child);

      // all vertices related to 'faces'
      nodes.clear();
      for(auto fc : face_child)
      {
         auto v=adjf2v->links(fc);
         nodes.insert(v.begin(),v.end());
      }

      //PRINT("nodes", nodes);

      // remove masters
      slaves_nodes.resize(nodes.size());
      master_nodes.resize(num_dofs_g);
      std::int32_t k = -1;
      std::int32_t l = -1;
      for (auto v : nodes)
         if (sj.isFineMasterNode(v))
            master_nodes[++l] = v;
         else
            slaves_nodes[++k] = v;
      slaves_nodes.resize(k + 1);

      auto slaves_dofs =
          dolfinx::fem::locate_dofs_topological(*topof, *dofmap, 0, std::span<const std::int32_t>(slaves_nodes), false);

      std::ranges::sort(slaves_dofs);

      //PRINT("slaves_dofs", slaves_dofs);

      // look for untreated
      std::vector<std::int32_t> to_treat;
      std::ranges::set_intersection(slaves_dofs, child_dofs, std::back_inserter(to_treat));
      std::int32_t to_treat_size=to_treat.size();

      //PRINT("to_treat", to_treat);

      // after removing slave nodes already treated it is possible that no more slaves exist and thus no more works has to be done
      if (to_treat_size<1) continue;

      // master
      master_nodes.resize(l + 1);
      auto master_dofs=dolfinx::fem::locate_dofs_topological(*topof,*dofmap,0,std::span<const std::int32_t>(master_nodes),false);
      //PRINT("master_dofs", master_dofs);

      // remove already treated
      for (auto const &i : to_treat) child_dofs.erase(i);
      // remove master to decrease complexity in set_intersection treatment
      // and find request
      if (dim > 2)
         for (auto const &i : master_dofs)
         {
            child_dofs.erase(i);
            slaves_child_dofs.erase(i);
         }
      else
         for (auto const &i : master_dofs) child_dofs.erase(i);

      // get the first cell related to the treated face
      assert(adjf2c->links(f).size() > 0);
      const std::int32_t c = adjf2c->links(f)[0];
      //PRINT("master cell",c);

      // coordinates of the cell
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(xc_dofmap, c, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (int i = 0; i < num_dofs_g; ++i)
      {
         const int pos = 3 * x_dofs[i];
         for (int j = 0; j < gdim; ++j) coord_dofs(i, j) = xc_g[pos + j];
      }
      //PRINT("coord_dofs", coord_dofs);

      // set x0 origine of the cell
      for (std::size_t i = 0; i < gdim; ++i) x0[i] = coord_dofs(0, i);

      // Jacobian of the transformation for this cell
      std::fill(J_b.begin(), J_b.end(), zero);
      dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi_master, coord_dofs, J);
      //PRINT("J",J_b);

      // inverse of the Jacobian for this cell
      dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);
      //PRINT("K",K_b);

      // view to FF coefficient
      //mdspan_t<const U, 2> coeff(phi_master_b.data(), 1, phi_master_shape[0]);
      //PRINT("coeff", coeff);

      // loop on master child nodes to set fine scale dof 
      std::fill(elem_loc_idx.begin(), elem_loc_idx.end(), -1);
#ifndef NDEBUG
      std::int8_t nb_master = 0;
#endif
      for (auto v : master_dofs)
      {
         //PRINT("v master",v);
         // grab physical node coordinate
         mdspan_t<const U, 2> x_child(&xf_d[3*v], 1, gdim);
         //PRINT("x_child", x_child);

         // compute node in reference element corresponding to master face
         dolfinx::fem::CoordinateElement<U>::pull_back_affine(X_child, K, x0, x_child);
         //PRINT("X_child", X_child);

         // get interpolation function values at reference location: coefficients of the mpc for this master nodes 
         cellc_cmap.tabulate(0, X_child_b, point_shape, phi_master_b);
         //PRINT("phi_master_b",phi_master_b);

         k = -1;
         for (auto &val : phi_master_b | std::views::take(phi_master_shape[0]))
         {
            ++k;
            if (val > onemeps)
            {
#ifndef NDEBUG
               ++nb_master;
#endif
               elem_loc_idx[k] = v;
               break; // nothing else to collect
            }
         }
      }
      assert(nb_master == num_dofs_dimm1 || nb_master == num_dofs_dimm1 - 1);
      //PRINT("elem_loc_idx",elem_loc_idx);
      k = -1;
      for (auto eli : elem_loc_idx)
      {
         ++k;
         if (eli < 0)
            elem_glob_idx[k] = -1;
         else if (eli < nbngl)
            idxmapd->local_to_global(std::span<const std::int32_t>(&eli, 1),
                                     std::span<std::int64_t>(elem_glob_idx.data() + k, 1));
         else
            elem_glob_idx[k] = global_dof_ghost[eli - nbngl];
      }
      //PRINT("elem_glob_idx",elem_glob_idx);
      for (auto &m : elem_glob_idx) m *= bs_d;
      //PRINT("elem_glob_idx",elem_glob_idx);

      // reserve containers
      // in 3D over reservation: if slave nodes of a face are on its edges they have only 2 masters
      // nodes, not 3 as num_dofs_dimm1 is in this case. If nodes of a face are inside the face there is no over
      // reservation. During face iteration over reservation may occurs but they may be corrected by next face reservation
      std::int32_t current_masters_size = masters.size();
      const std::int32_t delta = bs_d * to_treat_size ;
      masters.reserve(current_masters_size + delta * num_dofs_dimm1);
      assert(coeffs.size() == current_masters_size);
      coeffs.reserve(current_masters_size + delta * num_dofs_dimm1);
      // re-size containers
      // may over re-size
      std::int32_t nb_slaves = slaves.size();
      slaves.resize(slaves.size() + delta);
      offsets.resize(offsets.size() + delta);
      // view on them
      std::span<std::int32_t> slaves_(slaves.data()+nb_slaves,delta); 
      std::span<std::int32_t> offsets_(offsets.data()+nb_slaves+1,delta);
      //PRINT("delta", delta);

      // loop on child nodes (slaves nodes to link with master nodes)
      l=0;
      for (auto v : to_treat)
      {
         //PRINT("v",v);
         slaves_[l] = v * bs_d;

         // grab pysical node coordinate
         mdspan_t<const U, 2> x_child(&xf_d[3*v], 1, gdim);
         //PRINT("x_child", x_child);

         // compute node in reference element corresponding to master face
         dolfinx::fem::CoordinateElement<U>::pull_back_affine(X_child, K, x0, x_child);
         //PRINT("X_child", X_child);

         // get interpolation function values at reference location: coefficients of the mpc for this slave nodes 
         cellc_cmap.tabulate(0, X_child_b, point_shape, phi_master_b);
         //PRINT("phi_master_b",phi_master_b);

         // search mpc coef from FF values:
         //     * remove almost numerical 0. Here coefficient are in between 0 and 1 thus an absolute eps is ok
         //     * almost 1  not supposed to exist: error
         std::int8_t k = -1;
         for (auto &val : phi_master_b | std::views::take(phi_master_shape[0]))
         {
            ++k;
            if (val > eps)
            {
               assert(val < onemeps);
               coeffs.push_back(val);
               masters.push_back(k);
            }
            //PRINT("val", val);
         }
         offsets_[l] = masters.size();
         ++l;
      }

      // reset sizes and set views
      std::int32_t dm = masters.size() - current_masters_size;  // true augmentation
      std::int32_t dmb = dm * bs_d;
      std::int32_t ns = current_masters_size + dmb;
      owners.resize(ns);
      masters.resize(ns);
      coeffs.resize(ns);
      std::span<std::int32_t> owners_(owners.data() + current_masters_size, dmb);
      std::span<std::int64_t> masters_(masters.data() + current_masters_size, dmb);
      std::span<T> coeffs_(coeffs.data() + current_masters_size, dmb);

      for (std::int32_t k = 0; k < dm; ++k)
      {
         // master cell local id
         auto &m = masters_[k];

         // set owner
         if (elem_loc_idx[m] < nbngl)
            owners_[k] = pid;
         else
            owners_[k] = dof_ghost_owners[elem_loc_idx[m] - nbngl];

         // master global id
         m = elem_glob_idx[m];

         // duplicate other dimension
         for (std::int32_t b = 1; b < bs_d; ++b)
         {
            std::int32_t o = k + b * dm;
            coeffs_[o] = coeffs_[k];
            owners_[o] = owners_[k];
            masters_[o] = m + b;
         }
      }
      // duplicate slave/offset for other dimension
      for (std::int32_t k = 0; k < l; ++k)
      {
         for (std::int32_t b = 1; b < bs_d; ++b)
         {
            std::int32_t o = k + b * l;
            slaves_[o] = slaves_[k] + b;
            offsets_[o] = offsets_[k] + b * dm;
         }
      }
   }

   // In 2D it's finish. A slave dof is forcefully local to a process without any remote by construction: Fine scale mesh is
   // encapsulated in coarse scale mesh thus only master can be on process boundary. And slave on boundary hanging edge as no
   // counterpart in remote edges.
   //
   // In 3D slave dof can have remote on over process. And in this case same mpc must be defined so that on all processes, slave
   // get eliminated and contribution of elements of all processes are computed.
   //
   // =====================================
   // Exchange mpc in between process in 3D
   // =====================================
   if (dim>2)
   {
      auto ghost_dofd = idxmapd->ghosts();
      auto dest_rank=idxmapd->index_to_dest_ranks();
      std::unordered_map<int, std::vector<exchangedMPCData>> send_buff;
      std::int32_t k = 0;
      // loop on local slaves
      for (auto s : slaves)
      {
         // only exchange first component of blocks
         std::div_t d = std::div(s, bs_d);
         if (!d.rem)
         {
            // only slave having remote counterpart needs to be exchanged
            std::int32_t pos = d.quot;
            auto dests = dest_rank.links(pos);
            if(dests.size())
            {
               auto o1 = offsets[k];
               auto od = offsets[k + 1] - o1;
               // Normally only slave on edges are concerned by exchanging mechanism. Slave dof on face can't be in remote face as
               // it is a coarse size remote face. Regarding fine cell if they are connected to only one slave dof on face they
               // cannot be remote because they are by construction embedded in the macro element with slave face so in the same
               // process. Thus as slave on edges depends only on 2 master nodes (i.e. those on vertices of the edge) od must be
               // equal to 2
               assert(od==2);
               auto o2 = o1 + 1;
               // set data to exchange for this slave
               exchangedMPCData data_to_send(0,{coeffs[o1],coeffs[o2]},{masters[o1],masters[o2]},{owners[o1],owners[o2]}); 
               // switch to global id for slave
               if (pos<nbngl)
                  idxmapd->local_to_global(std::span<const std::int32_t>(&pos, 1),
                                           std::span<std::int64_t>(&(data_to_send.slave), 1));
               else
                  data_to_send.slave = ghost_dofd[pos-nbngl];

               // store in send_buff
               for (auto dest : dests) send_buff[dest].push_back(data_to_send);
            }
         }
         ++k;
      }

      // Data type to exchange mpc data
      MPI_Datatype exchangedMPCData_t;
      // and appropriate type
      MPI_Aint disp[4];
      exchangedMPCData trash;
      MPI_Aint base;
      MPI_Get_address(&trash,&base);
      MPI_Get_address(&trash.slave, &disp[0]);
      MPI_Get_address(&trash.coef, &disp[1]);
      MPI_Get_address(&trash.master, &disp[2]);
      MPI_Get_address(&trash.owner, &disp[3]);
      for (auto &d : std::span<MPI_Aint>(disp)) d -= base;
      int blocklen[4]={1,2,2,2};
      MPI_Datatype type[4]={MPI_INT64_T,dolfinx::MPI::mpi_t<T>,MPI_INT64_T,MPI_INT32_T};
      // verification to check assertion on padding is not wrong
      assert(sizeof(trash)>=2*sizeof(T)+3*sizeof(std::int64_t)+2*sizeof(std::int32_t));
      assert(sizeof(std::array<T,2>)>=2*sizeof(T));
      assert(sizeof(std::array<std::int64_t,2>)>=2*sizeof(std::int64_t));
      assert(sizeof(std::array<std::int32_t,2>)>=2*sizeof(std::int32_t));
      MPI_Type_create_struct(4, blocklen, disp, type, &exchangedMPCData_t);
      MPI_Type_commit(&exchangedMPCData_t);

      // exchange information
      twoscale::sendVectMPI3(
          send_buff, exchangedMPCData_t, 745, comm,
          // Functor to treat received mpc information
          [&idxmapd, &slaves_child_dofs, &bs_d, &slaves, &masters, &owners, &coeffs, &offsets](
              const std::vector<exchangedMPCData> &infos, int from) {
             // loop on received information from 'from'
             for (auto &mpc : infos)
             {
                // switch to local id for slave
                std::int32_t slave_loc;
                idxmapd->global_to_local(std::span<const std::int64_t>(&(mpc.slave), 1), std::span<std::int32_t>(&slave_loc, 1));

                // if slave not already treated locally
                if (slaves_child_dofs.find(slave_loc) == slaves_child_dofs.end())
                {
                   // add slave to treated to avoid that if another proc send this slave it is considered again 
                   slaves_child_dofs.insert(slave_loc);

                   // offset to use has starting value in loop
                   std::int32_t loffset = offsets.back();

                   // pass local slave block doff in component id
                   slave_loc*=bs_d;

                   // reservation
                   std::int32_t nss = slaves.size() + bs_d;       // nb components
                   std::int32_t nsm = masters.size() + bs_d * 2;  // nb components x 2 master per slave in exchangedMPCData
                   owners.reserve(nsm);
                   masters.reserve(nsm);
                   coeffs.reserve(nsm);
                   slaves.reserve(nss);
                   offsets.reserve(nss + 1);

                   // add all component
                   for (std::int32_t b = 0; b < bs_d; ++b)
                   {
                      loffset += 2;
                      slaves.push_back(slave_loc + b);
                      offsets.push_back(loffset);
                      masters.push_back(mpc.master[0] + b);
                      masters.push_back(mpc.master[1] + b);
                      coeffs.push_back(mpc.coef[0]);
                      coeffs.push_back(mpc.coef[1]);
                      owners.push_back(mpc.owner[0]);
                      owners.push_back(mpc.owner[1]);
                   }
                }
             }
          },
          []() {});

      MPI_Type_free(&exchangedMPCData_t);
   }

   if (0)
   {
      std::int32_t k = 0;
      for (auto s : slaves)
      {
         std::div_t d = std::div(s, bs_d);

         std::println("slave id in data_mpc : {} ({}th) ", s, k);

         if (!d.rem)
         {
            std::int32_t pos = d.quot;
            std::println("slave: {} ", pos);

            std::span<const U> x(&xf_d[3 * pos], &xf_d[3 * pos + gdim]);
            PRINT("x", x);
            auto o = offsets[k];
            auto od = offsets[k + 1] - o;
            std::span<const T> coeff(coeffs.begin() + o, od);
            std::span<const std::int64_t> mast(masters.begin() + o, od);
            std::span<const std::int32_t> ownersl(owners.begin() + o, od);
            std::int32_t l = 0;
            for (auto m : mast)
            {
               std::println("master {} owner {}", m, ownersl[l]);
               PRINT("coef", coeff[l]);
               std::ldiv_t dm = std::div(m, static_cast<std::int64_t>(bs_d));
               assert(!dm.rem);
               std::int64_t posm = dm.quot;
               PRINT("master glob block", posm);
               std::int32_t posml;
               idxmapd->global_to_local(std::span<const std::int64_t>(&posm, 1), std::span<std::int32_t>(&posml, 1));
               PRINT("master loc block", posml);

               std::span<const U> xm(&xf_d[3 * posml], &xf_d[3 * posml + gdim]);
               PRINT("xm", xm);
               ++l;
            }
         }
         else
         {
            std::int32_t pos = d.quot;
            std::println("slave: {} ", pos);
            std::span<const U> x(&xf_d[3 * pos], &xf_d[3 * pos + gdim]);
            PRINT("x", x);
            auto o = offsets[k];
            auto od = offsets[k + 1] - o;
            std::span<const T> coeff(coeffs.begin() + o, od);
            std::span<const std::int64_t> mast(masters.begin() + o, od);
            std::span<const std::int32_t> ownersl(owners.begin() + o, od);
            std::int32_t l = 0;
            for (auto m : mast)
            {
               std::println("master {} owner {}", m, ownersl[l]);
               PRINT("coef", coeff[l]);
               ++l;
            }
         }
         ++k;
      }
   }


   return std::move(data_mpc);
}


}  // namespace twoscale_dolfinx
#endif

