/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_COARSEMANAGER_IMP
#define TS_DOLFINX_COARSEMANAGER_IMP
#ifndef TS_DOLFINX_COARSEMANAGER
#error "Should not be included by hand"
#endif
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/petsc.h>
#include "dolfinx/mesh/utils.h"

#include <unordered_set>
#include <unistd.h>

#include "util.h"

namespace twoscale
{
template <typename M, typename V>
coarseManager<M, V>::coarseManager(M &&PS_, M &&PE_, std::vector<std::int32_t> enriched_dof_eliminated)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
coarseManager<M, V>::~coarseManager()
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
void coarseManager<M, V>::setStdCoarse(const M &Aff, const V &bf, bool use_imp_enriched)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
void coarseManager<M, V>::resetCoarseToStd()
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
void coarseManager<M, V>::updateEnrichCoarse(const M &Aff, const V &bf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
void coarseManager<M, V>::updateEImp()
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
template <typename T>
void coarseManager<M, V>::solve(T &xf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
template <typename T>
void coarseManager<M, V>::solveEImp(T &xf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename M, typename V>
template <typename T>
void coarseManager<M, V>::projectStdCoarse(T xcs, T xf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
#ifdef HAS_PETSC
template <typename P, typename F>
void coarseManager<Mat, Vec>::updateEnrichedOperator(twoscale_dolfinx::patchManager<P> &pm, std::shared_ptr<F> func)
{
   spdlog::info("update enriched operator");

   // prerequisites
   /* TODO: to clean
   // You can not call updateEnrichedOperator if  PEt was not stored 
   assert(state & PEt_STORED);
   */

   // =============================
   // generate enrichment operator
   // =============================
   //DO(MatRetrieveValues(PEt), "Problem while retrieving PEt", comm)
   state ^= PEt_ASS;
   pm.PEtUpdate(PEt, enriched_dof_id, func);
   state |= PEt_ASS;

   return;
}
#endif
} // namespace twoscale

namespace twoscale_dolfinx
{
#ifdef HAS_PETSC
namespace impl
{
template <typename F, typename U>
inline void setBlock(F &setter, std::int8_t br, int vb, const U &val, PetscScalar &zero_petsc, int &bs_f, std::vector<int> &c_,
                     std::vector<PetscScalar> &va_, std::vector<int>& ra_)

{
   // set matrix terms to add
   for (std::int32_t b = 0; b < br; ++b) va_[ra_[b] * br + b] = val;
   //std::println("br {} c_ {} ra_ {} va_ {}", br, std::span<const int>(c_.data(), br), std::span<const int>(ra_.data(), br), std::span<const PetscScalar>(va_.data(), br * bs_f));
   // add terms
   setter(std::span<const int>(&vb, 1), std::span<const int>(c_.data(), br), std::span<const PetscScalar>(va_.data(), br * bs_f));
   // reset va_
   for (std::int32_t b = 0; b < br; ++b) va_[ra_[b] * br + b] = zero_petsc;
   return;
};
template <typename F, typename U>
inline void setNonNullWNull(F &setter, const int &std_enr, std::int8_t k, int vb, const U &val, mdspan_t<int32_t, 3> &mix_coarse_dof,
                     PetscScalar &zero_petsc, int &bs_cs, int &bs_f, std::vector<int> &c_, std::vector<PetscScalar> &va_,
                     std::vector<int>& ra_)

{
   // if eliminated count future block size
   std::int32_t br = bs_cs;
   std::int32_t l = 0;
   for (std::int32_t b = 0; b < bs_cs; ++b)
   {
      auto idx = mix_coarse_dof(k, std_enr, b);
      if (idx < -1)
         --br;
      else
      {
         ra_[l]=b;
         c_[l++] = idx;
      }
   }
   // if block fully eliminated (br==0) skip
   // otherwise set with potential eliminated column
   if (br) setBlock(setter, br, vb, val, zero_petsc, bs_f, c_, va_, ra_);
};
}  // namespace impl
#ifdef TWOSCALE_PETSC_MATNEST
template <dolfinx::scalar T, std::floating_point U>
twoscale::coarseManager<Mat, Vec> generateCoarseManager(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &sj,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> coarse_space,
                                                        std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> &mpc,
                                                        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>> &bc)
{
   spdlog::info("Compute coarsening operator structure and initial values");

   // ===================
   // check coherance
   // ===================
   assert((sj.needs_mpc) ? (mpc != nullptr) : (mpc == nullptr));
   // TODO: mesh dim etc
   assert(&const_cast<const dolfinx::mesh::Mesh<U> &>(*sj.getCoarseMesh()) == &(*(coarse_space->mesh())));
   assert(&const_cast<const dolfinx::mesh::Mesh<U> &>(*sj.getFineMesh()) == &(*(fine_space->mesh())));

   // ================
   // collect mpi info
   // ================
   MPI_Comm comm = sj.getCoarseMesh()->comm();
   int pid = dolfinx::MPI::rank(comm);
   int nbproc = dolfinx::MPI::size(comm);

   // ===================
   // arithmetic constant
   // ===================
   // zero in U arithmetic
   const U zero = static_cast<U>(0.);
   // zero in PetscScalar arithmetic
   PetscScalar zero_petsc = static_cast<PetscScalar>(zero);
   // one in U arithmetic
   const U one = static_cast<U>(1.);
   // epsilon in U arithmetic : use this limit is ok as normally
   // child nodes are note supposed to be that close to coarse nodes
   // 1000 time numerical limit is expected to be well suited to identify properly
   const U eps = static_cast<U>(1000.) * std::numeric_limits<U>::epsilon();
   const U onemeps = one - eps;

   // ====================================
   // collect coarse mesh topo information
   // ====================================
   auto cdomain = sj.getCoarseMesh();
   auto topoc = cdomain->topology();
   int dim = topoc->dim();
   int dimm1 = dim - 1;
   std::int32_t nbccl = topoc->index_map(dim)->size_local();
   auto adj = topoc->connectivity(dim, 0);

   // =======================================
   // collect coarse element topo information
   // =======================================
   auto &geomc = cdomain->geometry();
   const auto &cellc_cmap = geomc.cmap();
   assert(cellc_cmap.is_affine());
   const std::size_t nb_nodes_elem = cellc_cmap.dim();
   const std::size_t gdim = geomc.dim();

   // ================================
   // collect coarse space information
   // ================================
   auto dofmapc = coarse_space->dofmap();
   auto bs_c = dofmapc->index_map_bs();
   auto bs_c2 = 2 * bs_c;
   auto idxmapc = dofmapc->index_map;
   std::int32_t nbdc = idxmapc->size_local();

   // ==============================
   // collect fine space information
   // ==============================
   auto dofmapf = fine_space->dofmap();
   auto idxmapf = dofmapf->index_map;
   auto bs_f = dofmapf->index_map_bs();
   std::int32_t nbdf = idxmapf->size_local();

   // ==============================
   // collect enrichment information
   // ==============================
   std::unordered_set<std::int32_t> to_enrich;
   {
      auto v = sj.getEnriched();
      to_enrich.insert(v.begin(), v.end());
   }
   {
      auto v = sj.getExtraEnriched();
      to_enrich.insert(v.begin(), v.end());
   }

   // std::println("to_enrich {} {}",to_enrich.size(),to_enrich);

   // =======================
   // set mpc filter function
   // =======================
   std::span<const std::int8_t> slave_flag;
   std::function<bool(const std::int32_t &)> mpc_filter = [](const std::int32_t &i) -> bool { return true; };
   if (mpc != nullptr)
   {
      slave_flag = mpc->is_slave();
      mpc_filter = [&slave_flag, &bs_f](const std::int32_t &i) -> bool { return (slave_flag[i * bs_f] < 1); };
   }

   // =======================
   // set BC filter function
   // and W vector
   // =======================
   std::unordered_set<std::int32_t> BC_idx_activated;
   Vec W = nullptr;
   Vec XDc = nullptr;
   if (bc.size() > 0)
   {
      // generate adhoc vector
      XDc = dolfinx::la::petsc::create_vector(*idxmapc, bs_c);
      // get view to it
      PetscScalar *XDc_;
      DO(VecGetArray(XDc, &XDc_), "problem while retriving  XDc_ array", comm)
      // temporary buffer 
      std::vector<T> buff((idxmapc->size_local() + idxmapc->num_ghosts())*bs_c);
      auto buff_ = std::span<T>(buff);

      // loop on bc
      for (auto &BC : bc)
      {
         assert(coarse_space->contains(*(BC->function_space())));
         // reset to zero buff
         std::ranges::fill(buff,zero_petsc);
         // set imposed value for this BC in buff
         BC->set(buff_,std::nullopt);
         // acumulate in XDc
         std::transform(buff.begin(),buff.end(),XDc_,XDc_,std::plus<T>());

         // collect indexes
         auto [BC_idx, ghost_idx] = BC->dof_indices();

         // switch to unordered_set
         BC_idx_activated.insert(BC_idx.begin(), BC_idx.end());
      }
      DO(VecRestoreArray(XDc, &XDc_), "problem while releasing  XDc_ array", comm)

      //VecView(XDc, PETSC_VIEWER_STDOUT_(comm));
      //PRINT("BC_idx_activated",BC_idx_activated);

      // check that non null Dirichlet exist (XDc_!=0). If no W=PSD.XDc=0 doesn't need to be created as it lead to zero results
      // A.W=0 or do not change computation W+Qt.S=Qt.S. In this case PSD is not assembled too.
      PetscReal norm_inf;
      DO(VecNorm(XDc, NORM_INFINITY, &norm_inf), "problem while computing XDc_ norm", comm)
      // generate W if norm not null
      if (norm_inf > std::numeric_limits<PetscReal>::epsilon()) W = dolfinx::la::petsc::create_vector(*idxmapf, bs_f);
      // otherwise remove XDc which is not needed anymore
      else
      {
         DO(VecDestroy(&XDc), "Problem to destroy XDc", comm)
      }
   }

   // ============================
   // collect fine dof coordinates
   // ============================
   auto xf_d = fine_space->tabulate_dof_coordinates(false);

   // ================================
   // collect coarse dof coordinates
   // ================================
   auto xc_d = coarse_space->tabulate_dof_coordinates(false);
   // PRINT("xc_d", xc_d);

   // ============================================
   // preparing container for per cell computation
   // ============================================
   // container to store element dofs in both space: standard and enriched
   // Somme element will not have any enriched dof or only some dof with null enrichment function
   std::vector<std::int32_t> mix_coarse(nb_nodes_elem * bs_c2);
   mdspan_t<int32_t, 3> mix_coarse_dof(mix_coarse.data(), nb_nodes_elem, 2, bs_c);

   // container to store enriched dof id
   assert(bs_f << sizeof(twoscale_dolfinx::enrichedDofIDs::eliminated_coarse_encoding));
   std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> enriched_dof_id;

   // container to store all child nodes
   std::unordered_set<std::int32_t> nodes;

   // container to store eliminated standard dofs blocked by Dirichlet BC 
   std::unordered_set<std::int32_t> Dirichlet_dofs;

   // container to store vertex coordinate of a coarse cell
   std::vector<U> coord_dofs_b(nb_nodes_elem * gdim);
   mdspan_t<U, 2> coord_dofs(coord_dofs_b.data(), nb_nodes_elem, gdim);

   // origin of the cell
   std::array<U, 3> x0;

   // reference coordinates
   std::array<U, 3> X_child_b;
   mdspan_t<U, 2> X_child(X_child_b.data(), 1, dim);

   // Geometry data at each point
   std::vector<U> J_b(gdim * dim);
   mdspan_t<U, 2> J(J_b.data(), gdim, dim);
   std::vector<U> K_b(dim * gdim);
   mdspan_t<U, 2> K(K_b.data(), dim, gdim);

   // point shape
   std::array<std::size_t, 2> point_shape = {1, dim};

   // cell interpolation function and its derivative for one point
   const std::array<std::size_t, 4> phi_coarse_shape = cellc_cmap.tabulate_shape(1, 1);
   std::vector<U> phi_coarse_b(std::reduce(phi_coarse_shape.begin(), phi_coarse_shape.end(), 1, std::multiplies{}));
   mdspan_t<const U, 4> phi_coarse(phi_coarse_b.data(), phi_coarse_shape);
   cellc_cmap.tabulate(1, std::vector<U>(dim), point_shape, phi_coarse_b);  // get for first node FF and derivative
   auto dphi_coarse = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
       phi_coarse, std::pair(1, dim + 1), 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);  // only derivative

   // ========================================
   // Full or partial enrichment specific data
   // ========================================
   std::shared_ptr<dolfinx::mesh::Mesh<U>> mesh_support;
   std::shared_ptr<dolfinx::fem::FunctionSpace<U>> enriched_space;
   std::shared_ptr<const dolfinx::fem::DofMap> dofmape;
   std::shared_ptr<const dolfinx::common::IndexMap> idxmape;
   std::unordered_map<std::int32_t, std::int32_t> enriched_cells;

   // If mpc not all coarse mesh are enriched. Thus we need to construct a adapted space based on a sub mesh (the support of
   // enriched nodes) of the coarse mesh. This sub space describe properly all enriched dof and also all extra enriched dof (on
   // boundary of the support).
   // PS and PE construction are based on 2 different spaces
   if (mpc != nullptr)
   {
      // ================================
      // Generate specific enriched space
      // ================================
      auto support = sj.getSupport();
      auto [mesh_support_, cell_map, vertice_map, coord_map] = dolfinx::mesh::create_submesh(*cdomain, dim, support);
      mesh_support = std::make_shared<dolfinx::mesh::Mesh<U>>(mesh_support_);
      enriched_space = std::make_shared<dolfinx::fem::FunctionSpace<U>>(
          dolfinx::fem::create_functionspace(mesh_support, coarse_space->element()));
      dofmape = enriched_space->dofmap();
      auto bs_e = dofmape->index_map_bs();
      assert(bs_e == bs_c);
      idxmape = dofmape->index_map;
      // =============================
      // Generate cell correspondences
      // =============================
      std::int32_t nbcel = support.size();
      std::vector<std::int32_t>sub_idx(nbcel);
      for (auto i : std::views::iota(0, nbcel)) sub_idx[i] = i;
      auto cell_coresp = cell_map.sub_topology_to_topology(sub_idx, false);
      enriched_cells.reserve(nbcel);
      for (std::int32_t i = 0; i < nbcel; ++i)
      {
         enriched_cells.emplace(cell_coresp[i],i);
      }
      // PRINT("cell_map", cell_map);
      // std::println("enriched_cells {}", enriched_cells);
   }
   // If no mpc all coarse mesh are enriched. Thus coarse standard and enriched space are the same
   // PE, PS are constructed based on this space
   else
   {
      idxmape = dofmapc->index_map;
   }
   // =========================
   // operator sparsity pattern
   // =========================
   dolfinx::la::SparsityPattern sps(comm, {idxmapf, idxmapc}, {bs_f, bs_c});
   dolfinx::la::SparsityPattern spe(comm, {idxmapf, idxmape}, {bs_f, bs_c});
   dolfinx::la::SparsityPattern spd(comm, {idxmapf, idxmapc}, {bs_f, bs_c});

   // PRINT("bs_c", bs_c);
   // PRINT("bs_f", bs_f);
   // associated function to feed sparcity patern
   auto addNonNull = [&W, &mix_coarse_dof, &bs_c, &sps, &spe, &spd](std::int8_t k, std::int32_t vb) {
      // column is a blocked field thus we must count number of eliminated dofs to see if full block is eliminated or not. If not
      // fully eliminated the block has to be keeped.
      // For PSR,PSD count
      std::int32_t nb_elim = 0;
      for (std::int32_t b = 0; b < bs_c; ++b)
         if (mix_coarse_dof(k, 0, b) < -1) ++nb_elim;
      // if not eliminated PSR  have it
      if (nb_elim < 1) sps.insert(vb, mix_coarse_dof(k, 0, 0) / bs_c);
      // if partially eliminated PSR and PSD have it
      else if (nb_elim < bs_c)
      {
         auto idx = mix_coarse_dof(k, 0, 0);
         if (idx > -1)
            idx /= bs_c;
         else
            idx = (-idx - 2) / bs_c;
         sps.insert(vb, idx);
         if (W != nullptr) spd.insert(vb, idx);
      }
      // if fully eliminated only PSD have it if W#0
      else if (W != nullptr)
      {
         auto idx = (-mix_coarse_dof(k, 0, 0) - 2) / bs_c;
         spd.insert(vb, idx);
      }
      // PE is eliminated only if associated node is not enriched and thus all component are eliminated. Testing first component
      // is then enough to decide if full bock need to be eliminated or not.
      auto idx = mix_coarse_dof(k, 1, 0);
      if (idx > -1) spe.insert(vb, idx / bs_c);
   };

   // ===========================================
   // loop on all cell to identify matrices graph
   // ===========================================
   for (std::int32_t cell_id = 0; cell_id < nbccl; ++cell_id)
   {
      nodes.clear();
      std::fill(J_b.begin(), J_b.end(), zero);

      // PRINT("cell", cell_id);

      //  cell children if any
      auto cell_child = sj.getChildren(cell_id);

#ifndef ALL_CHILD
      std::cout << "The macro ALL_CHILD is not set !!! MatNest only work with it !!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
#endif
      // std dof/topo node of the element
      auto std_dof = dofmapc->cell_dofs(cell_id);
      // PRINT("std_dof", std_dof);
      const auto std_topo = adj->links(cell_id);
      // PRINT("std_topo", std_topo);

      std::span<const std::int32_t> enr_dof;
      if (mpc != nullptr)
      {
         if (cell_child.size() > 1)
         {
            auto itf = enriched_cells.find(cell_id);
            assert(itf != enriched_cells.end());
            enr_dof = dofmape->cell_dofs(itf->second);
         }
      }
      else
         enr_dof = std_dof;
      // PRINT("enr_dof", enr_dof);
      bool enriched = (enr_dof.size() > 0);

      // coarse index setting
      // Get idx from both space
      for (std::int32_t k = 0, l = 0; auto x : std_dof)
      {
         std::int32_t idx_ = x * bs_c;
         auto *pmix_coarse = &mix_coarse[k];

         // PS operator have to be filtered by BC
         // loop on block index
         for (std::int32_t b = 0; b < bs_c; ++b)
         {
            auto idx = idx_ + b;
            // if BC eliminated
            if (BC_idx_activated.find(idx) != BC_idx_activated.end())
            {
               pmix_coarse[b] = -2 - idx;
               // only save local: remote do not need to be add ass diag terms as local do the job
               if (x<nbdc)
                  Dirichlet_dofs.insert(idx);
            }
            // if dof
            else
               pmix_coarse[b] = idx;
         }

         // if coarse node is enriched, PE operator must be set
         // otherwise set negative to ignore in PE
         // switch to enriched whatever
         k += bs_c;
         pmix_coarse = &mix_coarse[k];
         if (enriched)
         {
            // NOTE: when mpc enr_dof#std_dof then enr_dof definition is expected to follows std_dof definition. A priori
            // enr_dof correspond to the same cell as cell_id in the submesh. As there is no reason that the submesh change
            // topological definition we can expect that the topology of both cells are the same. And thus the space created on
            // both cell will create dof in same order (but not with same index).  l is then giving the index of enr_dof
            // coresponding to x.
            //
            idx_ = enr_dof[l] * bs_c;

            // start setting correspondence for PE update
            // NOTE: assertion is done here that std_dof and std_topo are in the same order.
            // i.e. that l correspond to a node on which x dof is leaving
            auto i_topo = std_topo[l];
            // init enriched dof ids if not already done
            if (enriched_dof_id.find(i_topo) == enriched_dof_id.end())
            {
               std::int64_t g;
               idxmape->local_to_global(std::span<const std::int32_t>(&enr_dof[l], 1), std::span<std::int64_t>(&g, 1));
               twoscale_dolfinx::enrichedDofIDs idxs;
               idxs.global_coarse_block_dof_idx = g*bs_c;
               idxs.global_fine_block_dof_idx = 0;
               // encode  eliminated dof in the block
               // In this version (mat_nest) we are not going to remove any enriched dof so we set the encoding to 0 (i.e. no
               // elimination). This encoding is kept to be compatible with the implementation of the original approach with mixed
               // field
               idxs.eliminated_coarse_encoding = 0;
               enriched_dof_id[i_topo] = idxs;
            }
         }
         else
            idx_ = -bs_c - 2;
         for (std::int32_t b = 0; b < bs_c; ++b) pmix_coarse[b] = idx_ + b;

         // increment
         k += bs_c;
         ++l;
      }
      // PRINT("mix_coarse", mix_coarse);
      // PRINT("mix_coarse_dof", mix_coarse_dof);
      //std::println("Dirichlet_dofs eliminated : {}", Dirichlet_dofs);

      // all local dof related to childs and not slave of an mpc relation
      for (auto fc : cell_child)
      {
         auto v = dofmapf->cell_dofs(fc);
         auto vf = v | std::views::filter(mpc_filter);
         // auto vf = v | std::views::filter([&nbdf](const std::int32_t &x) -> bool { return (x < nbdf); }) |
         // std::views::filter(mpc_filter);
         nodes.insert(vf.begin(), vf.end());
      }

      std::int32_t nodes_size = nodes.size();

      // PRINT("nodes_child", nodes);

      // if at least one childs to treat
      if (nodes_size)
      {
         // coordinates of the coarse element
         for (int i = 0; auto x : std_dof)
         {
            for (int j = 0; j < gdim; ++j) coord_dofs(i, j) = xc_d[3 * x + j];
            ++i;
         }
         // PRINT("coord_dofs", coord_dofs);

         // set x0
         for (std::size_t i = 0; i < gdim; ++i) x0[i] = coord_dofs(0, i);

         // Jacobian of the transformation for this element
         dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi_coarse, coord_dofs, J);

         // inverse of the Jacobian for this element
         dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);

         // loop on local child nodes (fine nodes to interpolate into coarse cell)
         for (auto v : nodes)
         {
            // PRINT("v", v);

            // grab physical node coordinate
            mdspan_t<const U, 2> x_child(&xf_d[3 * v], 1, gdim);
            // PRINT("x_child", x_child);

            // compute node in reference element corresponding to master face
            dolfinx::fem::CoordinateElement<U>::pull_back_affine(X_child, K, x0, x_child);
            // PRINT("X_child", X_child);

            // get interpolation function values at reference location: coefficients of the mpc for this slave nodes
            cellc_cmap.tabulate(0, X_child_b, point_shape, phi_coarse_b);
            // PRINT("phi_coarse_b", phi_coarse_b);

            // loop on FF
            std::int8_t k = -1;
            if (enriched)
            {
               for (auto &val : phi_coarse_b | std::views::take(phi_coarse_shape[0]))
               {
                  ++k;

                  // if val non null a interpolation exist in between fine and coarse scale
                  if (val > eps)
                  {
                     // depend on a coarse node k
                     addNonNull(k, v);
                     if (val > onemeps)
                     {
                        // if it correspond to a coarse enriched node grab fine dof global id for
                        // future selection of petsc terms
                        auto itf = enriched_dof_id.find(std_topo[k]);
                        // normaly it exist has we did create it in above loop
                        assert(itf != enriched_dof_id.end());
                        std::int64_t g;
                        idxmapf->local_to_global(std::span<const std::int32_t>(&v, 1), std::span<std::int64_t>(&g, 1));
                        itf->second.global_fine_block_dof_idx = bs_f * g;
                        // val of 1 implies others are null so no need to continue
                        break;
                     }
                  }
               }
            }
            else
            {
               for (auto &val : phi_coarse_b | std::views::take(phi_coarse_shape[0]))
               {
                  ++k;

                  // if val non null a interpolation exist in between fine and coarse scale
                  if (val > eps)
                  {
                     // depend on a coarse node k
                     addNonNull(k, v);
                     if (val > onemeps)
                        // val of 1 implies others are null so no need to continue
                        break;
                  }
               }
            }
         }
      }
   }
   // =======================
   // generate empty matrices
   // based on graph sps,spe
   // and spd
   // =======================
   sps.finalize();
   spd.finalize();
   spe.finalize();
   Mat PSR = dolfinx::la::petsc::create_matrix(comm, sps, MATAIJ);
   Mat PSD = dolfinx::la::petsc::create_matrix(comm, spd, MATAIJ);
   Mat PE = dolfinx::la::petsc::create_matrix(comm, spe, MATAIJ);

   // ===========================
   // matrix term setter function
   // ===========================
   auto setterr = dolfinx::la::petsc::Matrix::set_block_fn(PSR, INSERT_VALUES);
   auto setterd = dolfinx::la::petsc::Matrix::set_block_fn(PSD, INSERT_VALUES);
   auto settere = dolfinx::la::petsc::Matrix::set_block_fn(PE, INSERT_VALUES);

   // functor to set terms in matrices
   std::vector<int> c_(bs_c);
   std::vector<int> ra_(bs_f);
   int block_size = bs_f * bs_c;
   std::vector<PetscScalar> blockR_(block_size, 0.);
   std::vector<PetscScalar> blockD_(block_size, 0.);
   auto setNonNull = [&W, &mix_coarse_dof, &zero_petsc, &bs_c, &block_size, &setterr, &settere, &setterd, &c_, &ra_, &blockR_,
                      &blockD_, &comm](std::int8_t k, int vb, const U &val) {
      // PSR an PSD construction only if  XDc is not null ( in this case W=PSD.XDc=0 so we skip PSD construction)
      // set block matrix terms to add to PSR and/or PSD in any case
      std::int32_t nb_elim = 0;
      if (W != nullptr)
         for (std::int32_t b = 0; b < bs_c; ++b)
         {
            auto idx = mix_coarse_dof(k, 0, b);
            // eliminated => value for PSD and null for PSR
            if (idx < -1)
            {
               blockD_[b * bs_c + b] = val;
               ++nb_elim;
            }
            // kipped => value for PSR and nothing for PSD
            else
               blockR_[b * bs_c + b] = val;
         }
      else
         for (std::int32_t b = 0; b < bs_c; ++b)
         {
            auto idx = mix_coarse_dof(k, 0, b);
            // eliminated => count for PSD and nothing for PSR
            if (idx < -1)
            {
               ++nb_elim;
            }
            // kipped => value for PSR and nothing for PSD
            else
               blockR_[b * bs_c + b] = val;
         }

      // if block not eliminated  skip PSD and add PSR
      if (nb_elim < 1)
      {
         int idx = mix_coarse_dof(k, 0, 0) / bs_c;
         setterr(std::span<const int>(&vb, 1), std::span<const int>(&idx, 1),
                 std::span<const PetscScalar>(blockR_.data(), block_size));
         // reset to null block
         for (std::int32_t b = 0; b < bs_c; ++b) blockR_[b * bs_c + b] = zero_petsc;
      }
      // if block partially eliminated  add PSR and add PSD if W#0
      else if (nb_elim < bs_c)
      {
         int idx = mix_coarse_dof(k, 0, 0);
         if (idx > -1)
            idx /= bs_c;
         else
            idx = (-idx - 2) / bs_c;
         setterr(std::span<const int>(&vb, 1), std::span<const int>(&idx, 1),
                 std::span<const PetscScalar>(blockR_.data(), block_size));
         if (W != nullptr)
         {
            setterd(std::span<const int>(&vb, 1), std::span<const int>(&idx, 1),
                    std::span<const PetscScalar>(blockD_.data(), block_size));
            // reset to null block
            for (std::int32_t b = 0; b < bs_c; ++b) blockR_[b * bs_c + b] = blockD_[b * bs_c + b] = zero_petsc;
         }
         else
            for (std::int32_t b = 0; b < bs_c; ++b) blockR_[b * bs_c + b] = zero_petsc;
      }
      // if block fully eliminated  skip PSR and add PSD if W#0
      else if (W != nullptr)
      {
         int idx = (-mix_coarse_dof(k, 0, 0) - 2) / bs_c;
         setterd(std::span<const int>(&vb, 1), std::span<const int>(&idx, 1),
                 std::span<const PetscScalar>(blockD_.data(), block_size));
         // reset to null block
         for (std::int32_t b = 0; b < bs_c; ++b) blockD_[b * bs_c + b] = zero_petsc;
      }

      // PE construction
      // check first component
      int idx = mix_coarse_dof(k, 1, 0);
      // if not eliminated add block
      if (idx > -1)
      {
         for (std::int32_t b = 0; b < bs_c; ++b) blockR_[b * bs_c + b] = val;
         idx /= bs_c;
         settere(std::span<const int>(&vb, 1), std::span<const int>(&idx, 1),
                 std::span<const PetscScalar>(blockR_.data(), block_size));
         // reset to null block
         for (std::int32_t b = 0; b < bs_c; ++b) blockR_[b * bs_c + b] = zero_petsc;
      }
   };

   // ===================================
   // loop on all cells to set PSR&PSD&PE
   // ===================================
   for (std::int32_t cell_id = 0; cell_id < nbccl; ++cell_id)
   {
      nodes.clear();
      std::fill(J_b.begin(), J_b.end(), zero);

      // PRINT("cell", cell_id);

      //  cell children if any
      auto cell_child = sj.getChildren(cell_id);

#ifndef ALL_CHILD
      std::cout << "The macro ALL_CHILD is not set !!! MatNest only work with it !!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
#endif
      // std dof/topo node of the element
      auto std_dof = dofmapc->cell_dofs(cell_id);
      // PRINT("std_dof", std_dof);
      const auto std_topo = adj->links(cell_id);
      // PRINT("std_topo", std_topo);

      std::span<const std::int32_t> enr_dof;
      if (mpc != nullptr)
      {
         if (cell_child.size() > 1)
         {
            auto itf = enriched_cells.find(cell_id);
            assert(itf != enriched_cells.end());
            enr_dof = dofmape->cell_dofs(itf->second);
         }
      }
      else
         enr_dof = std_dof;
      // PRINT("enr_dof", enr_dof);
      bool enriched = (enr_dof.size() > 0);

      // coarse index setting
      // Get idx from both space
      for (std::int32_t k = 0, l = 0; auto x : std_dof)
      {
         std::int32_t idx_ = x * bs_c;
         auto *pmix_coarse = &mix_coarse[k];

         // PS operator have to be filtered by BC
         // loop on block index
         for (std::int32_t b = 0; b < bs_c; ++b)
         {
            auto idx = idx_ + b;
            // if BC eliminated
            if (BC_idx_activated.find(idx) != BC_idx_activated.end()) pmix_coarse[b] = -2 - idx;
            // if dof
            else
               pmix_coarse[b] = idx;
         }

         // if coarse node is enriched, PE operator must be set
         // otherwise set negative to ignore in PE
         // switch to enriched whatever
         k += bs_c;
         pmix_coarse = &mix_coarse[k];
         if (enriched)
         {
            // NOTE: when mpc enr_dof#std_dof then enr_dof definition is expected to follows std_dof definition. A priori
            // enr_dof correspond to the same cell as cell_id in the submesh. As there is no reason that the submesh change
            // topological definition we can expect that the topology of both cells are the same. And thus the space created on
            // both cell will create dof in same order (but not with same index).  l is then giving the index of enr_dof
            // coresponding to x.
            //
            idx_ = enr_dof[l] * bs_c;
         }
         else
            idx_ = -bs_c - 2;
         for (std::int32_t b = 0; b < bs_c; ++b) pmix_coarse[b] = idx_ + b;

         // increment
         k += bs_c;
         ++l;
      }
      // PRINT("mix_coarse", mix_coarse);
      // PRINT("mix_coarse_dof", mix_coarse_dof);

      // all local dof related to childs and not slave of an mpc relation
      for (auto fc : cell_child)
      {
         auto v = dofmapf->cell_dofs(fc);
         auto vf = v | std::views::filter(mpc_filter);
         // auto vf = v | std::views::filter([&nbdf](const std::int32_t &x) -> bool { return (x < nbdf); }) |
         // std::views::filter(mpc_filter);
         nodes.insert(vf.begin(), vf.end());
      }

      std::int32_t nodes_size = nodes.size();

      // PRINT("nodes_child", nodes);

      // if at least one childs to treat
      if (nodes_size)
      {
         // coordinates of the coarse element
         for (int i = 0; auto x : std_dof)
         {
            for (int j = 0; j < gdim; ++j) coord_dofs(i, j) = xc_d[3 * x + j];
            ++i;
         }
         // PRINT("coord_dofs", coord_dofs);

         // set x0
         for (std::size_t i = 0; i < gdim; ++i) x0[i] = coord_dofs(0, i);

         // Jacobian of the transformation for this element
         dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi_coarse, coord_dofs, J);

         // inverse of the Jacobian for this element
         dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);

         // loop on local child nodes (fine nodes to interpolate into coarse cell)
         for (auto v : nodes)
         {
            // PRINT("v", v);

            // grab physical node coordinate
            mdspan_t<const U, 2> x_child(&xf_d[3 * v], 1, gdim);
            // PRINT("x_child", x_child);

            // compute node in reference element corresponding to master face
            dolfinx::fem::CoordinateElement<U>::pull_back_affine(X_child, K, x0, x_child);
            // PRINT("X_child", X_child);

            // get interpolation function values at reference location: coefficients of the mpc for this slave nodes
            cellc_cmap.tabulate(0, X_child_b, point_shape, phi_coarse_b);
            // PRINT("phi_coarse_b", phi_coarse_b);

            // loop on FF
            std::int8_t k = -1;
            for (auto &val : phi_coarse_b | std::views::take(phi_coarse_shape[0]))
            {
               ++k;

               // if val non null simply set coefficient
               if (val > eps)
               {
                  // depend on a coarse node k
                  if (val > onemeps)
                  {
                     setNonNull(k, v, one);
                     // val of 1 implies others are null so no need to continue
                     break;
                  }
                  setNonNull(k, v, val);
               }
            }
         }
      }
   }

   // ===================
   // assemble PSR and PE
   // ===================
   DO(MatAssemblyBegin(PE, MAT_FINAL_ASSEMBLY), "PE begin assembly operator", comm)
   DO(MatAssemblyEnd(PE, MAT_FINAL_ASSEMBLY), "PE end assembly operator", comm)
   DO(MatAssemblyBegin(PSR, MAT_FINAL_ASSEMBLY), "PSR begin assembly operator", comm)
   DO(MatAssemblyEnd(PSR, MAT_FINAL_ASSEMBLY), "PSR end assembly operator", comm)

   // std::println("PSR");
   // MatView(PSR, PETSC_VIEWER_STDOUT_(comm));
   // std::println("PE");
   // MatView(PE, PETSC_VIEWER_STDOUT_(comm));

   // =============================================
   // only if W not null assemble PSD and compute W
   // =============================================
   if (W != nullptr)
   {
      // std::println("XDc");
      // VecView(XDc, PETSC_VIEWER_STDOUT_(comm));
      DO(MatAssemblyBegin(PSD, MAT_FINAL_ASSEMBLY), "PSD begin assembly operator", comm)
      DO(MatAssemblyEnd(PSD, MAT_FINAL_ASSEMBLY), "PSD end assembly operator", comm)
      // std::println("PSD");
      // MatView(PSD, PETSC_VIEWER_STDOUT_(comm));
      DO(MatMult(PSD, XDc, W), "Problem while computing W", comm)
      // std::println("W");
      // VecView(W, PETSC_VIEWER_STDOUT_(comm));
      // PSD,XDc not needed any more: clean memory
      DO(MatDestroy(&PSD), "Problem to destroy LS", comm)
      DO(VecDestroy(&XDc), "Problem to destroy XDc", comm)
   }

   // ============
   // transpose PE
   // ============
   // to be able to change PE easily in a distributed manner by row, transpose PE
   DO(MatTranspose(PE, MAT_INPLACE_MATRIX, &PE), "Problem while transposing  PE", comm)
   // std::println("PEt");
   // MatView(PE, PETSC_VIEWER_STDOUT_(comm));

   /*
   for (auto p : enriched_dof_id)
      std::cout << "topo node " << p.first << " gcbdi " << p.second.global_coarse_block_dof_idx << " gfbdi "
                << p.second.global_fine_block_dof_idx << std::endl;
                */

   // ======================
   // Prepare Dirichlet dofs
   // ======================
   // pack enrich_Dirichlet_dofs
   std::vector<std::int32_t> diag_Dirichlet_dofs(Dirichlet_dofs.begin(), Dirichlet_dofs.end());
   std::ranges::sort(diag_Dirichlet_dofs);
   auto nb_diag_Dirichlet=diag_Dirichlet_dofs.size();
   std::vector<std::int64_t> glob_diag_Dirichlet_dofs(nb_diag_Dirichlet);
   std::vector<std::int32_t> diag_Dirichlet_delta(nb_diag_Dirichlet);
   for (size_t i= 0;i<nb_diag_Dirichlet;++i)
   {
      diag_Dirichlet_delta[i]=diag_Dirichlet_dofs[i]%bs_c;
      diag_Dirichlet_dofs[i] /= bs_c;
   }
   idxmapc->local_to_global(std::span<const std::int32_t>(diag_Dirichlet_dofs.data(), nb_diag_Dirichlet),
                            std::span<std::int64_t>(glob_diag_Dirichlet_dofs.data(), nb_diag_Dirichlet));
   for (size_t i = 0; i < nb_diag_Dirichlet; ++i)
      glob_diag_Dirichlet_dofs[i] = glob_diag_Dirichlet_dofs[i] * bs_c + diag_Dirichlet_delta[i];
   // PRINT("diag_Dirichlet_dofs", diag_Dirichlet_dofs);
   // PRINT("diag_Dirichlet_dofs", glob_diag_Dirichlet_dofs);

   // note: here use of move for PETSC matrix/vector is more for semantic purpose then real implementation necessity as this
   // type are in fact pointers that could have been passed by copy.
   return twoscale::coarseManager<Mat, Vec>(std::move(PSR), std::move(PE), std::move(W), std::move(glob_diag_Dirichlet_dofs),
                                            std::move(enriched_dof_id));
}
#else
template <dolfinx::scalar T, std::floating_point U>
twoscale::coarseManager<Mat, Vec> generateCoarseManager(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &sj,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> coarse_enriched_space,
                                                        std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
                                                        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>> &bc)
{
   spdlog::info("Compute coarsening operator structure and values");

   // ===================
   // check coherance
   // ===================
   assert((sj.needs_mpc) ? (mpc != nullptr) : (mpc == nullptr));
   // TODO: mesh dim etc
   assert(&const_cast<const dolfinx::mesh::Mesh<U> &>(*sj.getCoarseMesh()) == &(*(coarse_enriched_space->mesh())));
   assert(&const_cast<const dolfinx::mesh::Mesh<U> &>(*sj.getFineMesh()) == &(*(fine_space->mesh())));

   // ================
   // collect mpi info
   // ================
   MPI_Comm comm = sj.getCoarseMesh()->comm();
   int pid = dolfinx::MPI::rank(comm);
   int nbproc = dolfinx::MPI::size(comm);

   // ===================
   // arithmetic constant
   // ===================
   // zero in U arithmetic
   const U zero = static_cast<U>(0.);
   // zero in PetscScalar arithmetic
   PetscScalar zero_petsc = static_cast<PetscScalar>(zero);
   // one in U arithmetic
   const U one = static_cast<U>(1.);
   // epsilon in U arithmetic : use this limit is ok as normally
   // child nodes are note supposed to be that close to coarse nodes
   // 1000 time numerical limit is expected to be well suited to identify properly
   const U eps = static_cast<U>(1000.) * std::numeric_limits<U>::epsilon();
   const U onemeps = one - eps;

   // =========================
   // collect space information
   // =========================
   auto dofmapf = fine_space->dofmap();
   auto idxmapf = dofmapf->index_map;
   auto bs_f = dofmapf->index_map_bs();
   std::int32_t nbdf = idxmapf->size_local();

   auto coarse_space_std_view = coarse_enriched_space->sub({0});
   auto [coarse_space_std, std_to_mix] = coarse_space_std_view.collapse();
   //std::println("std_to_mix {}: {}",std_to_mix.size(),std_to_mix);
   auto coarse_space_enrich_view = coarse_enriched_space->sub({1});
   auto [coarse_space_enrich, enr_to_mix] = coarse_space_enrich_view.collapse();
   //std::println("enr_to_mix {}: {}",enr_to_mix.size(),enr_to_mix);
   auto dofmapc = coarse_enriched_space->dofmap();
   auto idxmapc = dofmapc->index_map;
   auto dofmapcs = coarse_space_std.dofmap();
   auto idxmapcs = dofmapcs->index_map;
   auto bs_cs = dofmapcs->index_map_bs();
   auto bs_cs2 = 2 * bs_cs; // 2 here is related to mixed space having 2 subspace
   std::int32_t nbdcsl = idxmapcs->size_local();
   std::int32_t nbdcl = idxmapc->size_local();
   //std::println("nbdcsl {} ",nbdcsl);
   //std::println("g indices {} ",idxmapc->global_indices());
   assert(bs_f == bs_cs);
   //std::println("nb loc cs {}",idxmapcs->size_local());
   //std::println("nb goh cs {}",idxmapcs->num_ghosts());
   // NOTE: At current stage of development enriched and std space are expected to be the same and thus
   // provides the same idxs
   assert(idxmapcs->size_local() == coarse_space_enrich.dofmap()->index_map->size_local());
   assert(idxmapcs->num_ghosts() == coarse_space_enrich.dofmap()->index_map->num_ghosts());
   assert(dofmapcs->index_map_bs() == coarse_space_enrich.dofmap()->index_map_bs());

   // ==============================
   // collect enrichment information
   // ==============================
   std::unordered_set<std::int32_t> to_enrich;
   {
      auto v = sj.getEnriched();
      to_enrich.insert(v.begin(), v.end());
   }
   {
      auto v = sj.getExtraEnriched();
      to_enrich.insert(v.begin(), v.end());
   }

   //std::println("to_enrich {} {}",to_enrich.size(),to_enrich);

   // ======================================
   // collect coarse topological information
   // ======================================
   auto cdomain = sj.getCoarseMesh();
   auto topoc = cdomain->topology();
   int dim = topoc->dim();
   int dimm1 = dim - 1;
   auto idxmapcc = topoc->index_map(dim);
   std::int32_t nbccl = idxmapcc->size_local();
   auto adj = topoc->connectivity(dim, 0);
   assert(topoc->index_map(0)->size_local()==nbdcsl);

   // ======================================
   // collect coarse geometrical information
   // ======================================
   // Note:
   // Pass to tabulate_dof_coordinates would makes things more general
   // but in TS for now we use only order 1 Lagrange element so ....
   // Anyway tabulate_dof_coordinates is impossible with mixed space
   auto &geomc = cdomain->geometry();
   std::span<const U> xc_g = geomc.x();
   auto xc_dofmap = geomc.dofmap();
   const auto &cellc_cmap = geomc.cmap();
   assert(cellc_cmap.is_affine());
   const std::size_t nb_nodes_elem = cellc_cmap.dim();
   const std::size_t gdim = geomc.dim();

   // =======================
   // set mpc filter function
   // =======================
   std::span<const std::int8_t> slave_flag;
   std::function<bool(const std::int32_t &)> mpc_filter = [](const std::int32_t &i) -> bool { return true; };
   if (mpc != nullptr)
   {
      slave_flag = mpc->is_slave();
      mpc_filter = [&slave_flag, &bs_f](const std::int32_t &i) -> bool { return (slave_flag[i * bs_f] < 1); };
   }

   // =======================
   // set BC filter function
   // and W vector
   // =======================
   std::unordered_set<std::int32_t> BC_idx_activated;
   Vec W = nullptr;
   Vec XDc = nullptr;
   if (bc.size() > 0)
   {
      assert(bc.size() == 1);
      auto BC = bc[0];
      assert(coarse_enriched_space->contains(*(BC->function_space())));
      auto BC_values = BC->value();
      assert(BC_values.index() == 0);
      //std::span<const PetscScalar> x(std::get<0>(BC_values)->x()->array());
      Vec XDc_ = dolfinx::la::petsc::create_vector_wrap(*(std::get<0>(BC_values)->x()));
      // check that non null Dirichlet exist (XDc!=0). If not W=L.XDc=0 doesn't need to be created as it lead to zero results
      // AD.W=0 or do not change computation W+Qt.S=Qt.S. In this case L is also not assembled.
      PetscReal norm_inf;
      DO(VecNorm(XDc_, NORM_INFINITY, &norm_inf), "problem while computing XDc norm", comm)
      if(norm_inf>std::numeric_limits<PetscReal>::epsilon())
      {
         // generate W
         W = dolfinx::la::petsc::create_vector(*idxmapf,bs_f);
         // copy XD_c_ in XD_c to pass its storage property to coarseManager
         // NOTE: security purpose may 
         DO(VecDuplicate(XDc_, &XDc), "Problem while creating XDc by  duplication of XDc_", comm)
         DO(VecCopy(XDc_, XDc), "Problem while copying XDc_ in XDc", comm)
      }
      DO(VecDestroy(&XDc_), "Problem to destroy XDc_", comm)
      auto [BC_idx,BC_idx_ghost] = BC->dof_indices();
      //std::println("Bc indexes {} ghost {} val {} norm val {} eps {}",BC_idx,BC_idx_ghost,std::get<0>(BC_values)->x()->array(),norm_inf,std::numeric_limits<PetscReal>::epsilon());
      // switch to unordered_set 
      BC_idx_activated.insert(BC_idx.begin(),BC_idx.end());

   }


   // ============================
   // collect fine dof coordinates
   // ============================
   auto xf_d=fine_space->tabulate_dof_coordinates(false);

   // =========================
   // operator sparsity pattern
   // =========================
   dolfinx::la::SparsityPattern sps(comm, {idxmapf, idxmapc}, {dofmapf->index_map_bs(), dofmapc->index_map_bs()});
   dolfinx::la::SparsityPattern spe(comm, {idxmapf, idxmapc}, {dofmapf->index_map_bs(), dofmapc->index_map_bs()});
   dolfinx::la::SparsityPattern spl(comm, {idxmapf, idxmapc}, {dofmapf->index_map_bs(), dofmapc->index_map_bs()});
   dolfinx::la::SparsityPattern spce(comm, {idxmapc, idxmapc}, {dofmapc->index_map_bs(), dofmapc->index_map_bs()});

   // ============================================
   // preparing container for per cell computation
   // ============================================
   // container to store enriched dof id
   assert(bs_cs<<sizeof(twoscale_dolfinx::enrichedDofIDs::eliminated_coarse_encoding));
   std::unordered_map<std::int32_t,twoscale_dolfinx::enrichedDofIDs> enriched_dof_id;
   // container to store all child nodes
   std::unordered_set<std::int32_t> nodes;
   // container to store vertex coordinate of a coarse cell
   std::vector<U> coord_dofs_b(nb_nodes_elem * gdim);
   mdspan_t<U, 2> coord_dofs(coord_dofs_b.data(), nb_nodes_elem, gdim);
   // container to store element dofs in mixed space
   std::vector<std::int32_t> mix_coarse(nb_nodes_elem * bs_cs2);
   mdspan_t<int32_t, 3> mix_coarse_dof(mix_coarse.data(), nb_nodes_elem, 2, bs_cs);
   auto addNonNull = [&mix_coarse_dof, &bs_cs, &sps, &spe, &spl](std::int8_t k, std::int32_t vb) {
      for (std::int32_t b = 0; b < bs_cs; ++b)
      {
         auto idx = mix_coarse_dof(k, 0, b);
         if (idx > -1)
            sps.insert(vb, idx);
         else
            spl.insert(vb, -idx - 2);
         idx = mix_coarse_dof(k, 1, b);
         if (idx > -1) spe.insert(vb, idx);
      }
   };
   // container to store all dofs of enriched space that needs to be blocked by Dirichlet BC as not used by twoscale enriched or
   // extra enriched nodes
   std::unordered_set<std::int32_t> Dirichlet_dofs;
   // container to store all zero value that can be added by an element to Acc
   std::vector<PetscScalar> values(nb_nodes_elem*bs_cs2*nb_nodes_elem*bs_cs2,zero_petsc);
   std::vector<PetscInt> idxmn(nb_nodes_elem*bs_cs2);
   std::vector<std::int64_t> idxmng(nb_nodes_elem*bs_cs2);

   // functor to set mix_coarse filtered by BC in the context of graph creation
   // If any dof is eliminated return true otherwise return false
   auto set_mix_coarse_graph = [&Dirichlet_dofs, &bs_cs, &nbdcsl, &mix_coarse, &BC_idx_activated](
                                   std::int32_t x, std::int32_t k, std::vector<std::int32_t> &X_to_mix,
                                   const bool force) -> bool {
      std::int8_t nb_elim = 0;
      // filter  by BC
      // if local the associated eliminated diagonal term must be kept
      if (x < nbdcsl)
      {
         for (std::int32_t b = 0; b < bs_cs; ++b)
         {
            auto idx = X_to_mix[x * bs_cs + b];
            if (force || BC_idx_activated.find(idx) != BC_idx_activated.end())
            {

               mix_coarse[k + b] = -2 - idx;
               Dirichlet_dofs.insert(idx);
            }
            else
               mix_coarse[k + b] = idx;
         }
      }
      // if non local no diagonal treatement
      else
      {
         for (std::int32_t b = 0; b < bs_cs; ++b)
         {
            auto idx = X_to_mix[x * bs_cs + b];
            // when force is true (non enriched dof context a priori).
            if (force)
            {
               // Check BC, and if it is true return immediately true. Normally in this calling context it will be judged as a
               // problem.
               if (BC_idx_activated.find(idx) != BC_idx_activated.end()) return true;

               // otherwise eliminate without counting
               mix_coarse[k + b] = -2 - idx;
            }
            else if (BC_idx_activated.find(idx) != BC_idx_activated.end())
            {
               // when force false nb_elim will give returned info and calling context will judge if it is a problem or not
               ++nb_elim;

               mix_coarse[k + b] = -2 - idx;
            }
            // not eliminated
            else
               mix_coarse[k + b] = idx;
         }
      }

      return (nb_elim > 0);
   };

   // functor to set mix_coarse filtered by BC in the context of matrix assembly
   // If any dof is eliminated return true otherwise return false
   auto set_mix_coarse_ass = [&idxmn, &bs_cs, &nbdcsl, &mix_coarse, &BC_idx_activated](
                                 std::int32_t x, std::int32_t k, std::vector<std::int32_t> &X_to_mix, std::int32_t &nb_d_eff,
                                 std::int32_t &end_d_elim, bool force) -> void {
      std::int8_t nb_elim = 0;
      
      // filter  by BC
      // if local the associated eliminated diagonal term number must be kept
      if (x < nbdcsl)
      {
         for (std::int32_t b = 0; b < bs_cs; ++b)
         {
            auto idx = X_to_mix[x * bs_cs + b];
            if (force || (BC_idx_activated.find(idx) != BC_idx_activated.end()))
            {
               mix_coarse[k + b] = -2 - idx;
               idxmn[end_d_elim] = idx;
               --end_d_elim;
               ++nb_elim;
            }
            else
            {
               mix_coarse[k + b] = idx;
               idxmn[nb_d_eff] = idx;
               ++nb_d_eff;
            }
         }
      }
      else
      {
         for (std::int32_t b = 0; b < bs_cs; ++b)
         {
            auto idx = X_to_mix[x * bs_cs + b];
            if (force || (BC_idx_activated.find(idx) != BC_idx_activated.end()))
            {
               mix_coarse[k + b] = -2 - idx;
               ++nb_elim;
            }
            else
            {
               mix_coarse[k + b] = idx;
               idxmn[nb_d_eff] = idx;
               ++nb_d_eff;
            }
         }
      }

      return;
   };

   // origin of the cell
   std::array<U, 3> x0;

   // reference coordinates
   std::array<U, 3> X_child_b;
   mdspan_t<U, 2> X_child(X_child_b.data(), 1, dim);

   // Geometry data at each point
   std::vector<U> J_b(gdim * dim);
   mdspan_t<U, 2> J(J_b.data(), gdim, dim);
   std::vector<U> K_b(dim * gdim);
   mdspan_t<U, 2> K(K_b.data(), dim, gdim);

   // point shape
   std::array<std::size_t, 2> point_shape = {1, dim};


   // cell interpolation function and its derivative for one point
   const std::array<std::size_t, 4> phi_coarse_shape = cellc_cmap.tabulate_shape(1, 1);
   std::vector<U> phi_coarse_b(std::reduce(phi_coarse_shape.begin(), phi_coarse_shape.end(), 1, std::multiplies{}));
   mdspan_t<const U, 4> phi_coarse(phi_coarse_b.data(), phi_coarse_shape);
   cellc_cmap.tabulate(1, std::vector<U>(dim), point_shape, phi_coarse_b);  // get for first node FF and derivative
   auto dphi_coarse = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
       phi_coarse, std::pair(1, dim + 1), 0, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);  // only derivative

   // =============================================
   // loop on all cell to identify OP and Acc graph
   // =============================================
   for (std::int32_t cell_id = 0; cell_id < nbccl; ++cell_id)
   {
      nodes.clear();
      std::fill(J_b.begin(), J_b.end(), zero);

       // PRINT("cell", cell_id);

      //  cell children if any
      auto cell_child = sj.getChildren(cell_id);

#ifdef ALL_CHILD
      assert(cell_child.size() > 0);
#else
      // no children means not in extended support: operator is the identity for std an null for enrich
      // TODO
      if (cell_child.size() < 1) PRINT("cell not treated !!!", false);
      // childs : operator is the same for std and enrich at construction time
      else
#endif
      {
         // std dof of the element
         auto std_dof = dofmapcs->cell_dofs(cell_id);
         // PRINT("std_dof", std_dof);
         const auto std_topo = adj->links(cell_id);
         assert(std_dof.size()==std_topo.size()); // true in blocked context
         // PRINT("std_topo", std_topo);
         // coarse index to store enriched ids
         // NOTE: enriched dof of the element are the same as std dof in terms of local index for now
         for (std::int32_t k = 0, l = 0; auto x : std_dof)
         {
            // if coarse node is enriched, PE operator must be set and PS,PE operator filtered by BC
            // NOTE: assertion is done here that std_dof and std_topo are in the same order.
            // i.e. that l correspond to node on which x dof is leaving
            auto i_topo = std_topo[l];
            if (to_enrich.find(i_topo) != to_enrich.end())
            {
               // filter PC by BC
               set_mix_coarse_graph(x, k, std_to_mix,false);
               // filter PE by BC
               k += bs_cs;
               set_mix_coarse_graph(x, k, enr_to_mix,false);

               // init enriched dof ids if not already done and block not fully eliminated
               if (enriched_dof_id.find(i_topo) == enriched_dof_id.end())
               {
                  // encode  eliminated dof in the block
                  std::int64_t eliminated_coarse_encoding = 0;
                  std::int32_t nb_eliminated = 0;
                  for (std::int32_t b = 0; b < bs_cs; ++b)
                     if (mix_coarse[k + b] < 0)
                     {
                        ++nb_eliminated;
                        eliminated_coarse_encoding |= 1 << b;
                     }
                  if (nb_eliminated != bs_cs)
                  {
                     std::int64_t g;
                     idxmapc->local_to_global(std::span<const std::int32_t>(&enr_to_mix[x * bs_cs], 1),
                                              std::span<std::int64_t>(&g, 1));
                     twoscale_dolfinx::enrichedDofIDs idxs;
                     idxs.global_coarse_block_dof_idx=g;
                     idxs.global_fine_block_dof_idx = 0;
                     idxs.eliminated_coarse_encoding = eliminated_coarse_encoding;
                     enriched_dof_id[i_topo] = idxs;
                  }
               }
            }
            // old wrong: otherwise coarse is eliminated from PE and PS,PE should not have any dof filtered by BC from user 
            // otherwise coarse is eliminated from PE and PE should not have any dof filtered by BC from user 
            // but PS can  have dof filtered by BC from user 
            else
            {
               /* old
               // PS: fine scale BC take  precedence so any dof filtered by BC is an error
               if (set_mix_coarse_graph(x, k, std_to_mix,false)) 
               {
                  std::cout
                      << "Given boundary condition include std dofs not in support of enriched/extra enriched nodes. These BC "
                         "normally are coming from fine scale BC."
                      << std::endl;
                  MPI_Abort(MPI_COMM_WORLD, -1);

               }
               */
               set_mix_coarse_graph(x, k, std_to_mix,false); 
               
               // PE: elimination forced in this case so any dof filtered by BC is an error
               k += bs_cs;
               if (set_mix_coarse_graph(x, k, enr_to_mix,true)) 
               {
                  std::cout
                      << "Given boundary condition include enr dofs not in support of enriched/extra enriched nodes. These BC "
                         "normally are treated automatically by this function."
                      << std::endl;
                  MPI_Abort(MPI_COMM_WORLD, -1);

               }
              
            }
            k += bs_cs;
            ++l;
         }
         // PRINT("mix_coarse", mix_coarse);
         // PRINT("mix_coarse_dof", mix_coarse_dof);
         // std::println("Dirichlet_dofs eliminated : {}", Dirichlet_dofs);

         // set graph of Acc related to this element
         // loop on row
         for (std::int8_t r = 0; r < nb_nodes_elem; ++r)
         {
            // loop on bloc
            for (std::int32_t br = 0; br < bs_cs; ++br)
            {
               std::int32_t rows = mix_coarse_dof(r, 0, br);
               std::int32_t rowe = mix_coarse_dof(r, 1, br);
               // std row not eliminated
               if (rows > -1 )
               {
                  // enr row not eliminated
                  if (rowe > -1)
                  {
                     // loop on column
                     for (std::int8_t c = 0; c < nb_nodes_elem; ++c)
                     {
                        // loop on bloc
                        for (std::int32_t bc = 0; bc < bs_cs; ++bc)
                        {
                           std::int32_t cols = mix_coarse_dof(c, 0, bc);
                           if (cols > -1)
                           {
                              spce.insert(rows, cols);
                              spce.insert(rowe, cols);
                           }
                           std::int32_t cole = mix_coarse_dof(c, 1, bc);
                           if (cole > -1)
                           {
                              spce.insert(rows, cole);
                              spce.insert(rowe, cole);
                           }
                        }
                     }
                  }
                  // enr row  eliminated
                  else
                  {
                     // loop on column
                     for (std::int8_t c = 0; c < nb_nodes_elem; ++c)
                     {
                        // loop on bloc
                        for (std::int32_t bc = 0; bc < bs_cs; ++bc)
                        {
                           std::int32_t cols = mix_coarse_dof(c, 0, bc);
                           if (cols > -1) spce.insert(rows, cols);
                           std::int32_t cole = mix_coarse_dof(c, 1, bc);
                           if (cole > -1) spce.insert(rows, cole);
                           // diagonal eliminated term must be added (diag 1 value)
                           else if ((cole < -1) && (cole == rowe))
                           {
                              cole = -(rowe + 2);
                              spce.insert(cole, cole);
                           }
                        }
                     }
                  }
               }
               // std row eliminated
               else
               {
                  // enr row not eliminated
                  if (rowe > -1)
                  {
                     // loop on column
                     for (std::int8_t c = 0; c < nb_nodes_elem; ++c)
                     {
                        // loop on bloc
                        for (std::int32_t bc = 0; bc < bs_cs; ++bc)
                        {
                           std::int32_t cols = mix_coarse_dof(c, 0, bc);
                           if (cols > -1) spce.insert(rowe, cols);
                           // diagonal eliminated term must be added (diag 1 value)
                           else if ((cols < -1) && (cols == rows))
                           {
                              cols = -(rows + 2);
                              spce.insert(cols, cols);
                           }
                           std::int32_t cole = mix_coarse_dof(c, 1, bc);
                           if (cole > -1) spce.insert(rowe, cole);
                        }
                     }
                  }
                  // enr row  eliminated
                  else
                  {
                     // loop on column
                     // only diagonal eliminated term must be added (diag 1 value)
                     for (std::int8_t c = 0; c < nb_nodes_elem; ++c)
                     {
                        // loop on bloc
                        for (std::int32_t bc = 0; bc < bs_cs; ++bc)
                        {
                           std::int32_t cols = mix_coarse_dof(c, 0, bc);
                           if ((cols < -1) && (cols == rows))
                           {
                              cols = -(rows + 2);
                              spce.insert(cols, cols);
                           }
                           std::int32_t cole = mix_coarse_dof(c, 1, bc);
                           if ((cole < -1) && (cole == rowe))
                           {
                              cole = -(rowe + 2);
                              spce.insert(cole, cole);
                           }
                        }
                     }
                  }
               }
            }
         }


         // all local dof related to childs and not slave of an mpc relation
         for (auto fc : cell_child)
         {
            auto v = dofmapf->cell_dofs(fc);
             auto vf = v | std::views::filter(mpc_filter);
            //auto vf = v | std::views::filter([&nbdf](const std::int32_t &x) -> bool { return (x < nbdf); }) | std::views::filter(mpc_filter);
            nodes.insert(vf.begin(), vf.end());
         }

         std::int32_t nodes_size = nodes.size();

         // PRINT("nodes_child", nodes);

         // if at least one childs to treat
         if (nodes_size)
         {
            // coordinates of the element
            auto x_dofs =
                MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(xc_dofmap, cell_id, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (int i = 0; i < nb_nodes_elem; ++i)
            {
               const int pos = 3 * x_dofs[i];
               for (int j = 0; j < gdim; ++j) coord_dofs(i, j) = xc_g[pos + j];
            }
            // PRINT("coord_dofs", coord_dofs);

            // set x0
            for (std::size_t i = 0; i < gdim; ++i) x0[i] = coord_dofs(0, i);

            // Jacobian of the transformation for this element
            dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi_coarse, coord_dofs, J);

            // inverse of the Jacobian for this element
            dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);

            // loop on local child nodes (fine nodes to interpolate into coarse cell)
            for (auto v : nodes)
            {
               // PRINT("v", v);

               //assert(v < nbdf);

               // grab physical node coordinate
               mdspan_t<const U, 2> x_child(&xf_d[3 * v], 1, gdim);
               //PRINT("x_child", x_child);

               // compute node in reference element corresponding to master face
               dolfinx::fem::CoordinateElement<U>::pull_back_affine(X_child, K, x0, x_child);
               // PRINT("X_child", X_child);

               // get interpolation function values at reference location: coefficients of the mpc for this slave nodes
               cellc_cmap.tabulate(0, X_child_b, point_shape, phi_coarse_b);
               // PRINT("phi_coarse_b", phi_coarse_b);

               // loop on FF
               std::int8_t k = -1;
               for (auto &val : phi_coarse_b | std::views::take(phi_coarse_shape[0]))
               {
                  ++k;

                  // if val non null a interpolation exist in between fine and coarse scale
                  if (val > eps)
                  {
                     // depend on a coarse node k
                     addNonNull(k, v);
                     if (val > onemeps)
                     {
                        // if it correspond to a coarse enriched node grab fine dof global id for 
                        // future selection of petsc terms
                        auto i_topo = std_topo[k];
                        if (to_enrich.find(i_topo) != to_enrich.end())
                        {
                           auto itf = enriched_dof_id.find(i_topo);
                           // if BC fully eliminate enriched dof of the block the find is empty
                           // skip action in this case
                           if(itf != enriched_dof_id.end())
                           {
                              std::int64_t g;
                              idxmapf->local_to_global(std::span<const std::int32_t>(&v, 1), std::span<std::int64_t>(&g, 1));
                              itf->second.global_fine_block_dof_idx = bs_f * g;

                              //std::println("a sauver {} ({}) k {} cell_id {} macro {} - {} ", g * bs_f, v, k, cell_id,
                               //            enriched_dof_id, i_topo);
                           }
                        
                        }
                        // val of 1 implies others are null so no need to continue
                        break;
                     }
                  }
               }
            }
         }
      }
   }

   // =======================
   // generate empty matrices
   // based on graph sps,spe 
   // and spce
   // =======================
   sps.finalize();
   spl.finalize();
   spe.finalize();
   spce.finalize();
   Mat PS = dolfinx::la::petsc::create_matrix(comm, sps, MATAIJ);
   Mat LS = dolfinx::la::petsc::create_matrix(comm, spl, MATAIJ);
   Mat PE = dolfinx::la::petsc::create_matrix(comm, spe, MATAIJ);
   Mat Acc = dolfinx::la::petsc::create_matrix(comm, spce, MATAIJ);

   // ===========================
   // matrix term setter function
   // ===========================
   auto setters = dolfinx::la::petsc::Matrix::set_block_fn(PS, INSERT_VALUES);
   auto setterl = dolfinx::la::petsc::Matrix::set_block_fn(LS, INSERT_VALUES);
   auto settere = dolfinx::la::petsc::Matrix::set_block_fn(PE, INSERT_VALUES);
   // Note: use of bs_f is juste to visualize what is related to rows compaire to bs_cs related to column but assert above already
   // check that bs_f=bs_cs.
   std::vector<int> c_(bs_cs);
   std::vector<int> ra_(bs_f);
   std::vector<PetscScalar> va_(bs_f * bs_cs, 0.);
   auto setNonNull = [&W, &mix_coarse_dof, &zero_petsc, &bs_cs, &bs_f, &setters, &settere, &setterl, &c_, &ra_,  &va_,
                      &comm](std::int8_t k, int vb, const U &val) {
      // PS construction. Case where XDc is nul thus W=LS.XDc=0 so we skip LS construction
      if (W == nullptr) impl::setNonNullWNull(setters, 0, k, vb, val, mix_coarse_dof, zero_petsc, bs_cs, bs_f, c_, va_,ra_);
      // PS an LS construction
      else
      {
         // if dof are eliminated count future block size for PS and LS 
         std::int32_t brp = bs_cs;
         std::int32_t brl = 0;
         PetscInt vb_petsc = vb;
         for (std::int32_t b = 0; b < bs_cs; ++b)
         {
            auto idx = mix_coarse_dof(k, 0, b);
            if (idx < -1)
            {
               c_[--brp] = -idx - 2;
               ra_[brp]=b;
            }
            else
            {
               ra_[brl] = b;
               c_[brl++] = idx;
            }
         }
         assert(brp==brl);


         // for PS if block fully eliminated (brp==0) skip
         // otherwise set with potential eliminated column 
         if (brp) impl::setBlock(setters, brp, vb, val, zero_petsc, bs_f, c_, va_, ra_);
         

         // if eliminated dofs, add them to LS
         if (brp<bs_cs)
         {
            brl = bs_cs - brp;
            // reverse to obtain correct column ordering and information packed at begin
            std::ranges::reverse(c_);
            std::ranges::reverse(ra_);
            // set
            impl::setBlock(setterl, brl, vb, val, zero_petsc, bs_f, c_, va_, ra_);
         }
      }

      // PE construction: XDc assumed to be null on e-set so only LS exist and no Le create nor used (Le.0=0 ... so remove computation)
      impl::setNonNullWNull(settere, 1, k, vb, val, mix_coarse_dof, zero_petsc, bs_cs, bs_f, c_, va_,ra_);
   };

   // =================================
   // loop on all cell to set PS&LS&PE&Acc
   // =================================
   for (std::int32_t cell_id = 0; cell_id < nbccl; ++cell_id)
   {
      nodes.clear();
      std::fill(J_b.begin(), J_b.end(), zero);

      // PRINT("cell", cell_id);

      //  cell childs if any
      auto cell_child = sj.getChildren(cell_id);

#ifdef ALL_CHILD
      assert(cell_child.size() > 0);
#else
      // no childs means not in extended support: operator is the identity for std an null for enrich
      if (cell_child.size() < 1) PRINT("cell not treated !!!", false);
      // childs : operator is the same for std and enrich at construction time
      else
#endif
      {
         // TODO reshape dofmapc->cell_dofs(cell_id) to suppress unused enriched dof ?

         // std dof of the element
         auto std_dof = dofmapcs->cell_dofs(cell_id);
         // PRINT("std_dof", std_dof);
         const auto std_topo = adj->links(cell_id);
         // PRINT("std_topo", std_topo);
         // NOTE: enriched dof of the element are the same as std dof in terms of local index for now
         std::int32_t nb_d_eff = 0;
         std::int32_t end_d_elim = idxmn.size() - 1;
         for (std::int32_t k = 0, l = 0; auto x : std_dof)
         {
            // if coarse node is enriched, PE operator must be set and PS,PE operator filtered by BC.
            // Acc diag eliminated dof has to be identified.
            if (to_enrich.find(std_topo[l]) != to_enrich.end())
            {
               // filter PC by BC
               set_mix_coarse_ass(x, k, std_to_mix, nb_d_eff, end_d_elim, false);
               // filter PE by BC
               k += bs_cs;
               set_mix_coarse_ass(x, k, enr_to_mix, nb_d_eff, end_d_elim, false);
            }
            // otherwise coarse is eliminated from PE and PE,PS are not filtered  by BC
            // Acc diag eliminated dof has to be identified.
            else
            {
               // PC normally not filtered by BC (already checked)
               set_mix_coarse_ass(x, k, std_to_mix, nb_d_eff, end_d_elim, false);
               k += bs_cs;
               // PE force elimination and normally not filtered by BC (already checked)
               set_mix_coarse_ass(x, k, enr_to_mix, nb_d_eff, end_d_elim, true);
            }
            k += bs_cs;
            ++l;
         }
         // to avoid local_to_global to get in trouble with old uncleaned index
         if ((nb_d_eff - 1) < end_d_elim) std::fill(&idxmn[nb_d_eff], &idxmn[end_d_elim + 1], 0);

         // PRINT("nb_d_eff", nb_d_eff);
         // PRINT("end_d_elim", end_d_elim);
         // PRINT("mix_coarse", mix_coarse);
         // PRINT("mix_coarse_dof", mix_coarse_dof);
         // PRINT("idxmn", idxmn);

         // assemble Acc bloc
         idxmapc->local_to_global(std::span<const PetscInt>(idxmn), std::span<std::int64_t>(idxmng));
         std::ranges::transform(idxmng, idxmn.begin(), [](const std::int64_t &x) { return static_cast<PetscInt>(x); });
         // PRINT("idxmn", idxmn);
         DO(MatSetValues(Acc, nb_d_eff, idxmn.data(), nb_d_eff, idxmn.data(), values.data(), INSERT_VALUES),
            "Elemental assembly in Acc imposible", comm)

         // assemble Acc dirichlet diag terms
         // TODO : local ??
         for (std::int32_t ie = idxmn.size() - 1; ie > end_d_elim; --ie)
            DO(MatSetValues(Acc, 1, &idxmn[ie], 1, &idxmn[ie], &zero_petsc, INSERT_VALUES),
               "Problem while inserting eliminated enriched diagonal term in Acc", comm)

         // all dof related to childs
         for (auto fc : cell_child)
         {
            auto v = dofmapf->cell_dofs(fc);
            auto vf = v | std::views::filter(mpc_filter);
            // auto vf = v | std::views::filter([&nbdf](const std::int32_t &x) -> bool { return (x < nbdf); }) |
            // std::views::filter(mpc_filter);
            nodes.insert(vf.begin(), vf.end());
         }
         std::int32_t nodes_size = nodes.size();

         // PRINT("nodes_child", nodes);

         // if at least one childs to treat
         if (nodes_size)
         {
            // coordinates of the element
            auto x_dofs =
                MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(xc_dofmap, cell_id, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (int i = 0; i < nb_nodes_elem; ++i)
            {
               const int pos = 3 * x_dofs[i];
               for (int j = 0; j < gdim; ++j) coord_dofs(i, j) = xc_g[pos + j];
            }
            // PRINT("coord_dofs", coord_dofs);

            // set x0
            for (std::size_t i = 0; i < gdim; ++i) x0[i] = coord_dofs(0, i);

            // Jacobian of the transformation for this element
            dolfinx::fem::CoordinateElement<U>::compute_jacobian(dphi_coarse, coord_dofs, J);

            // inverse of the Jacobian for this element
            dolfinx::fem::CoordinateElement<U>::compute_jacobian_inverse(J, K);

            // loop on child nodes (fine nodes to interpolate into coarse cell)
            for (auto v : nodes)
            {
               // PRINT("v", v);

               // grab physical node coordinate
               mdspan_t<const U, 2> x_child(&xf_d[3 * v], 1, gdim);
               // PRINT("x_child", x_child);

               // compute node in reference element corresponding to master face
               dolfinx::fem::CoordinateElement<U>::pull_back_affine(X_child, K, x0, x_child);
               // PRINT("X_child", X_child);

               // get interpolation function values at reference location: coefficients of the mpc for this slave nodes
               cellc_cmap.tabulate(0, X_child_b, point_shape, phi_coarse_b);
               // PRINT("phi_coarse_b", phi_coarse_b);

               // loop on FF
               std::int8_t k = -1;
               for (auto &val : phi_coarse_b | std::views::take(phi_coarse_shape[0]))
               {
                  ++k;

                  // if val non null simply set coefficient
                  if (val > eps)
                  {
                     // depend on a coarse node k
                     if (val > onemeps)
                     {
                        setNonNull(k, v, one);
                        // val of 1 implies others are null so no need to continue
                        break;
                     }
                     setNonNull(k, v, val);
                  }
               }
            }
         }
      }
   }

   // assemble PS,PE and Acc
   DO(MatAssemblyBegin(PS, MAT_FINAL_ASSEMBLY), "PS begin assembly operator", comm)
   DO(MatAssemblyEnd(PS, MAT_FINAL_ASSEMBLY), "PS end assembly operator", comm)
   DO(MatAssemblyBegin(PE, MAT_FINAL_ASSEMBLY), "PE begin assembly operator", comm)
   DO(MatAssemblyEnd(PE, MAT_FINAL_ASSEMBLY), "PE end assembly operator", comm)
   DO(MatAssemblyBegin(Acc, MAT_FINAL_ASSEMBLY), "Acc begin assembly Matrix", comm)
   DO(MatAssemblyEnd(Acc, MAT_FINAL_ASSEMBLY), "Acc end assembly Matrix", comm)

   // std::println("Acc");
   // MatView(Acc, PETSC_VIEWER_STDOUT_(comm));
   // std::println("PS");
   // MatView(PS, PETSC_VIEWER_STDOUT_(comm));
   // std::println("PE");
   // MatView(PE, PETSC_VIEWER_STDOUT_(comm));

   // only if W not null
   if (W != nullptr)
   {
      // std::println("XDc");
      // VecView(XDc, PETSC_VIEWER_STDOUT_(comm));
      DO(MatAssemblyBegin(LS, MAT_FINAL_ASSEMBLY), "LS begin assembly operator", comm)
      DO(MatAssemblyEnd(LS, MAT_FINAL_ASSEMBLY), "LS end assembly operator", comm)
      // std::println("LS");
      // MatView(LS, PETSC_VIEWER_STDOUT_(comm));
      DO(MatMult(LS, XDc, W), "Problem while computing W", comm)
      // std::println("W");
      // VecView(W, PETSC_VIEWER_STDOUT_(comm));
      // Ls not needed any more: clean memory
      DO(MatDestroy(&LS), "Problem to destroy LS", comm)
   }

   // to be able to change PE easely in a distributed maner by row, transpose PE
   DO(MatTranspose(PE, MAT_INPLACE_MATRIX, &PE), "Problem while transposing  PE", comm)
   // MatView(PE, PETSC_VIEWER_STDOUT_(comm));

   // pack enrich_Dirichlet_dofs
   std::vector<std::int32_t> diag_Dirichlet_dofs(Dirichlet_dofs.begin(), Dirichlet_dofs.end());
   std::ranges::sort(diag_Dirichlet_dofs);
   // PRINT("diag_Dirichlet_dofs", diag_Dirichlet_dofs);

   // note: here use of move for PETSC matrix/vector is more for semantic purpose then real implementation necessity as this
   // type are in fact pointers that could have been passed by copy.
   return twoscale::coarseManager<Mat, Vec>(std::move(PS), std::move(PE), std::move(Acc), std::move(W), std::move(XDc),
                                            std::move(diag_Dirichlet_dofs), std::move(enriched_dof_id));
}
#endif
#endif

}  // namespace twoscale_dolfinx
#endif
