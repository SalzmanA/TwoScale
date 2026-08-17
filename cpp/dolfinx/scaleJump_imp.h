/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_SCALE_JUMP_IMP
#define TS_DOLFINX_SCALE_JUMP_IMP
#ifndef TS_DOLFINX_SCALE_JUMP
#error "Should not be included by hand"
#endif
#include <unistd.h>

#include <algorithm>
#include <array>
#include <execution>
#include <iostream>
#include <ranges>
#include <unordered_map>
#include <vector>

//#include "dolfinx/refinement/refine.h"
#include "dolfinx/common/Scatterer.h"
//#include "dolfinx/common/log.h"
#include "dataTsPerMacro.h"
#include "extraPartitioners.h"
#include "extraRefine.h"
#include "extraUtils.h"
#include "mpi.h"

namespace twoscale
{
template <typename M>
const std::shared_ptr<M> scaleJump<M>::getFineMesh() const
{
   return fine_mesh;
}
template <typename M>
const std::shared_ptr<M> scaleJump<M>::getCoarseMesh() const
{
   return coarse_mesh;
}

template <typename M>
std::span<const std::int32_t> scaleJump<M>::getChildren(std::int32_t coarse_cell_idx) const
{
   size_t s = cell_child_offset[coarse_cell_idx];
   size_t ds = cell_child_offset[coarse_cell_idx + 1] - s;
   if (ds)
   {
      auto beg = &cell_child[s];
      return std::span<const std::int32_t>(beg, ds);
   }
   else
      return std::span<std::int32_t>();
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getFaceChilds(std::int32_t coarse_face_idx) const
{
   size_t s = face_child_offset[coarse_face_idx];
   size_t ds = face_child_offset[coarse_face_idx + 1] - s;
   if (ds)
   {
      auto beg = &face_child[s];
      return std::span<const std::int32_t>(beg, ds);
   }
   else
      return std::span<std::int32_t>();
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getSurroundingFaceChilds(std::int32_t coarse_face_idx) const
{
   size_t s = sface_child_offset[coarse_face_idx];
   size_t ds = sface_child_offset[coarse_face_idx + 1] - s;
   if (ds)
   {
      auto beg = &sface_child[s];
      return std::span<const std::int32_t>(beg, ds);
   }
   else
      return std::span<std::int32_t>();
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getEnriched() const
{
   return std::span<const std::int32_t>(enriched_nodes.data(), enriched_nodes.size());
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getExtraEnriched() const
{
   return std::span<const std::int32_t>(extra_enriched_nodes.data(), extra_enriched_nodes.size());
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getSupport() const
{
   return std::span<const std::int32_t>(support.data(), support.size());
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getSurroundingCells() const
{
   return std::span<const std::int32_t>(surrounding_cells.data(), surrounding_cells.size());
}
template <typename M>
bool scaleJump<M>::isFineMasterNode(std::int32_t i) const
{
   return (fine_node_master.find(i) != fine_node_master.end());
}
template <typename M>
std::span<const std::int32_t> scaleJump<M>::getCoarseMaster() const
{
   return std::span<const std::int32_t>(coarse_interface_master.data(), coarse_interface_master.size());
}
template <typename M>
std::int32_t scaleJump<M>::getMaxNumberOfChildFaces() const
{
   return max_nb_face_child;
}

}  // namespace twoscale
namespace twoscale_dolfinx
{
namespace impl
{
template <typename K, typename T>
void find_in(std::span<T> x, std::int32_t *idx, std::int32_t nbl, std::span<T> xc, std::int32_t *idxc, std::int32_t nblc,
             std::vector<std::int32_t> &surrounding_cells, std::vector<std::int32_t> &count_child,
             std::vector<std::int32_t> &cell_child, std::shared_ptr<const graph::AdjacencyList<std::int32_t>> adj1,
             std::shared_ptr<const graph::AdjacencyList<std::int32_t>> adj2,
             std::shared_ptr<const graph::AdjacencyList<std::int32_t>> adj3,
             std::shared_ptr<const graph::AdjacencyList<std::int32_t>> adj4, std::unordered_set<std::int32_t> &fine_node_master,
             std::vector<std::int32_t> &coarse_interface_master, K tree_check)
{
   for (auto cell_id : surrounding_cells)
   {
      // PRINT("surrounding cell_id", cell_id);

      // at fine level
      size_t s = count_child[cell_id];
      size_t ds = count_child[cell_id + 1] - s;
      assert(ds == 1);
      auto fcell_id = cell_child[s];
      // PRINT("fcell_id", fcell_id);
      auto l = adj1->links(fcell_id);
      for (auto f : l)
      {
         // PRINT("face_id", f);
         auto v = adj2->links(f);
         // std::int32_t found = -1;
         for (auto n : v)
         {
            // PRINT("node", n);
            std::int32_t i;
            if (n < nbl)
               i = n;
            else
               i = idx[n];
            // PRINT("node", i);
            // PRINT("x", std::span<T>(&x[i*3],3));
            if (tree_check(&x[i * 3]))
            {
               fine_node_master.insert(n);
            }
         }
      }

      // at coarse level
      auto lc = adj3->links(cell_id);
      for (auto f : lc)
      {
         // PRINT("face_id", f);
         auto v = adj4->links(f);
         bool found = true;
         for (auto n : v)
         {
            // PRINT("node", n);
            std::int32_t i;
            if (n < nblc)
               i = n;
            else
               i = idxc[n];
            if (!tree_check(&xc[i * 3]))
            {
               found = false;
               break;
            }
         }
         if (found)
         {
            // PRINT("face is master",f);
            coarse_interface_master.push_back(f);
         }
      }
   }
}
template <typename T>
class Coord3DToGdimCoordPred
{
  public:
   Coord3DToGdimCoordPred(int gdim_) : gdim(gdim_), encoded(1), count(0)
   {
      assert(gdim_ <= 3 && gdim_ > 0);
      encoded = encoded << (1 + gdim_);
   }
   bool operator()(const T &x) const
   {
      switch (count | encoded)
      {
         case 4:
         case 8:
         case 16:
         {
            count = 1;
            return true;
         }
         case 5:
         {
            count = 2;
            return false;
         }
         case 9:
         case 17:
         {
            count = 2;
            return true;
         }
         case 6:
         case 10:
         {
            count = 0;
            return false;
         }
         case 18:
         {
            count = 0;
            return true;
         }
      };
   }
   int dim() const { return gdim; }

  private:
   const int gdim;
   mutable std::int8_t encoded, count;
};
template <typename T>
std::vector<T> Coord3DToGdimCoord(std::span<T> x, Coord3DToGdimCoordPred<T> &pred)
{
   std::size_t sx = x.size();
   assert(!(sx % 3));
   sx /= 3;
   std::vector<T> res;
   res.reserve(sx * pred.dim());
   std::ranges::copy(std::ranges::views::filter(x, pred), std::back_inserter(res));
   return res;
}
template <typename T, dolfinx::mesh::MarkerFn<T> W>
void topDownWeight(dolfinx::mesh::Mesh<T> &mesh, std::vector<std::int32_t> &support, int dim, W crit, std::uint8_t &control,
                   std::uint8_t level, std::vector<std::int32_t> &weights)
{
   spdlog::info("Compute accurately weight per cells related to scale jump constructed with enriched and crit");

   // ===============
   // isolate support
   // ===============
   spdlog::info("Support isolate as a submesh for weight");
   auto [mesh_support, cell_map, vertice_map, coord_map] =
       dolfinx::mesh::create_submesh(mesh, dim, std::span<std::int32_t>(support.data(), support.size()));

   // ==============
   // refine support
   // ==============
   spdlog::info("Refine support for weight");
   std::vector<std::int32_t> parent;
   dolfinx::refinement::Option opt = dolfinx::refinement::Option::parent_cell;

   // For first level always refine all element (see http://doi.org/10.1016/j.cma.2023.115914 page 12 end section 3.8)
   {
      auto topo = mesh_support.topology_mutable();
      topo->create_entities(1);
      auto idx = topo->index_map(1);
      auto nb_edges = idx->size_local() + idx->num_ghosts();
      std::vector<std::int32_t> all_edges(nb_edges);
      std::iota(all_edges.begin(), all_edges.end(), 0);

      auto refined_data = dolfinx::refinement::refine(mesh_support, std::span<std::int32_t>(all_edges.data(), nb_edges), opt);

      assert(std::get<1>(refined_data).has_value());

      //std::println("parent  cell 00 {}: {}", std::get<1>(refined_data)->size(), *std::get<1>(refined_data));

      // passe in coarse mesh numbering
      std::transform(std::execution::seq, std::get<1>(refined_data)->begin(), std::get<1>(refined_data)->end(),
                     std::get<1>(refined_data)->begin(), [&cell_map](std::int32_t x) { return cell_map[x]; });
      parent = std::move(*std::get<1>(refined_data));

      //std::println("parent  cell 0 control {} {}: {}", (int)control, parent.size(), parent);

      mesh_support = std::move(std::get<0>(refined_data));
   }

   // loop on remaining level
   while (++control < level)
   {
      auto edges = locate_entities(mesh_support, 1, crit);

      auto refined_data = dolfinx::refinement::refine(mesh_support, std::span<std::int32_t>(edges.data(), edges.size()), opt);

      assert(std::get<1>(refined_data).has_value());

      std::transform(std::execution::seq, std::get<1>(refined_data)->begin(), std::get<1>(refined_data)->end(),
                     std::get<1>(refined_data)->begin(), [&parent](std::int32_t x) { return parent[x]; });

      parent = std::move(*std::get<1>(refined_data));

      //std::println("parent  cell 1 control {} {}: {}", (int)control, parent.size(), parent);

      mesh_support = std::move(std::get<0>(refined_data));
   }

   // ========================
   // generated element weight
   // ========================
   // std::cout<<pid << "nb elements   " << nb_cells << std::endl;

   auto delta= 1 << dim;
   for (auto p : parent)
   {
      weights[p]+=delta;
   }

   return;
}
}  // namespace impl
template <typename T, dolfinx::mesh::MarkerFn<T> U, dolfinx::mesh::MarkerFn<T> W, typename Z>
twoscale::scaleJump<dolfinx::mesh::Mesh<T>> topDown(dolfinx::mesh::Mesh<T> &mesh, U enriched, W crit, std::uint8_t &control,
                                                    std::uint8_t level,  bool clustering_dual_graph ,bool accurate_weight,
                                                    dolfinx::mesh::MeshTags<Z> const *tags)
{
   spdlog::info("Compute scale jump for a given macro mesh with a geometrical function to identify enriched nodes");

   MPI_Comm comm = mesh.comm();
   auto pid = dolfinx::MPI::rank(comm);
   auto nbproc = dolfinx::MPI::size(comm);

   FILES(pid)
   // std::string no = "proc_" + std::to_string(pid) + "_output.txt";
   // freopen(no.c_str(), "w", stdout);

   // ========================================================
   //  Identify enriched node on given coarse distributed mesh
   // ========================================================
   auto enriched_nodes = locate_entities(mesh, 0, enriched);
   // std::cout<<pid<<"enriched_nodes   "<<enriched_nodes.size()<<" "; for (auto s : cnriched_nodes) std::cout << s << " "
   // ;std::cout<<std::endl;

   // Look if all nodes (local+ghost) are enriched 
   std::int8_t need_balance = (enriched_nodes.size() != mesh.topology()->index_map(0)->size_local() + mesh.topology()->index_map(0)->num_ghosts());
   MPI_Allreduce(MPI_IN_PLACE, &need_balance, 1, MPI_INT8_T, MPI_MAX, comm);
   // TODO
   // when all nodes are enriched this function is not optimal.
   // This need_balance should be used to simplify
   // For now it avoid rebalancing, considering that refinement will not perturbate the load balancing. In this senario it might
   // be true as enriching all nodes is most of the time associated with refining at least all domain and refining further is also
   // expect on all domain. It would be surprising that refining locally is intended in this case.
   if (!need_balance)
      std::cout << "Alternate method would make sense !!!" << std::endl;


   // ==============================================
   //  Load balance coarse mesh
   // ==============================================
   std::vector<std::int32_t> support;
   int dim = 0;
   if (dolfinx::MPI::size(comm) > 1 && need_balance > 0)
   {
      spdlog::info("Load balance macro mesh to force all proces to work on at least one macro element");
      // ==============================================
      // Gather all cells in supports of enriched nodes
      // ==============================================
      std::unordered_set<std::int32_t> u_support;
      auto topo = mesh.topology_mutable();
      dim = topo->dim();
      topo->create_connectivity(0, dim);
      auto adj = topo->connectivity(0, dim);
      // std::cout<<pid << "nb nodes " << adj->num_nodes() << std::endl;
      for (auto ei : enriched_nodes)
      {
         const auto li = adj->links(ei);
         u_support.insert(li.begin(), li.end());
      }
      support.reserve(u_support.size());
      support.insert(support.end(), u_support.begin(), u_support.end());

      // std::cout<<pid << "support   " << support.size() << " "; for (auto s : support) std::cout << s << " "; std::cout <<
      // std::endl;

      // element weight
      auto nb_cells = topo->index_map(dim)->size_local();
      std::cout<<pid << "nb elements   " << nb_cells << std::endl;
      std::vector<std::int32_t> weights;
      int nbw;



      bool agresive_weight=false;

      //std::cout << pid << "agresive_weight   " << agresive_weight << " clustering_dual_graph " << clustering_dual_graph << " accurate_weight " << accurate_weight << std::flush << std::endl;

      if (clustering_dual_graph)
      {
         weights.resize(nb_cells, 1);
         if (accurate_weight)
         {
            // Elements of support get refined at most 'level' time in area located by crit. When done weight per cell is
            // simply the number of child cells created by refinement.
            auto control_save = control;
            impl::topDownWeight(mesh, support, dim, crit, control, level, weights);
            control = control_save;
         }
         else
         {
            // =======================================================
            // Add element connected to extra enriched nodes to support
            // =======================================================
            // It must be added to force mixed patch to be has much as possible on the same proc.
            // If not taken into account depending on the way element out of the support are distributed we can have those patches
            // almost always distributed
            //
            // after some experiment the side effect is that in somme cases almost all surrounding cells are in only one proc !
            // impl::extendSupport(topo,adj,dim,support,u_support);

            // ==================================
            // generated agressive element weight
            // ==================================
            //  All element of extended support got a "huge" weight to force nice load balancing of this part
            auto sg = topo->index_map(dim)->size_global();
            std::int32_t mxw = std::max(1 << (dim * level), static_cast<std::int32_t>(std::lround(sg * 0.9)));
            for (auto s : support) weights[s] = mxw;
         }

         {
            // =======================================
            // load balance mesh with agressive weight
            // =======================================
#ifdef HAS_PARMETIS
            const auto partitioner = dolfinx::graph::parmetis::partitionerWithNodeWeight(weights, 1);
#else
#ifdef HAS_KAHIP
            const auto partitioner = dolfinx::graph::kahip::partitionerWithNodeWeight(weights);
#else
#error "PARMETIS or KAHIP required"
#endif
#endif
            auto cell_part = dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::none, partitioner);
            auto &geo = mesh.geometry();
            const int gdim = geo.dim();
            const auto &x = geo.x();
            const auto nblv = geo.index_map()->size_local();
            const auto locx = std::span<T>(x.begin(), nblv * 3);
            const std::vector<std::int32_t> &adjl = topo->connectivity(dim, 0)->array();
            std::vector<std::int64_t> cells(adjl.size());
            topo->index_map(0)->local_to_global(std::span<const std::int32_t>(adjl.data(), adjl.size()),
                                                std::span<std::int64_t>(cells.data(), cells.size()));
            impl::Coord3DToGdimCoordPred<T> pred(gdim);
            auto locxgdim = impl::Coord3DToGdimCoord<T>(locx, pred);
            auto coarse_mesh = dolfinx::mesh::create_mesh(comm, comm, cells, geo.cmap(), comm, locxgdim, {nblv, gdim}, cell_part);

            // ==========
            // reset mesh
            // ==========
            mesh = std::move(coarse_mesh);
            comm = mesh.comm();
            assert(pid == dolfinx::MPI::rank(comm));
            assert(nbproc == dolfinx::MPI::size(comm));
         }

         // ==============================================
         // Gather all cells in supports of enriched nodes
         // ==============================================
         u_support.clear();
         support.clear();
         auto topon = mesh.topology_mutable();
         topon->create_connectivity(0, dim);
         auto adjn = topon->connectivity(0, dim);
         auto new_enriched_nodes = locate_entities(mesh, 0, enriched);
         for (auto ei : new_enriched_nodes)
         {
            const auto li = adjn->links(ei);
            u_support.insert(li.begin(), li.end());
         }
         support.reserve(u_support.size());
         support.insert(support.end(), u_support.begin(), u_support.end());
         // if (!accurate_weight)
         // impl::extendSupport(topon,adjn,dim,support,u_support);

         //if(0)
         {
            // ============================================================
            // load balance mesh with clustered dual graph based on support
            // ============================================================
            const auto partitioner_fix = dolfinx::graph::partitionerClustering(u_support);
            auto cell_part = dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::none, partitioner_fix);
            auto &geo = mesh.geometry();
            const int gdim = geo.dim();
            const auto &x = geo.x();
            const auto nblv = geo.index_map()->size_local();
            const auto locx = std::span<T>(x.begin(), nblv * 3);
            const std::vector<std::int32_t> &adjl = topon->connectivity(dim, 0)->array();
            std::vector<std::int64_t> cells(adjl.size());
            topon->index_map(0)->local_to_global(std::span<const std::int32_t>(adjl.data(), adjl.size()),
                                                std::span<std::int64_t>(cells.data(), cells.size()));
            impl::Coord3DToGdimCoordPred<T> pred(gdim);
            auto locxgdim = impl::Coord3DToGdimCoord<T>(locx, pred);
            auto coarse_mesh = dolfinx::mesh::create_mesh(comm, comm, cells, geo.cmap(), comm, locxgdim, {nblv, gdim}, cell_part);

            // ==========
            // reset mesh
            // ==========
            mesh = std::move(coarse_mesh);
            comm = mesh.comm();
            assert(pid == dolfinx::MPI::rank(comm));
            assert(nbproc == dolfinx::MPI::size(comm));
         }
      }
      else
      {
         if (agresive_weight)
         {
            // ===================================
            // generated aggressive element weight
            // ===================================
            nbw = 1;
            weights.resize(nb_cells, 1);
            auto sg = topo->index_map(dim)->size_global();
            std::int32_t mxw = std::max(1 << (dim * level), static_cast<std::int32_t>(std::lround(sg * 0.9)));
            for (auto s : support) weights[s] = mxw;

            std::println("weights 1  {}: {}", nb_cells, std::span<std::int32_t>(weights.begin(), weights.begin() + nb_cells));
         }
         else
         {
            // ========================
            // generated element weight
            // ========================
            nbw = 2;
            weights.resize(2 * nb_cells, 1);
            // To avoid that remaining coarse mesh get placed only in few process a supplementary weight is set : 0
            // on support 1 on other part
            auto *pw = &weights[nb_cells];
            for (auto s : support) pw[s] = 0;
            if (accurate_weight)
            {
               // Elements of support get refined at most 'level' time in area located by crit. When done weight per cell is
               // simply the number of child cells created by refinement.
               auto control_save = control;
               impl::topDownWeight(mesh, support, dim, crit, control, level, weights);
               control = control_save;
            }
            else
            {
               // elements of support is supposed to be uniformly refined 'level' times. Thus an approximate weight is imposed to
               // all support cells based on number of level and on dimension.

               std::int32_t mxw = 1 << (dim * level);
               // std::int32_t mxw = (1<<dim)*level*level;
               // std::int32_t mxw = dim*level;
               // std::int32_t mxw = dim+level;
               for (auto s : support) weights[s] = mxw;
            }

            // std::println("weights 1  {}: {}", nb_cells, std::span<std::int32_t>(weights.begin(), weights.begin() + nb_cells));
            // std::println("weights 2  {}: {}", nb_cells, std::span<std::int32_t>(weights.begin() + nb_cells, weights.end()));
         }

         // =================
         // load balance mesh
         // =================
         // Durty !?!?! creating a new mesh from the given one is simple from an implementation point of view but looks quite
         // costly. Maybe there is a way to do things differently but it is  note obvious from source investigation.
#ifdef HAS_PARMETIS
         const auto partitioner = dolfinx::graph::parmetis::partitionerWithNodeWeight(weights, nbw);
#else
#ifdef HAS_KAHIP
         if (nbw>1)
         {
            // As a last resort, consider using mono weight optimization.
            // weights = 2*weight1+weight2
            // 2 : just to be sure that support being well dispatched compare to remaining cells
            // TODO check 2 is mandatory, maybe 1 is ok
            if (!pid)
            {
               std::cout << "Warning: use mono weight optimization for load balancing" << std::endl;
               std::cout << "   Use KAHIP but switching to PARMETIS will normally provides better balancing" << std::endl;
            }
            auto w1 = std::span<std::int32_t>(weights.begin(), weights.begin() + nb_cells);
            auto w2 = std::span<std::int32_t>(weights.begin() + nb_cells, weights.end());
            std::ranges::transform(w1, w2, weights.begin(),
                                   [](std::int32_t &i, std::int32_t &j) -> std::int32_t { return 2 * i + j; });
            weights.resize(nb_cells);
         }
         const auto partitioner = dolfinx::graph::kahip::partitionerWithNodeWeight(weights);
#else
#error "PARMETIS or KAHIP required"
#endif
#endif
         auto cell_part = dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::none, partitioner);
         // auto cell_part = dolfinx::mesh::create_cell_partitioner();
         auto &geo = mesh.geometry();
         const int gdim = geo.dim();
         const auto &x = geo.x();
         const auto nblv = geo.index_map()->size_local();
         const auto locx = std::span<T>(x.begin(), nblv * 3);
         const std::vector<std::int32_t> &adjl = topo->connectivity(dim, 0)->array();
         std::vector<std::int64_t> cells(adjl.size());
         // std::cout<<pid << "adjl   " << adjl.size() << ": "; for (auto s : adjl) std::cout << s << " "; std::cout << std::endl;
         topo->index_map(0)->local_to_global(std::span<const std::int32_t>(adjl.data(), adjl.size()),
                                             std::span<std::int64_t>(cells.data(), cells.size()));
         // std::cout<<pid << "cells   " << cells.size() << ": "; for (auto s : cells) std::cout << s << " "; std::cout <<
         // std::endl; std::cout<<pid << "x   " << x.size() << ": "; for (auto s : x) std::cout << s << " "; std::cout <<
         // std::endl; PRINT("locx",locx);
         impl::Coord3DToGdimCoordPred<T> pred(gdim);
         // when used many times filter view retain first iterator which put the mess in internal count: thus here coords has to
         // be copyed. A filter view cannot be passed to create_mesh that need to use it maybe more then once. And also view do
         // not provides value_type. Maybe to avoid this durty copy another c++ 23/26 view may be used, warped in something
         // providing value_type.
         auto locxgdim = impl::Coord3DToGdimCoord<T>(locx, pred);
         // PRINT("locxgdim",locxgdim);
         // if (tags)
         //   auto coarse_mesh = dolfinx::mesh::create_mesh(comm, comm, cells, geo.cmap(), comm, locxgdim, {nblv, gdim},
         //   cell_part,*tag);
         // else
         auto coarse_mesh = dolfinx::mesh::create_mesh(comm, comm, cells, geo.cmap(), comm, locxgdim, {nblv, gdim}, cell_part);

         // ==================================
         // generated edge weight for support
         // ==================================
         if (0 && agresive_weight)
         {
            // By generating hight edge weight in between element of the support only locally we force these element to remain in
            // those proc. The node weight can be ignored. Passing Parmetis with those weight lead to rebalance all element except
            // support
            // ==============================================
            // Gather all cells in supports of enriched nodes
            // ==============================================
            u_support.clear();
            support.clear();
            auto topoi = coarse_mesh.topology_mutable();
            dim = topoi->dim();
            auto nb_cellsi = topoi->index_map(dim)->size_local();
            topoi->create_connectivity(0, dim);
            auto adj = topoi->connectivity(0, dim);
            std::cout << pid << "nb nodes " << adj->num_nodes() << std::endl;
            auto new_enriched_nodes = locate_entities(coarse_mesh, 0, enriched);
            for (auto ei : new_enriched_nodes)
            {
               const auto li = adj->links(ei);
               u_support.insert(li.begin(), li.end());
            }
            support.reserve(u_support.size());
            support.insert(support.end(), u_support.begin(), u_support.end());
            // ===============================================
            // compute node weight for edge weight computation
            // ===============================================
            weights.resize(nb_cellsi);
            std::fill(weights.begin(), weights.end(), 1);
            std::int32_t mxw = 1 << (dim * level);
            // auto sg = topoi->index_map(dim)->size_global();
            // std::int32_t mxw = std::max(1 << (dim * level), static_cast<std::int32_t>(std::lround(sg * 0.9)));
            for (auto s : support) weights[s] = mxw;
            std::println("weights 1b  {}: {}", nb_cellsi, std::span<std::int32_t>(weights.begin(), weights.begin() + nb_cellsi));
            // =================
            // load balance mesh
            // =================
#ifdef HAS_KAHIP
         const auto partitioner = dolfinx::graph::kahip::partitioner_with_weight(weights);
#else
#ifdef HAS_PARMETIS
         const auto partitioner = dolfinx::graph::parmetis::partitionerWithEdgeWeight(weights);
#else
#error "PARMETIS or KAHIP required"
#endif
#endif
         auto cell_part = dolfinx::mesh::create_cell_partitioner(dolfinx::mesh::GhostMode::none, partitioner);
         // auto cell_part = dolfinx::mesh::create_cell_partitioner();
         auto &geo = coarse_mesh.geometry();
         const int gdim = geo.dim();
         const auto &x = geo.x();
         const auto nblv = geo.index_map()->size_local();
         const auto locx = std::span<T>(x.begin(), nblv * 3);
         const std::vector<std::int32_t> &adjl = topoi->connectivity(dim, 0)->array();
         std::vector<std::int64_t> cells(adjl.size());
         topoi->index_map(0)->local_to_global(std::span<const std::int32_t>(adjl.data(), adjl.size()),
                                              std::span<std::int64_t>(cells.data(), cells.size()));
         impl::Coord3DToGdimCoordPred<T> pred(gdim);
         auto locxgdim = impl::Coord3DToGdimCoord<T>(locx, pred);
         auto new_coarse_mesh = dolfinx::mesh::create_mesh(comm, comm, cells, geo.cmap(), comm, locxgdim, {nblv, gdim}, cell_part);
         coarse_mesh = std::move(new_coarse_mesh);
         }
         /*
         std::vector<std::int32_t> cellid(nb_cells);
         std::iota(cellid.begin(), cellid.end(), 0);
         auto on = dolfinx::mesh::cell_normals(mesh, dim, std::span<const std::int32_t>(cellid.data(), cellid.size()));
         auto nnb_cells = coarse_mesh.topology()->index_map(dim)->size_local();
         cellid.resize(nnb_cells);
         std::iota(cellid.begin(), cellid.end(), 0);
         auto nn = dolfinx::mesh::cell_normals(coarse_mesh, dim, std::span<const std::int32_t>(cellid.data(), cellid.size()));
         std::println("normal old {} new {}",on, nn);
         */

         // ==========
         // reset mesh
         // ==========
         mesh = std::move(coarse_mesh);
         comm = mesh.comm();
         assert(pid == dolfinx::MPI::rank(comm));
         assert(nbproc == dolfinx::MPI::size(comm));
      }

      // ========================================================
      // enriched nodes needs to be found in new distributed mesh
      // ========================================================
      auto new_enriched_nodes = locate_entities(mesh, 0, enriched);
      enriched_nodes = std::move(new_enriched_nodes);
      std::cout << pid << "enriched_nodes   " << enriched_nodes.size() << " ";
      for (auto s : enriched_nodes) std::cout << s << " ";
      std::cout << std::endl;
   }

   if (0)
   {
      auto &geoo = mesh.geometry();
      const int gdim = geoo.dim();
      const auto &xo = geoo.x();
      PRINT("gdim", gdim);
      PRINT("xo size", xo.size());
      auto nbno = xo.size() / 3;
      auto nbnol = geoo.index_map()->size_local();
      auto idxmapgo = geoo.index_map();

      size_t i = 0;
      for (; i < nbnol * 3; i += 3)
      {
         std::cout << pid << " 0 ball node " << i / 3 << ": ";
         for (size_t k = 0; k < gdim; ++k) std::cout << xo[i + k] << " ";
         std::cout << std::endl;
      }
      for (; i < nbno * 3; i += 3)
      {
         std::cout << pid << " 0 balr node " << i / 3 << ": ";
         for (size_t k = 0; k < gdim; ++k) std::cout << xo[i + k] << " ";
         std::cout << std::endl;
      }

      std::vector<std::int32_t> loc_idx(nbno);
      std::iota(loc_idx.begin(), loc_idx.end(), 0);
      std::vector<std::int64_t> glob_idx(nbno);

      idxmapgo->local_to_global(std::span<const std::int32_t>(loc_idx.data(), loc_idx.size()),
                                std::span<std::int64_t>(glob_idx.data(), glob_idx.size()));
      PRINT(" 0 g index g ", glob_idx);

      auto topoo = mesh.topology();
      auto idxmap0o = topoo->index_map(0);

      idxmap0o->local_to_global(std::span<const std::int32_t>(loc_idx.data(), loc_idx.size()),
                                std::span<std::int64_t>(glob_idx.data(), glob_idx.size()));
      PRINT(" 0 g index 0 ", glob_idx);

      auto adj = topoo->connectivity(dim, 0);
      std::cout << pid << adj->str();
      const std::vector<std::int32_t> &adjn = adj->array();
      PRINT(" 0 adj   ", adjn);

      auto ghostg = idxmapgo->ghosts();
      PRINT(" 0 ghost  g ", ghostg);

      auto ghost0 = idxmap0o->ghosts();
      PRINT(" 0 ghost  0 ", ghost0);

      mesh.topology_mutable()->create_entities(1);
      auto idxmap1o = topoo->index_map(1);
      auto ghost1 = idxmap1o->ghosts();
      PRINT(" 0 ghost  1 ", ghost1);
      auto adj1 = topoo->connectivity(dim, 1);
      std::cout << pid << adj1->str();
      const std::vector<std::int32_t> &adjn1 = adj1->array();
      glob_idx.resize(adjn1.size());
      idxmap1o->local_to_global(std::span<const std::int32_t>(adjn1.data(), adjn1.size()),
                                std::span<std::int64_t>(glob_idx.data(), glob_idx.size()));
      PRINT(" 0 g index 1 ", glob_idx);
   }

   // ===========================================
   // generate coarse cell to 'face' connectivity
   // ===========================================
   int dimm1, nb_face_per_cell;
   {
      auto topo = mesh.topology_mutable();
      dim = topo->dim();
      dimm1 = dim - 1;
      if (dimm1)
      {
         topo->create_connectivity(dim, dimm1);
         topo->create_connectivity(dimm1, dim);
      }
      assert(mesh::is_simplex(topo->cell_type()));
      nb_face_per_cell = dim + 1;
   }

   // std::cout<<pid<<"nb cell   "<< mesh.topology()->index_map(dim)->size_local()<<std::endl;
   // std::cout<<pid<<"nb local nodes   "<< mesh.topology()->index_map(0)->size_local()<<std::endl;
   // std::cout<<pid<<"nb ghost nodes   "<< mesh.topology()->index_map(0)->num_ghosts()<<std::endl;
   // std::cout<<pid<<"nb nodes   "<<mesh.topology()->index_map(0)->size_local()+
   // mesh.topology()->index_map(0)->num_ghosts()<<std::endl;

   // ======================================
   // collect coarse topological information
   // ======================================
   auto topoc = mesh.topology();
   auto idxmap0c = topoc->index_map(0);
   auto adjcv_c = topoc->connectivity(dim, 0);
   auto nbncl = idxmap0c->size_local();
   auto nbnc = nbncl + idxmap0c->num_ghosts();
   dolfinx::common::Scatterer coarse_scatter_topo(*idxmap0c, 1);
   auto ghost_topo_0c = idxmap0c->ghosts();

   // ================================================================
   // create support/extra_enriched_nodes/surrounding_cells containers
   // ================================================================
   support.clear();
   std::unordered_set<std::int32_t> u_macro_nodes;
   std::vector<std::int32_t> extra_enriched_nodes;
   std::vector<std::int32_t> surrounding_cells;
   std::int32_t nbcct = 0;
   std::unordered_set<std::int32_t> u_support;
   {
      // ==============================================
      // Gather all cells in supports of enriched nodes
      // ==============================================
      spdlog::info("Gather enriched node support");
      auto topo = mesh.topology_mutable();
      topo->create_connectivity(0, dim);
      auto adj = topo->connectivity(0, dim);
      // std::cout<<pid << "nb nodes " << adj->num_nodes() << std::endl;
      for (auto ei : enriched_nodes)
      {
         const auto li = adj->links(ei);
         u_support.insert(li.begin(), li.end());
      }
      support.reserve(u_support.size());
      support.insert(support.end(), u_support.begin(), u_support.end());

      // PRINT("support",support);

      // ==============================
      // looks for extra enriched nodes
      // ==============================
      spdlog::info("Looking for extra enriched nodes");
      auto adj_cell = topo->connectivity(dim, 0);
      nbcct = adj_cell->num_nodes();  // no ghost for now
      std::unordered_set<std::int32_t> tmp;
      for (auto si : support)
      {
         const auto li = adj_cell->links(si);
         tmp.insert(li.begin(), li.end());
         u_macro_nodes.insert(li.begin(), li.end());
      }
      for (auto ei : enriched_nodes) tmp.erase(ei);

      // extra enriched node need to be exchanged as cell adjacency do no pass thru process boundary.
      // create mark container
      std::vector<std::int8_t> mark(nbnc, -1);
      // mark all
      for (auto t : tmp) mark[t] = 1;

      // PRINT("extra mark 0",std::ranges::count(mark,1));
      // PRINT("extra mark 0",mark);

      // local marked by comm from ghost
      coarse_scatter_topo.scatter_rev(std::span<std::int8_t>(mark.begin(), mark.begin() + nbncl),
                                      std::span<const std::int8_t>(mark.begin() + nbncl, mark.end()),
                                      [](const std::int8_t &i, const std::int8_t &j) {
                                         if (j > -1)
                                            return j;
                                         else
                                            return i;
                                      });

      // PRINT("extra mark 1",std::ranges::count(mark,1));
      // PRINT("extra mark 1",mark);

      // ghost marked by comm from local
      coarse_scatter_topo.scatter_fwd(std::span<const std::int8_t>(mark.begin(), mark.begin() + nbncl),
                                      std::span<std::int8_t>(mark.begin() + nbncl, mark.end()));

      // PRINT("extra mark 2",std::ranges::count(mark,1));
      // PRINT("extra mark 2",mark);

      extra_enriched_nodes.reserve(std::ranges::count(mark, 1));
      std::int32_t k = -1;
      for (auto m : mark)
      {
         ++k;
         if (m > -1) extra_enriched_nodes.push_back(k);
      }

      // PRINT("extra",extra_enriched_nodes);

      if (0)
      {
         std::vector<std::int32_t> tmp(extra_enriched_nodes);
         std::vector<std::int64_t> tmpg(extra_enriched_nodes.size());
         idxmap0c->local_to_global(std::span<const std::int32_t>(tmp.data(), tmp.size()),
                                   std::span<std::int64_t>(tmpg.data(), tmpg.size()));
         auto idxmapg = mesh.geometry().index_map();
         idxmapg->global_to_local(tmpg, std::span<std::int32_t>(tmp));
         const auto &xc = mesh.geometry().x();
         for (auto ex : tmp)
         {
            PRINT("ex", ex);
            PRINT("x", std::span<T>(&xc[ex * 3], mesh.geometry().dim()));
         }
      }
      // ============================================
      // Gather all macro cells connected to supports
      // ============================================
      tmp.clear();
      spdlog::info("Gather macro cells connected to support");
      for (auto ei : extra_enriched_nodes)
      {
         const auto li = adj->links(ei);
         for (auto cell : li)
         {
            if (u_support.find(cell) == u_support.end()) tmp.insert(cell);
         }
      }

      surrounding_cells.reserve(tmp.size());
      surrounding_cells.insert(surrounding_cells.end(), tmp.begin(), tmp.end());

      // std::cout<<pid << "surrounding_cells   " << tmp.size() << " "; for (auto s : tmp) std::cout << s << " "; std::cout <<
      // std::endl;
   }

   // ===============
   // isolate support
   // ===============
   spdlog::info("Support isolate as a submesh");
#if 0
   dolfinx::mesh::Mesh<T> mesh_support(mesh);
#else
   auto [mesh_support, cell_map, vertice_map, coord_map] =
       dolfinx::mesh::create_submesh(mesh, dim, std::span<std::int32_t>(support.data(), support.size()));

   if (cell_map.size() == 0)
   {
      std::cerr << "Two scale suppose that support of enriched nodes is well balanced across process." << std::endl;
      std::cerr << "Apparently process " << pid << " do not hold any cell of the support and thus do not participate."
                << std::endl;
      std::cerr << "Rebalancing coarse mesh did not succed. Reducing number of process is now the only solution." << std::endl;

      MPI_Abort(comm, -1);
   }
   // std::cout<<pid<<"Cell map    "<<cell_map.size()<<" "; for (auto s : cell_map) std::cout << s << " " ;std::cout<<std::endl;
   // std::cout<<pid<<"Vertice map "<<vertice_map.size()<<" "; for (auto s : vertice_map) std::cout << s << " "
   // ;std::cout<<std::endl; std::cout<<pid<<"Coord map   "<<coord_map.size()<<" "; for (auto s : coord_map) std::cout << s << " "
   // ;std::cout<<std::endl;

   // =======================
   // renumber enriched nodes
   // =======================
   // std::cout<<pid<<"enriched_nodes sub   "<<enriched_nodes.size()<<" "; for (auto s : enriched_nodes) std::cout << s << " "
   // ;std::cout<<std::endl;

#endif

   // ==============
   // refine support
   // ==============
   spdlog::info("Refine support");
   std::vector<std::int32_t> parent;
   std::vector<std::int8_t> face_parent;
   dolfinx::refinement::Option opt;
   if (dim > 1)
      opt = dolfinx::refinement::Option::parent_cell_and_facet;
   else
      opt = dolfinx::refinement::Option::parent_cell;
   // For first level always refine all element (see http://doi.org/10.1016/j.cma.2023.115914 page 12 end section 3.8)
   {
      auto topo = mesh_support.topology_mutable();
      topo->create_entities(1);
      auto idx = topo->index_map(1);
      auto nb_edges = idx->size_local() + idx->num_ghosts();
      std::vector<std::int32_t> all_edges(nb_edges);
      std::iota(all_edges.begin(), all_edges.end(), 0);
      // auto all_edges = locate_entities(mesh_support, 1, crit);
      // nb_edges = all_edges.size();

      // std::cout<<pid << "all_edges  " << all_edges.size() << " "; for (auto s : all_edges) std::cout << s << " "; std::cout <<
      // std::endl;

      auto refined_data = dolfinx::refinement::refine(mesh_support, std::span<std::int32_t>(all_edges.data(), nb_edges), opt);

      assert(std::get<1>(refined_data).has_value());

      // std::cout<<pid << "parent  cell 00     " << std::get<1>(refined_data)->size() << ": "; for (auto s :
      // *std::get<1>(refined_data)) std::cout << s << " "; std::cout << std::endl;

      // passe in coarse mesh numbering
      std::transform(std::execution::seq, std::get<1>(refined_data)->begin(), std::get<1>(refined_data)->end(),
                     std::get<1>(refined_data)->begin(), [&cell_map](std::int32_t x) { return cell_map[x]; });
      parent = std::move(*std::get<1>(refined_data));

      // std::cout<<pid << "parent  cell 0     " << parent.size() << ": "; for (auto s : parent) std::cout << s << " "; std::cout
      // << std::endl;

      if (dimm1)
      {
         // in 2D/3D face are edge/face
         assert(std::get<2>(refined_data).has_value());
         // simple move to keep memory footprint low by using int8 for face_parent: ok as parent give the right cell to do
         // connection with macro face
         face_parent = std::move(*std::get<2>(refined_data));
         // std::cout<<pid << "parent  face 0      " << face_parent.size() << ": "; for (auto s : face_parent) std::cout << (int)s
         // << " "; std::cout << std::endl;
      }
      else
      {
         // in 1D face are node
         // nothing given by refine thus it has do be done by hand here
         // TODO
         throw -1;
      }

      mesh_support = std::move(std::get<0>(refined_data));
   }

   // loop on remaining level
   while (++control < level)
   {
      auto edges = locate_entities(mesh_support, 1, crit);

      auto refined_data = dolfinx::refinement::refine(mesh_support, std::span<std::int32_t>(edges.data(), edges.size()), opt);

      assert(std::get<1>(refined_data).has_value());

      if (dimm1)
      {
         assert(std::get<2>(refined_data).has_value());
         // std::cout<<pid << "parent  face " << ((int)control-1)*2+1<<" control "<<std::get<2>(refined_data)->size() << ": "; for
         // (auto s : *std::get<2>(refined_data)) std::cout << (int)s << " "; std::cout << std::endl;

         auto &pc = *std::get<1>(refined_data);
         auto &pf = *std::get<2>(refined_data);
         for (std::int32_t c = 0, nbc = pf.size() / nb_face_per_cell; c < nbc; ++c)
         {
            std::span<const std::int8_t> faceup(face_parent.data() + pc[c] * nb_face_per_cell, nb_face_per_cell);
            std::span<std::int8_t> facedown(pf.data() + c * nb_face_per_cell, nb_face_per_cell);
            for (auto &f : facedown)
            {
               if (f > -1) f = faceup[f];
            }
         }
         // std::cout<<pid << "parent  face " << ((int)control-1)*2+2<<" control "<<std::get<2>(refined_data)->size() << ": "; for
         // (auto s : *std::get<2>(refined_data)) std::cout << (int)s << " "; std::cout << std::endl;

         face_parent = std::move(*std::get<2>(refined_data));
      }

      std::transform(std::execution::seq, std::get<1>(refined_data)->begin(), std::get<1>(refined_data)->end(),
                     std::get<1>(refined_data)->begin(), [&parent](std::int32_t x) { return parent[x]; });

      parent = std::move(*std::get<1>(refined_data));

      // std::cout<<pid << "parent  cell 1 control "<<(int)control<<" " << parent.size() << ": "; for (auto s : parent) std::cout
      // << s << " "; std::cout << std::endl;

      mesh_support = std::move(std::get<0>(refined_data));
   }

   spdlog::info("Connect refined support into coarse");
   // =========================
   // collect nodes information
   // =========================
   auto &geoc = mesh.geometry();
   const auto &xc = geoc.x();
   const int gdim = geoc.dim();
   assert(gdim == dim);
   auto idxmapgc = geoc.index_map();
   assert(idxmapgc->size_local() == nbncl);
   assert(nbnc == xc.size() / 3);
   auto topof = mesh_support.topology();
   auto &geof = mesh_support.geometry();
   const auto &xf = geof.x();
   assert(gdim == geof.dim());
   auto nbnf = xf.size() / 3;
   auto nbnfl = geof.index_map()->size_local();
   auto idxmapgf = geof.index_map();
   auto idxmap0f = topof->index_map(0);
   auto ghost_geof = idxmapgf->ghosts();
   assert(idxmap0f->size_local() == nbnfl);
   assert(idxmap0f->size_local() + idxmap0f->num_ghosts() == nbnf);

   std::unordered_map<std::int32_t, std::int32_t> macro_node_child;

   // topo to geom ghost index
   std::vector<std::int32_t> ghostc_geo_index_in_topo_order(ghost_topo_0c.size());
   idxmapgc->global_to_local(ghost_topo_0c, ghostc_geo_index_in_topo_order);
   // PRINT("ghostc_geo_index_in_topo_order",ghostc_geo_index_in_topo_order);
   // PRINT("ghost_topo_0c",ghost_topo_0c);
   auto in_geoc_index = [&nbncl, ghostc_geo_index_in_topo_order](const std::int32_t &idx) {
      if (idx < nbncl)
         return idx;
      else
         return ghostc_geo_index_in_topo_order[idx - nbncl];
   };
   // geom to topo  ghost index
   std::vector<std::int32_t> ghostf_topo_index_in_geo_order(ghost_geof.size());
   idxmap0f->global_to_local(ghost_geof, ghostf_topo_index_in_geo_order);
   auto in_topof_index = [&nbnfl, ghostf_topo_index_in_geo_order](const std::int32_t &idx) {
      if (idx < nbnfl)
         return idx;
      else
         return ghostf_topo_index_in_geo_order[idx - nbnfl];
   };

   if (0)
   {
      mesh_support.topology_mutable()->create_entities(dimm1);
      mesh_support.topology_mutable()->create_connectivity(dim, dimm1);
      const std::vector<std::int32_t> &face_f = topof->connectivity(dim, dimm1)->array();
      std::cout << pid << "face_f  " << face_f.size() << ": ";
      for (auto s : face_f) std::cout << s << " ";
      std::cout << std::endl;
      auto idxmap1n = topof->index_map(dimm1);
      std::vector<std::int64_t> facesf(face_f.size());
      idxmap1n->local_to_global(std::span<const std::int32_t>(face_f.data(), face_f.size()),
                                std::span<std::int64_t>(facesf.data(), facesf.size()));
      std::cout << pid << "facesf  " << facesf.size() << ": ";
      for (auto s : facesf) std::cout << s << " ";
      std::cout << std::endl;
   }
   if (0)
   {
      size_t i = 0;
      for (; i < nbncl * 3; i += 3)
         std::cout << pid << "macrol node " << i / 3 << ": " << xc[i] << " " << xc[i + 1] << " " << xc[i + 2] << std::endl;
      for (; i < nbnc * 3; i += 3)
         std::cout << pid << "macror node " << i / 3 << ": " << xc[i] << " " << xc[i + 1] << " " << xc[i + 2] << std::endl;
   }

   // ====================================
   // Identify macro nodes in refined mesh
   // ====================================
   class equal_key
   {
      T eps = 1.e-10;

     public:
      bool operator()(const T &x1, const T &x2) const
      {
         if (x1 < x2 + eps && x1 > x2 - eps)
            return true;
         else
            return false;
      }
   };
   // to reduce memory impact optimize geo-tree dimension
   switch (gdim)
   {
      case 1:
      {
         // generate a 1d-tree of macro coordinates of the support
         // no need to store nor search for y,z-component which are considered to be null.
         // embedded 1D mesh inside a 2D/3D is not considered here
         assert(xc[1] == 0.);
         assert(xc[2] == 0.);
         std::unordered_map<T, std::int32_t, std::hash<T>, equal_key> cgeo_coord_tree;
         for (auto macro_node : u_macro_nodes)
         {
            std::int32_t h = in_geoc_index(macro_node);
            std::int32_t i = h * 3;
            auto &xk = cgeo_coord_tree[xc[i]];
            // xk = macro_node;
            xk = h;
         }
         // found in refined mesh all nodes corresponding to macro nodes of the support
         for (size_t i = 0; i < nbnf; ++i)
         {
            size_t k = i * 3U;
            // if(i<nbnfl) std::cout<<pid<<"microl  node "<<i<<": "<<xf[k]<<" "<<xf[k+1]<<" "<<xf[k+2]<<std::endl;
            // else std::cout<<pid<<"micror  node "<<i<<": "<<xf[k]<<" "<<xf[k+1]<<" "<<xf[k+2]<<std::endl;
            auto itfx = cgeo_coord_tree.find(xf[k]);
            if (itfx != cgeo_coord_tree.end())
            {
               if (macro_node_child.find(itfx->second) != macro_node_child.end())
               {
                  std::cerr << "One fine scale node " << i << " is to be connected with coarse node " << itfx->second
                            << " which is already connected with " << macro_node_child[itfx->second] << std::endl
                            << "Abort in process " << pid << std::endl;

                  MPI_Abort(comm, -1);
               }
               // store fine topologic index
               macro_node_child[itfx->second] = in_topof_index(i);
               // std::cout << pid << "micro node " << i << ": " << xf[k] << " " << xf[k + 1] << " " << xf[k + 2] << " found macro
               // " << itfx->second << std::endl;
            }
         }
         break;
      }
      case 2:
      {
         // generate a 2d-tree of macro coordinates of the support
         // no need to store nor search for z-component which is considered to be null.
         // embedded 2D mesh inside a 3D is not considered here
         assert(xc[2] == 0.);
         std::unordered_map<T, std::unordered_map<T, std::int32_t, std::hash<T>, equal_key>, std::hash<T>, equal_key>
             cgeo_coord_tree;
         for (auto macro_node : u_macro_nodes)
         {
            std::int32_t h = in_geoc_index(macro_node);
            std::int32_t i = h * 3;
            auto &xk = cgeo_coord_tree[xc[i]];
            auto &yk = xk[xc[i + 1]];
            // yk = macro_node;
            yk = h;
         }

         // found in refined mesh all nodes corresponding to macro nodes of the support
         for (size_t i = 0; i < nbnf; ++i)
         {
            size_t k = i * 3U;
            // if(i<nbnfl) std::cout<<pid<<"microl  node "<<i<<": "<<xf[k]<<" "<<xf[k+1]<<" "<<xf[k+2]<<std::endl;
            // else std::cout<<pid<<"micror  node "<<i<<": "<<xf[k]<<" "<<xf[k+1]<<" "<<xf[k+2]<<std::endl;
            auto itfx = cgeo_coord_tree.find(xf[k]);
            if (itfx != cgeo_coord_tree.end())
            {
               auto itfy = itfx->second.find(xf[k + 1]);
               if (itfy != itfx->second.end())
               {
                  if (macro_node_child.find(itfy->second) != macro_node_child.end())
                  {
                     std::cerr << "One fine scale node " << i << " is to be connected with coarse node " << itfy->second
                               << " which is already connected with " << macro_node_child[itfy->second] << std::endl
                               << "Abort in process " << pid << std::endl;

                     MPI_Abort(comm, -1);
                  }
                  // store fine topologic index
                  macro_node_child[itfy->second] = in_topof_index(i);
                  // std::cout << pid << "micro node " << i << " " << in_topof_index(i) << ": " << xf[k] << " " << xf[k + 1] << "
                  // " << xf[k + 2] << " found macro " << itfy->second << std::endl;
               }
            }
         }
         break;
      }
      case 3:
      {
         // generate a 3d-tree of macro coordinates of the support
         std::unordered_map<
             T, std::unordered_map<T, std::unordered_map<T, std::int32_t, std::hash<T>, equal_key>, std::hash<T>, equal_key>,
             std::hash<T>, equal_key>
             cgeo_coord_tree;
         for (auto macro_node : u_macro_nodes)
         {
            std::int32_t h = in_geoc_index(macro_node);
            std::int32_t i = h * 3;
            auto &xk = cgeo_coord_tree[xc[i]];
            auto &yk = xk[xc[i + 1]];
            auto &zk = yk[xc[i + 2]];
            zk = h;
         }
         // found in refined mesh all nodes corresponding to macro nodes of the support
         for (size_t i = 0; i < nbnf; ++i)
         {
            size_t k = i * 3U;
            // if(i<nbnfl) std::cout<<pid<<"microl  node "<<i<<": "<<xf[k]<<" "<<xf[k+1]<<" "<<xf[k+2]<<std::endl;
            // else std::cout<<pid<<"micror  node "<<i<<": "<<xf[k]<<" "<<xf[k+1]<<" "<<xf[k+2]<<std::endl;
            auto itfx = cgeo_coord_tree.find(xf[k]);
            if (itfx != cgeo_coord_tree.end())
            {
               auto itfy = itfx->second.find(xf[k + 1]);
               if (itfy != itfx->second.end())
               {
                  auto itfz = itfy->second.find(xf[k + 2]);
                  if (itfz != itfy->second.end())
                  {
                     if (macro_node_child.find(itfz->second) != macro_node_child.end())
                     {
                        std::cerr << "One fine scale node " << i << " is to be connected with coarse node " << itfz->second
                                  << " which is already connected with " << macro_node_child[itfz->second] << std::endl
                                  << "Abort in process " << pid << std::endl;

                        MPI_Abort(comm, -1);
                     }
                     // store fine topologic index
                     macro_node_child[itfz->second] = in_topof_index(i);
                     // std::cout << pid << "micro node " << i << ": " << xf[k] << " " << xf[k + 1] << " " << xf[k + 2] << " found
                     // macro " << itfz->second << std::endl;
                  }
               }
            }
         }
         break;
      }
      default:
      {
         std::cerr << "Dimension " << dim << " not meaningful." << std::endl;

         MPI_Abort(comm, -1);
      }
   };

   // for (auto p : macro_node_child) std::cout << pid << "macro_node_child[ " << p.first << "]=" << p.second << std::endl;

   dolfinx::common::Scatterer coarse_scatter(*idxmapgc, 1);
   std::int32_t nbncl_not_in_f;
   std::int64_t nbnnl;
   std::vector<T> xn;
   {
      // =================================================
      // collect local macro node identified in micro mesh
      // =================================================
      // note: Somme macro node are only seen to be connect to fine mesh in
      // process that refine macro element that is connected to them. But in an
      // other process that do not hold refined macro element these macro node
      // looks independent. And if by bad luck they are local to this other process
      // it can be retained as new node: this results in presence of this node as
      // a fine node in one process and a macro node in an other ....
      // The only way to avoid such case is to collect remote identification
      //
      // create a marker container to identify macro node linked to a child
      std::vector<std::int8_t> mark_c(nbnc, -1);
      for (auto p : macro_node_child) mark_c[p.first] = 1;

      // PRINT("mark_c 0 ",mark_c);

      // gather remote identified node localy as a mark
      coarse_scatter.scatter_rev(std::span<std::int8_t>(mark_c.begin(), mark_c.begin() + nbncl),
                                 std::span<const std::int8_t>(mark_c.begin() + nbncl, mark_c.end()),
                                 [](const std::int8_t &i, const std::int8_t &j) {
                                    if (j > -1)
                                       return j;
                                    else
                                       return i;
                                 });

      // PRINT("mark_c 1 ",mark_c);

      // now all local coarse node identified with a fine node on this proc or remotely are marked thus remaining value of -1
      // imply a macro node not in the support
      nbncl_not_in_f = std::count(mark_c.begin(), mark_c.begin() + nbncl, -1);

      std::cout << pid << "nbncl          " << nbncl << std::endl;
      std::cout << pid << "nbncl_not_in_f " << nbncl_not_in_f << std::endl;

      // ===========================================
      // Generate new coordinates of the merged mesh
      // ===========================================
      nbnnl = nbnfl + nbncl_not_in_f;
      // copy local nodes of the fine mesh first
      xn.reserve(nbnnl * gdim);
      {
         impl::Coord3DToGdimCoordPred<T> pred(gdim);
         // auto filt = xf | std::views::take(nbnfl * 3) | std::views::filter(pred);
         // xn.insert(xn.end(), filt.begin(), filt.end());
         std::int32_t i = 0;
         std::int32_t il = nbnfl * 3;
         for (auto xi : xf)
         {
            if (i < il)
            {
               if (pred(xi)) xn.push_back(xi);
            }
            else
               break;
            ++i;
         }
      }
      // PRINT("xf   ",xf);
      // PRINT("xn fl",xn);
      // copy local nodes of the coarse mesh skipping coarse node identified in fine mesh
      for (std::int32_t i = 0; i < nbncl; ++i)
      {
         if (mark_c[i] < 0)
         {
            size_t j = i * 3;
            xn.insert(xn.end(), &xc[j], &xc[j + gdim]);
         }
      }
      assert(xn.size() == nbnnl * gdim);
      // PRINT("xn   ",xn);
   }

   // ==============
   // Compute offset
   // ==============
   std::int64_t offset_n = 0;
   MPI_Exscan(&nbnnl, &offset_n, 1, MPI_INT64_T, MPI_SUM, comm);

   // std::cout << pid << "nbnnl " << nbnnl << " nbnf " << nbnf << " nbnfl " << nbnfl << " nbncl " << nbncl << " nbncl_not_in_f "
   // << nbncl_not_in_f << " offset_n "<<offset_n<< std::endl;

   // ========================
   // collect cell information
   // ========================
   const std::vector<std::int32_t> &cell_f = topof->connectivity(dim, 0)->array();
   std::int32_t nbaf = cell_f.size();
   std::int32_t nbcs = support.size();
   std::vector<std::int32_t> cell_c;
   {
      cell_c.reserve((nbcct - nbcs) * dim);
      std::int32_t nbcf = parent.size();
      parent.resize(nbcf + nbcct - nbcs);
      std::int32_t k = nbcf - 1;
      for (std::int32_t cell_id = 0; cell_id < nbcct; ++cell_id)
      {
         if (u_support.find(cell_id) == u_support.end())
         {
            const auto li = adjcv_c->links(cell_id);
            cell_c.insert(cell_c.end(), li.begin(), li.end());
            parent[++k] = cell_id;
         }
      }
   }

   // PRINT("parent  cell 2 f+c", parent);
   // PRINT("cell c", cell_c);

   std::int32_t nbac = cell_c.size();
   std::int32_t nban = nbaf + nbac;

   // ==========================
   // create global idx for fine
   // ==========================
   // create local idx
   std::vector<std::int32_t> loc_idx(nbnfl);
   std::iota(loc_idx.begin(), loc_idx.end(), 0);
   // create new global idx mapping container
   std::vector<std::int64_t> glob_idx_f(nbnf);

   // for local index it is simply a question of offsetting with new offset
   std::transform(loc_idx.begin(), loc_idx.end(), glob_idx_f.begin(), [&offset_n](std::int32_t x) { return x + offset_n; });
   // for ghost index simply scatter new global index to ghost location
   // using topological ghost indexing to fit to cell node idexing
   {
      dolfinx::common::Scatterer fine_scatter(*idxmap0f, 1);
      fine_scatter.scatter_fwd(std::span<const std::int64_t>(glob_idx_f.begin(), glob_idx_f.begin() + nbnfl),
                               std::span<std::int64_t>(glob_idx_f.begin() + nbnfl, glob_idx_f.end()));

      // PRINT("g index f after offset", glob_idx_f);
   }

   // ============================
   // create global idx for coarse
   // ============================
   // init global coarse index to -1
   std::vector<std::int64_t> glob_idx_c(nbnc, -1);
   // set macro node identified in fine with new global fine idx using glob_idx_f
   for (auto p : macro_node_child) glob_idx_c[p.first] = glob_idx_f[p.second];

   // PRINT("g index c fgi ",glob_idx_c);

   // gather remote identified node localy with correct new idx
   coarse_scatter.scatter_rev(std::span<std::int64_t>(glob_idx_c.begin(), glob_idx_c.begin() + nbncl),
                              std::span<const std::int64_t>(glob_idx_c.begin() + nbncl, glob_idx_c.end()),
                              [](const std::int64_t &i, const std::int64_t &j) {
                                 // PRINT("i", i);
                                 // PRINT("j", j);
                                 if (j > -1)  // remote give info
                                 {
                                    if (i < 0)  // local note identified => remote do
                                       return j;
                                    else
                                    {
                                       assert(i == j);
                                       return j;  // i or j are the same
                                    }
                                 }
                                 else
                                    return i;  // i what ever it is remain the same
                              });

   // PRINT("g index c2 ", glob_idx_c);

   // for local index 3 situation as to be handled:
   //    * the local node is an enriched node and as such is eliminated by cell selection. No special treatement as anyway it is
   //    treate in next item.
   //    * the local node is having a fine node equivalent: fine node idx is already correct so no action
   //    * the local node is outside support: it must be re-indexed (packing) and offset by offset_n + nbnfl
   std::int64_t k = -1;
   std::int64_t loffset = offset_n + nbnfl;
   for (std::int32_t i = 0; i < nbncl; ++i)
   {
      auto &v = glob_idx_c[i];
      if (v < 0) v = ++k + loffset;
   }
   // PRINT("g index c3", glob_idx_c);
   assert(k <= nbncl_not_in_f);
   // for ghost index simply scatter new global index to ghost location with topological ghost index to fit to cell node idexing
   coarse_scatter_topo.scatter_fwd(std::span<const std::int64_t>(glob_idx_c.begin(), glob_idx_c.begin() + nbncl),
                                   std::span<std::int64_t>(glob_idx_c.begin() + nbncl, glob_idx_c.end()));

   // PRINT("g index c4", glob_idx_c);

   // ====================================
   // Generate new cell of the merged mesh
   // ====================================
   std::vector<std::int64_t> cells(nban);
   std::ranges::transform(cell_f, cells.begin(), [&glob_idx_f](std::int32_t x) { return glob_idx_f[x]; });
   std::ranges::transform(cell_c, cells.begin() + nbaf, [&glob_idx_c](std::int32_t x) { return glob_idx_c[x]; });
   // std::vector<std::int64_t> cellst(cells.begin()+nbaf,cells.end());
   // std::vector<std::int64_t> cellst(cells.begin(),cells.begin()+nbaf);
   // std::cout<<pid << "cells test   " << cellst.size() << ": "; for (auto s : cellst) std::cout << s << " "; std::cout <<
   // std::endl; PRINT("cell_f   ",cell_f); PRINT("cells (f)",std::span<std::int64_t>(cells.data(),nbaf)); PRINT("cell_c ",cell_c);
   // PRINT("cells (c)",std::span<std::int64_t>(cells.begin()+nbaf,cells.end()));

   // ========================
   // Generate new merged mesh
   // ========================
   // dolfinx::mesh::CellPartitionFunction nofunc;
   // mesh_support = std::move(dolfinx::mesh::create_mesh(comm, comm, cells, geoc.cmap(), comm, xn, {nbnnl, 3}, nofunc));
   auto [merged_mesh, remap] = dolfinx::mesh::create_mesh(comm, comm, cells, geoc.cmap(), comm, xn, {nbnnl, gdim});

   // auto [yo, yoremap] = dolfinx::mesh::create_mesh(comm, comm, cellst, geoc.cmap(), comm, xn, {nbnnl, gdim});

   // std::cout << pid << "remap   " << remap.size() << ": "; for (auto s : remap) std::cout << s << " "; std::cout << std::endl;

   // ============
   // connectivity
   // ============
   merged_mesh.topology_mutable()->create_entities(dimm1);
   merged_mesh.topology_mutable()->create_connectivity(0, dim);
   std::int32_t nbcn = parent.size();
   // ============
   // parent/child
   // ============
   assert(nbcn == remap.size());
   std::vector<std::int32_t> count_child(nbcct + 1, 0);
   std::vector<std::int32_t> cell_child;
   std::vector<std::int32_t> count_face_child;
   std::vector<std::int32_t> face_child;
   std::int32_t max_nb_face_child = 0;
   std::int32_t nbfc;
   {
      std::vector<std::int32_t> macro_child_index(nbcct + 1, -1);
#ifdef ALL_CHILD
      std::int32_t nbcwd = nbcct;
      std::iota(macro_child_index.begin(), macro_child_index.end(), 0);
#else
      std::int32_t nbcwd = surrounding_cells.size() + support.size();
      {
         std::vector<std::int32_t> indices(nbcwd);
         std::copy(support.begin(), support.end(), indices.begin());
         std::copy(surrounding_cells.begin(), surrounding_cells.end(), indices.begin() + support.size());
         std::ranges::sort(indices);
         {
            std::int32_t k = -1;
            std::ranges::for_each(indices, [&macro_child_index, &k](const std::int32_t &x) { macro_child_index[x] = ++k; });
         }
      }
#endif

      // treat face first as algo use parent in mesh_support order
      if (dimm1)
      {
         std::vector<std::int32_t> loc_face_idx;
         std::int32_t nbfwd;
         std::vector<std::int32_t> macro_face_child_index;
         {
            // from adjency generate local face idx
            auto adj = topoc->connectivity(dim, dimm1);
            auto idxmapfc = topoc->index_map(dimm1);
            loc_face_idx = adj->array();

            nbfc = idxmapfc->size_local() + idxmapfc->num_ghosts();
            macro_face_child_index.resize(nbfc + 1, -1);
            auto set_idk = [&nb_face_per_cell, &loc_face_idx, &macro_face_child_index](const std::int32_t cell_id) {
               std::span<const std::int32_t> faces(loc_face_idx.data() + cell_id * nb_face_per_cell, nb_face_per_cell);
               for (auto f : faces)
               {
                  if (macro_face_child_index[f] < 0) macro_face_child_index[f] = 1;
               }
            };
            for (auto cell_id : support) set_idk(cell_id);
            // surrounding and non support are note described in face_parent and thus can't be integrated in this process:
            // for (auto cell_id : surrounding_cells) set_idk(cell_id);  won't do anything a priori (to check if no side effect)
            // A face with no child is its own child in this context
            nbfwd = -1;
            for (auto &v : macro_face_child_index)
               if (v > 0) v = ++nbfwd;
            ++nbfwd;

            // std::cout << pid << " macro_face_child_index " << nbfwd << " " << macro_face_child_index.size() << ": "; for (auto
            // s : macro_face_child_index) std::cout << s << " "; std::cout << std::endl;
         }
         count_face_child.resize(nbfc + 1, 0);
         std::unordered_map<std::int32_t, std::int32_t> unique_fine_face_on_macro;
         // count face child
         for (std::int32_t c = 0, nbc = face_parent.size() / nb_face_per_cell; c < nbc; ++c)
         {
            std::int32_t pc = parent[c];
            std::span<const std::int32_t> faceup(loc_face_idx.data() + pc * nb_face_per_cell, nb_face_per_cell);
            std::span<std::int8_t> facedown(face_parent.data() + c * nb_face_per_cell, nb_face_per_cell);
            for (auto f : facedown)
            {
               if (f > -1)
               {
                  // std::cout<<pid<<"elem "<<c<<" relate to macro face "<<(int)f<<" of macro element "<<pc<<" with macro face idx
                  // "<<faceup[f]<<std::endl;
                  std::int32_t mfi = faceup[f];
                  bool inc = false;
                  auto itf = unique_fine_face_on_macro.find(mfi);
                  if (itf != unique_fine_face_on_macro.end())
                  {
                     if (itf->second == pc) inc = true;
                  }
                  else
                  {
                     unique_fine_face_on_macro[mfi] = pc;
                     inc = true;
                  }
                  if (inc)
                  {
                     std::int32_t idx_covering_face = macro_face_child_index[mfi];
                     if (idx_covering_face > -1) ++count_face_child[idx_covering_face];
                  }
               }
            }
         }

         // PRINT("count_face_child", count_face_child);

         // count into offset
         for (std::int32_t face_id = 0; face_id < nbfwd; ++face_id)
         {
            auto &next = count_face_child[face_id + 1];
            max_nb_face_child = std::max(max_nb_face_child, next);
            next += count_face_child[face_id];
         }
         // PRINT("count_face_child", count_face_child);
         for (std::int32_t face_id = nbfwd; face_id > 0; --face_id) count_face_child[face_id] = count_face_child[face_id - 1];
         count_face_child[0] = 0;

         // PRINT("count_face_child", count_face_child);
         // PRINT("max_nb_face_child", max_nb_face_child);

         // get face for merged_mesh
         merged_mesh.topology_mutable()->create_connectivity(dim, dimm1);
         auto topon = merged_mesh.topology();
         const std::vector<std::int32_t> &face_n = topon->connectivity(dim, dimm1)->array();

         // fill face_child
         face_child.resize(count_face_child[nbfwd]);
         for (std::int32_t c = 0, nbc = face_parent.size() / nb_face_per_cell; c < nbc; ++c)
         {
            std::int32_t pc = parent[c];
            std::span<const std::int32_t> faceup(loc_face_idx.data() + pc * nb_face_per_cell, nb_face_per_cell);
            std::span<std::int8_t> facedown(face_parent.data() + c * nb_face_per_cell, nb_face_per_cell);
            std::int8_t k = -1;
            for (auto f : facedown)
            {
               ++k;
               if (f > -1)
               {
                  std::int32_t mfi = faceup[f];
                  bool inc = false;
                  auto itf = unique_fine_face_on_macro.find(mfi);
                  if (itf != unique_fine_face_on_macro.end())
                  {
                     if (itf->second == pc) inc = true;
                  }
                  else
                  {
                     unique_fine_face_on_macro[mfi] = pc;
                     inc = true;
                  }
                  if (inc)
                  {
                     std::int32_t idx_covering_face = macro_face_child_index[mfi];
                     if (idx_covering_face > -1)
                     {
                        face_child[count_face_child[idx_covering_face]] = face_n[remap[c] * nb_face_per_cell + k];
                        ++count_face_child[idx_covering_face];
                     }
                  }
               }
            }
         }

         // reset offset and expand to give zero child for macro face outside support
         std::int32_t current_offset = 0;
         for (auto &mci : macro_face_child_index | std::views::take(nbfc))
         {
            std::int32_t idx_covering_cell = mci;
            mci = current_offset;
            if (idx_covering_cell > -1) current_offset = count_face_child[idx_covering_cell];
         }
         macro_face_child_index[nbfc] = current_offset;
         count_face_child = std::move(macro_face_child_index);

         // std::cout<<pid << "count_face_child  " << count_face_child.size() << ": "; for (auto s : count_face_child) std::cout
         // << s << " "; std::cout << std::endl;
      }

      // update parent in merged_mesh order and count cell child
      std::vector<std::int32_t> nparent(nbcn);
      for (std::int32_t cell_id = 0; cell_id < nbcn; ++cell_id)
      {
         nparent[remap[cell_id]] = parent[cell_id];
         // TODO to simplify if ALL_CHILD
         std::int32_t idx_covering_cell = macro_child_index[parent[cell_id]];
         if (idx_covering_cell > -1) ++count_child[idx_covering_cell];
      }
      parent = std::move(nparent);

      // count into offset
      for (std::int32_t cell_id = 0; cell_id < nbcwd; ++cell_id) count_child[cell_id + 1] += count_child[cell_id];
      for (std::int32_t cell_id = nbcwd; cell_id > 0; --cell_id) count_child[cell_id] = count_child[cell_id - 1];
      count_child[0] = 0;

      // PRINT("count_child",count_child);

      // fill cell_child
      cell_child.resize(count_child[nbcwd]);
      for (std::int32_t cell_id = 0; cell_id < nbcn; ++cell_id)
      {
         // TODO to simplify if ALL_CHILD
         std::int32_t idx_covering_cell = macro_child_index[parent[cell_id]];
         if (idx_covering_cell > -1)
         {
            cell_child[count_child[idx_covering_cell]] = cell_id;
            ++count_child[idx_covering_cell];
         }
      }
      // reset offset and expand to give zero child for macro element outside extanded support
      // TODO to simplify if ALL_CHILD
      std::int32_t current_offset = 0;
      for (auto &mci : macro_child_index | std::views::take(nbcct))
      {
         std::int32_t idx_covering_cell = mci;
         mci = current_offset;
         if (idx_covering_cell > -1) current_offset = count_child[idx_covering_cell];
      }
      macro_child_index[nbcct] = current_offset;
      count_child = std::move(macro_child_index);
   }

   if (0)
   {
      std::cout << pid << "parent  cell 3 mer " << parent.size() << ": ";
      for (auto s : parent) std::cout << s << " ";
      std::cout << std::endl;
      std::cout << pid << "cell_child  " << cell_child.size() << ": ";
      for (auto s : cell_child) std::cout << s << " ";
      std::cout << std::endl;
      for (std::int32_t cell_id = 0; cell_id < nbcct; ++cell_id)
      {
         std::cout << pid << "child  " << cell_id << " " << count_child[cell_id + 1] - count_child[cell_id] << ": ";
         for (int k = count_child[cell_id]; k < count_child[cell_id + 1]; ++k) std::cout << cell_child[k] << " ";
         std::cout << std::endl;
      }
   }

   if (0)
   {
      std::cout << pid << "cells   " << cells.size() << ": ";
      for (auto s : cells) std::cout << s << " ";
      std::cout << std::endl;
      auto topon = merged_mesh.topology();
      const std::vector<std::int32_t> &cell_n = topon->connectivity(dim, 0)->array();
      std::vector<std::int64_t> cellsn(cell_n.size());
      auto idxmap0n = topon->index_map(0);
      idxmap0n->local_to_global(std::span<const std::int32_t>(cell_n.data(), cell_n.size()),
                                std::span<std::int64_t>(cellsn.data(), cellsn.size()));
      std::cout << pid << "cellsn  " << cellsn.size() << ": ";
      for (auto s : cellsn) std::cout << s << " ";
      std::cout << std::endl;
      merged_mesh.topology_mutable()->create_entities(dimm1);
      merged_mesh.topology_mutable()->create_connectivity(dim, dimm1);
      const std::vector<std::int32_t> &face_n = topon->connectivity(dim, dimm1)->array();
      std::cout << pid << "face_n  " << face_n.size() << ": ";
      for (auto s : face_n) std::cout << s << " ";
      std::cout << std::endl;
      auto idxmap1n = topon->index_map(dimm1);
      std::vector<std::int64_t> facesn(face_n.size());
      idxmap1n->local_to_global(std::span<const std::int32_t>(face_n.data(), face_n.size()),
                                std::span<std::int64_t>(facesn.data(), facesn.size()));
      std::cout << pid << "facesn  " << facesn.size() << ": ";
      for (auto s : facesn) std::cout << s << " ";
      std::cout << std::endl;
   }

   // ==============================================================================================
   // identify in coarse mesh the set of master 'face' that need to be used for mpc setting, if any.
   // identify in merged mesh the set of master node that need to be used for mpc setting, if any.
   // ==============================================================================================
   // only meaningful in 2D/3D
   std::unordered_set<std::int32_t> fine_node_master;
   std::vector<std::int32_t> coarse_interface_master;
   if (dimm1)
   {
      // prepare data for impl::find_in
      auto topon = merged_mesh.topology();
      auto geon = merged_mesh.geometry();
      auto idxmapgn = geon.index_map();
      const auto &x_n = geon.x();
      // PRINT("x_n",x_n);
      auto adj1 = topon->connectivity(dim, dimm1);
      auto adj2 = topon->connectivity(dimm1, 0);
      auto adj3 = topoc->connectivity(dim, dimm1);
      auto adj4 = topoc->connectivity(dimm1, 0);
      auto idxmap0n = topon->index_map(0);
      auto ghostn_t = idxmap0n->ghosts();
      // PRINT1("ghostn",ghostn_t)
      std::vector<std::int32_t> geon_loc_idx(ghostn_t.size(), 0);
      std::int32_t nbnnnl = idxmapgn->size_local();
      assert(nbnnnl == idxmap0n->size_local());
      // PRINT0("nbnnnl",nbnnnl)
      int32_t *png = geon_loc_idx.data() - nbnnnl;
      idxmapgn->global_to_local(ghostn_t, std::span<std::int32_t>(geon_loc_idx));
      // PRINT1("geon_loc_idx",geon_loc_idx)

      // create geo-tree made of extra_enriched_nodes (macro nodes)
      auto ghost_t = idxmap0c->ghosts();
      std::vector<std::int32_t> geo_loc_idx(ghost_t.size(), 0);
      int32_t *pg = geo_loc_idx.data() - nbncl;
      idxmapgc->global_to_local(ghost_t, std::span<std::int32_t>(geo_loc_idx));
      // PRINT1("geo_loc_idx",geo_loc_idx)
      // to reduce memory impact optimize geo-tree dimension
      switch (gdim)
      {
         case 2:
         {
            // generate a 2d-tree of macro coordinates of the support
            // no need to store nor search for z-component which is considered to be null.
            // embedded 2D mesh inside a 3D is not considered here
            assert(xc[2] == 0.);
            std::unordered_map<T, std::unordered_set<T, std::hash<T>, equal_key>, std::hash<T>, equal_key> cgeo_coord_tree;
            for (auto macro_node : extra_enriched_nodes)
            {
               std::int32_t i;
               if (macro_node < nbncl)
                  i = macro_node * 3;
               else
                  i = pg[macro_node] * 3;
               auto &xk = cgeo_coord_tree[xc[i]];
               xk.insert(xc[i + 1]);
            }
            if (0)
            {
               for (auto x : cgeo_coord_tree)
               {
                  PRINT("=========\nx", x.first);
                  for (auto y : x.second)
                  {
                     PRINT("y", y);
                  }
               }
            }
            impl::find_in(x_n, png, nbnnnl, xc, pg, nbncl, surrounding_cells, count_child, cell_child, adj1, adj2, adj3, adj4,
                          fine_node_master, coarse_interface_master, [&cgeo_coord_tree](const T *geo) -> bool {
                             auto itfx = cgeo_coord_tree.find(geo[0]);
                             if (itfx != cgeo_coord_tree.end())
                             {
                                auto itfy = itfx->second.find(geo[1]);
                                return (itfy != itfx->second.end());
                             }
                             else
                                return false;
                          });
            break;
         }
         case 3:
         {
            // generate a 3d-tree of macro coordinates of the support
            std::unordered_map<T, std::unordered_map<T, std::unordered_set<T, std::hash<T>, equal_key>, std::hash<T>, equal_key>,
                               std::hash<T>, equal_key>
                cgeo_coord_tree;
            for (auto macro_node : extra_enriched_nodes)
            {
               std::int32_t i;
               if (macro_node < nbncl)
                  i = macro_node * 3;
               else
                  i = pg[macro_node] * 3;
               auto &xk = cgeo_coord_tree[xc[i]];
               auto &yk = xk[xc[i + 1]];
               yk.insert(xc[i + 2]);
            }
            if (0)
            {
               for (auto x : cgeo_coord_tree)
               {
                  PRINT("=========\nx", x.first);
                  for (auto y : x.second)
                  {
                     PRINT("===\ny", y.first);
                     for (auto z : y.second)
                     {
                        PRINT("z", z);
                     }
                  }
               }
            }
            impl::find_in(x_n, png, nbnnnl, xc, pg, nbncl, surrounding_cells, count_child, cell_child, adj1, adj2, adj3, adj4,
                          fine_node_master, coarse_interface_master, [&cgeo_coord_tree](const T *geo) -> bool {
                             auto itfx = cgeo_coord_tree.find(geo[0]);
                             if (itfx != cgeo_coord_tree.end())
                             {
                                auto itfy = itfx->second.find(geo[1]);
                                if (itfy != itfx->second.end())
                                {
                                   auto itfz = itfy->second.find(geo[2]);
                                   return (itfz != itfy->second.end());
                                }
                                else
                                   return false;
                             }
                             else
                                return false;
                          });
            break;
         }
         default:
         {
            std::cerr << "Dimension " << dim << " not meaningful." << std::endl;

            MPI_Abort(comm, -1);
         }
      };
      // PRINT("fine_node_master", fine_node_master);
      // PRINT("coarse_interface_master", coarse_interface_master);
      // assert(fine_interface_master.size()==coarse_interface_master.size());

      // At this point communication is required to complete sets:
      // Some master faces can remotely be identified but locally not as their surrounding cell are not present in this process
      // Only done at coarse level as fine level do not have ghost master 'face' when at process boundary (because there is no
      // cell connected to this 'face' in remote process apparently no ghost 'face' is created even thought in this process
      // there is one)
      // TODO TODO TODO
      // should use MPI3 comunication as master faces are certainnely a small subset of faces that can be exchanged in between
      // processes It would reduce the amount of comunication but there will be any way at least 2 exchange to treate dicovered
      // first and new after.
      // TODO TODO TODO
      {
         auto idxmap1c = topoc->index_map(dimm1);
         std::int32_t nbfcl = idxmap1c->size_local();
         std::int32_t nbfc = nbfcl + idxmap1c->num_ghosts();
         auto remote_face = idxmap1c->index_to_dest_ranks();
         std::vector<std::int8_t> exchange(nbfc, -1);

         // Before exchanging filter out master faces not connected to support locally (only easy identification)
         // and mark with number of support cell attached to face
         auto adj5 = topoc->connectivity(dimm1, dim);
         std::ranges::remove_if(coarse_interface_master, [&u_support, &adj5, &remote_face, &exchange](const std::int32_t &idf) {
            auto cells_on_face = adj5->links(idf);

            // PRINT("idf", idf);
            // PRINT("cells_on_face",cells_on_face);
            // PRINT("remote_face.num_links(idf)",remote_face.num_links(idf));

            // 0 cell connected to this cell is not supposed to exist
            // check as used in else below
            assert(cells_on_face.size() > 0);
            // Investigate only face connected to 2 cells owned by this proc
            if (cells_on_face.size() > 1)
            {
               // no cell in the support: eliminate as in fact it is just a face connected to master node but not a master
               // face (possible in 3D)
               if (std::ranges::count_if(cells_on_face, [&u_support](const std::int32_t &idc) {
                      // PRINT("idc ", idc);
                      // PRINT("u_support ", (u_support.find(idc) != u_support.end()));
                      return (u_support.find(idc) != u_support.end());
                   }) < 1)
                  return true;
               // otherwise mark it for creation after exchange
               else
                  exchange[idf] = 1;
            }
            // only one cell
            // checked during exchange
            // if face on domain boundary in between exchange test will treat this case
            else
            {
               exchange[idf] = (u_support.find(cells_on_face[0]) != u_support.end());
            }

            return false;  // keep
         });

         // PRINT("exchange 1 count",std::ranges::count(exchange,1));
         // PRINT("exchange 1 count",std::ranges::count(exchange,0));
         // PRINT("exchange 1",exchange);

         // ghost to local
         dolfinx::common::Scatterer coarse_face_scatterer(*idxmap1c, 1);
         coarse_face_scatterer.scatter_rev(std::span<std::int8_t>(exchange.begin(), exchange.begin() + nbfcl),
                                           std::span<const std::int8_t>(exchange.begin() + nbfcl, exchange.end()),
                                           [](const std::int8_t &i, const std::int8_t &j) -> std::int8_t {
                                              if (j > -1)
                                              {
                                                 if (i > -1)
                                                    return i + j;
                                                 else
                                                    return j;
                                              }
                                              else
                                                 return i;
                                           });

         // PRINT("exchange 2 count",std::ranges::count(exchange,1));
         // PRINT("exchange 2 count",std::ranges::count(exchange,0));
         // PRINT("exchange 2",exchange);

         // for all local marked face with 0 check connect to a support face in this proc
         // treat face on domain boundary as well (this kind of face are always local so filtering with nbfcl is ok)
         for (std::int32_t f = 0; f < nbfc; ++f)
         {
            if (f < nbfcl && (!exchange[f]))
            {
               auto cells_on_face = adj5->links(f);
               assert(cells_on_face.size() == 1);
               exchange[f] += (u_support.find(cells_on_face[0]) != u_support.end());
            }
         }

         // PRINT("exchange 3 count",std::ranges::count(exchange,1));
         // PRINT("exchange 3 count",std::ranges::count(exchange,0));
         // PRINT("exchange 3",exchange);

         // local to ghost
         // all local with 1 are ok. local with 0 not modified by ghost remain indeterminated
         coarse_face_scatterer.scatter_fwd(std::span<const std::int8_t>(exchange.begin(), exchange.begin() + nbfcl),
                                           std::span<std::int8_t>(exchange.begin() + nbfcl, exchange.end()));

         // PRINT("exchange 4 count",std::ranges::count(exchange,1));
         // PRINT("exchange 4 count",std::ranges::count(exchange,0));
         // PRINT("exchange 4", exchange);

         // For ghost face it is possible that owner face was only connected to a surrounding cell and in this proc it was not
         // seen as a master faces. In this case the code is zero but connected cell has to be tested.
         for (std::int32_t f = 0; f < nbfc; ++f)
         {
            if (!(f < nbfcl))
            {
               if (!exchange[f])
               {
                  auto cells_on_face = adj5->links(f);
                  assert(cells_on_face.size() == 1);
                  exchange[f] += (u_support.find(cells_on_face[0]) != u_support.end());
               }
            }
         }

         // PRINT("exchange 5 count",std::ranges::count(exchange,1));
         // PRINT("exchange 5 count", std::ranges::count(exchange, 0));
         // PRINT("exchange 5",exchange);

         // ghost to local again as previous step can have change ghost and thus mus be propagated to owner
         coarse_face_scatterer.scatter_rev(std::span<std::int8_t>(exchange.begin(), exchange.begin() + nbfcl),
                                           std::span<const std::int8_t>(exchange.begin() + nbfcl, exchange.end()),
                                           [](const std::int8_t &i, const std::int8_t &j) -> std::int8_t {
                                              if (j > -1)
                                                 return j;
                                              else
                                                 return i;
                                           });

         // PRINT("exchange 6 count",std::ranges::count(exchange,1));
         // PRINT("exchange 6 count",std::ranges::count(exchange,0));
         // PRINT("exchange 6",exchange);

         // reshape coarse_master
         // std::unordered_set<std::int32_t> cim(coarse_interface_master.begin(), coarse_interface_master.end());
         // std::int32_t k = coarse_interface_master.size();
         coarse_interface_master.resize(std::ranges::count(exchange, 1));
         std::int32_t k = 0;
         for (std::int32_t f = 0; f < nbfc; ++f)
            if (exchange[f] > 0) coarse_interface_master[k++] = f;
      }
      // Some master node can remotely be identified but locally not or the inverse.
      // exchange
      {
         dolfinx::common::Scatterer fine_scatterer(*idxmap0n, 1);
         std::vector<std::int8_t> mark(nbnnnl + ghostn_t.size(), -1);
         for (auto v : fine_node_master) mark[v] = 1;
         // mix remote and local
         fine_scatterer.scatter_rev(std::span<std::int8_t>(mark.begin(), mark.begin() + nbnnnl),
                                    std::span<const std::int8_t>(mark.begin() + nbnnnl, mark.end()),
                                    [](const std::int8_t &i, const std::int8_t &j) {
                                       if (j > -1)
                                          return j;
                                       else
                                          return i;
                                    });
         // impose local mixed to ghost
         fine_scatterer.scatter_fwd(std::span<const std::int8_t>(mark.begin(), mark.begin() + nbnnnl),
                                    std::span<std::int8_t>(mark.begin() + nbnnnl, mark.end()));
         // complete fine_node_master
         std::int32_t k = 0;
         for (auto m : mark)
         {
            if (m > -1) fine_node_master.insert(k);
            ++k;
         }
      }

#if 0
  std::span src = map->src();
    std::span dest = map->dest();
    MPI_Dist_graph_create_adjacent(
        map->comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
        dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm[d]);

    // Number and values to send and receive
    const int num_indices = global[d].size();
    std::vector<int> size_recv;
    size_recv.reserve(1); // ensure data is not a nullptr
    size_recv.resize(src.size());
    MPI_Neighbor_allgather(&num_indices, 1, MPI_INT, size_recv.data(), 1,
                           MPI_INT, comm[d]);

#endif
   }
   // PRINT("fine_node_master",fine_node_master);
   // PRINT("coarse_interface_master",coarse_interface_master);

   // ========================================================================
   // generate child for surrounding faces (needed to identify patch boundary)
   // ========================================================================
   std::vector<std::int32_t> count_sface_child;
   std::vector<std::int32_t> sface_child;
   {
      // prepare data for loop
      std::int32_t nb_nodes_per_face = dim;
      std::int32_t nb_coord_per_face = nb_nodes_per_face * gdim;
      auto topon = merged_mesh.topology();
      auto idxmap0n = topon->index_map(0);
      auto geon = merged_mesh.geometry();
      const auto &x_n = geon.x();
      auto idxmapgn = geon.index_map();
      auto ghostn = idxmap0n->ghosts();
      std::vector<std::int32_t> geon_loc_idx(ghostn.size());
      std::int32_t nbnnnl = idxmapgn->size_local();
      assert(nbnnnl == idxmap0n->size_local());
      int32_t *png = geon_loc_idx.data() - nbnnnl;
      idxmapgn->global_to_local(ghostn, std::span<std::int32_t>(geon_loc_idx));
      auto adjcf_f = topon->connectivity(dim, dimm1);
      auto adjfv_f = topon->connectivity(dimm1, 0);

      auto adjcf_c = topoc->connectivity(dim, dimm1);
      auto adjfv_c = topoc->connectivity(dimm1, 0);
      std::vector<T> coarse_coord;
      std::vector<T> fine_coord;
      fine_coord.reserve(nb_face_per_cell * nb_coord_per_face);
      coarse_coord.reserve(nb_coord_per_face);

      equal_key is_same_coord;
      std::vector<bool> not_matching(nb_face_per_cell, true);
      std::vector<std::int32_t> fine_coord_idx(nb_face_per_cell, 0);
      // Predicate to filter coarse face
      auto pred = [&count_sface_child, &sface_child](const std::int32_t &f) -> bool {
         return (sface_child[count_sface_child[f]] < 0);
      };

      // container to store count of child for identified surrounding faces. No needs to really count them we now that there is
      // only one child face in this context.
      count_sface_child.resize(nbfc + 1, 0);
      auto *ps = &count_sface_child[1];
      for (auto cell_id : surrounding_cells)
      {
         auto lc = adjcf_c->links(cell_id);
         for (auto f : lc) ps[f] = 1;
      }
      for (std::int32_t face_id = 1, endf = nbfc + 1; face_id < endf; ++face_id)
         count_sface_child[face_id] += count_sface_child[face_id - 1];
      count_sface_child[0] = 0;
      sface_child.resize(count_sface_child[nbfc], -1);

      // loop on surrounding cells
      for (auto cell_id : surrounding_cells)
      {
         // reset
         fine_coord.clear();
         std::fill(not_matching.begin(), not_matching.end(), true);

         // get child faces list
         size_t s = count_child[cell_id];
         size_t ds = count_child[cell_id + 1] - s;
         assert(ds == 1);
         auto fcell_id = cell_child[s];
         auto lf = adjcf_f->links(fcell_id);

         // get coarse faces list
         auto lc = adjcf_c->links(cell_id);

         // fill not_matching so that only fine face not already associated with a coarse face be considered in the process
         int k = 0;
         for (auto f : lc)
         {
            auto ff = sface_child[count_sface_child[f]];
            if (ff > -1)
            {
               auto it = std::find(lf.begin(), lf.end(), ff);
               assert(it != lf.end());
               not_matching[it - lf.begin()] = false;
               ++k;
            }
         }

         // if all coarse face are identified nothing to do => pass to next cell
         if (k == nb_face_per_cell) continue;

         // get fine coordinates (not already identified)
         k = 0;
         for (auto f : lf)
         {
            fine_coord_idx[k] = fine_coord.size();
            if (not_matching[k])
            {
               auto v = adjfv_f->links(f);
               for (auto n : v)
               {
                  std::int32_t i;
                  if (n < nbnnnl)
                     i = n;
                  else
                     i = png[n];
                  std::span<T> point_coord(&x_n[i * 3], gdim);
                  fine_coord.insert(fine_coord.end(), point_coord.begin(), point_coord.end());
               }
            }
            ++k;
         }

         // loop on coarse faces list (not already identified)
         for (auto f : lc | std::views::filter(pred))
         {
            // get coarse coordinates
            coarse_coord.clear();
            {
               auto v = adjfv_c->links(f);
               for (auto n : v)
               {
                  std::int32_t i = in_geoc_index(n);
                  std::span<T> point_coord(&xc[i * 3], gdim);
                  coarse_coord.insert(coarse_coord.end(), point_coord.begin(), point_coord.end());
               }
            }

            // loop on fine face to identify matching faces
            int kf = 0;
            for (auto ff : lf)
            {
               // if not already matched compare coordinates
               if (not_matching[kf])
               {
                  std::span<T> facef_coord(&fine_coord[fine_coord_idx[kf]], nb_coord_per_face);
                  // loop on nodes of the coarse face
                  int vt = 0;
                  for (int cc = 0; cc < nb_nodes_per_face; ++cc)
                  {
                     auto c0_c = &coarse_coord[cc * gdim];
                     // loop on nodes of the fine face
                     for (int cf = 0; cf < nb_nodes_per_face; ++cf)
                     {
                        auto c0_f = &facef_coord[cf * gdim];
                        int ct = 0;
                        while (ct < gdim && is_same_coord(c0_c[ct], c0_f[ct])) ++ct;
                        if (ct == gdim)
                        {
                           ++vt;
                           break;
                        }
                     }
                  }
                  if (vt == nb_nodes_per_face)
                  {
                     not_matching[kf] = false;
                     sface_child[count_sface_child[f]] = ff;
                     break;
                  }
               }
               ++kf;
            }
         }
      }
   }
   assert(!std::ranges::count(sface_child, -1));
   // PRINT("sface_child",sface_child);

   std::int32_t nb_surround = surrounding_cells.size();
   MPI_Allreduce(MPI_IN_PLACE, &nb_surround, 1, MPI_INT32_T, MPI_MAX, comm);

   return twoscale::scaleJump<dolfinx::mesh::Mesh<T>>(
       std::move(merged_mesh), std::move(mesh), (nb_surround > 0), std::move(count_child), std::move(cell_child),
       std::move(count_face_child), std::move(face_child), max_nb_face_child, std::move(count_sface_child),
       std::move(sface_child), std::move(parent), std::move(enriched_nodes), std::move(extra_enriched_nodes), std::move(support),
       std::move(surrounding_cells), std::move(fine_node_master), std::move(coarse_interface_master));
}


}  // namespace twoscale_dolfinx
#endif
