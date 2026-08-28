/*
 * Copyright (C) 2010-2024 Garth N. Wells and Paul T. Kühner
 *
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * This code is extracted from DOLFINx and slightly modified to take into account element reordering in create_mesh.
 * use for that custom create_mesh that provides cell mapping.
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_DOLFINX_EXTRA_REFINE
#define TS_DOLFINX_EXTRA_REFINE


#include "dolfinx/refinement/refine.h"
#include "extraUtils.h"

namespace dolfinx::refinement
{
/// @brief Refine a mesh with markers.
///
/// The refined mesh is not re-partitioned across processes.
///
/// Parent-child relationships can be optionally computed. Parent-child
/// relationships can be used to create MeshTags on the refined mesh
/// from MeshTags on the parent mesh.
///
///
///
/// @param[in] mesh Input mesh to be refined.
/// @param[in] edges Indices of the edges that should be split in the
/// refinement. If not provided (`std::nullopt`), uniform refinement is
/// performed.
/// @param[in] option Control the computation of parent facets, parent
/// cells. If an option is not selected, an empty list is returned.
/// @return New mesh, and optional parent cell indices and parent facet
/// indices.

template <std::floating_point T>
std::tuple<mesh::Mesh<T>, std::optional<std::vector<std::int32_t>>, std::optional<std::vector<std::int8_t>>> refine(
    const mesh::Mesh<T>& mesh, std::optional<std::span<const std::int32_t>> edges, Option option = Option::none)
{
   auto topology = mesh.topology();
   assert(topology);
   if (!mesh::is_simplex(topology->cell_type())) throw std::runtime_error("Refinement only defined for simplices");

   auto [cell_adj, new_vertex_coords, xshape, parent_cell, parent_facet] =
       (topology->cell_type() == mesh::CellType::interval) ? interval::compute_refinement_data(mesh, edges, option)
                                                           : plaza::compute_refinement_data(mesh, edges, option);

   auto [ mesh1, remap] = mesh::create_mesh(mesh.comm(), mesh.comm(), cell_adj.array(), mesh.geometry().cmap(), 
                                                        mesh.comm(), new_vertex_coords, xshape);

   if (parent_cell.has_value())
   {
      auto& pc = *parent_cell;
      auto cpc=pc;
      assert(pc.size()==remap.size());
      for (std::int32_t i = 0, nbc = pc.size(); i < nbc; ++i) pc[remap[i]] = cpc[i];
   }
   if (parent_facet.has_value())
   {
      auto& pf = *parent_facet;
      auto cpf=pf;
      const int num_cell_facet = mesh::cell_num_entities(topology->cell_type(),topology->dim()-1);
      assert(pf.size()/num_cell_facet==remap.size());

      for (std::int32_t i=0,nbc=remap.size();i<nbc;++i)
      {
         auto &cpfi=cpf[i*num_cell_facet];
         std::copy(&cpfi,(&cpfi)+num_cell_facet,&pf[remap[i]*num_cell_facet]);
      }
   }

   // Report the number of refined cells
   const int D = topology->dim();
   const std::int64_t n0 = topology->index_map(D)->size_global();
   const std::int64_t n1 = mesh1.topology()->index_map(D)->size_global();
   spdlog::info("Number of cells increased from {} to {} ({}% increase).", n0, n1,
                100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

   return {std::move(mesh1), std::move(parent_cell), std::move(parent_facet)};
}
} // namespace dolfinx::refinement
#endif
