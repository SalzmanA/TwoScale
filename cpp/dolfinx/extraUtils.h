/*
 * Copyright (C) 2019-2020 Garth N. Wells
 *
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * This code is extracted from DOLFINx 0.9.0 and slightly modified to return element reordering.
 *
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_EXTRA_UTILS
#define TS_DOLFINX_EXTRA_UTILS
#include <dolfinx/mesh/utils.h>

namespace dolfinx::mesh
{
/// @brief Create a distributed mesh from mesh data using a provided
/// graph partitioning function for determining the parallel
/// distribution of the mesh.
///
/// From mesh input data that is distributed across processes, a
/// distributed mesh::Mesh is created. If the partitioning function is
/// not callable, i.e. it does not store a callable function, no
/// re-distribution of cells is done.
///
/// @param[in] comm Communicator to build the mesh on.
/// @param[in] commt Communicator that the topology data (`cells`) is
/// distributed on. This should be `MPI_COMM_NULL` for ranks that should
/// not participate in computing the topology partitioning.
/// @param[in] cells Cells on the calling process. Each cell (node in
/// the `AdjacencyList`) is defined by its 'nodes' (using global
/// indices) following the Basix ordering. For lowest order cells this
/// will be just the cell vertices. For higher-order cells, other cells
/// 'nodes' will be included. See dolfinx::io::cells for examples of the
/// Basix ordering.
/// @param[in] element Coordinate element for the cells.
/// @param[in] commg Communicator for geometry
/// @param[in] x Geometry data ('node' coordinates). Row-major storage.
/// The global index of the `i`th node (row) in `x` is taken as `i` plus
/// the process offset  on`comm`, The offset  is the sum of `x` rows on
/// all processed with a lower rank than the caller.
/// @param[in] xshape Shape of the `x` data.
/// @return A mesh distributed on the communicator `comm`.
template <typename U>
std::tuple<Mesh<typename std::remove_reference_t<typename U::value_type>>, std::vector<std::int32_t>> create_mesh(
    MPI_Comm comm, MPI_Comm commt, std::span<const std::int64_t> cells,
    const fem::CoordinateElement<typename std::remove_reference_t<typename U::value_type>>& element, MPI_Comm commg, const U& x,
    std::array<std::size_t, 2> xshape)
{
  CellType celltype = element.cell_shape();
  const fem::ElementDofLayout doflayout = element.create_dof_layout();

  const int num_cell_vertices = mesh::num_cell_vertices(element.cell_shape());
  std::size_t num_cell_nodes = doflayout.num_dofs();

  // Note: `extract_topology` extracts topology data, i.e. just the
  // vertices. For P1 geometry this should just be the identity
  // operator. For other elements the filtered lists may have 'gaps',
  // i.e. the indices might not be contiguous.
  //
  // `extract_topology` could be skipped for 'P1 geometry' elements

  // -- Partition topology across ranks of comm
  std::vector<std::int64_t> cells1;
  std::vector<std::int64_t> original_idx1;
  std::vector<int> ghost_owners;

  cells1 = std::vector<std::int64_t>(cells.begin(), cells.end());
  assert(cells1.size() % num_cell_nodes == 0);
  std::int64_t offset = 0;
  std::int64_t num_owned = cells1.size() / num_cell_nodes;
  MPI_Exscan(&num_owned, &offset, 1, MPI_INT64_T, MPI_SUM, comm);
  original_idx1.resize(num_owned);
  std::iota(original_idx1.begin(), original_idx1.end(), offset);

  // Extract cell 'topology', i.e. extract the vertices for each cell
  // and discard any 'higher-order' nodes
  std::vector<std::int64_t> cells1_v = extract_topology(celltype, doflayout, cells1);
  spdlog::info("Extract basic topology: {}->{}", cells1.size(),
               cells1_v.size());

  auto pid = dolfinx::MPI::rank(comm);

  // Build local dual graph for owned cells to (i) get list of vertices
  // on the process boundary and (ii) apply re-ordering to cells for
  // locality
  std::vector<std::int64_t> boundary_v;
  std::vector<std::int32_t> remap;
  {
     std::int32_t num_owned_cells = cells1_v.size() / num_cell_vertices - ghost_owners.size();
     std::vector<std::int32_t> cell_offsets(num_owned_cells + 1, 0);
     for (std::size_t i = 1; i < cell_offsets.size(); ++i) cell_offsets[i] = cell_offsets[i - 1] + num_cell_vertices;
     spdlog::info("Build local dual graph");
     auto [graph, unmatched_facets, max_v, facet_attached_cells] =
         build_local_dual_graph(std::vector{celltype}, {std::span(cells1_v.data(), num_owned_cells * num_cell_vertices)});
     remap = graph::reorder_gps(graph);
     // const std::vector<int> remap = graph::reorder_gps(graph);

     // Create re-ordered cell lists (leaves ghosts unchanged)
     /*
      */
     std::vector<std::int64_t> _original_idx(original_idx1.size());
     for (std::size_t i = 0; i < remap.size(); ++i) _original_idx[remap[i]] = original_idx1[i];
     std::copy_n(std::next(original_idx1.cbegin(), num_owned_cells), ghost_owners.size(),
                 std::next(_original_idx.begin(), num_owned_cells));
     impl::reorder_list(std::span(cells1_v.data(), remap.size() * num_cell_vertices), remap);
     impl::reorder_list(std::span(cells1.data(), remap.size() * num_cell_nodes), remap);
     original_idx1 = _original_idx;

     // Boundary vertices are marked as 'unknown'
     boundary_v = unmatched_facets;
     std::ranges::sort(boundary_v);
     auto [unique_end, range_end] = std::ranges::unique(boundary_v);
     boundary_v.erase(unique_end, range_end);

     // Remove -1 if it occurs in boundary vertices (may occur in mixed
     // topology)
     if (!boundary_v.empty() > 0 and boundary_v[0] == -1) boundary_v.erase(boundary_v.begin());
  }

  // Create Topology
  Topology topology = create_topology(comm, cells1_v, original_idx1,
                                      ghost_owners, celltype, boundary_v);


  // Create connectivities required higher-order geometries for creating
  // a Geometry object
  for (int e = 1; e < topology.dim(); ++e)
    if (doflayout.num_entity_dofs(e) > 0)
      topology.create_entities(e);
  if (element.needs_dof_permutations())
    topology.create_entity_permutations();

  // Build list of unique (global) node indices from cells1 and
  // distribute coordinate data
  std::vector<std::int64_t> nodes1 = cells1;
  dolfinx::radix_sort(nodes1);
  auto [unique_end, range_end] = std::ranges::unique(nodes1);
  nodes1.erase(unique_end, range_end);

  std::vector coords
      = dolfinx::MPI::distribute_data(comm, nodes1, commg, x, xshape[1]);

  // Create geometry object
  Geometry geometry
      = create_geometry(topology, element, nodes1, cells1, coords, xshape[1]);

  return {Mesh(comm, std::make_shared<Topology>(std::move(topology)), std::move(geometry)), std::move(remap)};
}
/*
    TODO TODO

     when using partitioner mesh is redistributed and thus tags needs to be redistributed !!!!
     Complex as tag if on 'face' must be distributed using the cell on which it sets:
            args !!!! does cell face order is maintened during distribution ?
                      does face exist at this point ?
                      Should we create faces before distribution ?
                      A priori transfer_cell_meshtag can't be used in this case ?
                      .....

template <typename U,typename D>
Mesh<typename std::remove_reference_t<typename U::value_type>> create_mesh(
    MPI_Comm comm, MPI_Comm commt, std::span<const std::int64_t> cells,
    const fem::CoordinateElement<
        typename std::remove_reference_t<typename U::value_type>>& element,
    MPI_Comm commg, const U& x, std::array<std::size_t, 2> xshape,
    const CellPartitionFunction& partitioner,mesh::MeshTags<D>* tags)
{
}
*/

}  // namespace dolfinx::mesh
#endif
