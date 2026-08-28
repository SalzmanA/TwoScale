/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_SCALE_JUMP
#define TS_DOLFINX_SCALE_JUMP
#include <memory>
#include <unordered_set>

#include "dolfinx/mesh/Mesh.h"
#include "dolfinx/mesh/MeshTags.h"
#include "dolfinx/mesh/utils.h"
#include "dolfinx/graph/AdjacencyList.h"
#include "dolfinx/common/Scatterer.h"

// still under reflection if all child should be constructed or not:
// pros:
//    all child construction simplify operator creation as all macro got at least a child and thus operator can proceed also those
//    child from outside the group support&surrounding.
//    It is not clear for now how fine matrix part related to element external to group support&surrounding can be transferred at
//    coarse level in the case of restricted set of child: assembly ???
// cons:
//    Add count operation (anyway these operation scale well ...)
//    take more memory if support&surrounding group small compares to coarse number of element
//    Operator multiplication apply on a larger set of terms of the fine matrix thus cost more if again   support&surrounding
//    group is small compares to coarse number of element
//
//    if set, this macro switch to the creation of at least one child for all macro cell
//    Implementation has still to be optimized if this macro is set because all algorithms are considering restricted set of child
//    (useless test to remove).
#define  ALL_CHILD

#if 1
#include "debug.h"
#define FILES(pid)                                                 \
   std::string no = "proc_" + std::to_string(pid) + "_output.txt"; \
   freopen(no.c_str(), "w", stdout);
#else
#define FILES(pid)
#endif





namespace twoscale
{
/// Class that store fine and coarse mesh with the connections in between them.
/// @tparam M The mesh type
template <typename M>
class scaleJump
{
  public:
   /// Class constructor
   /// @param fine_mesh_ The fine mesh instance of type M
   /// @param coarse_mesh_ The coarse mesh instance of type M
   /// @param needs_mpc_  A boolean indicating if mpc are required or not
   /// @param cell_child_offset_, cell_child_ The vector cell_child_ store fine scale cell local id packed by group corresponding
   /// to coarse scale cell.
   /// @note User won't call it directelly a priori but will use twoscale_dolfinx::topDown function
   scaleJump(M &&fine_mesh_, M &&coarse_mesh_, bool needs_mpc_, std::vector<std::int32_t> &&cell_child_offset_,
             std::vector<std::int32_t> &&cell_child_, std::vector<std::int32_t> &&face_child_offset_,
             std::vector<std::int32_t> &&face_child_, std::int32_t max_nb_face_child_,
             std::vector<std::int32_t> &&sface_child_offset_, std::vector<std::int32_t> &&sface_child_,
             std::vector<std::int32_t> &&cell_parent_, std::vector<std::int32_t> &&enriched_nodes_,
             std::vector<std::int32_t> &&extra_enriched_nodes_, std::vector<std::int32_t> &&support_,
             std::vector<std::int32_t> &&surrounding_cells_, std::unordered_set<std::int32_t> &&fine_node_master_,
             std::vector<std::int32_t> &&coarse_interface_master_)
       : needs_mpc(needs_mpc_),
         coarse_mesh(std::make_shared<M>(std::forward<M>(coarse_mesh_))),
         fine_mesh(std::make_shared<M>(std::forward<M>(fine_mesh_))),
         cell_child_offset(std::move(cell_child_offset_)),
         cell_child(std::move(cell_child_)),
         face_child_offset(std::move(face_child_offset_)),
         face_child(std::move(face_child_)),
         sface_child_offset(std::move(sface_child_offset_)),
         sface_child(std::move(sface_child_)),
         max_nb_face_child(max_nb_face_child_),
         cell_parent(std::move(cell_parent_)),
         enriched_nodes(std::move(enriched_nodes_)),
         extra_enriched_nodes(std::move(extra_enriched_nodes_)),
         support(std::move(support_)),
         surrounding_cells(std::move(surrounding_cells_)),
         fine_node_master(std::move(fine_node_master_)),
         coarse_interface_master(std::move(coarse_interface_master_))
   {
#ifndef NDEBUG
   int rcomp;
   MPI_Comm_compare(coarse_mesh->comm(), fine_mesh->comm(), &rcomp);
   assert(rcomp == MPI_CONGRUENT);
#endif
      std::cout << "in the Jump constructor" << std::endl;
   }
   /// Function that give a shared pointer of the fine scale mesh
   const std::shared_ptr<M> getFineMesh() const;
   /// Function that give a shared pointer of the coarse scale mesh
   const std::shared_ptr<M> getCoarseMesh() const;
   /// Function that return a list of fine scale cells local ids. These cells are embodied in coarse scale cell corresponding to
   /// local index given as argument. If no fine cell is associated to this coarse cell it return an empty list.
   /// @param coarse_cell_idx Index of the coarse scale cell for which we want embedded fine scale cells (*children*)
   /// @warning In the context of a distributed mesh, a 'local index' is a numbering system used by each process to identify
   /// entities that it store. This is usually complemented by a global numbering system covering all processes. Here the FEniCs
   /// numbering is implicitly used.
   std::span<const std::int32_t> getChildren(std::int32_t coarse_cell_idx) const;
   /// Function that return a list of fine scale faces (in 3D and edges in 2D or nodes in 1D) local ids. These faces are embodied in
   /// coarse scale face corresponding to local index given as argument. If no fine face is associated to this coarse face it
   /// return an empty list.
   /// @param coarse_face_idx Index of the coarse scale face for which we want embedded fine scale faces (*children*)
   /// @warning In the context of a distributed mesh, a 'local index' is a numbering system used by each process to identify 
   /// entities that it store. This is usually complemented by a global numbering system covering all processes. Here the FEniCs
   /// numbering is implicitly used.
   std::span<const std::int32_t> getFaceChilds(std::int32_t coarse_face_idx) const;
   std::span<const std::int32_t> getSurroundingFaceChilds(std::int32_t coarse_cell_idx) const;
   /// Function that return a list of coarse scale nodes local ids. These nodes corresponds to the enriched nodes provided by the
   /// user at creation of the instance.
   std::span<const std::int32_t> getEnriched() const;
   /// Function that return a list of coarse scale nodes local ids. These nodes corresponds to the enriched nodes added to treat
   /// blending problem at interfaces of enriched/non enriched area.
   std::span<const std::int32_t> getExtraEnriched() const;
   /// Function that return a list of coarse scale cells local ids. These cells corresponds to the support of the enriched nodes
   /// given by scaleJump::getEnriched
   std::span<const std::int32_t> getSupport() const;
   /// Function that return a list of coarse scale cells local ids. These cells corresponds to the support of the enriched nodes
   /// given by scaleJump::getExtraEnriched minus the cells provided by scaleJump::getSupport
   std::span<const std::int32_t> getSurroundingCells() const;
   /// Function that return true if index given as argument correspond to a local fine scale id of a nodes which is a master node
   /// in a MPC relationship.
   /// @param i fine scale local index of the node to test
   bool isFineMasterNode(std::int32_t i) const;
   std::span<const std::int32_t> getCoarseMaster() const;
   std::int32_t getMaxNumberOfChildFaces() const;
   /// Member indicating if MPC are mandatory or not (i.e. if mesh jump exist in fine scale mesh)
   const bool needs_mpc;

  private:
   // meshes
   std::shared_ptr<M>  coarse_mesh;
   std::shared_ptr<M> fine_mesh;
   // childs of macro
   std::vector<std::int32_t> cell_child_offset;
   std::vector<std::int32_t> cell_child;
   std::vector<std::int32_t> face_child_offset;
   std::vector<std::int32_t> face_child;
   std::vector<std::int32_t> sface_child_offset;
   std::vector<std::int32_t> sface_child;
   std::int32_t max_nb_face_child;
   // parent
   std::vector<std::int32_t> cell_parent;
   // coarse set
   std::vector<std::int32_t> enriched_nodes;
   std::vector<std::int32_t> extra_enriched_nodes;
   std::vector<std::int32_t> support;
   std::vector<std::int32_t> surrounding_cells;
   // master set
   std::unordered_set<std::int32_t> fine_node_master; 
   std::vector<std::int32_t> coarse_interface_master;
};

}  // namespace twoscale
namespace twoscale_dolfinx
{
namespace impl
{
void extendSupport(std::shared_ptr<const dolfinx::mesh::Topology> topo,
                   std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> adj, int dim, std::vector<std::int32_t> &support,
                   std::unordered_set<std::int32_t> &u_support);
}

/// Function creating a scale jump in a top down manner. Starting from
///  * a coarse mesh distributed on some process
///  * a set of coarse enriched nodes
///  * a geometric criterion functor to locate the areas where mesh needs to be refined.
///  * a level of refinement
///
///  this function do the following:
///  1. identify the support of all enriched nodes
///  2. load balance the original coarse mesh distribution so that each procces get a peace of the support
///  3. isolate the sub-mesh coresponding to the support
///  4. from this new distributed sub-mesh loop 'level of refinement' time to refine mesh locate in areas identified by criterion
///  5. reconnect this refined sub-mesh with ignored part of the coarse mesh. It form a new composite mesh made of coarse and fine
///  element that will be connected later by MPC relation for dangling nodes.
///  6. compute all relation between new coarse mesh and new composite mesh for future MPC creation and TwoScale processing.
///
/// @note User is responsible to provides a crit functor in coherence with  enriched nodes support (i.e. if refined area are not
/// in the support, refinement except the first level won't be done)
///
/// @tparam T The mesh geometry scalar type (float or double).
/// @tparam U Based on the dolfinx concept represents a function for geometry marking
/// @tparam W Based on the dolfinx concept represents a function for geometry marking
/// @tparam Z The type that is stored by MeshTags argument
///
/// @param[in] mesh The coarse mesh in its original distribution.
/// @param[in] enrich The set of enriched node provided by the user as a Dolfinx Marker function
/// @param[in] crit The refinement area selection functor provided by the user as a Dolfinx Marker function with an extra control
/// parameter. This parameter may be used by crit to tune area selection (e.g. focus on a specific location while mesh
/// discretization becomes smaller)
/// @param[in] control The control parameter that drive the refinement area selection functor crit. It is set to current
/// refinement level during refinement loop and passed to crit as second argument.
/// @param[in] level The level of refinement (i.e. the number of time areas identified by crit are refined by
/// dolfinx::refinement::refine)
/// @param[in] clustering_dual_graph A Boolean to force usage of clustering strategy
/// @param[in] accurate_weight A Boolean forcing accurate weight computation by refining mesh with coarse mesh in its
/// original distribution. Once weights are computed this refined mesh is dropped. By default (i.e. accurate_weight==false) an
/// uniform estimate based on level and dimension  is used for all weights of all support cells.
/// @param[in] tags Optional MeshTags object to propagate during refinement
/// @return A scaleJump object encapsulating all new meshes and their relationship.
///
/// @warning tags argument is completely ignored for now. Implementation has to be done and certainnely somme API change
/// to provide to the user this MeshTags as it won't make sens to incude it in scaleJump object.
template <typename T, dolfinx::mesh::MarkerFn<T> U, dolfinx::mesh::MarkerFn<T> W, typename Z>
twoscale::scaleJump<dolfinx::mesh::Mesh<T>> topDown(dolfinx::mesh::Mesh<T> &mesh, U enriched, W crit, std::uint8_t &control,
                                                    std::uint8_t level, bool clustering_dual_graph = true,
                                                    bool accurate_weight = false,
                                                    dolfinx::mesh::MeshTags<Z> const *tags = nullptr);

}  // namespace twoscale_dolfinx

#include "scaleJump_imp.h"
#endif
