/*
 * Copyright (C) 2019-2020 Garth N. Wells, Chris Richardson and Igor A. Baratta
 *
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * This is partially extracted from partitioners.cpp of dolfinx 0.9.0 project 
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include "extraPartitioners.h"

#include <dolfinx/common/MPI.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/log.h>

#include <algorithm>
#include <execution>
#include <vector>
#include <unordered_map>

#include "util.h"

using namespace dolfinx;
namespace
{
// 
//
// 
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later
template <typename T>
graph::AdjacencyList<int> compute_destination_ranks(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& graph,
                                                    const std::vector<T>& node_disp, const std::vector<T>& part)
{
   common::Timer timer("Extend graph destination ranks for halo");

   const int rank = dolfinx::MPI::rank(comm);
   const std::int64_t range0 = node_disp[rank];
   const std::int64_t range1 = node_disp[rank + 1];
   assert(static_cast<std::int32_t>(range1 - range0) == graph.num_nodes());

   // Wherever an owned 'node' goes, so must the nodes connected to it by
   // an edge ('node1'). Task is to let the owner of node1 know the extra
   // ranks that it needs to send node1 to.
   std::vector<std::array<std::int64_t, 3>> node_to_dest;
   for (int node0 = 0; node0 < graph.num_nodes(); ++node0)
   {
      // Wherever 'node' goes to, so must the attached 'node1'
      for (auto node1 : graph.links(node0))
      {
         if (node1 < range0 or node1 >= range1)
         {
            auto it = std::ranges::upper_bound(node_disp, node1);
            int remote_rank = std::distance(node_disp.begin(), it) - 1;
            node_to_dest.push_back({remote_rank, node1, static_cast<std::int64_t>(part[node0])});
         }
         else
            node_to_dest.push_back({rank, node1, static_cast<std::int64_t>(part[node0])});
      }
   }

   std::ranges::sort(node_to_dest);
   auto [unique_end, range_end] = std::ranges::unique(node_to_dest);
   node_to_dest.erase(unique_end, range_end);

   // Build send data and buffer
   std::vector<int> dest, send_sizes;
   std::vector<std::int64_t> send_buffer;
   {
      auto it = node_to_dest.begin();
      while (it != node_to_dest.end())
      {
         // Current destination rank
         dest.push_back((*it)[0]);

         // Find iterator to next destination rank and pack send data
         auto it1 = std::find_if(it, node_to_dest.end(), [r0 = dest.back()](auto& idx) { return idx[0] != r0; });
         send_sizes.push_back(2 * std::distance(it, it1));
         for (auto itx = it; itx != it1; ++itx)
         {
            send_buffer.push_back((*itx)[1]);
            send_buffer.push_back((*itx)[2]);
         }

         it = it1;
      }
   }

   // Prepare send displacements
   std::vector<int> send_disp(send_sizes.size() + 1, 0);
   std::partial_sum(send_sizes.begin(), send_sizes.end(), std::next(send_disp.begin()));

   // Discover src ranks. ParMETIS/KaHIP are not scalable (holding an
   // array of size equal to the comm size), so no extra harm in using
   // non-scalable neighbourhood detection (which might be faster for
   // small rank counts).
   const std::vector<int> src = dolfinx::MPI::compute_graph_edges_pcx(comm, dest);

   // Create neighbourhood communicator
   MPI_Comm neigh_comm;
   MPI_Dist_graph_create_adjacent(comm, src.size(), src.data(), MPI_UNWEIGHTED, dest.size(), dest.data(), MPI_UNWEIGHTED,
                                  MPI_INFO_NULL, false, &neigh_comm);

   // Determine receives sizes
   std::vector<int> recv_sizes(dest.size());
   send_sizes.reserve(1);
   recv_sizes.reserve(1);
   MPI_Neighbor_alltoall(send_sizes.data(), 1, MPI_INT, recv_sizes.data(), 1, MPI_INT, neigh_comm);

   // Prepare receive displacements
   std::vector<int> recv_disp(recv_sizes.size() + 1, 0);
   std::partial_sum(recv_sizes.begin(), recv_sizes.end(), std::next(recv_disp.begin()));

   // Send/receive data
   std::vector<std::int64_t> recv_buffer(recv_disp.back());
   MPI_Neighbor_alltoallv(send_buffer.data(), send_sizes.data(), send_disp.data(), MPI_INT64_T, recv_buffer.data(),
                          recv_sizes.data(), recv_disp.data(), MPI_INT64_T, neigh_comm);
   MPI_Comm_free(&neigh_comm);

   // Prepare (local node index, destination rank) array. Add local data,
   // then add the received data, and the make unique.
   std::vector<std::array<int, 2>> local_node_to_dest;
   for (auto d : part)
   {
      local_node_to_dest.push_back({static_cast<int>(local_node_to_dest.size()), static_cast<int>(d)});
   }
   for (std::size_t i = 0; i < recv_buffer.size(); i += 2)
   {
      std::int64_t idx = recv_buffer[i];
      int d = recv_buffer[i + 1];
      assert(idx >= range0 and idx < range1);
      std::int32_t idx_local = idx - range0;
      local_node_to_dest.push_back({idx_local, d});
   }

   {
      std::ranges::sort(local_node_to_dest);
      auto [unique_end, range_end] = std::ranges::unique(local_node_to_dest);
      local_node_to_dest.erase(unique_end, range_end);
   }
   // Compute offsets
   std::vector<std::int32_t> offsets(graph.num_nodes() + 1, 0);
   {
      std::vector<std::int32_t> num_dests(graph.num_nodes(), 0);
      for (auto x : local_node_to_dest) ++num_dests[x[0]];
      std::partial_sum(num_dests.begin(), num_dests.end(), std::next(offsets.begin()));
   }

   // Fill data array
   std::vector<int> data(offsets.back());
   {
      std::vector<std::int32_t> pos = offsets;
      for (auto [x0, x1] : local_node_to_dest) data[pos[x0]++] = x1;
   }

   graph::AdjacencyList<int> g(std::move(data), std::move(offsets));

   // Make sure the owning rank comes first for each node
   for (std::int32_t i = 0; i < g.num_nodes(); ++i)
   {
      auto d = g.links(i);
      auto it = std::find(d.begin(), d.end(), part[i]);
      assert(it != d.end());
      std::iter_swap(d.begin(), it);
   }

   return g;
}
// end blunt copy from dolfinx
}  // namespace

namespace dolfinx::graph
{
namespace parmetis
{
#ifdef HAS_PARMETIS
extern "C"
{
#include <parmetis.h>
}
// This part is an adaptation of graph::parmetis::partitioner from partitioners.cpp of dolfinx project
// node weight 
graph::partition_fn partitionerWithNodeWeight(std::vector<std::int32_t>& weights, int nb_weights, double imbalance,
                                          std::array<int, 3> options)
{
   return [imbalance, options, &weights, nb_weights](MPI_Comm comm, idx_t nparts, const graph::AdjacencyList<std::int64_t>& graph,
                                                     bool ghosting) {
      spdlog::info("Compute node weighted graph partition using ParMETIS");
      common::Timer timer("Compute Node weighted graph partition (ParMETIS)");

      if (nparts == 1 and dolfinx::MPI::size(comm) == 1)
      {
         // Nothing to be partitioned
         return regular_adjacency_list(std::vector<std::int32_t>(graph.num_nodes(), 0), 1);
      }

      // Note: ParMETIS fails (crashes) if a rank does not have any graph
      // data. Therefore we split the communicator such that ParMETIS
      // partitioning happens only on ranks that have data. Ideallt we
      // wouldn't need to do this.
      constexpr bool split_comm = true;
      MPI_Comm pcomm = MPI_COMM_NULL;
      if (split_comm)
      {
         int rank = dolfinx::MPI::rank(comm);
         int color = graph.num_nodes() > 0 ? 1 : MPI_UNDEFINED;
         int ierr = MPI_Comm_split(comm, color, rank, &pcomm);
         dolfinx::MPI::check_error(comm, ierr);
      }
      else
         pcomm = comm;

      std::vector<idx_t> part(graph.num_nodes());
      std::vector<idx_t> node_disp;
      if (pcomm != MPI_COMM_NULL)
      {
         // Build adjacency list data
         const int psize = dolfinx::MPI::size(pcomm);
         const idx_t num_local_nodes = graph.num_nodes();
         node_disp = std::vector<idx_t>(psize + 1, 0);
         MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<idx_t>(), node_disp.data() + 1, 1,
                       dolfinx::MPI::mpi_type<idx_t>(), pcomm);
         std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
         std::vector<idx_t> array(graph.array().begin(), graph.array().end());
         std::vector<idx_t> offsets(graph.offsets().begin(), graph.offsets().end());

         /*
         std::cout<<graph.str();
         std::cout<<"node_disp "<<node_disp.size()<<":"; for (auto s :node_disp) std::cout<<" "<<s; std::cout<<std::endl;
         std::cout<<"offsets "<<offsets.size()<<":"; for (auto s :offsets) std::cout<<" "<<s; std::cout<<std::endl;
         std::cout<<"array "<<array.size()<<":"; for (auto s :array) std::cout<<" "<<s; std::cout<<std::endl;
         std::cout<<"weights "<<weights.size()<<":"; for (auto s :weights) std::cout<<" "<<s; std::cout<<std::endl;
         */

         // check weight/graph consistency
         if (num_local_nodes * nb_weights != weights.size())
         {
            throw std::runtime_error("Graph and weights are incoherent : "+std::to_string(num_local_nodes * nb_weights)+"#"+std::to_string(weights.size()));
         }

         // Options and data for ParMETIS
         std::array<idx_t, 3> opts = {options[0], options[1], options[2]};
         idx_t ncon = static_cast<idx_t>(nb_weights);
         idx_t* elmwgt = nullptr;
         bool convertion = !std::is_same<idx_t, std::int32_t>::value;
         if (convertion)
         {
            elmwgt = new idx_t[num_local_nodes * ncon];
            std::transform(std::execution::unseq, weights.begin(), weights.end(), elmwgt,
                           [](std::int32_t x) { return static_cast<idx_t>(x); });
         }
         else
         {
            elmwgt = weights.data();
            
         }
         idx_t wgtflag(2), edgecut(0), numflag(0);
         std::vector<real_t> tpwgts(ncon * nparts, 1.0 / static_cast<real_t>(nparts));
         std::vector<real_t> ubvec(ncon, static_cast<real_t>(imbalance));

         // Partition
         common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
         int err = ParMETIS_V3_PartKway(node_disp.data(), offsets.data(), array.data(), elmwgt, nullptr, &wgtflag, &numflag,
                                        &ncon, &nparts, tpwgts.data(), ubvec.data(), opts.data(), &edgecut, part.data(), &pcomm);
         if (err != METIS_OK)
         {
            throw std::runtime_error("ParMETIS_V3_PartKway failed. Error code: " + std::to_string(err));
         }

         if (convertion) delete elmwgt;
      }

      if (ghosting and pcomm != MPI_COMM_NULL)
      {
         // FIXME: Is it implicit that the first entry is the owner?
         graph::AdjacencyList<int> dest = compute_destination_ranks(pcomm, graph, node_disp, part);
         if (split_comm) MPI_Comm_free(&pcomm);
         return dest;
      }
      else
      {
         if (split_comm and pcomm != MPI_COMM_NULL) MPI_Comm_free(&pcomm);
         return regular_adjacency_list(std::vector<int>(part.begin(), part.end()), 1);
      }
   };
}
// edge weight
graph::partition_fn partitionerWithEdgeWeight(std::vector<std::int32_t>& weights, double imbalance, std::array<int, 3> options)
{
   return [imbalance, options, &weights](MPI_Comm comm, idx_t nparts, const graph::AdjacencyList<std::int64_t>& graph,
                                                     bool ghosting) {
      spdlog::info("Compute edge weighted graph partition using ParMETIS");
      common::Timer timer("Compute edge weighted graph partition (ParMETIS)");

      if (nparts == 1 and dolfinx::MPI::size(comm) == 1)
      {
         // Nothing to be partitioned
         return regular_adjacency_list(std::vector<std::int32_t>(graph.num_nodes(), 0), 1);
      }

      // Note: ParMETIS fails (crashes) if a rank does not have any graph
      // data. Therefore we split the communicator such that ParMETIS
      // partitioning happens only on ranks that have data. Ideallt we
      // wouldn't need to do this.
      constexpr bool split_comm = true;
      MPI_Comm pcomm = MPI_COMM_NULL;
      if (split_comm)
      {
         int rank = dolfinx::MPI::rank(comm);
         int color = graph.num_nodes() > 0 ? 1 : MPI_UNDEFINED;
         int ierr = MPI_Comm_split(comm, color, rank, &pcomm);
         dolfinx::MPI::check_error(comm, ierr);
      }
      else
         pcomm = comm;

      std::vector<idx_t> part(graph.num_nodes());
      std::vector<idx_t> node_disp;
      if (pcomm != MPI_COMM_NULL)
      {
         // Build adjacency list data
         const int psize = dolfinx::MPI::size(pcomm);
         const idx_t num_local_nodes = graph.num_nodes();
         node_disp = std::vector<idx_t>(psize + 1, 0);
         MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<idx_t>(), node_disp.data() + 1, 1,
                       dolfinx::MPI::mpi_type<idx_t>(), pcomm);
         std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
         std::vector<idx_t> array(graph.array().begin(), graph.array().end());
         std::vector<idx_t> offsets(graph.offsets().begin(), graph.offsets().end());

         /*
         std::cout<<graph.str();
         std::cout<<"node_disp "<<node_disp.size()<<":"; for (auto s :node_disp) std::cout<<" "<<s; std::cout<<std::endl;
         std::cout<<"offsets "<<offsets.size()<<":"; for (auto s :offsets) std::cout<<" "<<s; std::cout<<std::endl;
         std::cout<<"array "<<array.size()<<":"; for (auto s :array) std::cout<<" "<<s; std::cout<<std::endl;
         std::cout<<"weights "<<weights.size()<<":"; for (auto s :weights) std::cout<<" "<<s; std::cout<<std::endl;
         */

         const idx_t num_edges = array.size();

         // check weight/graph consistency
         if (num_local_nodes != weights.size())
         {
            throw std::runtime_error("Graph and node weights for edge weight are incoherent : "+std::to_string(num_local_nodes )+"#"+std::to_string(weights.size()));
         }

         // Options and data for ParMETIS
         //std::array<idx_t, 3> opts = {options[0], options[1], options[2]};
         std::array<idx_t, 3> opts = {1, 127, 2};
         idx_t ncon = 1;
         idx_t* adjwgt = nullptr;
         adjwgt = new idx_t[num_edges];
         const int rank = dolfinx::MPI::rank(pcomm);
         idx_t start = node_disp[rank];
         idx_t end = node_disp[rank + 1]-start;
         idx_t null_wgt=0;
         idx_t ewgt=600;

         for (idx_t node = 0; node < num_local_nodes; ++node)
         {
            idx_t wn = weights[node];
            ewgt=wn;
            if (wn < 2)
            {
               for (idx_t edge = offsets[node], le = offsets[node + 1]; edge < le; ++edge)
               {
                  assert(edge < num_edges);
                  adjwgt[edge] = null_wgt;
               }
            }
            else
            {
               for (idx_t edge = offsets[node], le = offsets[node + 1]; edge < le; ++edge)
               {
                  assert(edge < num_edges);
                  auto target = array[edge] - start;
                  // if local target
                  if (target < end && !(target < 0))
                  {
                     assert(target < num_local_nodes);
                     adjwgt[edge] = (weights[target] == wn) ?  ewgt : null_wgt;
                  }
                  else
                     adjwgt[edge] = null_wgt;
                  std::cout << "node " << node << " edge " << edge << " linked " << array[edge] << " " << target << " wn " << wn
                            << " wnlinked "
                            << " adjw " << adjwgt[edge] << std::endl;
               }
            }
         }

         bool convertion = !std::is_same<idx_t, std::int32_t>::value;
         idx_t* vwgt = nullptr;
         vwgt = new idx_t[num_local_nodes * ncon];
         std::transform(std::execution::unseq, weights.begin(), weights.end(), vwgt,
                        [](std::int32_t x) { return static_cast<idx_t>((x>1)?0:1); });
         
         idx_t wgtflag(3), edgecut(0), numflag(0);
         std::vector<real_t> tpwgts(nparts, 1.0 / static_cast<real_t>(nparts));
         std::vector<real_t> ubvec(1, static_cast<real_t>(nparts/2.));
         //std::vector<real_t> ubvec(1, static_cast<real_t>(imbalance));

         // Partition
         common::Timer timer1("ParMETIS: call ParMETIS_V3_PartKway");
         int err = ParMETIS_V3_PartKway(node_disp.data(), offsets.data(), array.data(), vwgt, adjwgt, &wgtflag, &numflag, &ncon,
                                        &nparts, tpwgts.data(), ubvec.data(), opts.data(), &edgecut, part.data(), &pcomm);
         if (err != METIS_OK)
         {
            throw std::runtime_error("ParMETIS_V3_PartKway failed. Error code: " + std::to_string(err));
         }

         delete vwgt;
         delete adjwgt;
      }

      if (ghosting and pcomm != MPI_COMM_NULL)
      {
         // FIXME: Is it implicit that the first entry is the owner?
         graph::AdjacencyList<int> dest = compute_destination_ranks(pcomm, graph, node_disp, part);
         if (split_comm) MPI_Comm_free(&pcomm);
         return dest;
      }
      else
      {
         if (split_comm and pcomm != MPI_COMM_NULL) MPI_Comm_free(&pcomm);
         return regular_adjacency_list(std::vector<int>(part.begin(), part.end()), 1);
      }
   };
}
//-----------------------------------------------------------------------------
#endif
}  // namespace parmetis


#ifdef HAS_KAHIP
#include <parhip_interface.h>

//----------------------------------------------------------------------------
graph::partition_fn graph::kahip::partitionerWithNodeWeight(std::vector<std::int32_t>& weights,int mode, int seed,
                                              double imbalance,
                                              bool suppress_output)
{
  return [&weights, mode, seed, imbalance, suppress_output](
             MPI_Comm comm, int nparts,
             const graph::AdjacencyList<std::int64_t>& graph, bool ghosting)
  {
    spdlog::info("Compute graph partition using (parallel) KaHIP with node weight");

    // KaHIP integer type
    using T = unsigned long long;

    common::Timer timer("Compute graph partition (KaHIP)");


    // Build adjacency list data
    common::Timer timer1("KaHIP: build adjacency data");
    std::vector<T> node_disp(dolfinx::MPI::size(comm) + 1, 0);
    const T num_local_nodes = graph.num_nodes();
    MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<T>(),
                  node_disp.data() + 1, 1, dolfinx::MPI::mpi_type<T>(), comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
    std::vector<T> array(graph.array().begin(), graph.array().end());
    std::vector<T> offsets(graph.offsets().begin(), graph.offsets().end());
    timer1.stop();

    common::Timer timer2("KaHIP: build vwgt data");
    // check weight/graph consistency
    if (graph.num_nodes() != weights.size())
    {
       throw std::runtime_error("Graph and node weights for node weight are incoherent : " + std::to_string(num_local_nodes) +
                                "#" + std::to_string(weights.size()));
    }
    
    T* vwgt = new T[num_local_nodes];
    std::transform(std::execution::unseq, weights.begin(), weights.end(), vwgt,
                   [](std::int32_t x) { return static_cast<T>(x); });

    timer2.stop();

    // Call KaHIP to partition graph
    common::Timer timer3("KaHIP: call ParHIPPartitionKWay");
    std::vector<T> part(graph.num_nodes());
    int edgecut = 0;
    //double _imbalance = imbalance;
    double _imbalance = 0.5;
    ParHIPPartitionKWay(node_disp.data(), offsets.data(), array.data(), vwgt,
                        NULL, &nparts, &_imbalance, suppress_output, seed,
                        mode, &edgecut, part.data(), &comm);
    timer3.stop();

    delete vwgt;

    if (ghosting)
      return compute_destination_ranks(comm, graph, node_disp, part);
    else
    {
      return regular_adjacency_list(std::vector<int>(part.begin(), part.end()), 1);
    }
  };
}
//----------------------------------------------------------------------------
graph::partition_fn graph::kahip::partitioner_with_weight(std::vector<std::int32_t>& weights,int mode, int seed,
                                              double imbalance,
                                              bool suppress_output)
{
  return [&weights, mode, seed, imbalance, suppress_output](
             MPI_Comm comm, int nparts,
             const graph::AdjacencyList<std::int64_t>& graph, bool ghosting)
  {
    spdlog::info("Compute graph partition using (parallel) KaHIP with weight");

    // KaHIP integer type
    using T = unsigned long long;

    common::Timer timer("Compute graph partition (KaHIP)");


    // Build adjacency list data
    common::Timer timer1("KaHIP: build adjacency data");
    std::vector<T> node_disp(dolfinx::MPI::size(comm) + 1, 0);
    const T num_local_nodes = graph.num_nodes();
    MPI_Allgather(&num_local_nodes, 1, dolfinx::MPI::mpi_type<T>(),
                  node_disp.data() + 1, 1, dolfinx::MPI::mpi_type<T>(), comm);
    std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
    std::vector<T> array(graph.array().begin(), graph.array().end());
    std::vector<T> offsets(graph.offsets().begin(), graph.offsets().end());
    timer1.stop();

    common::Timer timer2("KaHIP: build v&adj wgt data");
    // check weight/graph consistency
    if (graph.num_nodes() != weights.size())
    {
       throw std::runtime_error("Graph and node weights for edge weight are incoherent : " + std::to_string(num_local_nodes) +
                                "#" + std::to_string(weights.size()));
    }
    const T num_edges = array.size();
    T * adjwgt = nullptr;
    adjwgt = new T[num_edges];
    const int rank = dolfinx::MPI::rank(comm);
    T start = node_disp[rank];
    T end = node_disp[rank + 1] - start;
    T null_wgt = 1;
    T ewgt;
    for (T node = 0; node < num_local_nodes; ++node)
    {
       T wn = weights[node];
       ewgt = wn;
       if (wn < 2)
       {
          for (T edge = offsets[node], le = offsets[node + 1]; edge < le; ++edge)
          {
             assert(edge < num_edges);
             adjwgt[edge] = null_wgt;
          }
       }
       else
       {
          for (T edge = offsets[node], le = offsets[node + 1]; edge < le; ++edge)
          {
             assert(edge < num_edges);
             // if local target
             if ((array[edge] >= start) && (array[edge] < end))
             {
                T target = array[edge] - start;
                assert(target < num_local_nodes);
                //adjwgt[edge] = (weights[target] == wn) ? ewgt : null_wgt;
                adjwgt[edge] = (weights[target] == wn) ? null_wgt : ewgt;
             }
             else
                adjwgt[edge] = null_wgt;
             std::cout << "node " << node << " edge " << edge << " linked " << array[edge] <<  " wn " << wn
                       << " wnlinked "
                       << " adjw " << adjwgt[edge] << std::endl;
          }
       }
    }
    T* vwgt = new T[num_local_nodes];
    std::transform(std::execution::unseq, weights.begin(), weights.end(), vwgt,
                   [&ewgt](std::int32_t x) { return static_cast<T>((x > 1) ? 600. : 1); });

    timer2.stop();

    // Call KaHIP to partition graph
    common::Timer timer3("KaHIP: call ParHIPPartitionKWay");
    std::vector<T> part(graph.num_nodes());
    int edgecut = 0;
    //double _imbalance = imbalance;
    double _imbalance = 0.5;
    ParHIPPartitionKWay(node_disp.data(), offsets.data(), array.data(), vwgt,
                        adjwgt, &nparts, &_imbalance, suppress_output, seed,
                        mode, &edgecut, part.data(), &comm);
    timer3.stop();

         delete vwgt;

    if (ghosting)
      return compute_destination_ranks(comm, graph, node_disp, part);
    else
    {
      return regular_adjacency_list(std::vector<int>(part.begin(), part.end()), 1);
    }
  };
}
//----------------------------------------------------------------------------
#endif

graph::partition_fn partitionerImposed(const AdjacencyList<int>& dest)
{
   return [&dest](MPI_Comm comm, int nparts, const graph::AdjacencyList<std::int64_t>& graph, bool ghosting) {
      // Return partion set by user
      return dest;
   };
}
graph::partition_fn partitionerClustering(const std::unordered_set<std::int32_t>& local_cluster)
{
   return [&local_cluster](MPI_Comm comm, int nparts, const graph::AdjacencyList<std::int64_t>& graph, bool ghosting) {
      if (ghosting)
      {
         throw std::runtime_error("partitionerClustering is not implement to treat ghost elements");
      }

      auto pid = dolfinx::MPI::rank(comm);
      auto nbproc = dolfinx::MPI::size(comm);

      // ===========================
      // create clustered dual graph
      // ===========================
      // curent node shift
      auto nn = graph.num_nodes();
      std::vector<std::int32_t> node_disp(nbproc + 1);
      MPI_Allgather(&nn, 1, MPI_INT32_T, node_disp.data() + 1, 1, MPI_INT32_T, comm);
      std::partial_sum(node_disp.begin(), node_disp.end(), node_disp.begin());
      auto startn = node_disp[pid];
      auto endn = node_disp[pid + 1];
      auto endnm1 = endn - 1;

      // node renumbering
      // cluster always the first node (i.e. 0)
      auto cluster_size = local_cluster.size();
      std::int32_t new_nbnode = nn - cluster_size + 1;
      std::vector<std::int32_t> new_local_node_id(nn);
      std::int32_t k = 1;
      for (std::int32_t n = 0; n < nn; ++n)
      {
         if (local_cluster.find(n) == local_cluster.end())
         {
            assert(k < new_nbnode);
            new_local_node_id[n] = k;
            ++k;
         }
         else
            new_local_node_id[n] = 0;
      }
      assert(k == new_nbnode);

      // new node shift
      std::vector<std::int32_t> new_node_disp(nbproc + 1);
      MPI_Allgather(&new_nbnode, 1, MPI_INT32_T, new_node_disp.data() + 1, 1, MPI_INT32_T, comm);
      std::partial_sum(new_node_disp.begin(), new_node_disp.end(), new_node_disp.begin());

      // container to store/exchange new edges
      std::unordered_map<int, std::vector<std::int64_t>> tmp_data;
      std::unordered_set<std::int64_t> data_cluster;
      std::vector<std::unordered_set<std::int64_t>> to_send(nbproc);

      // loop on local_cluster node to merge them in the local cluster
      std::int32_t lnd = new_node_disp[pid];
      for (auto n : local_cluster)
      {
         for (auto ln : graph.links(n))
         {
            // local link node id
            auto ln_loc = ln - startn;

            // if remote
            if (ln > endnm1 || ln_loc < 0)
            {
               // compute proc which owns this  node
               int rr = std::distance(node_disp.begin(), std::ranges::upper_bound(node_disp, ln)) - 1;
               // prepare to send link as a cluster connected link
               to_send[rr].insert(ln);
            }
            // if local
            else
            {
               // keep outside cluster link only.
               if (local_cluster.find(ln_loc) == local_cluster.end()) data_cluster.insert(new_local_node_id[ln_loc] + lnd);
            }
         }
      }
      // reshape prepared data to send them
      std::unordered_map<int, std::vector<std::int64_t>> send_buff;
      std::unordered_map<int, std::vector<std::int64_t>> rsend_buff;
      k = 0;
      for (auto& s : to_send)
      {
         if (s.size())
         {
            send_buff[k].insert(send_buff[k].end(), s.begin(), s.end());
            s.clear();
         }
         ++k;
      }
      to_send.clear();

      // exchange
      twoscale::sendVectMPI3(
          send_buff, MPI_INT64_T, 369, comm,
          // Functor to treat received link
          [&data_cluster, &new_node_disp, &new_local_node_id, &tmp_data, &rsend_buff, startn, endn, lnd](
              const std::vector<std::int64_t>& infos, int from) {
             // remote cluster id
             std::int32_t rnd = new_node_disp[from];

             // loop on received information from 'from'
             for (auto ln : infos)
             {
                auto ln_loc = ln - startn;

                // verify that ln is local to this proc
                assert(ln_loc >= 0 && ln < endn);

                // get new local node id
                auto nln = new_local_node_id[ln_loc];
                // not a clustered node
                if (nln > 0) tmp_data[nln].push_back(rnd);
                //  a clustered node is simply adding remote cluster as a link to local cluster
                else
                   data_cluster.insert(rnd);
                // in all case from must add to ist own connection this node
                rsend_buff[from].push_back(nln + lnd);
             }
          },
          []() {});
      send_buff.clear();

      // exchange back
      twoscale::sendVectMPI3(
          rsend_buff, MPI_INT64_T, 469, comm,
          // Functor to treat received link
          [&data_cluster](const std::vector<std::int64_t>& infos, int from) {
             // all received data are connection to local cluster from remote nodes  either not in remote cluster or from a
             // remote cluster
             data_cluster.insert(infos.begin(), infos.end());
          },
          []() {});
      rsend_buff.clear();

      // special container for pair identification
      std::vector<std::unordered_set<std::pair<std::int64_t, std::int64_t>, twoscale::hashPair<std::int64_t, std::int64_t>,
                                     twoscale::equalPair<std::int64_t, std::int64_t>>>
          to_sendf(nbproc);
      // loop on all local nodes
      for (std::int32_t n = 0; n < nn; ++n)
      {
         // if node outside cluster
         // (node in cluster already treated in previous exchanges)
         if (local_cluster.find(n) == local_cluster.end())
         {
            // get new local node id
            auto nln = new_local_node_id[n];
            assert(nln > 0);  // check not cluster node

            auto l = graph.links(n);
            for (auto ln : l)
            {
               // local link node id
               auto ln_loc = ln - startn;

               // if remote
               if (ln > endnm1 || ln_loc < 0)
               {
                  // compute proc which owns this  node
                  int rr = std::distance(node_disp.begin(), std::ranges::upper_bound(node_disp, ln)) - 1;
                  // prepare to send link
                  // TODO filtering should be done to remove target that are already treated. If in tmp_data it is already
                  // treated. Problem: searching in tmp_data coast if not sorted ! But compare to exchange one extra data and
                  // potentially one extra vector ...
                  to_sendf[rr].insert(std::make_pair(ln, nln + lnd));
               }
               // if local
               else
               {
                  tmp_data[nln].push_back(new_local_node_id[ln_loc] + lnd);
               }
            }
         }
      }

      // reshape prepared data to send them
      k = 0;
      for (auto& s : to_sendf)
      {
         if (s.size())
         {
            send_buff[k].reserve(send_buff[k].size()+2*s.size());
            for (auto& p : s)
            {
               send_buff[k].push_back(p.first);
               send_buff[k].push_back(p.second);
            }
            s.clear();
         }
         ++k;
      }
      to_sendf.clear();

      // exchange
      twoscale::sendVectMPI3(
          send_buff, MPI_INT64_T, 259, comm,
          // Functor to treat received link
          [&new_node_disp, &new_local_node_id, &tmp_data, &rsend_buff, startn, endn, lnd](const std::vector<std::int64_t>& infos,
                                                                                          int from) {
             // loop on received information from 'from' by pair
             assert(infos.size() % 2 == 0);
             auto nb = infos.size();
             for (size_t i = 0; i < nb; i += 2)
             {
                auto ln = infos[i];
                auto ln_loc = ln - startn;

                // verify that ln is local to this proc
                assert(ln_loc >= 0 && ln < endn);

                // get new local node id
                auto nln = new_local_node_id[ln_loc];
                // clustered node are already treated but we did not make the effort of filtering already treated link
                // thus this nln can be part of local cluster and should not be added to tmp_data
                // TODO filtering to_sendf !!!!! or reshape in one pass for all nodes
                if (nln > 0) tmp_data[nln].push_back(infos[i + 1]);
             }
          },
          []() {});
      send_buff.clear();

      // sort all links and pack them into data
      std::vector<std::int64_t> data;
      std::vector<std::int32_t> offsets(new_nbnode + 1);
      offsets[0] = 0;
      // data_cluster is an unordered_set so it is not sorted => copy+sort
      data.insert(data.end(), data_cluster.begin(), data_cluster.end());
      std::ranges::sort(data);
      // tmp_data has to be sorted
      for (k = 1; k < new_nbnode; ++k)
      {
         offsets[k] = data.size();
         auto& list = tmp_data.at(k);
         std::ranges::sort(list);
         data.insert(data.end(), list.begin(), list.end());
      }
      offsets[k] = data.size();

      // create reduced dual graph
      const dolfinx::graph::AdjacencyList<std::int64_t> dual_graph_clust(data, offsets);
      //std::cout << dual_graph_clust.str();

      // ==============================================
      // generated  element weight integrating clusters
      // ==============================================
      std::vector<std::int32_t> weights(new_nbnode, 1);
      // clusters are first
      weights[0] = cluster_size * 100;  // to maintain cluster where they are, ugly but looks like working in tested case.

      // =================================
      // Load balance dual clustered graph
      // =================================
#ifdef HAS_PARMETIS
      const auto partitioner_clust = dolfinx::graph::parmetis::partitionerWithNodeWeight(weights, 1);
#else
#ifdef HAS_KAHIP
      const auto partitioner_clust = dolfinx::graph::kahip::partitionerWithNodeWeight(weights);
#else
#error "PARMETIS or KAHIP required"
#endif
#endif
      auto dest_cluster = partitioner_clust(comm, nbproc, dual_graph_clust, false);
      //std::cout << dest_cluster.str();

      // ==================================
      // generated unclustered distribution
      // ==================================
      std::vector<int> datauc(nn);
      std::vector<std::int32_t> offsetsuc(nn + 1);
      // check that there is only one cluster per process
      auto dri = dest_cluster.links(0);
      if (1)
      {
         int drclust = dri[0];
         MPI_Allreduce(MPI_IN_PLACE, &drclust, 1, MPI_INT, MPI_SUM, comm);
         //std::println("drclust {}", drclust);
         // sum of all targets is normally (nbproc-1)! if all cluster are targeted to be in each process of the group
         for (k = 0; k < nbproc; ++k) drclust -= k;
         if (drclust)
         {
            std::cerr << "Clustering strategy fails !! At least two clusters  are on same process ! delta=" << drclust
                      << std::endl;
            MPI_Abort(comm, -1342);
         }
      }
      offsetsuc[0] = 0;
      datauc[0] = dri[0];
      for (k = 1; k < nn; ++k)
      {
         auto dr = dest_cluster.links(new_local_node_id[k]);
         datauc[k] = dr[0];
         offsetsuc[k] = k;
      }
      offsetsuc[k] = k;
      const dolfinx::graph::AdjacencyList<int> dest_unclust(datauc, offsetsuc);
      //std::cout << dest_unclust.str();
      return dest_unclust;
   };
}

}  // namespace dolfinx::graph
