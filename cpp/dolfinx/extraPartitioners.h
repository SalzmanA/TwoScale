/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include "dolfinx/graph/partition.h"
#include <array>
#include <unordered_set>

namespace dolfinx::graph
{
#ifdef HAS_PARMETIS
namespace parmetis
{
graph::partition_fn partitionerWithNodeWeight(std::vector<std::int32_t>& weights, int nb_weights = 1, double imbalance = 1.05,
                                              std::array<int, 3> options = {1, 0, 5});

/// Function that provide graph::partition_fn function to partition a mesh with Parmetis with weights for edges contructed on the
/// fly from given node weight: An edge got a weight equal to the weight of its connecte nodes if both are equal and zero
/// otherwise
graph::partition_fn partitionerWithEdgeWeight(std::vector<std::int32_t>& weights, double imbalance = 1.05,
                                              std::array<int, 3> options = {1, 0, 5});
}  // namespace parmetis
#endif
#ifdef HAS_KAHIP
namespace kahip
{
graph::partition_fn partitionerWithNodeWeight(std::vector<std::int32_t>& weights,int mode = 1, int seed = 1,
                                double imbalance = 0.03,
                                bool suppress_output = true);
graph::partition_fn partitioner_with_weight(std::vector<std::int32_t>& weights,int mode = 1, int seed = 1,
                                double imbalance = 0.03,
                                bool suppress_output = true);
} // namespace kahip
#endif

graph::partition_fn partitionerImposed(const AdjacencyList<int>& dest);
graph::partition_fn partitionerClustering(const std::unordered_set<std::int32_t>& local_cluster);

}  // namespace dolfinx::graph
