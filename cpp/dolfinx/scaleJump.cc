/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include <iostream>
#include "scaleJump.h"
namespace twoscale_dolfinx
{
namespace impl
{
void extendSupport(std::shared_ptr<const dolfinx::mesh::Topology> topo,
                   std::shared_ptr<const dolfinx::graph::AdjacencyList<std::int32_t>> adj, int dim, std::vector<std::int32_t> &support,
                   std::unordered_set<std::int32_t> &u_support)
{
   auto idxmap0ct = topo->index_map(0);
   auto adj_cell = topo->connectivity(dim, 0);
   dolfinx::common::Scatterer coarse_scatter_topot(*idxmap0ct, 1);
   std::unordered_set<std::int32_t> tmp;
   auto nbncl = idxmap0ct->size_local();
   auto nbnc = nbncl + idxmap0ct->num_ghosts();
   for (auto si : support)
   {
      const auto li = adj_cell->links(si);
      tmp.insert(li.begin(), li.end());
   }
   std::vector<std::int8_t> mark(nbnc, -1);
   for (auto t : tmp) mark[t] = 1;
   twoscale_dolfinx::scatter_rev_fwd(coarse_scatter_topot,std::span<std::int8_t>(mark.begin(), mark.begin() + nbncl),
                                    std::span<std::int8_t>(mark.begin() + nbncl, mark.end()),
                                    [](const std::int8_t &i, const std::int8_t &j) {
                                       if (j > -1)
                                          return j;
                                       else
                                          return i;
                                    });

   std::int32_t k = -1;
   for (auto m : mark)
   {
      ++k;
      if (m > -1)
      {
         const auto li = adj->links(k);
         u_support.insert(li.begin(), li.end());
      }
   }
   support.clear();
   support.reserve(u_support.size());
   support.insert(support.end(), u_support.begin(), u_support.end());
}
}
}  // namespace twoscale_dolfinx
