/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_BIND_COARSEMANAGER_H
#define TS_BIND_COARSEMANAGER_H
#include "bind_util.h"
#include <nanobind/stl/vector.h>


#include "twoscale/dolfinx/coarseManager.h"
#include "twoscale/dolfinx/enrichedFunctions.h"


namespace implbindts
{

template <typename M, typename V, typename P, typename F>
void declare_coarseManager(nb::module_ &m, std::string type)
{
   std::string pyclass_name = std::string("coarseManager_") + type;
   nb::class_<twoscale::coarseManager<M, V>>(m, pyclass_name.c_str(), "typed coarseManager object")
       .def("setStdCoarse", &twoscale::coarseManager<M, V>::setStdCoarse)
       .def("resetCoarseToStd", &twoscale::coarseManager<M, V>::resetCoarseToStd)
       .def("updateEnrichedOperator",
            [](twoscale::coarseManager<M, V> &self, twoscale_dolfinx::patchManager<P> &pm, std::shared_ptr<F> func) -> void {
               self.updateEnrichedOperator(pm, func );
            })
       .def("updateEnrichCoarse", &twoscale::coarseManager<M, V>::updateEnrichCoarse)
       .def("solve", &twoscale::coarseManager<M, V>::solve)
       .def("updateEImp", &twoscale::coarseManager<M, V>::updateEImp)
       .def("solveEImp", &twoscale::coarseManager<M, V>::solveEImp)
       .def("projectStdCoarse", &twoscale::coarseManager<M, V>::projectStdCoarse);
}


template <typename T, typename U>
void declare_generate_coarseManager(nb::module_ &m)
{
#if defined(HAS_PETSC)
   m.def(
       "generateCoarseManager",
       [](const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &j, std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space,
          std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> coarse_enriched_space,
          std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>> &mpc,
          const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>> &bc) {
          return twoscale_dolfinx::generateCoarseManager(j, fine_space, coarse_enriched_space, mpc, bc);
       },
       nb::arg("j"), nb::arg("fine_space"), nb::arg("coarse_enriched_space"), nb::arg("mpc").none(), nb::arg("bc"));
#endif
}
}  // namespace implbindts
namespace bindts
{
void setCoarseManager(nb::module_ &m);
}  // namespace bindts
#endif
