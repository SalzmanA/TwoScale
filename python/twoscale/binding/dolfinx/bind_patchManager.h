/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_BIND_PATCHMANAGER_H
#define TS_BIND_PATCHMANAGER_H
#include "bind_util.h"
#include "twoscale/dolfinx/patchManager.h"
#include "twoscale/dolfinx/scaleJump.h"

namespace implbindts
{
template <typename P>
void declare_patchManager(nb::module_ &m, std::string type)
{
   std::string pyclass_name = std::string("patchManager_") + type;
   nb::class_<twoscale_dolfinx::patchManager<P>>(m, pyclass_name.c_str(), "typed patchManager object")
#ifdef HAS_PETSC
       .def("generateProblems", &twoscale_dolfinx::patchManager<P>::generateProblems)
       .def("solveProblems", &twoscale_dolfinx::patchManager<P>::solveProblems)
       .def("grabPatchSolution", &twoscale_dolfinx::patchManager<P>::grabPatchSolution)
#endif
       .def("numberOfSequence", &twoscale_dolfinx::patchManager<P>::numberOfSequence)
       ;
}

template <typename P, typename U>
void declare_generate_patch(nb::module_ &m)
{
   m.def(
       "generatePatchManager",
       [](const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &sj, std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space)
           -> twoscale_dolfinx::patchManager<P> { return twoscale_dolfinx::generatePatchManager<P, U>(sj, fine_space); },
       nb::arg("sj"), nb::arg("fine_space"));
}
}  // namespace implbindts
namespace bindts
{
void setPatchManager(nb::module_ &m);

}  // namespace bindts
#endif
