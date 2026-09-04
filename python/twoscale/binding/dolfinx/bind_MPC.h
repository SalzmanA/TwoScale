/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_BIND_MPC_H
#define TS_BIND_MPC_H
#include "bind_util.h"
#include "twoscale/dolfinx/MPC.h"

namespace implbindts
{
template <typename T, typename U>
void declare_generate_MPC(nb::module_ &m)
{
   m.def(
       "generateMPC",
       [](const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &j, std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> space)
           -> dolfinx_mpc::mpc_data<T> { return twoscale_dolfinx::generateMPC<T>(j, space); },
       nb::arg("j"), nb::arg("field"));
}
}  // namespace implbindts
namespace bindts
{
void setMPC(nb::module_ &m);
}  // namespace bindts
#endif
