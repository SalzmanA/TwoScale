/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_MPC
#define TS_DOLFINX_MPC
#include <dolfinx/fem/Function.h>
#include <dolfinx_mpc/utils.h>

#include <memory>

#include "debug.h"
#include "scaleJump.h"
// namespace twoscale
namespace twoscale_dolfinx
{
/// @tparam T The field scalar type (e.g. float,std::complex<double>,...).
/// @tparam U The mesh geometry/space scalar type (float or double).
/// @param j The scaleJump object describing coarse/fine mesh and their relations
/// @param space The fine scale disretization space
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_t<T>>
dolfinx_mpc::mpc_data<T> generateMPC(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &j,
                                                    std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> space);

}  // namespace twoscale_dolfinx

#include "MPC_imp.h"
#endif
