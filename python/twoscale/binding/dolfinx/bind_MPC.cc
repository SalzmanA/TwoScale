/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#include "bind_MPC.h"

namespace bindts
{
void setMPC(nb::module_ &m)
{
   // generate function
   implbindts::declare_generate_MPC<float,float>(m);
   implbindts::declare_generate_MPC<std::complex<float>,float>(m);
   implbindts::declare_generate_MPC<double,double>(m);
   implbindts::declare_generate_MPC<std::complex<double>,double>(m);
}
}  // namespace bindts
