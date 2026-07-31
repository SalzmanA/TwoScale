/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#include "bind_patchManager.h"

// namespace implbindts
namespace bindts
{
void setPatchManager(nb::module_ &m)
{
   // patchManager class
   implbindts::declare_patchManager<twoscale::patchMPIDirect>(m, "MPIDirect");

   // generate function
   implbindts::declare_generate_patch<twoscale::patchMPIDirect, float>(m);
   implbindts::declare_generate_patch<twoscale::patchMPIDirect, double>(m);
}
}  // namespace bindts
