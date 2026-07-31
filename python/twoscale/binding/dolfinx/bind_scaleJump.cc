/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#include "bind_scaleJump.h"

namespace bindts
{
void setScaleJump(nb::module_ &m)
{
   // scaleJump class
   implbindts::declare_scaleJump<dolfinx::mesh::Mesh<float>>(m, "float32");
   implbindts::declare_scaleJump<dolfinx::mesh::Mesh<double>>(m, "float64");

   // topDown function
   implbindts::declare_topDown<float, std::int64_t>(m);
   implbindts::declare_topDown<double, std::int64_t>(m);
   implbindts::declare_topDown<float, float>(m);
   implbindts::declare_topDown<double, float>(m);
   implbindts::declare_topDown<float, double>(m);
   implbindts::declare_topDown<double, double>(m);

}
}  // namespace bindts
