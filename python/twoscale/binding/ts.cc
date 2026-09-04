/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include <nanobind/nanobind.h>
namespace nb = nanobind;

namespace bindts
{
void setDolfinx(nb::module_ &m);
}

NB_MODULE(ts_cpp, m)
{
   m.attr("__version__") = TWOSCALE_VERSION;
#ifdef TWOSCALE_DOLFINX
   nb::module_ dolfinx = m.def_submodule("dolfinx", "twoscale dolfinx version module ");
   bindts::setDolfinx(dolfinx);
#endif
}
