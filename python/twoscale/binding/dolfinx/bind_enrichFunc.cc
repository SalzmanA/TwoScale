/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#include "bind_enrichFunc.h"

  // namespace implbindts
namespace bindts
{
void setEnrichFunc(nb::module_ &m)
{
   // Base Class
   implbindts::declare_enrichedFunctions<float>(m, "float32");
   implbindts::declare_enrichedFunctions<double>(m, "float64");
   implbindts::declare_enrichedFunctions<std::complex<float>>(m, "complex32");
   implbindts::declare_enrichedFunctions<std::complex<double>>(m, "complex64");

   // generator
   implbindts::declare_generate_enrich<float, float>(m);
   implbindts::declare_generate_enrich<std::complex<float>, float>(m);
   implbindts::declare_generate_enrich<double,double>(m);
   implbindts::declare_generate_enrich<std::complex<double>, double>(m);
}
}  // namespace bindts
