/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_BIND_ENRICHFUNC_H
#define TS_BIND_ENRICHFUNC_H
#include "bind_util.h"

#include "dolfinx/fem/Function.h"
#include "twoscale/dolfinx/enrichedFunctions.h"

namespace implbindts
{

template <typename T>
void declare_enrichedFunctions(nb::module_ &m, std::string type)
{
   std::string pyclass_name = std::string("enrichedFunction_") + type;
   nb::class_<twoscale::enrichedFunction<T>>(m, pyclass_name.c_str(), "typed enrichedFunction object");
       //.def("operator()", &twoscale::enrichedFunction<T>::operator());
}

template <typename T, typename U>
void declare_generate_enrich(nb::module_ &m)
{
   m.def("generateEnrichedShiftFunction",
         [](std::shared_ptr<const dolfinx::fem::Function<T>> field) -> std::shared_ptr<twoscale::enrichedFunction<T>> {
            return twoscale_dolfinx::generateEnrichedShiftFunction(field);
         });
}
}  // namespace implbindts
namespace bindts
{
void setEnrichFunc(nb::module_ &m);
}  // namespace bindts
#endif
