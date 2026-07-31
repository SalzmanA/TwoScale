/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_PER_MACRO
#define TS_DOLFINX_PER_MACRO
#ifdef TWOSCALE_DOLFINX
namespace twoscale_dolfinx
{
class dataTsPerMacro
{
  public:
   double residual;  //!< residual for this element
};
}  // namespace twoscale
#else
#error "no implementation available"
#endif
#endif
