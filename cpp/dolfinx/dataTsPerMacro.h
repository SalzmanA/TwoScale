/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
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
