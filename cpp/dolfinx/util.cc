/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include "util.h"
namespace twoscale_dolfinx
{
bool useNest()
{
#ifdef TWOSCALE_PETSC_MATNEST
   return true;
#else
   return false;
#endif
}
}  // namespace twoscale_dolfinx
