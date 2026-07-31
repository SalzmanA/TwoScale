/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
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
