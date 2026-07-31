/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#include "bind_coarseManager.h"

namespace bindts
{
void setCoarseManager(nb::module_ &m)
{

   // coarseManager class
#if defined(HAS_PETSC)
   implbindts::declare_coarseManager<Mat, Vec, twoscale::patchMPIDirect, twoscale::enrichedFunction<PetscScalar>>(m, "petsc");

   implbindts::declare_generate_coarseManager<PetscScalar,PetscReal>(m);
#endif
}
}  // namespace bindts
