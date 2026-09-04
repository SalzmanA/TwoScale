/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#include "bind_util.h"
#include "twoscale/dolfinx/util.h"
namespace bindts
{
void setMPC(nb::module_ &m);
void setPatchManager(nb::module_ &m);
void setCoarseManager(nb::module_ &m);
void setScaleJump(nb::module_ &m);
void setEnrichFunc(nb::module_ &m);
void setUtil(nb::module_ &m);
}  // namespace bindts

namespace bindts
{
void setDolfinx(nb::module_ &m)
{
   m.doc() = "Twoscale implementation based on dolfinx";

   // scaleJump class and generator
   setScaleJump(m);

   // coarseManager class and generator
   setCoarseManager(m);

   // enriched function Class and generator
   setEnrichFunc(m);

   // patchManager class and generator
   setPatchManager(m);

   // MPC generator
   setMPC(m);

   // Util 
   // as it may be very limited in time no setUtil implementation
   m.def("useNest",&twoscale_dolfinx::useNest);
}
}  // namespace bindts
