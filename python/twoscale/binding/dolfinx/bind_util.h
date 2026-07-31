/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_BIND_UTIL_H
#define TS_BIND_UTIL_H
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>

#if defined(HAS_PETSC)
#include <petscmat.h>
#if defined(HAS_DOLFINX_PY)
// dolfinx warper to encapsulate Mat and Vec petsc types into nanobind type_caster
#include "caster_petsc.h"
#endif
#endif

#if (NB_VERSION_MAJOR > 2) || (NB_VERSION_MINOR > 1)
#define IA(d) intArray(d.data(), {d.size()})
#else
#define IA(d) intArray(d.data(), {d.size()}, {})
#endif
#define RETURNIA(d) return IA(d);

namespace nb = nanobind;
namespace implbindts
{
using intArray = nb::ndarray<const std::int32_t, nb::numpy, nb::ndim<1>, nb::c_contig>;
}
#endif
