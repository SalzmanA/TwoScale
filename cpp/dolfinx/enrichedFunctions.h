/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_ENRICHED_FUNCTION
#define TS_DOLFINX_ENRICHED_FUNCTION

#include <cassert>
#include <memory>

namespace twoscale
{
template <typename T>
class enrichedFunction
{

  public:
   typedef T value_t;
   virtual void operator()(std::span<const T> field, std::int32_t enriched_idx, std::span<T> func);
};
template <typename T, int bs>
class enrichedShiftFunction : public enrichedFunction<T>
{
  public:
   void operator()(std::span<const T> field, std::int32_t enriched_idx, std::span<T> func) override;

  private:
   std::array<T, bs> shifts;
};

}  // namespace twoscale
namespace twoscale_dolfinx
{
/// @tparam T The field scalar type (e.g. float,std::complex<double>,...).
template <dolfinx::scalar T>
std::shared_ptr<twoscale::enrichedFunction<T>>  generateEnrichedShiftFunction(std::shared_ptr<const dolfinx::fem::Function<T>> fine_field);
}

#include "enrichedFunctions_imp.h"
#endif
