/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_ENRICHED_FUNCTION_IMP
#define TS_DOLFINX_ENRICHED_FUNCTION_IMP
#ifndef TS_DOLFINX_ENRICHED_FUNCTION
#error "Should not be included by hand"
#endif


namespace twoscale
{
template <typename T>
void enrichedFunction<T>::operator()(std::span<const T> field, std::int32_t enriched_idx, std::span<T> func)
{
   assert(false);  // should never be called => derived yes
}
template <typename T, int bs>
void enrichedShiftFunction<T,bs>::operator()(std::span<const T> field, std::int32_t enriched_idx, std::span<T> func)
{
   auto& shift = shifts;
   auto *ptr=&field[enriched_idx*bs];
   for (std::int32_t b = 0; b < bs; ++b) shifts[b] = ptr[b];
   int i = 0;
   std::ranges::transform(field, func.begin(), [&shift, &i](const T& x) {
      if (!(i < bs)) i = 0;
      return x - shift[i++];
   });
}

}  // namespace twoscale
namespace twoscale_dolfinx
{
template <dolfinx::scalar T>
std::shared_ptr<twoscale::enrichedFunction<T>> generateEnrichedShiftFunction(std::shared_ptr<const dolfinx::fem::Function<T>> fine_field)
{
   twoscale::enrichedFunction<T>* pt;
   switch(fine_field->function_space()->dofmap()->index_map_bs())
   {
      case 1:
         pt = new twoscale::enrichedShiftFunction<T, 1>();
         break;
      case 2:
         pt = new twoscale::enrichedShiftFunction<T, 2>();
         break;
      case 3:
         pt = new twoscale::enrichedShiftFunction<T, 3>();
         break;
   }
   return std::shared_ptr<twoscale::enrichedFunction<T>>(pt);
}
}

#endif
