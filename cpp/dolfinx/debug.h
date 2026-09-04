/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DEBUG
#define TS_DEBUG
#include <cstdint>
#include <iostream>
#include <ranges>
#include <span>
#include <string>
#include <vector>
#include "cpp23.h"
// concept ===========================================================
// any type compatible with a range having size method but not beiing std::vector<std::int8_t>>&
template <typename P>
concept size_range_of_anything_but_int8 =
    std::ranges::sized_range<P> && not std::same_as<std::remove_reference_t<P>, std::vector<std::int8_t>>;
// generic ============================
template <typename P>
void PRINT(const std::string& msg, P&& to_print)
{
   std::cout << msg << ": " << to_print << std::endl;
   //std::println("{}: {}",msg,to_print); //KO specialization not working for std::complex<double> at least ??
   return;
}
// specialization =====================
template <>
void PRINT(const std::string& msg, std::vector<std::int8_t>& to_print);

// overloading ========================
template <size_range_of_anything_but_int8 P>
void PRINT(const std::string& msg, P&& to_print)
{
   std::cout << msg << " " << to_print.size() << ": ";
   for (auto s : to_print) std::cout << s << " ";
   std::cout << std::endl;
   return;
}

#include "basix/mdspan.hpp"

template <typename T>
void PRINT(const std::string& msg,
           MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>& to_print)
{
   assert(to_print.rank() == 4);
   std::cout << msg << " rank 4-" << to_print.extent(0) << "x" << to_print.extent(1) << "x" << to_print.extent(2) << "x"
             << to_print.extent(3) << ": " << std::endl;
   for (size_t i = 0; i < to_print.extent(0); ++i)
   {
      for (size_t j = 0; j < to_print.extent(1); ++j)
      {
         std::cout << " ";
         for (size_t k = 0; k < to_print.extent(2); ++k)
         {
            std::cout << " ";
            for (size_t l = 0; l < to_print.extent(3); ++l) std::cout << " " << to_print(i, j, k, l);
            std::cout << std::endl;
         }
      }
   }
   return;
}
template <typename T>
void PRINT(const std::string& msg,
           MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>& to_print)
{
   assert(to_print.rank() == 3);
   std::cout << msg << " rank 3-" << to_print.extent(0) << "x" << to_print.extent(1) << "x" << to_print.extent(2) << ": " << std::endl;
   for (size_t i = 0; i < to_print.extent(0); ++i)
   {
      for (size_t j = 0; j < to_print.extent(1); ++j)
      {
         std::cout << " ";
         for (size_t k = 0; k < to_print.extent(2); ++k) std::cout << " " << to_print(i, j, k);
         std::cout << std::endl;
      }
   }
   return;
}
template <typename T>
void PRINT(const std::string& msg,
           MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>& to_print)
{
   assert(to_print.rank() == 2);
   std::cout << msg << " rank 2-" << to_print.extent(0) << "x" << to_print.extent(1) << ": " << std::endl;
   for (size_t i = 0; i < to_print.extent(0); ++i)
   {
      for (size_t j = 0; j < to_print.extent(1); ++j)
      {
         std::cout << " " << to_print(i, j);
      }
      std::cout << std::endl;
   }
   return;
}
template <typename T>
void PRINT(const std::string& msg,
           MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 1>>& to_print)
{
   assert(to_print.rank() == 1);
   std::cout << msg << " rank 1-" << to_print.extent(0) << ": " << std::endl;
   for (size_t i = 0; i < to_print.extent(0); ++i)
   {
      std::cout << " " << to_print(i);
   }
   std::cout << std::endl;
   return;
}

#endif
