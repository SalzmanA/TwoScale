/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_CPP23
#define TS_CPP23
// concept ===========================================================
template <typename P>
concept size_range =
    std::ranges::sized_range<P>;
// mimic c++23 print if not c++23 compiler ===========================
#ifndef __cpp_lib_print
#include <stdio.h>
#include <format>
#include <complex>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <utility>
namespace debug_imp
{
template <class T>
concept cplx_c = std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>;
}
namespace std
{
template <typename... _Args>
inline void print(ostream& __os , format_string<_Args...> __fmt, _Args&&... __args)
{
#if __GNUC__ > 13
   __os << std::format(__fmt, std::forward<_Args>(__args)...);
#endif
}
template <typename... _Args>
inline void println(ostream& __os, format_string<_Args...> __fmt, _Args&&... __args)
{
#if __GNUC__ > 13
   __os << std::format(__fmt, std::forward<_Args>(__args)...) << "\n";
#endif
}

template <typename... _Args>
inline void print(FILE* __out, format_string<_Args...> __fmt, _Args&&... __args)
{
#if __GNUC__ > 13
   std::string buffer;
   std::format_to(std::back_inserter(buffer), __fmt, std::forward<_Args>(__args)...);
   fprintf(__out, "%s", buffer.c_str());
#endif
}
template <typename... _Args>
inline void println(FILE* __out, format_string<_Args...> __fmt, _Args&&... __args)
{
#if __GNUC__ > 13
   std::string buffer;
   std::format_to(std::back_inserter(buffer), __fmt, std::forward<_Args>(__args)...);
   fprintf(__out, "%s\n", buffer.c_str());
#endif
}

template <typename... _Args>
inline void print(format_string<_Args...> __fmt, _Args&&... __args)
{
   std::print(stdout, __fmt, std::forward<_Args>(__args)...);
}

template <typename... _Args>
inline void println(format_string<_Args...> __fmt, _Args&&... __args)
{
   std::println(stdout, __fmt, std::forward<_Args>(__args)...);
}

// formatter specialization =====================
// may be mandatory also in c++23. Wait and see
template <size_range P, __format::__char _CharT>
struct formatter<P, _CharT>
{
   bool to_int = false;
   template <class ParseContext>
   constexpr ParseContext::iterator parse(ParseContext& ctx)
   {
      auto it = ctx.begin();
      if (it == ctx.end()) return it;
      if (*it == 'd')
      {
         to_int = true;
         ++it;
      }
      if (it != ctx.end() && *it != '}') throw std::format_error("Invalid format args for range.");
      return it;
   }
    template<class FmtContext>
    FmtContext::iterator format(P& to_print, FmtContext& ctx) const
    {
       std::ostringstream out;
       out<<"[";
       bool do_print=true;
       if (to_int)
          for (auto s : to_print) 
          {
             if (do_print)
             {
                out << std::setw(2) << std::format((std::is_integral_v<typename P::value_type>) ? "{:d}" : "{}", s);
                do_print = false;
             }
             else
                out << "," << std::setw(2) << std::format((std::is_integral_v<typename P::value_type>) ? "{:d}" : "{}", s);
          }
       else
       {
          for (auto s : to_print)
          {
             if (do_print)
             {
                do_print = false;
                out << std::setw(2) << std::format("{}", s);
             }
             else
                out << "," << std::setw(2) << std::format("{}", s);
          }
       }
       out<<"]";

       return std::ranges::copy(std::move(out).str(), ctx.out()).out;
    }
};
template <typename _A, __format::__char _CharT>
struct formatter<std::complex<_A>, _CharT>
{
   bool with_i = false;
   template <class ParseContext>
   constexpr ParseContext::iterator parse(ParseContext& ctx)
   {
      auto it = ctx.begin();
      if (it == ctx.end()) return it;
      if (*it == '#')
      {
         with_i = true;
         ++it;
      }
      if (it != ctx.end() && *it != '}') throw std::format_error("Invalid format args for complex.");
      return it;
   }
   template <class FmtContext>
   FmtContext::iterator format(std::complex<_A>& to_print, FmtContext& ctx) const
   {
      std::ostringstream out;
      if (with_i)
         if (to_print.imag() < 0.)
            out << to_print.real() << to_print.imag() << "i";
         else
            out << to_print.real() << "+" << to_print.imag() << "i";
      else
         out << "(" << to_print.real() << "," << to_print.imag() << ")";

      return std::ranges::copy(std::move(out).str(), ctx.out()).out;
    }
};
template < __format::__char _CharT, typename... _Ts>
struct formatter<std::pair<_Ts...>,_CharT>
{
   bool to_int = false;
   template <class ParseContext>
   constexpr ParseContext::iterator parse(ParseContext& ctx)
   {
      auto it = ctx.begin();
      if (it == ctx.end()) return it;
      if (*it == 'd') 
      {
         to_int = true;
         ++it;
      }
      if (it != ctx.end() && *it != '}') throw std::format_error("Invalid format args for pair.");
      return it;
   }
    template<class FmtContext>
    FmtContext::iterator format(std::pair<_Ts...>& to_print, FmtContext& ctx) const
    {
       std::ostringstream out;
       if (to_int)
       {
          out << std::format((std::is_integral_v<typename std::pair<_Ts...>::first_type>) ? "[{:d}:" : "[{}:",
                             std::get<0>(to_print));
          out << std::format((std::is_integral_v<typename std::pair<_Ts...>::second_type>) ? "{:d}]" : "{}]",
                             std::get<1>(to_print));
       }
       else
          out << std::format("[{}:{}]", std::get<0>(to_print) , std::get<1>(to_print));

       return std::ranges::copy(std::move(out).str(), ctx.out()).out;
    }
};

}  // namespace std
#else
#error "use of c++23 print needs to be checked"
// use c++23 print if  c++23 compiler ===========================
#include <ostream>
#include <print>
#endif

#endif
