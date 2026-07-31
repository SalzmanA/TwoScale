/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include "debug.h"
#include <iomanip>

// unique specialization implementation
template <>
void PRINT(const std::string& msg, std::vector<std::int8_t>& to_print)
{
   std::cout << msg << " " << to_print.size() << ": ";
   for (auto s : to_print) std::cout <<std::setw(2)<< (int)s << " ";
   std::cout << std::endl;
   return;
}
