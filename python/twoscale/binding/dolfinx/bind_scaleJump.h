/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */

#ifndef TS_BIND_SCALEJUMP_H
#define TS_BIND_SCALEJUMP_H
#include "bind_util.h"
#include "twoscale/dolfinx/scaleJump.h"

namespace implbindts
{
template <typename M>
void declare_scaleJump(nb::module_ &m, std::string type)
{
   std::string pyclass_name = std::string("scaleJump_") + type;
   nb::class_<twoscale::scaleJump<M>>(m, pyclass_name.c_str(), "typed scaleJump object")
       .def_ro("needs_mpc", &twoscale::scaleJump<M>::needs_mpc,
               "bool indicating if MPC needs to be created (i.e. refined mesh is in a submesh or not)")
       .def("getFineMesh", &twoscale::scaleJump<M>::getFineMesh)
       .def("getCoarseMesh", &twoscale::scaleJump<M>::getCoarseMesh)
       .def("isFineMasterNode", &twoscale::scaleJump<M>::isFineMasterNode)
       .def(
           "getChildren",
           [](twoscale::scaleJump<M> &self, std::int32_t idx_coarse) -> intArray {
              auto childs = self.getChildren(idx_coarse);
              RETURNIA(childs)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getFaceChilds",
           [](twoscale::scaleJump<M> &self, std::int32_t idx_coarse) -> intArray {
              auto childs = self.getFaceChilds(idx_coarse);
              RETURNIA(childs)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getSurroundingFaceChilds",
           [](twoscale::scaleJump<M> &self, std::int32_t idx_coarse) -> intArray {
              auto childs = self.getSurroundingFaceChilds(idx_coarse);
              RETURNIA(childs)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getEnriched",
           [](twoscale::scaleJump<M> &self) -> intArray {
              auto enriched =self.getEnriched();
              RETURNIA(enriched)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getExtraEnriched",
           [](twoscale::scaleJump<M> &self) -> intArray {
              auto enriched =self.getExtraEnriched();
              RETURNIA(enriched)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getSupport",
           [](twoscale::scaleJump<M> &self) -> intArray {
              auto cells =self.getSupport();
              RETURNIA(cells)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getSurroundingCells",
           [](twoscale::scaleJump<M> &self) -> intArray {
              auto cells =self.getSurroundingCells();
              RETURNIA(cells)
           },
           nb::rv_policy::reference_internal)
       .def(
           "getCoarseMaster",
           [](twoscale::scaleJump<M> &self) -> intArray {
              auto faces =self.getCoarseMaster();
              RETURNIA(faces)
           },
           nb::rv_policy::reference_internal);
}

template <typename T, typename Z>
void declare_topDown(nb::module_ &m)
{
   m.def(
       "topDown",
       [](dolfinx::mesh::Mesh<T> &mesh,
          std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(nb::ndarray<const T, nb::ndim<2>, nb::numpy>)> enriched,
          std::function<nb::ndarray<bool, nb::ndim<1>, nb::c_contig>(nb::ndarray<const T, nb::ndim<2>, nb::numpy>, std::uint8_t)>
              crit,
          std::uint8_t level, bool clustering_dual_graph, bool accurate_weight,
          dolfinx::mesh::MeshTags<Z> const *tags) -> twoscale::scaleJump<dolfinx::mesh::Mesh<T>> {
          // transcript enriched python callable into a c++ function following MarkerFn concept via a lambda
          auto cpp_enriched = [&enriched](auto x) {
             nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(x.data_handle(), {x.extent(0), x.extent(1)}, nb::handle());
             auto marked = enriched(x_view);
             return std::vector<std::int8_t>(marked.data(), marked.data() + marked.size());
          };
          // Control variable passed to c++ function to drive crit behavior
          std::uint8_t control=0u;
          // transcript crit python callable into a c++ function following MarkerFn concept via a lambda
          auto cpp_crit = [&crit,&control](auto x) {
             nb::ndarray<const T, nb::ndim<2>, nb::numpy> x_view(x.data_handle(), {x.extent(0), x.extent(1)}, nb::handle());
             auto marked = crit(x_view,control);
             return std::vector<std::int8_t>(marked.data(), marked.data() + marked.size());
          };

          // call c++ function with adapted argument
          return twoscale_dolfinx::topDown(mesh, cpp_enriched, cpp_crit, control, level, clustering_dual_graph, accurate_weight,
                                           tags);
       },
       nb::arg("mesh"), nb::arg("enriched"), nb::arg("crit"), nb::arg("level"), nb::arg("clustering_dual_graph"),
       nb::arg("accurate_weight"), nb::arg("tags").none());
}

}  // namespace implbindts
namespace bindts
{
void setScaleJump(nb::module_ &m);
}
#endif
