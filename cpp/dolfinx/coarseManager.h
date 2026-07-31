/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_COARSEMANAGER
#define TS_DOLFINX_COARSEMANAGER
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx_mpc/MultiPointConstraint.h>

#include "debug.h"
#include "scaleJump.h"
#include "patchManager.h"

#ifdef HAS_PETSC
#include <petscmat.h>
#include <petscvec.h>   
#include <petscksp.h>
#else
#error "PETSc is required. No alternative for now"
#endif

namespace twoscale
{
// TODO this class may  be set as a concept and the specialization viewed as an implementation of this concept. But its not clear
// as method argument change in specialization.
//
/// @tparam M The matrix type 
/// @tparam V The vector type 
template <typename M, typename V>
class coarseManager
{
  public:
   coarseManager(M &&PS_, M &&PE_, std::vector<std::int32_t> diag_dof_eliminated);
   ~coarseManager();
   void setStdCoarse(const M &Aff, const V &bf, bool use_imp_enriched=false);
   void resetCoarseToStd();
   void updateEnrichCoarse(const M &Aff, const V &bf);
   void updateEImp();
   template <typename T>
   void solveEImp(T &xf);
   template <typename T>
   void solve(T &xf);
   template <typename T>
   void projectStdCoarse(T xcin, T xf);
};
#ifdef HAS_PETSC
/// Specialization of twoscale::coarseManager class with M,V being respectively a PETSc Mat and Vec
template <>
class coarseManager<Mat,Vec>
{
  public:
#ifdef TWOSCALE_PETSC_MATNEST
   /// Class constructor that check communicator coherance of its arguments and initialize privates members
   /// @param PSR Standard part restriction to passe from coarse to fine scale (interpolation coefficients).
   /// @param PE Enriched part of operator Q to passe from coarse to fine scale (interpolation coefficients).
   /// @param W Vector representing Dirichlet BC values imposed at coarse level and projected on fine scale field (PSD.XDc with PSD
   /// eliminate column of PS and XDc Vector representing Dirichlet BC values imposed at coarse level )
   /// @param diag_dof_eliminated Set of index (global) coresponding to all eliminated rows/columns that will be used to
   /// fill diagonal term of \f$PSR^t.A_{ff}.PSR\f$ block with one.
   /// @param enriched_dof_id_ Map of global dofs index pair (enriched coarse dof,coresponding fine dof) indexed by topological
   /// index of enriched nodes at coarse level (passed to twoscale_dolfinx::patchManager to update \f$PE^t\f$)
   ///
   /// @note User won't call it directelly a priori but will use twoscale_dolfinx::generateCoarseManager function
   coarseManager(Mat &&PSR, Mat &&PE, Vec &&W, std::vector<std::int64_t> diag_dof_eliminated,
                 std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &&enriched_dof_id_);
#else
   /// Class constructor that check communicator coherance of its arguments and initialize privates members
   /// @param PS Standard part of operator Q to passe from coarse to fine scale (interpolation coefficiants).
   /// @param PE Enriched part of operator Q to passe from coarse to fine scale (interpolation coefficiants).
   /// @param Acc Coarse enriched system matrix (structure with null values)
   /// @param W Vector representing Dirichlet BC values imposed at coarse level and projected on fine scale field (L.XDc with L
   /// eliminate column of PS)
   /// @param XDc Vector representing Dirichlet BC values imposed at coarse level
   /// @param diag_dof_eliminated Set of index (local to proc) coresponding to all eliminated rows/columns that will be used to
   /// fill diagonal term of Acc with one.
   /// @param enriched_dof_id_ Map of global dofs index pair (enriched coarse dof,coresponding fine dof) indexed by topological
   /// index of enriched nodes at coarse level (passed to twoscale_dolfinx::patchManager to update \f$PE^t\f$)
   ///
   /// @note User won't call it directelly a priori but will use twoscale_dolfinx::generateCoarseManager function
   coarseManager(Mat &&PS, Mat &&PE, Mat &&Acc, Vec &&W, Vec &&XDc, std::vector<std::int32_t> diag_dof_eliminated,
                 std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &&enriched_dof_id_);
#endif
   coarseManager(coarseManager && other);
   ~coarseManager();
   void setStdCoarse(Mat Aff, Vec bf, bool use_imp_enriched=false);
   void resetCoarseToStd();
   void updateEnrichCoarse(Mat Aff, Vec bf);
   void updateEImp();
   void solveEImp(Vec xf);
   /// Update PE, the enriched part of the Q operator. When called for the first time it transfert PE coefficiant into patch
   /// object owning the row corresponding to coarse enriched node. In following call those coefficiants are used directely
   /// from patches.
   /// @param pm Patch manager used to update PE.
   /// @param func Function applyed to patch solution to obtain enrichment function
   template <typename P, typename F>
   void updateEnrichedOperator(twoscale_dolfinx::patchManager<P> &pm, std::shared_ptr<F> func);
   void solve(Vec xf);
   /// Project, using the standard scale operator, an enriched vector at coarse scale into a vector at fine scale
   /// @param xcs vector at coarse scale representing coarse enriched field
   /// @param xf vector at fine scale representing fine field
   void projectStdCoarse(Vec xcs, Vec xf);

  private:
#ifdef TWOSCALE_PETSC_MATNEST
   Mat PSR, PEt, Acc, AffPSR, Acc_blocks[4], Acc_std;
   Vec W, BMZ, bc, xc, bc_std, bc_enr, xc_std, xc_enr, S, bc_contrib;
   KSP solver, solver_std_only;
#else
   Mat PS, PEt, Acc, PEtAff, AffPE, PEtAffPS, PStAffPE, PEtAffPE;
   Vec W,BMZ,bc,xc,bc_store,XDc;
   KSP solver;
#endif
   PetscScalar one,mone;
   std::vector<PetscInt> dde;
   std::unordered_map<std::int32_t,twoscale_dolfinx::enrichedDofIDs> enriched_dof_id;
#ifndef TWOSCALE_PETSC_MATNEST
   PetscInt mc,Mc;
#endif
   bool moved;
   std::uint16_t state;
   MPI_Comm comm;
   enum
   {
      INIT = 0u,
      ACC_STORED = 1u,
      BC_STORED = 2u,
      ACC_STD = 4u,
      BC_STD = 8u,
      PEt_STORED = 16u,
      ACC_ASS = 32u,
      BC_ASS = 64u,
      PEt_ASS = 128u,
#ifdef TWOSCALE_PETSC_MATNEST
      AffPSR_CREA = 256u,
#else
      PEtAff_CREA = 256u,
#endif
      PEtAffPS_CREA= 512u,
      PStAffPE_CREA= 1024u,
      PEtAffPE_CREA= 2048u
   } managerState;
};
#endif
}  // namespace twoscale
namespace twoscale_dolfinx
{
#ifdef HAS_PETSC
//
/// @note This function implemented is only available with PETSc 
///
/// @tparam T The field scalar type (e.g. float,std::complex<double>,...).
/// @tparam U The mesh geometry/space scalar type (float or double).
///
/// Function that generate a twoscale::coarseManager<Mat, Vec> object from arguments.
#ifdef TWOSCALE_PETSC_MATNEST
/// The coarse scale system (\f$A_{cc}\f$,\f$B_{c}\f$) is in PETSc nested format managed by coarseManager instance created by this
/// function. Only operators used to compute sub blocks are created by this function and provided to coarseManager instance (via
/// the constructor). The field to describe coarse enriched system are split in two independent fields:
/// * A simple coarse regular field to deal with standard dof (std). Its space (blocked) is given by the user (coarse_space).
/// Boundary conditions (Dirichlet) given by the user (bc) are  eliminated at operator level and define two set :
///   * D for Dirichlet eliminated dofs
///   * R for Retained free dofs
/// * A simple coarse regular field to deal with enriched dof (enr). This field is defined only on the support of enriched nodes
/// defined in j (given by user as parameter). Its space is created automatically from coarse_space characteristics and either the
/// sub or full coarse mesh. If all nodes are enriched coarse_space is used for this field.
///
/// The operator that are constructed are thus:
///   * \f$P_{SR}\f$ size nb_fine x (nb_R): pass from retained standard dof to fine scale dof (given by user: fine_space)
///   * \f$P_E^t\f$  size nb_fine x (nb_enr): pass from enriched dof to fine scale dof (given by user: fine_space)
///
/// If bc use non null imposed value the following are also create by this function:
/// * \f$XD_{c}\f$ size nb_D  the vector of imposed value on coarse_space
/// * \f$P_{SD}\f$ size nb_fine x (nb_D) a temporary coarse to fine operator
/// * \f$W=P_{SD}.XR_D\f$  size nb_fine : vector of coarse bc projected on fine field
///
/// @param j The twoscale::scaleJump instance that give all jump information (mesh transition information)
/// @param fine_space The fine scale discretization space
/// @param coarse_space The coarse scale discretization space (regular space i.e. not enriched)
/// @param mpc if any, multi point constrain used to connect fine field dof at support interface
/// @param bc a vector of Dirichlet  boundary conditions, if any, to be imposed at coarse scale 
///
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
twoscale::coarseManager<Mat, Vec> generateCoarseManager(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &j,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> coarse_space,
                                                        std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
                                                        const std::vector< std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>> > &bc);
#else
/// It mainly compute structure of coarse matrix \f$A_{cc}\f$ and scale operator to pass from fine to coarse scale.
///
/// @param j The twoscale::scaleJump instance that give all jump information (mesh transition information)
/// @param fine_space The fine scale discretization space
/// @param coarse_enriched_space The coarse scale enriched disretization space
/// @param mpc if any, multi point constrain used to connect fine field dof at support interface
/// @param bc a vector of Dirichlet  boundary conditions, if any, to be imposed at coarse scale on enriched field
template <dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
twoscale::coarseManager<Mat, Vec> generateCoarseManager(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &j,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space,
                                                        std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> coarse_enriched_space,
                                                        std::shared_ptr<dolfinx_mpc::MultiPointConstraint<T, U>>& mpc,
                                                        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T, U>>> &bc);
#endif
#endif

}  // namespace twoscale_dolfinx

#include "coarseManager_imp.h"
#endif
