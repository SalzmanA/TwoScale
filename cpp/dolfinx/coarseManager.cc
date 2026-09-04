/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include "coarseManager.h"

#include "debug.h"
#include "mpi.h"
#include "util.h"

#ifdef HAS_PETSC
#include "petscis.h"
#endif

//#define TWOSCALE_SCOREP_PROFILE
//#define TWOSCALE_HAND_PROFILE
#ifdef TWOSCALE_HAND_PROFILE
#include <dolfinx/common/TimeLogger.h>
#include <dolfinx/common/Timer.h>
#endif


namespace twoscale
{
#ifdef HAS_PETSC
#ifdef TWOSCALE_SCOREP_PROFILE
void f_PEtAff1(Mat PEt, Mat Aff,  Mat *PEtAff, MPI_Comm comm)
{
   DO(MatMatMult(PEt, Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, PEtAff), "Problem while computing the product PEt.Aff", comm)
}
void f_PEtAff2(Mat PEt, Mat Aff, Mat *PEtAff, MPI_Comm comm)
{
   DO(MatMatMult(PEt, Aff, MAT_REUSE_MATRIX, PETSC_DETERMINE, PEtAff), "Problem while computing the product PEt.Aff", comm)
}
#endif
// ==========================================================================
#ifdef TWOSCALE_PETSC_MATNEST
coarseManager<Mat, Vec>::coarseManager(Mat &&PSR_, Mat &&PEt_, Vec &&W_, std::vector<std::int64_t> diag_dof_eliminated,
                                       std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &&enriched_dof_id_)
    : PSR(PSR_),
      PEt(PEt_),
      AffPSR(nullptr),
      Acc_std(nullptr),
      W(W_),
      BMZ(nullptr),
      bc(nullptr),
      xc(nullptr),
      bc_std(nullptr),
      bc_enr(nullptr),
      xc_std(nullptr),
      xc_enr(nullptr),
      S(nullptr),
      bc_contrib(nullptr),
      solver(nullptr),
      solver_std_only(nullptr),
      one(1.0),
      mone(-1.0),
      dde(diag_dof_eliminated.begin(), diag_dof_eliminated.end()),
      enriched_dof_id(std::move(enriched_dof_id_)),
      moved(false),
      state(INIT)
{
   PetscErrorCode ierr;
   DO(PetscObjectGetComm((PetscObject)PSR, &comm), "Problem while getting the communicator of PSR, thus no choice abort on world",
      MPI_COMM_WORLD)

   int rcomp;
   MPI_Comm comm_test;
   DO(PetscObjectGetComm((PetscObject)PEt, &comm_test), "Problem while getting the communicator of PEt", comm)
   MPI_Comm_compare(comm, comm_test, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT), "Communicator must be at least congruent for coarsening operators",
           comm)


   //std::println("PEt matrix");
   //MatView(PEt, PETSC_VIEWER_STDOUT_(comm));

   // Init solver
#if 0
   DO(KSPCreate(comm, &solver), "Problem while creating solver", comm)
   DO(KSPSetOptionsPrefix(solver, "ts_coarse_"), "Problem while setting prefix to solver", comm)
   //DO(KSPSetFromOptions(solver), "Problem while setting 'from' option  solver", comm)
   DO(KSPSetType(solver,KSPPREONLY), "Problem while setting solver type", comm)
   PC pc;
   DO(KSPGetPC(solver, &pc), "Problem while getting PC", comm)
   DO(PCSetType(pc, PCLU), "Problem while setting PC type", comm)
   DO(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS), "Problem while setting PCLU type", comm)
#else
   DO(KSPCreate(comm, &solver), "Problem while creating solver", comm)
   DO(KSPSetOptionsPrefix(solver, "ts_coarse_"), "Problem while setting prefix to solver", comm)
   DO(KSPSetTolerances(solver, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, 2000), "Problem while setting tolerances to solver", comm)
   DO(KSPSetType(solver, KSPGMRES),"Problem while imposing solver to be GMRES",comm);
   PC pc;
   DO(KSPGetPC(solver, &pc), "Problem while getting PC", comm)
   DO(PCSetOptionsPrefix(pc, "ts_coarse_"), "Problem while setting prefix to pc", comm)
   DO(PCSetType(pc, PCFIELDSPLIT), "Problem while setting fieldsplit pc", comm)
   DO(PCFieldSplitSetType(pc, PC_COMPOSITE_ADDITIVE), "Problem while setting fieldsplit pc as additive", comm)
   DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_0_pc_type", "lu"),"Problem while setting fieldsplit0 pc type", comm)
   DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_0_pc_factor_mat_solver_type", "mumps"),"Problem while setting fieldsplit0 pc lu type", comm)
   DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_1_pc_type", "lu"),"Problem while setting fieldsplit1 pc type", comm)
   DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_1_pc_factor_mat_solver_type", "mumps"),"Problem while setting fieldsplit1 pc lu type", comm)
   //DO(PetscOptionsSetValue(NULL, "-ts_coarse_ksp_view", NULL),"Problem while setting ksp_view", comm)
   //DO(PetscOptionsSetValue(NULL, "-ts_coarse_ksp_monitor", NULL),"Problem while setting ksp_monitor", comm)
   //DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_0_ksp_monitor", NULL),"Problem while setting ksp_monitor sf0", comm)
   //DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_1_ksp_monitor", NULL),"Problem while setting ksp_monitor sf1", comm)
   //DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_0_ksp_view", NULL),"Problem while setting ksp_view sf0", comm)
   //DO(PetscOptionsSetValue(NULL, "-ts_coarse_fieldsplit_1_ksp_view", NULL),"Problem while setting ksp_view sf1", comm)
#endif
   DO(KSPSetFromOptions(solver),"Problem while setting ksp option",comm)

#ifdef TWOSCALE_OUTPUT_MATVECT
   Mat PST;
   DO(MatTranspose(PSR, MAT_INITIAL_MATRIX, &PST), "Problem while transposing  PS", comm)
   PetscViewer PST_viewer;
   DO(PetscViewerBinaryOpen(comm, "PSt.mat", FILE_MODE_WRITE, &PST_viewer), "openning file PSt.mat fails", comm)
   DO(PetscViewerBinarySetUseMPIIO(PST_viewer, PETSC_TRUE), "switching to MPIIO viewer fails", comm)
   DO(MatView(PST, PST_viewer),"save PSt.mat fails",comm)
   DO(PetscViewerDestroy(&PST_viewer), "Destroy PST_viewer fails", comm)
   //MatView(PST, PETSC_VIEWER_STDOUT_(comm));
   DO(MatDestroy(&PST), "Problem to destroy PST", comm)
#endif
   
}
// ==========================================================================
coarseManager<Mat, Vec>::coarseManager(coarseManager<Mat, Vec> &&other)
    : PSR(other.PSR),
      PEt(other.PEt),
      AffPSR(other.AffPSR),
      Acc_std(other.Acc_std),
      W(other.W),
      BMZ(other.BMZ),
      bc(other.bc),
      xc(other.xc),
      bc_std(other.bc_std),
      bc_enr(other.bc_enr),
      xc_std(other.xc_std),
      xc_enr(other.xc_enr),
      S(other.S),
      bc_contrib(other.bc_contrib),
      solver(other.solver),
      solver_std_only(other.solver_std_only),
      one(other.one),
      mone(other.mone),
      dde(std::move(other.dde)),
      enriched_dof_id(std::move(other.enriched_dof_id)),
      moved(false),
      state(other.state),
      comm(std::move(other.comm))
{
   //std::cout << " in coarseManager move" << std::endl;
   other.moved = true;
}
// ==========================================================================
coarseManager<Mat, Vec>::~coarseManager()
{
   //std::cout << " in ~coarseManager : moved " << moved << std::endl;

   // if moved instance all members are now owned by another instance thus pointers should not be freed
   if (!moved)
   {
      // Normally PS,PEt and Acc should'nt be null
      assert(PSR != nullptr);
      assert(PEt != nullptr);
      DO(MatDestroy(&PSR), "Problem to destroy PS", comm)
      DO(MatDestroy(&PEt), "Problem to destroy PEt", comm)
      if (AffPSR != nullptr) DO(MatDestroy(&AffPSR), "Problem to destroy AffPSR", comm)
      // for vector depending on state they are allocated or not thus test is done
      if (W != nullptr) DO(VecDestroy(&W), "Problem to destroy W", comm)
      if (BMZ != nullptr) DO(VecDestroy(&BMZ), "Problem to destroy BMZ", comm)
      if (bc != nullptr) DO(VecDestroy(&bc), "Problem to destroy bc", comm)
      if (xc != nullptr) DO(VecDestroy(&xc), "Problem to destroy xc", comm)
      if (solver != nullptr) DO(KSPDestroy(&solver), "Problem to destroy solver", comm)
#ifdef TWOSCALE_PETSC_MATNEST
      if (solver_std_only != nullptr) DO(KSPDestroy(&solver_std_only), "Problem to destroy solver_std_only", comm)
      if (Acc_std != nullptr) DO(MatDestroy(&Acc_std), "Problem to destroy Acc_std", comm)
      if (S != nullptr) DO(VecDestroy(&S), "Problem to destroy S", comm)
      if (bc_contrib != nullptr) DO(VecDestroy(&bc_contrib), "Problem to destroy bc_contrib", comm)
#endif
   }
}
#else
coarseManager<Mat, Vec>::coarseManager(Mat &&PS_, Mat &&PEt_, Mat &&Acc_, Vec &&W_, Vec &&XDc_,
                                       std::vector<std::int32_t> diag_dof_eliminated,
                                       std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &&enriched_dof_id_)
    : PS(PS_),
      PEt(PEt_),
      Acc(Acc_),
      PEtAff(nullptr),
      AffPE(nullptr),
      PEtAffPS(nullptr),
      PStAffPE(nullptr),
      PEtAffPE(nullptr),
      W(W_),
      BMZ(nullptr),
      bc(nullptr),
      xc(nullptr),
      bc_store(nullptr),
      XDc(XDc_),
      solver(nullptr),
      one(1.0),
      mone(-1.0),
      dde(diag_dof_eliminated.begin(), diag_dof_eliminated.end()),
      enriched_dof_id(std::move(enriched_dof_id_)),
      mc(0),
      Mc(0),
      moved(false),
      state(INIT)
{
   PetscErrorCode ierr;
   DO(PetscObjectGetComm((PetscObject)PS, &comm), "Problem while getting the communicator of PS, thus no choice abort on world",
      MPI_COMM_WORLD)

   int rcomp;
   MPI_Comm comm_test;
   DO(PetscObjectGetComm((PetscObject)PEt, &comm_test), "Problem while getting the communicator of PEt", comm)
   MPI_Comm_compare(comm, comm_test, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT), "Communicator must be at least congruent for coarsening operators",
           comm)

   DO(PetscObjectGetComm((PetscObject)Acc, &comm_test), "Problem while getting the communicator of Acc", comm)
   MPI_Comm_compare(comm, comm_test, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT),
           "Communicator must be at least congruent for coarsening operators and Acc", comm)

   // get number of rows
   DO(MatGetLocalSize(Acc, &mc, nullptr), "Problem while retrieving Acc row local size", comm)
   DO(MatGetSize(Acc, &Mc, nullptr), "Problem while retrieving Acc row global size", comm)

   /*
    * remove store : transfered to patches
    *
   // store PEt initial values
   DO(MatSetOption(PEt, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE), "Problem while setting option  MAT_NEW_NONZERO_LOCATIONS to PEt",
      comm)
   DO(MatStoreValues(PEt), "Problem while storing PEt", comm)
   state |= PEt_STORED;
   */

   //std::println("PEt matrix");
   //MatView(PEt, PETSC_VIEWER_STDOUT_(comm));

   // Init solver
   DO(KSPCreate(comm, &solver), "Problem while creating solver", comm)
   DO(KSPSetOptionsPrefix(solver, "ts_coarse_"), "Problem while setting prefix to solver", comm)
   //DO(KSPSetFromOptions(solver), "Problem while setting 'from' option  solver", comm)
   DO(KSPSetType(solver,KSPPREONLY), "Problem while setting solver type", comm)
   PC pc;
   DO(KSPGetPC(solver, &pc), "Problem while getting PC", comm)
   DO(PCSetType(pc, PCLU), "Problem while setting PC type", comm)
   DO(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS), "Problem while setting PCLU type", comm)
   //DO(KSPSetOperators(solver,Acc,Acc), "Problem while setting operator  solver", comm)

   //ISLocalToGlobalMapping rmapping;
   //ISLocalToGlobalMapping cmapping;
   //MatGetLocalToGlobalMapping(PE, nullptr, &cmapping);
   //PRINT("dde",dde);
   //ISLocalToGlobalMappingApply(cmapping, dde.size(), dde.data(), dde.data());

#ifdef TWOSCALE_OUTPUT_MATVECT
   Mat PST;
   DO(MatTranspose(PS, MAT_INITIAL_MATRIX, &PST), "Problem while transposing  PS", comm)
   PetscViewer PST_viewer;
   DO(PetscViewerBinaryOpen(comm, "PSt.mat", FILE_MODE_WRITE, &PST_viewer), "openning file PSt.mat fails", comm)
   DO(PetscViewerBinarySetUseMPIIO(PST_viewer, PETSC_TRUE), "switching to MPIIO viewer fails", comm)
   DO(MatView(PST, PST_viewer),"save PSt.mat fails",comm)
   DO(PetscViewerDestroy(&PST_viewer), "Destroy PST_viewer fails", comm)
   //MatView(PST, PETSC_VIEWER_STDOUT_(comm));
   DO(MatDestroy(&PST), "Problem to destroy PST", comm)
#endif
   //MatView(PS, PETSC_VIEWER_STDOUT_(comm));
   
}
// ==========================================================================
coarseManager<Mat, Vec>::coarseManager(coarseManager<Mat, Vec> &&other)
    : PS(other.PS),
      PEt(other.PEt),
      Acc(other.Acc),
      PEtAff(other.PEtAff),
      AffPE(other.AffPE),
      PEtAffPS(other.PEtAffPS),
      PStAffPE(other.PStAffPE),
      PEtAffPE(other.PEtAffPE),
      W(other.W),
      BMZ(other.BMZ),
      bc(other.bc),
      xc(other.xc),
      bc_store(other.bc_store),
      XDc(other.XDc),
      one(other.one),
      mone(other.mone),
      solver(other.solver),
      dde(std::move(other.dde)),
      enriched_dof_id(std::move(other.enriched_dof_id)),
      mc(other.mc),
      Mc(other.Mc),
      moved(false),
      state(other.state),
      comm(std::move(other.comm))
{
   //std::cout << " in coarseManager move" << std::endl;
   other.moved = true;
}
// ==========================================================================
coarseManager<Mat, Vec>::~coarseManager()
{
   //std::cout << " in ~coarseManager : moved " << moved << std::endl;

   // if moved instance all members are now owned by another instance thus pointers should not be freed
   if (!moved)
   {
      // Normally PS,PEt and Acc should'nt be null
      assert(PS != nullptr);
      assert(PEt != nullptr);
      assert(Acc != nullptr);
      DO(MatDestroy(&PS), "Problem to destroy PS", comm)
      DO(MatDestroy(&PEt), "Problem to destroy PEt", comm)
      DO(MatDestroy(&Acc), "Problem to destroy Acc", comm)
      if (PEtAff != nullptr) DO(MatDestroy(&PEtAff), "Problem to destroy PEtAff", comm)
      if (AffPE != nullptr) DO(MatDestroy(&AffPE), "Problem to destroy AffPE", comm)
      if (PEtAffPS != nullptr) DO(MatDestroy(&PEtAffPS), "Problem to destroy PEtAffPS", comm)
      if (PStAffPE != nullptr) DO(MatDestroy(&PStAffPE), "Problem to destroy PStAffPE", comm)
      if (PEtAffPE != nullptr) DO(MatDestroy(&PEtAffPE), "Problem to destroy PEtAffPE", comm)
      // for vector depending on state they are allocated or not thus test is done
      if (W != nullptr) DO(VecDestroy(&W), "Problem to destroy W", comm)
      if (BMZ != nullptr) DO(VecDestroy(&BMZ), "Problem to destroy BMZ", comm)
      if (bc != nullptr) DO(VecDestroy(&bc), "Problem to destroy bc", comm)
      if (xc != nullptr) DO(VecDestroy(&xc), "Problem to destroy xc", comm)
      if (bc_store != nullptr) DO(VecDestroy(&bc_store), "Problem to destroy bc_store", comm)
      if (XDc != nullptr) DO(VecDestroy(&XDc), "Problem to destroy XDc", comm)
      if (solver != nullptr) DO(KSPDestroy(&solver), "Problem to destroy solver", comm)
   }
}
#endif
// ==========================================================================
void coarseManager<Mat, Vec>::setStdCoarse(Mat Aff, Vec bf, bool use_imp_enriched)
{
   // prerequisites
   // assert(state & PEt_STORED);
   
#ifndef TWOSCALE_PETSC_MATNEST
   CHECKTS(use_imp_enriched, "imposing enriched dofs to one only possible with matnest strategy", comm)
#endif

   // NOTE: on return ACC_STORED, ACC_STD, BC_STORED, BC_STD are set
   // and ACC_ASS, BC_ASS unset
   spdlog::info("Generate/update matrices standard part of the coarse system");

   // check communicators coherance of provided Mat/Vec with coarseManager
   {
      int rcomp;
      MPI_Comm comm_test;
      DO(PetscObjectGetComm((PetscObject)Aff, &comm_test), "Problem while getting the communicator of Aff", comm)
      MPI_Comm_compare(comm, comm_test, &rcomp);
      CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT),
              "Communicator must be at least congruent for Aff matrix argument and coarsening operator", comm)
      DO(PetscObjectGetComm((PetscObject)bf, &comm_test), "Problem while getting the communicator of bf", comm)
      MPI_Comm_compare(comm, comm_test, &rcomp);
      CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT),
              "Communicator must be at least congruent for bf matrix argument and coarsening operator", comm)
   }

   // =================================================
   // force Aff to be considered as sym for performance
   // =================================================
   DO(MatSetOption(Aff, MAT_SYMMETRIC, PETSC_TRUE), "Problem while setting option  MAT_SYMMETRIC to Aff", comm)

   // ============
   // reset status
   // ============
   // TODO: memory leak ????
   state ^= ACC_STD;
   state ^= ACC_ASS;
   state ^= BC_STD;
   state ^= BC_ASS;

   // ================================
   // set or reset coarse matrix with
   // standard x standard bloc creation
   // ================================
#ifdef TWOSCALE_PETSC_MATNEST
   assert(!(state & ACC_STORED)); // for now can only be called once TODO manage memory leak

   // compute block in to step to save AffPSR done once here
   assert(!(state & AffPSR_CREA));
   DO(MatMatMult(Aff, PSR, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AffPSR), "Problem while computing the product Aff.PSR", comm)
   DO(MatTransposeMatMult(PSR, AffPSR, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Acc_blocks[0]),
      "Problem while computing the diagonal standard sub bloc of Acc", comm)
   state |= AffPSR_CREA;

   // Add 1 to null diagonal terms
   // Force temporarily reallocation as null diagonal terms are forcefully not in the graph
   DO(MatSetOption(Acc_blocks[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE),"set NEW NONZERO option",comm)
   for (auto enrich : dde)
      DO(MatSetValues(Acc_blocks[0], 1, &enrich, 1, &enrich, &one, INSERT_VALUES),
         "Problem while inserting eliminated enriched diagonal term in Acc_blocks[0]", comm)
   DO(MatAssemblyBegin(Acc_blocks[0], MAT_FINAL_ASSEMBLY), "PSRtAffPSR begin assembly Matrix", comm)
   DO(MatAssemblyEnd(Acc_blocks[0], MAT_FINAL_ASSEMBLY), "PSRtAffPSR end assembly Matrix", comm)
   // reset reallocation to have all Matrices in the same state in the TS lib
   DO(MatSetOption(Acc_blocks[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE),"set NEW NONZERO option",comm)

   // std::println("PSRtAffPSR ==============");
   // MatView(Acc_blocks[0], PETSC_VIEWER_STDOUT_(comm));
   if (!(state & ACC_STORED))
   {
      DO(MatSetOption(Acc_blocks[0], MAT_SYMMETRIC, PETSC_TRUE), "Problem while setting option  MAT_SYMMETRIC to Acc", comm)
      DO(MatSetOption(Acc_blocks[0], MAT_SYMMETRY_ETERNAL, PETSC_TRUE),
         "Problem while setting option  MAT_SYMMETRY_ETERNAL to Acc", comm)
   }

   // To be able to create full Acc matrix all of its sub block must be created so that vector obtained from Acc get correct
   // graph. Thus each block using PEt are computed with current PEt coefficient that are not forcefully multiplied by enrichment
   // function. This is very stupid as these computation are done for nothing for the numeric part !!!! TODO see if this can be
   // done differently.
   //

   // for now call only once this function with this instance so PEtAffPS_CREA should not be set. If so something as to be done
   // (cleaning,using ??)
   assert(!(state & PEtAffPS_CREA));
   DO(MatMatMult(PEt, AffPSR, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Acc_blocks[2]),
      "Problem while computing the product PEt x Aff.PS", comm)
   state |= PEtAffPS_CREA;
   // for now call only once this function with this instance so PStAffPE_CREA should not be set. If so something as to be done
   // (cleaning,using ??)
   assert(!(state & PStAffPE_CREA));
   DO(MatTranspose(Acc_blocks[2], MAT_INITIAL_MATRIX, &Acc_blocks[1]), "Problem while transposing  PEt.Aff.PS", comm)
   state |= PStAffPE_CREA;
   // for now call only once this function with this instance so PEtAffPE_CREA should not be set. If so something as to be done
   // (cleaning,using ??)
   assert(!(state & PEtAffPE_CREA));
   DO(MatRARt(Aff, PEt, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Acc_blocks[3]), "Problem while computing the product PEt.Aff.PE",
      comm)
   state |= PEtAffPE_CREA;

   // std::println("PEt =====================");
   // MatView(PEt, PETSC_VIEWER_STDOUT_(comm));
   // std::println("AffPSR ==================");
   // MatView(AffPSR, PETSC_VIEWER_STDOUT_(comm));
   
   // std::println("PEt,AffPSR info ==================");
   // PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(comm), PETSC_VIEWER_ASCII_INFO_DETAIL );
   // MatView(PEt, PETSC_VIEWER_STDOUT_(comm));
   // MatView(AffPSR, PETSC_VIEWER_STDOUT_(comm));
   // PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(comm), PETSC_VIEWER_DEFAULT );

   // All blocks are created thus Acc can be created
   DO(MatCreateNest(comm, 2, NULL, 2, NULL, Acc_blocks, &Acc), "Problem while creating Acc nested matrix from blocks", comm)
   DO(MatNestSetVecType(Acc, VECNEST), "Problem while setting vector type for Acc", comm)

   // Operator are set at this level for all the duration of the instance
   DO(KSPSetOperators(solver,Acc,Acc), "Problem while setting operator  solver", comm)

   // This function if setting use_imp_enriched to true prepare mix computation with free and imposed enriched dofs
   // As this adding memory consumption only init extra matrix/vector/solver if user want them.
   // Set at this level for all the duration of the instance
   if (use_imp_enriched)
   {
      // init solver
      DO(KSPCreate(comm, &solver_std_only), "Problem while creating solver_std_only", comm)
      DO(KSPSetOptionsPrefix(solver_std_only, "ts_coarse_std_"), "Problem while setting prefix to solver_std_only", comm)
      DO(KSPSetType(solver_std_only, KSPPREONLY), "Problem while setting solver_std_only type", comm)
      PC pc_std_only;
      DO(KSPGetPC(solver_std_only, &pc_std_only), "Problem while getting PC", comm)
      DO(PCSetType(pc_std_only, PCLU), "Problem while setting PC type", comm)
      DO(PCFactorSetMatSolverType(pc_std_only, MATSOLVERMUMPS), "Problem while setting PCLU type", comm)
      DO(KSPSetFromOptions(solver_std_only), "Problem while setting ksp option", comm)

      // init S
      DO(MatCreateVecs(PEt, &S, NULL), "Problem while creating S", comm)

      // copy A stdxstd bloc
      DO(MatConvert(Acc_blocks[0], MATSAME, MAT_INITIAL_MATRIX, &Acc_std), "Problem while duplicating Acc_blocks[0] into Acc_std",
         comm)

      // assign operator
      DO(KSPSetOperators(solver_std_only, Acc_std, Acc_std), "Problem while setting operator  solver_std_only", comm)
   }
#else
   Mat tmp;
   DO(MatPtAP(Aff, PS, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &tmp),
      "Problem while computing the diagonal standard sub bloc of Acc", comm)
   if (state & ACC_STORED)
      DO(MatZeroEntries(Acc), "Problem while resetting to zero Acc", comm)
   else
   {
      DO(MatSetOption(Acc, MAT_SYMMETRIC, PETSC_TRUE), "Problem while setting option  MAT_SYMMETRIC to Acc", comm)
      DO(MatSetOption(Acc, MAT_SYMMETRY_ETERNAL, PETSC_TRUE), "Problem while setting option  MAT_SYMMETRY_ETERNAL to Acc", comm)
   }

   // std::println("ACC_STORED {}", state & ACC_STORED);
   // MatView(tmp, PETSC_VIEWER_STDOUT_(comm));
   // MatView(Acc, PETSC_VIEWER_STDOUT_(comm));

   DO(MatAXPY(Acc, one, tmp, SUBSET_NONZERO_PATTERN), "Problem while adding  (PSt.Aff.PS)t to Acc", comm)
   DO(MatDestroy(&tmp), "Problem to destroy (PSt.Aff.PS) ", comm)
   tmp = nullptr;

   // =============================================================
   // Treating diagonal zero terms related to Dirichlet elimination
   // =============================================================
   // note: correspond to IDc
   // note: Eliminated with MATNEST
   //
   // dde in local numbering is fine (restricted to local anyway)
   // std::println("dde {}: {}",dde.size(),dde);
   for (auto enrich : dde)
      DO(MatSetValuesLocal(Acc, 1, &enrich, 1, &enrich, &one, INSERT_VALUES),
         "Problem while inserting eliminated enriched diagonal term in Acc", comm)

   DO(MatAssemblyBegin(Acc, MAT_FINAL_ASSEMBLY), "Acc begin assembly Matrix", comm)
   DO(MatAssemblyEnd(Acc, MAT_FINAL_ASSEMBLY), "Acc end assembly Matrix", comm)
#endif
   state |= ACC_STD;

   // ==========================================================
   // set or reset coarse rhs with standard bloc creation taking
   // into account W
   // ==========================================================
   // NOTE: W provides Dirichlet non null values projected at
   // fine scale that needs to be used to add correct
   // contribution to rhs: B-A.W
   assert(!(state & BC_STORED));
#ifdef TWOSCALE_PETSC_MATNEST
   assert(bc == nullptr);
   assert(xc == nullptr);
   DO(MatCreateVecs(Acc, &bc, &xc), "Problem while creating  bc and xc", comm)
   DO(VecNestGetSubVec(bc, 0, &bc_std), "Problem while extracting standard sub vector from bc", comm)
   DO(VecNestGetSubVec(bc, 1, &bc_enr), "Problem while extracting enriched sub vector from bc", comm)
   DO(VecNestGetSubVec(xc, 0, &xc_std), "Problem while extracting standard sub vector from xc", comm)
   DO(VecNestGetSubVec(xc, 1, &xc_enr), "Problem while extracting enriched sub vector from xc", comm)
   /*
   std::println("bc ==================");
   VecView(bc, PETSC_VIEWER_STDOUT_(comm));
   std::println("xc ==================");
   VecView(xc, PETSC_VIEWER_STDOUT_(comm));
   */
   if (use_imp_enriched)
      DO(VecDuplicate(bc_std, &bc_contrib), "Problem while creating bc_contrib by  duplication of bc_std", comm)

#else
   assert(bc == nullptr);
   assert(xc == nullptr);
   assert(Mc > 0);
   DO(VecCreate(comm, &bc), "Problem while creating  bc", comm)
   DO(VecSetType(bc, VECSTANDARD), "Problem while setting bc type", comm)
   DO(VecSetSizes(bc, mc, Mc), "Problem while setting bc size", comm)
   ISLocalToGlobalMapping cmapping;
   DO(MatGetLocalToGlobalMapping(Acc, nullptr, &cmapping), "Problem while collecting local to global mapping of Acc column ",
      comm)
   DO(VecSetLocalToGlobalMapping(bc, cmapping), "Problem while setting local to global mapping for bc", comm)
   assert(bc_store == nullptr);
   DO(VecDuplicate(bc, &bc_store), "Problem while creating bc_store by  duplication of bc", comm)
   DO(VecDuplicate(bc, &xc), "Problem while creating xc by  duplication of bc", comm)
   DO(VecZeroEntries(xc), "Problem while resetting to zero xc", comm)
#endif

   if (W != nullptr)
   {
      // TODO memory leak ?? called more then once smash BMZ ??? put above ?!
      DO(VecDuplicate(bf, &BMZ), "Problem while creating BMZ by  duplication of bf", comm)
      DO(MatMult(Aff, W, BMZ), "Problem while computing A.W", comm)
      DO(VecAYPX(BMZ, mone, bf), "Problem while computing B-A.W", comm)
#ifdef TWOSCALE_PETSC_MATNEST
      DO(MatMultTranspose(PSR, BMZ, bc_std),
         "Problem while computing the standard part of bc  with W contribution  (PSRt.(B-A.W))", comm)
#else
      DO(MatMultTranspose(PS, BMZ, bc), "Problem while computing the standard with W contribution sub bloc of bc (PSt.(BD-AD.W))",
         comm)
      DO(VecAXPY(bc, one, XDc), "Problem while computing XDc+PSt.(BD-AD.W)", comm)
#endif
   }
   else
#ifdef TWOSCALE_PETSC_MATNEST
      DO(MatMultTranspose(PSR, bf, bc_std),
         "Problem while computing the standard part of bc  (PSRt.B)", comm)
#else
      DO(MatMultTranspose(PS, bf, bc), "Problem while computing the standard sub bloc of bc", comm)

   // =================
   // saving cst vector
   // =================
   DO(VecCopy(bc, bc_store), "Problem while saving bc in bc_store", comm)
#endif
   state |= BC_STD;
   state |= BC_STORED;

   // std::cout<<" bf give bc to store"<<std::endl;
   // VecView(bf, PETSC_VIEWER_STDOUT_(comm));
   // VecView(bc, PETSC_VIEWER_STDOUT_(comm));


   // Note with MatNest de facto the Acc_blocks[0] will never be updated thus there is no need to do an extra backup of this block
#ifndef TWOSCALE_PETSC_MATNEST
   // =================
   // saving cst matrix
   // =================
   // NOTE in twoscale iteration the standard x standard diagonal block, the boundary condition and the fixed enriched dof are
   // unchanged thus this part of the matrix can be saved. As the full non zero pattern is set updates won't lead to problem when
   // adding enriched part.
   DO(MatSetOption(Acc, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE), "Problem while setting option  MAT_NEW_NONZERO_LOCATIONS to Acc",
      comm)
   DO(MatStoreValues(Acc), "Problem while storing Acc", comm)
#endif
   state |= ACC_STORED;

   // std::cout<<"Acc from PS"<<std::endl;
   // MatView(Aff, PETSC_VIEWER_STDOUT_(comm));
   // MatView(Acc, PETSC_VIEWER_STDOUT_(comm));
   // MatView(PS, PETSC_VIEWER_STDOUT_(comm));

   return;
}
// ==========================================================================
void coarseManager<Mat, Vec>::resetCoarseToStd()
{
   spdlog::info("update matrices standard part of the coarse system");

   // prerequisites
   CHECKTS(((!(state & ACC_STORED)) || (!(state & BC_STORED))),
           "You can not do an update if at least one call to setStdCoarse was not done", comm)

#ifdef TWOSCALE_PETSC_MATNEST
   // Note: with MatNest de facto the Acc_blocks[0] and bc_std are never updated thus there is no need to retrieve any backup that
   // do not exist anyway
   // Just set state to be consistent with non nested approach
   state |= ACC_STD;
   state |= BC_STD;
#else
   //MatView(Acc, PETSC_VIEWER_STDOUT_(comm));

   // ===============================
   // reset coarse matrix with saved
   // ===============================
   DO(MatRetrieveValues(Acc), "Problem while retriving Acc", comm)
   state |= ACC_STD;

   //std::cout<<"reset Acc by Acc_store"<<std::endl;
   //MatView(Acc, PETSC_VIEWER_STDOUT_(comm));

   // ================================
   // set or reset coarse rhs with
   // standardx1 bloc
   // ================================
   assert(state & BC_STORED);
   DO(VecCopy(bc_store, bc), "Problem while saving bc_store in bc", comm)
   state |= BC_STD;
#endif

   //std::cout<<"reset bc by bc_store"<<std::endl;
   //VecView(bc, PETSC_VIEWER_STDOUT_(comm));

   return;
}
// ==========================================================================
void coarseManager<Mat, Vec>::updateEnrichCoarse(Mat Aff, Vec bf)
{
   spdlog::info("update matrices enriched part of the coarse system");
#ifdef TWOSCALE_HAND_PROFILE
   auto logger=common::TimeLogger();
   auto dt = common::Timer();
#endif

   // prerequisites
   // You can not call updateEnrichCoarse if at least one call to setStdCoarse/resetCoarseToStd was not done so that Acc and bc are in standard mode
   assert(state&ACC_STD);
   assert(state&BC_STD);
   // You can not call updateEnrichCoarse if at least one call to updateEnrichedOperator  was not done so that PEt is in assembly mode
   assert(state & PEt_ASS);

   // MatView(PEt, PETSC_VIEWER_STDOUT_(comm));

#ifdef TWOSCALE_HAND_PROFILE
   dt.start();
#endif


#ifndef TWOSCALE_PETSC_MATNEST
   // ================
   // product PEt.Aff
   // ================
#ifdef TWOSCALE_SCOREP_PROFILE
   if (state & PEtAff_CREA)
      f_PEtAff2(PEt, Aff, &PEtAff,comm);
   else
   {
      f_PEtAff1(PEt, Aff, &PEtAff,comm);
      state |= PEtAff_CREA;
   }
#else
   if (state & PEtAff_CREA)
   {
      DO(MatMatMult(PEt, Aff, MAT_REUSE_MATRIX, PETSC_DETERMINE, &PEtAff), "Problem while computing the product PEt.Aff", comm)
      DO(MatTranspose(PEtAff, MAT_REUSE_MATRIX, &AffPE), "Problem while transposing  PEt.Aff", comm)
   }
   else
   {
      DO(MatMatMult(PEt, Aff, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &PEtAff), "Problem while computing the product PEt.Aff", comm)
      DO(MatTranspose(PEtAff, MAT_INITIAL_MATRIX, &AffPE), "Problem while transposing  PEt.Aff", comm)
      state |= PEtAff_CREA;
   }
#endif

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   auto elapsed=dt. elapsed();
   logger.register_timing("PEtAff", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif
#endif

   // ===========================================
   // non diagonal bloc creation and addition
   // standard x enriched and enriched x standard
   // ===========================================

#ifdef TWOSCALE_PETSC_MATNEST
   assert(state & AffPSR_CREA);
   assert(state & PEtAffPS_CREA);
   DO(MatMatMult(PEt, AffPSR, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Acc_blocks[2]),
      "Problem while computing the product PEt.Aff.PS", comm)
#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   auto elapsed=dt. elapsed();
   logger.register_timing("PEtAffPS", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif
   DO(MatTranspose(Acc_blocks[2], MAT_REUSE_MATRIX, &Acc_blocks[1]), "Problem while transposing  PEt.Aff.PS", comm)
#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("PEtAffPS transpose", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif
#else
   if (state & PEtAffPS_CREA)
      DO(MatMatMult(PEtAff, PS, MAT_REUSE_MATRIX, PETSC_DETERMINE, &PEtAffPS), "Problem while computing the product PEt.Aff.PS", comm)
   else
   {
      DO(MatMatMult(PEtAff, PS, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &PEtAffPS), "Problem while computing the product PEt.Aff.PS",
         comm)
      state |= PEtAffPS_CREA;
   }

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("PEtAffPS", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif

   DO(MatAXPY(Acc, one, PEtAffPS, SUBSET_NONZERO_PATTERN), "Problem while adding  PEt.Aff.PS to Acc", comm)


#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("Acc+PEtAffPS", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif


   if (state & PStAffPE_CREA)
      DO(MatTranspose(PEtAffPS, MAT_REUSE_MATRIX, &PStAffPE), "Problem while transposing  PEt.Aff.PS", comm)
   else
   {
      DO(MatTranspose(PEtAffPS, MAT_INITIAL_MATRIX, &PStAffPE), "Problem while transposing  PEt.Aff.PS", comm)
      state |= PStAffPE_CREA;
   }

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("PEtAffPS transpose", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif

   DO(MatAXPY(Acc, one, PStAffPE, SUBSET_NONZERO_PATTERN), "Problem while adding  PSt.Aff.PE to Acc", comm)

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("Acc + PStAffPE", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif
#endif

   // =============================================
   // enriched diagonal bloc creation and addition
   // enriched x enriched
   // =============================================
#ifdef TWOSCALE_PETSC_MATNEST

   // std::println("PEt =====================");
   // MatView(PEt, PETSC_VIEWER_STDOUT_(comm));

   assert(state & PEtAffPE_CREA);
   DO(MatRARt(Aff, PEt, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Acc_blocks[3]), "Problem while computing the product PEt.Aff.PE",
      comm)
#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("PEtAffPE", elapsed[0], elapsed[1], elapsed[2]);
#endif
#else
   if (state & PEtAffPE_CREA)
      DO(MatMatMult(PEt, AffPE, MAT_REUSE_MATRIX, PETSC_DETERMINE, &PEtAffPE), "Problem while computing the product PEt.Aff.PE",
         comm)
   else
   {
      DO(MatMatMult(PEt, AffPE, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &PEtAffPE), "Problem while computing the product PEt.Aff.PE",
         comm)
      state |= PEtAffPE_CREA;
   }

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("PEtAffPE", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif

   DO(MatAXPY(Acc, one, PEtAffPE, SUBSET_NONZERO_PATTERN), "Problem while adding  PEt.Aff.PE to Acc", comm)

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.register_timing("Acc + PEtAffPE", elapsed[0], elapsed[1], elapsed[2]);
#endif
#endif

   state |= ACC_ASS;
   state ^= ACC_STD;


#if defined(TWOSCALE_OUTPUT_MATVECT) && !(defined(TWOSCALE_PETSC_MATNEST))
   PetscViewer ACC_viewer;
   DO(PetscViewerBinaryOpen(comm, "ACC.mat", FILE_MODE_WRITE, &ACC_viewer), "openning file ACC.mat fails", comm)
   DO(PetscViewerBinarySetUseMPIIO(ACC_viewer, PETSC_TRUE), "switching to MPIIO viewer fails", comm)
   DO(MatView(Acc, ACC_viewer),"save ACC.mat fails",comm)
   DO(PetscViewerDestroy(&ACC_viewer), "Destroy ACC_viewer fails", comm)
#endif

   // ==============
   // product PEt.BMZ
   // ==============
#ifdef TWOSCALE_PETSC_MATNEST
   /*
   std::println("bc_enr ==================");
   VecView(bc_enr, PETSC_VIEWER_STDOUT_(comm));
   std::println("BMZ =====================");
   VecView(BMZ, PETSC_VIEWER_STDOUT_(comm));
   std::println("PEt matrix ==============");
   MatView(PEt, PETSC_VIEWER_STDOUT_(comm));
   PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(comm), PETSC_VIEWER_ASCII_INFO_DETAIL );
   MatView(PEt, PETSC_VIEWER_STDOUT_(comm));
   */
   if(W!=nullptr)
      DO(MatMult(PEt, BMZ, bc_enr),
         "Problem while computing the enriched part of bc  with W contribution  (PEt.(B-A.W))", comm)
   else
      DO(MatMult(PEt, bf, bc_enr),
         "Problem while computing the enriched part of bc (PEt.B)", comm)
#else
   if(W!=nullptr)
      DO(MatMultAdd(PEt, BMZ, bc, bc), "Problem while computing the enriched sub bloc of bc", comm)
   else
      DO(MatMultAdd(PEt, bf, bc, bc), "Problem while computing the enriched sub bloc of bc", comm)
#endif
   state |= BC_ASS;
   state ^= BC_STD;

#if defined(TWOSCALE_OUTPUT_MATVECT) && !(defined(TWOSCALE_PETSC_MATNEST))
   PetscViewer BC_viewer;
   DO(PetscViewerBinaryOpen(comm, "BC.mat", FILE_MODE_WRITE, &BC_viewer), "openning file BC.mat fails", comm)
   DO(PetscViewerBinarySetUseMPIIO(BC_viewer, PETSC_TRUE), "switching to MPIIO viewer fails", comm)
   DO(VecView(bc, BC_viewer),"save BC.mat fails",comm)
   DO(PetscViewerDestroy(&BC_viewer), "Destroy BC_viewer fails", comm)
#endif

#ifdef TWOSCALE_HAND_PROFILE
   logger.list_timings(comm,{TimingType::wall,TimingType::user},dolfinx::Table::Reduction::average);
#endif

   return;
}
// ==========================================================================
void coarseManager<Mat, Vec>::updateEImp()
{
   spdlog::info("update right and side of the coarse standard system");
#ifdef TWOSCALE_PETSC_MATNEST
#ifdef TWOSCALE_HAND_PROFILE
   auto logger=common::TimeLogger();
   auto dt = common::Timer();
#endif

   // prerequisites
   // You can not call updateEImp if at least one call to setStdCoarse/resetCoarseToStd was not done so that Acc and bc are in standard mode
   assert(state&ACC_STD);
   assert(state&BC_STD);
   // You can not call updateEImp if at least one call to updateEnrichedOperator  was not done so that PEt is in assembly mode
   assert(state & PEt_ASS);
   // You can not call updateEImp if AffPSR is not created
   assert(state & AffPSR_CREA);

   // MatView(PEt, PETSC_VIEWER_STDOUT_(comm));
   
   // =========================
   // force enriched dof to one
   // =========================
   DO(VecSet(xc_enr, 1.), "Problem while setting  to one enriched dofs", comm)

#ifdef TWOSCALE_HAND_PROFILE
   dt.start();
#endif
   // ========================================
   // Current PE multiplication by unit vector
   // ========================================
   DO(MatMultTranspose(PEt,xc_enr , S), "Problem while computing  PE.one", comm)

#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   auto elapsed=dt. elapsed();
   logger.register_timing("PE.ONE", elapsed[0], elapsed[1], elapsed[2]);
   dt.start();
#endif
   // ==============================
   // non diagonal bloc contribution
   // ==============================
   DO(MatMultTranspose(AffPSR,S , bc_contrib), "Problem while computing  PSRt.Aff.PE.one", comm)
#ifdef TWOSCALE_HAND_PROFILE
   dt.stop();
   elapsed=dt. elapsed();
   logger.list_timings(comm,{TimingType::wall,TimingType::user},dolfinx::Table::Reduction::average);
#endif
#else
   assert(0);
#endif

   return;
}
// ==========================================================================
void coarseManager<Mat, Vec>::solveEImp(Vec xf)
{
#ifdef TWOSCALE_PETSC_MATNEST
   // prerequisites
   // TODO
   // register updateEImp operation that generate bc_contrib
   // You can not call solve if at least one call to updateEnrichedOperator  was not done so that PEt is in assembly mode
   assert(state & PEt_ASS);

   DO(VecAYPX(bc_contrib,-1.,bc_std), "Problem while copying W+S in xf", comm)

   DO(KSPSolve(solver_std_only,bc_contrib, xc_std), "Problem while solving sytem Acc.xc=bc", comm)


   if (W != nullptr)
      DO(VecWAXPY(xf,1.,S,W), "Problem while copying W+S in xf", comm)
   else
      DO(VecCopy(S, xf), "Problem while copying S in xf", comm)
   DO(MatMultAdd(PSR, xc_std, xf, xf), "Problem while projecting standard to xf", comm)
#else
   assert(0);
#endif
}
// ==========================================================================
void coarseManager<Mat, Vec>::solve(Vec xf)
{
   // prerequisites
   // You can not call solve if at least one call to updateEnrichCoarse was not done so that Acc and bc are in assembly mode
   assert(state&ACC_ASS);
   assert(state & BC_ASS);
   // You can not call solve if at least one call to updateEnrichedOperator  was not done so that PEt is in assembly mode
   assert(state & PEt_ASS);
#ifndef TWOSCALE_PETSC_MATNEST
   DO(KSPSetOperators(solver,Acc,Acc), "Problem while setting operator  solver", comm)
#endif

   //std::println("current xc");
   //VecView(xc, PETSC_VIEWER_STDOUT_(comm));
   //std::println("bc");
   //VecView(bc, PETSC_VIEWER_STDOUT_(comm));

   DO(KSPSolve(solver,bc, xc), "Problem while solving sytem Acc.xc=bc", comm)

   //std::println("xc solve");
   //VecView(xc, PETSC_VIEWER_STDOUT_(comm));
   //VecView(bc, PETSC_VIEWER_STDOUT_(comm));
   //MatView(Acc, PETSC_VIEWER_STDOUT_(comm));
   //MatView(PEt, PETSC_VIEWER_STDOUT_(comm));

   if (W != nullptr)
      DO(VecCopy(W, xf), "Problem while copying W in xf", comm)
   else
      DO(VecZeroEntries(xf), "Problem while resetting to zero xf", comm)
#ifdef TWOSCALE_PETSC_MATNEST
   DO(MatMultAdd(PSR, xc_std, xf, xf), "Problem while projecting standard to xf", comm)
   DO(MatMultTransposeAdd(PEt, xc_enr, xf, xf), "Problem while projecting enriched to xf", comm)
#else
   DO(MatMultAdd(PS, xc, xf, xf), "Problem while projecting standard to xf", comm)
   DO(MatMultTransposeAdd(PEt, xc, xf, xf), "Problem while projecting enriched to xf", comm)
#endif
   //TODO scater forward xf ? here would be more clean no ? not clear
   //std::println("xf solve");
   //VecView(xf, PETSC_VIEWER_STDOUT_(comm));
}
// ==========================================================================
void coarseManager<Mat, Vec>::projectStdCoarse(Vec xcin, Vec xf)
{
   // xc -> xf
   if (W != nullptr)
      DO(VecCopy(W, xf), "Problem while copying W in xf", comm)
   else
      DO(VecZeroEntries(xf), "Problem while resetting to zero xf", comm)

   //std::println("xf ini");
   //VecView(xf, PETSC_VIEWER_STDOUT_(comm));

   assert(state & BC_STORED);
#ifdef TWOSCALE_PETSC_MATNEST
   //DO(MatMultAdd(PSR, xc_std, xf, xf), "Problem while projecting standard to xf", comm)
   std::println("In projectStdCoarse !!!!! WARNING !!!!!!!! use bluntelly xcin argument is maybe dangerous: to check");
   DO(MatMultAdd(PSR, xcin, xf, xf), "Problem while projecting standard to xf", comm)
#else
   DO(MatMultAdd(PS, xcin, xf, xf), "Problem while projecting standard to xf", comm)
#endif


   //std::println("project xc in xf");
   //VecView(xcin, PETSC_VIEWER_STDOUT_(comm));
   //VecView(xf, PETSC_VIEWER_STDOUT_(comm));
}

#endif
}  // namespace twoscale
