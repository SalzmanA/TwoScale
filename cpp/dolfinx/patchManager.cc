/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#include "patchManager.h"

#include <iostream>
#include <utility>
#include <vector>
#include <type_traits>

#include "dataTsPerMacro.h"
#include "scaleJump.h"

using std::cout, std::endl, std::vector;  

////////////////////////////GRRRRRRRRR////////////////////////////
void dump(Mat Aff,std::span<std::int32_t> rows,std::span<std::int32_t> cols)
{
   cout<<"start dump HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH"<<endl;
   PetscInt rstart, rend, ncols;
   const PetscInt *icols[1];
   const PetscScalar *rvals[1];
   MatGetOwnershipRange(Aff, &rstart, &rend);

   for (PetscInt r = rstart; r < rend; ++r)
   {
      MatGetRow(Aff, r, &ncols, icols, rvals);
      for (PetscInt i = 0; i < ncols; ++i)
      {
         printf("i=%d; j=%d; val=%e; PetscCall(MatSetValues(A, 1, &i, 1, &j, &val, INSERT_VALUES));\n", r, icols[0][i],
                rvals[0][i]);
      }
      MatRestoreRow(Aff, r, &ncols, icols, rvals);
   }
    MatGetSize(Aff, &rstart, &rend);
    cout << "PetscInt M=" << rstart << ", N=" << rend << ";" << endl;
    MatGetLocalSize(Aff, &rstart, &rend);
    cout << "PetscInt m=" << rstart << ", n=" << rend << ";" << endl;
    cout << "size_r=" << rows.size() << ";\nselr[" << rows.size() << "]={";
    for (auto r : rows | std::views::take(rows.size() - 1)) cout << r << ",";
    cout << rows[rows.size() - 1] << "};" << endl;
    cout << "size_c=" << cols.size() << ";\nselc[" << cols.size() << "]={";
    for (auto r : cols | std::views::take(cols.size() - 1)) cout << r << ",";
    cout << cols[cols.size() - 1] << "};" << endl;
    cout << "end   dump HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH" << endl;
}

////////////////////////////GRRRRRRRRR////////////////////////////

namespace twoscale
{
patch::patch(std::int32_t enriched_id_, std::int64_t id_, std::uint16_t state_, std::vector<PetscInt> &&dofs_,
             std::vector<PetscInt> &&dofs_to_exchange_, std::vector<PetscInt> &&Diri_dofs_,
             std::vector<PetscInt> &&Diri_dofs_to_exchange_, int owner_, std::vector<int> &&remotes_)
    : dofs(std::move(dofs_)),
      dofs_to_exchange(std::move(dofs_to_exchange_)),
      Diri_dofs(std::move(Diri_dofs_)),
      Diri_dofs_to_exchange(std::move(Diri_dofs_to_exchange_)),
      remotes(std::move(remotes_)),
      id(id_),
      weight(dofs_.size()),
      enriched_id(enriched_id_),
      owner(owner_),
      state(state_),
      pid(-1),
      nbproc(0),
      pidp(-1),
      nbprocp(0),
      i_seq(-1)
{
   assert((state & DUMMY) ? true : ((state & LOCAL) ? (!(state & DIST)) : (state & DIST)));
   if (state & LOCAL) state |= GLOBAL_WEIGHT;
   if (state & DUMMY) state |= LOCAL;
   // security: patchState encoding base on this assertion (i.e. 16 bits available at least)
   static_assert(std::is_same_v<std::underlying_type_t<patchState>, unsigned int>);
}
void patch::printID() const { cout << " " << id << endl; }
std::int64_t patch::getID() const { return id; }
std::int32_t patch::getSID() const { return enriched_id; }
std::span<const int> patch::getRemote() const { return std::span<const int>(remotes); }
void patch::initPatchForComm(MPI_Comm comm,MPI_Comm univ)
{
   // set rank and size
   MPI_Comm_rank(univ, &pidp);
   MPI_Comm_size(univ, &nbprocp);
   MPI_Comm_rank(comm, &pid);
   MPI_Comm_size(comm, &nbproc);


   // set target only for distributed patch
   if (state & DIST)
   {
      // prepare info to send
      std:: unordered_map<int,std::vector<PetscInt>> send_buff;
      {
         // get rank translation
         std::unordered_map<int, int> univ_ranks;
         {
            std::vector<int> receive_rank(nbprocp);
            MPI_Allgather(&pid, 1, MPI_INT, receive_rank.data(), 1, MPI_INT, univ);
            for (int i = 0; i < nbprocp; ++i) univ_ranks[receive_rank[i]] = i;
         }

         // fill send buffer with ghost activated dof in this proc
         std::int32_t nb = dofs_to_exchange.size();
         assert(!(nb % 2));
         for (std::int32_t i = 0; i < nb; i += 2)
         {
            auto itr = univ_ranks.find(dofs_to_exchange[i + 1]);
            assert(itr != univ_ranks.end());
            send_buff[(*itr).second].push_back(dofs_to_exchange[i]);
         }
         nb = Diri_dofs_to_exchange.size();
         assert(!(nb % 2));
         for (std::int32_t i = 0; i < nb; i += 2)
         {
            auto itr = univ_ranks.find(Diri_dofs_to_exchange[i + 1]);
            assert(itr != univ_ranks.end());
            send_buff[(*itr).second].push_back(-Diri_dofs_to_exchange[i] - 1);  // -1 required to disociate 0
         }
      }

      // send info and treat them
      auto &dofs_=dofs;
      auto &dofs_to_exchange_=dofs_to_exchange;
      auto &Diri_dofs_=Diri_dofs;
      auto &Diri_dofs_to_exchange_=Diri_dofs_to_exchange;
      sendVectMPI3(
          send_buff, MPIU_INT, id, univ,
          [&dofs_, &Diri_dofs_](const std::vector<PetscInt> &infos, int from) {
             // loop on infos
             for (auto info : infos)
             {
                if (info < 0)
                {
                   info = -info - 1;
                   auto itf = std::ranges::lower_bound(Diri_dofs_, info);
                   if (itf != Diri_dofs_.end())
                   {
                      if (*itf != info)
                      {
                         // durty
                         Diri_dofs_.insert(itf, info);
                      }
                   }
                   else
                      Diri_dofs_.push_back(info);
                }
                else
                {
                   auto itf = std::ranges::lower_bound(dofs_, info);
                   if (itf != dofs_.end())
                   {
                      if (*itf != info)
                      {
                         // durty
                         dofs_.insert(itf, info);
                      }
                   }
                   else
                      dofs_.push_back(info);
                }
             }
          },
          [&dofs_to_exchange_, &Diri_dofs_to_exchange_](void) {
             // somme work during comunication: clear to_exchange comunicator not needed any more
             dofs_to_exchange_.clear();
             Diri_dofs_to_exchange_.clear();
          });
   }

   // generate free dof group
   free_dofs.clear();
   std::ranges::set_difference(dofs, Diri_dofs, std::back_inserter(free_dofs));

   // change state
   state |= INIT;
}
void patch::print() const
{
   cout<<"Patch "<<id<<" ("<<enriched_id<<") ";
   if (state & LOCAL)
      cout << "local ";
   else
      cout << "distributed ";
   cout << "patch ";
   if (state & GLOBAL_WEIGHT)
      cout << "Global ";
   else
      cout << "Local ";
   cout << "weight:" << weight ;
   if (state & DIST)
   {
      cout << " owner: " << owner ;
      cout << " remote:";
      for (auto r : remotes) cout << " " << r;
   }
   cout << endl;
}
bool patch::isDist() const { return (state & DIST); }
float patch::getWeight() const { return weight; }
patchMPIDirect::patchMPIDirect(std::int32_t enriched_id_, std::int64_t id_, std::uint16_t state_, std::vector<PetscInt> &&dofs_,
                               std::vector<PetscInt> &&dofs_to_exchange_, std::vector<PetscInt> &&Diri_dofs_,
                               std::vector<PetscInt> &&Diri_dofs_to_exchange_, int owner_, std::vector<int> &&remotes_)
    : patch(enriched_id_, id_, state_, std::forward<std::vector<PetscInt>>(dofs_),
            std::forward<std::vector<PetscInt>>(dofs_to_exchange_), std::forward<std::vector<PetscInt>>(Diri_dofs_),
            std::forward<std::vector<PetscInt>>(Diri_dofs_to_exchange_), owner_, std::forward<std::vector<int>>(remotes_)),
      idxqf(nullptr),
      idxdf(nullptr),
      Aqq(nullptr),
      Aqd(nullptr),
      bq0(nullptr),
      bq(nullptr),
      xd(nullptr),
      xdloc(nullptr),
      xq(nullptr),
      computef(nullptr),
      enrich(nullptr),
      solver(nullptr),
      idr(-1),
      idx_dof_f_enriched(-1),
      scatter2e(nullptr)
{
}
patchMPIDirect::~patchMPIDirect()
{
   if (idxqf != nullptr) std::cout << "in ~patchMPIDirect with non cleared data" << std::endl;
}
void patchMPIDirect::clear(MPI_Comm comm, MPI_Comm univ)
{
   if (idxqf != nullptr)
   {
      DO(ISDestroy(&idxqf), "problem while deleting q idx", comm)
      idxqf=nullptr;
   }
   if (idxdf != nullptr)
   {
      DO(ISDestroy(&idxdf), "problem while deleting d idx", comm)
      idxdf=nullptr;
   }
   if (Aqq != nullptr)
   {
      DO(MatDestroy(&Aqq), "Problem to destroy Aqq ", comm)
      Aqq=nullptr;
   }
   if (Aqd != nullptr)
   {
      DO(MatDestroy(&Aqd), "Problem to destroy Aqd ", comm)
      Aqd=nullptr;
   }
   if (bq0 != nullptr)
   {
      DO(VecDestroy(&bq0), "Problem to destroy bq0", comm)
      bq0 = nullptr;
   }
   if (bq != nullptr)
   {
      DO(VecDestroy(&bq), "Problem to destroy bq", comm)
      bq = nullptr;
   }
   if (xd != nullptr)
   {
      DO(VecDestroy(&xd), "Problem to destroy xd", comm)
      xd = nullptr;
   }
   if (xdloc != nullptr)
   {
      DO(VecDestroy(&xdloc), "Problem to destroy xdloc", comm)
      xdloc = nullptr;
   }
   if (xq != nullptr)
   {
      DO(VecDestroy(&xq), "Problem to destroy xq", comm)
      xq = nullptr;
   }
   computef = nullptr;
   if (enrich != nullptr)
   {
      DO(VecDestroy(&enrich), "Problem to destroy enrich", comm)
      enrich = nullptr;
   }
   if (solver != nullptr)
   {
      DO(KSPDestroy(&solver), "Problem to destroy solver", comm)
      solver = nullptr;
   }
   if (scatter2e != nullptr)
   {
      DO(VecScatterDestroy(&scatter2e), "Problem to destroy scatter2e", comm)
      scatter2e = nullptr;
   }
}
void patchMPIDirect::generateProblem(Mat Aff, Vec bf, Vec computef_, MPI_Comm comm, MPI_Comm univ, std::int32_t nbdl,
                                     std::int32_t bs)
{
   assert(state&INIT);
   //cout<<"================================================"<<endl;
   //std::println("Patch id {}: local rank {} and size {}, global rank {} and size {}", id, pidp, nbprocp, pid, nbproc);
   //std::println("dofs      {}: {}\nDiri_dofs {}: {}\nfree_dofs {}: {}", dofs.size(), dofs, Diri_dofs.size(), Diri_dofs, free_dofs.size(), free_dofs);
   if (!(state & DUMMY) && (state & PRBSET))
   {
      std::cout << "TODO: cleanning to avoid memory leak has to be donne if generateProblem is called more than once: "
                << __FILE__ << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
   }
   state ^= PRBSET;
   state ^= PRBSOLVED;

// TODO :to dump or to integrate more cleanly: if generatePatchManager give local dof number this should be implemented here
// For now generatePatchManager gives global filtered dof so this is to dump
#if 0
   ISLocalToGlobalMapping rmapping, cmapping;
   MatGetLocalToGlobalMapping(Aff, &rmapping, &cmapping);
   ISLocalToGlobalMappingView(rmapping,PETSC_VIEWER_STDOUT_(comm));
   ISLocalToGlobalMappingView(cmapping,PETSC_VIEWER_STDOUT_(comm));
   std::ranges::transform(dofs, dofs.begin(), [](const std::int32_t &x) -> std::int32_t { return 2 * x; });
   std::ranges::transform(Diri_dofs, Diri_dofs.begin(), [](const std::int32_t &x) -> std::int32_t { return 2 * x; });
   std::ranges::transform(free_dofs, free_dofs.begin(), [](const std::int32_t &x) -> std::int32_t { return 2 * x; });
   ISLocalToGlobalMappingApply(cmapping, dofs.size(), dofs.data(), dofs.data());
   ISLocalToGlobalMappingApply(cmapping, Diri_dofs.size(), Diri_dofs.data(), Diri_dofs.data());
   ISLocalToGlobalMappingApply(cmapping, free_dofs.size(), free_dofs.data(), free_dofs.data());
   std::ranges::transform(dofs, dofs.begin(), [](const std::int32_t &x) -> std::int32_t { return x/2; });
   std::ranges::transform(Diri_dofs, Diri_dofs.begin(), [](const std::int32_t &x) -> std::int32_t { return x/2; });
   std::ranges::transform(free_dofs, free_dofs.begin(), [](const std::int32_t &x) -> std::int32_t { return x/2; });
   PRINT("dofs", dofs);
   PRINT("Diri_dofs", Diri_dofs);
   PRINT("free_dofs", free_dofs);
#endif

   PetscInt M,N;
   //cout<<"================================================"<<endl;
   // check Aff is distributed
   bool Aff_dist=(dolfinx::MPI::size(comm)>1);

   // generate q-set index and extract Aqq
   // TODO: cleanning submat if this function is called more than once
   IS idxq;
   DO(ISCreateBlock(univ, bs, free_dofs.size(), free_dofs.data(), PETSC_USE_POINTER, &idxq), "problem while creating q-set idx", comm)
   if (Aff_dist)
      DO(MatCreateSubMatricesMPI(Aff, 1, &idxq, &idxq, MAT_INITIAL_MATRIX, &submat[0]), "Problem while creating Aqq", comm)
   else
      DO(MatCreateSubMatrices(Aff, 1, &idxq, &idxq, MAT_INITIAL_MATRIX, &submat[0]), "Problem while creating Aqq", comm)
   Aqq = *submat[0];
   //DO(MatGetSize(Aqq, &M, &N),"Problem while retriving Aqq dimension",comm)
   //std::print("Aqq: M {} N {}", M, N);
   //DO(MatGetOwnershipRange(Aqq, &M, &N),"Problem while retriving Aqq owner range",comm)
   //std::println(" rstard {} rend {}", M, N);
   //MatView(Aqq, PETSC_VIEWER_STDOUT_(univ));

   // generate d-set index and extract Aqd
   IS idxd;
   DO(ISCreateBlock(univ, bs, Diri_dofs.size(), Diri_dofs.data(), PETSC_USE_POINTER, &idxd), "problem while creating d-set idx",
      comm)
   if (Aff_dist)
      DO(MatCreateSubMatricesMPI(Aff, 1, &idxq, &idxd, MAT_INITIAL_MATRIX, &submat[1]), "Problem while creating Aqd", comm)
   else
      DO(MatCreateSubMatrices(Aff, 1, &idxq, &idxd, MAT_INITIAL_MATRIX, &submat[1]), "Problem while creating Aqd", comm)
   DO(ISDestroy(&idxq), "problem while deleting q-set idx", comm)
   DO(ISDestroy(&idxd), "problem while deleting d-set idx", comm)
   Aqd = *submat[1];
   //DO(MatGetSize(Aqd, &M, &N), "Problem while retriving Aqd dimension", comm)
   //std::println("Aqd: M {} N {}", M, N);
   //MatView(Aqd, PETSC_VIEWER_STDOUT_(univ));

   // intermediate idx for copy
   DO(ISCreateBlock(comm, bs, free_dofs.size(), free_dofs.data(), PETSC_USE_POINTER, &idxqf), "problem while creating q-set idx (comm)", comm)
   DO(ISCreateBlock(comm, bs, Diri_dofs.size(), Diri_dofs.data(), PETSC_USE_POINTER, &idxdf), "problem while creating d-set idx (comm)", comm)

   // create bq0,bq,xq,xd
   DO(VecCreateMPI(univ, bs*Diri_dofs.size(), PETSC_DETERMINE, &xd), "Problem while creating  xd", comm)
   DO(VecCreateMPI(univ, bs*free_dofs.size(), PETSC_DETERMINE, &bq0), "Problem while creating  bq0", comm)
   DO(VecDuplicate(bq0, &xq), "Problem while creating  xq", comm)
   DO(VecDuplicate(bq0, &bq), "Problem while creating  bq", comm)

   // intermediate scatter/local for copy
   DO(VecCreateLocalVector(xd, &xdloc), "Problem while creating  xdloc", comm)

   // set computef
   computef=computef_;

   // extract bf into bq0
   // It is possible to use VecISCopy in this context as all dofs by construction are owned by proc that select them. The only
   // case where dof are from other proc is related to dof owned by proc not in support of patch. This case correspond to
   // sub-domain connected to the support of the patch. In this case only TS Dirichlet dof are impacted not free dof that are
   // inside the support (not on its frontier).
   DO(VecISCopy(bf, idxqf,  SCATTER_REVERSE, bq0), "Problem while copying bf in bq0", comm)

   //DO(VecGetSize(bq0, &M), "Problem while retriving bq0 dimension", comm)
   //std::print("bq0 M {}", M);
   //DO(VecGetOwnershipRange(bq0, &M, &N),"Problem while retriving bq0 owner range",comm)
   //std::println(" rstard {} rend {}", M, N);

   // set solver
   DO(KSPCreate(univ, &solver), "Problem while creating solver", comm)
   DO(KSPSetOperators(solver, Aqq, Aqq), "Problem while setting solver operator", comm)
#if 1
   DO(KSPSetType(solver,KSPPREONLY), "Problem while setting solver type", comm)
   PC pc;
   DO(KSPGetPC(solver, &pc), "Problem while getting PC", comm)
   DO(PCSetType(pc, PCLU), "Problem while setting PC type", comm)
   DO(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS), "Problem while setting PCLU type", comm)

   //KSPView(solver, PETSC_VIEWER_STDOUT_(univ));
#else
   DO(KSPSetOptionsPrefix(solver, "ts_patch_"), "Problem while setting prefix to solver", comm)
   DO(KSPSetTolerances(solver, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, 2000), "Problem while setting tolerances to solver", comm)
   DO(KSPSetType(solver, KSPCG),"Problem while imposing solver to be GMRES",comm);
   PC pc;
   DO(KSPGetPC(solver, &pc), "Problem while getting PC", comm)
   DO(PCSetOptionsPrefix(pc, "ts_patch_"), "Problem while setting prefix to pc", comm)
   DO(PCSetType(pc, PCHYPRE), "Problem while setting Hypre pc", comm)
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_type", "boomeramg"), "Problem while setting boomeramg option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_coarsen_type", "HMIS"), "Problem while setting HMIS option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_relax_type_up", "l1scaled-SOR/Jacobi"), "Problem while setting relax up option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_relax_type_down", "l1scaled-SOR/Jacobi"), "Problem while setting relax down option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_relax_type_coarse", "Gaussian-elimination"), "Problem while setting relax coarse option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_interp_type", "ext+i"), "Problem while setting interp type option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_agg_nl", "0"), "Problem while setting  agressive coarsening option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_agg_nl", "0"), "Problem while setting  agressive coarsening option to hypre pc", comm);
   DO(PetscOptionsSetValue(NULL, "-ts_patch_pc_hypre_boomeramg_strong_threshold", "0.25"), "Problem while setting  st option to hypre pc", comm);
   //DO(PetscOptionsSetValue(NULL, "-ts_patch_ksp_monitor", NULL),"Problem while setting ksp_monitor", comm)
   DO(KSPSetFromOptions(solver),"Problem while setting ksp option",comm)
#endif

   // change state
   state |= PRBSET;
}
void patchMPIDirect::solveProblem(Vec xf, MPI_Comm comm, MPI_Comm univ, std::int32_t bs)
{
   assert(state&PRBSET);
   state ^= PRBSOLVED;
   //if (1 && (i_seq == 1172 || i_seq == 1138 || i_seq == 1129))
   //{
   //   cout<<"================================================"<<endl;
   //   std::println("Patch id {}: local rank {} and size {}, global rank {} and size {}", id, pidp, nbprocp, pid, nbproc);
   //   std::println("dofs      {}: {}\nDiri_dofs {}: {}\nfree_dofs {}: {}", dofs.size(), dofs, Diri_dofs.size(), Diri_dofs, free_dofs.size(), free_dofs);
   //}


   // extract xf into xd: unfortunately VecISCopy can't be used in this context. idxdf/Diri_dofs can have dof not owned by current
   // proc (i.e. dof owned by a proc not used in the patch). 
   // Use scatter ...
   // TODO: save scatter2d to avoid its reconstruction. Normaly xf is allways the same ?! no ?? Works like that but durty
   DO(VecGetLocalVector(xd, xdloc), "Problem while getting  xdloc", comm)
   VecScatter scatter2d;
   DO(VecScatterCreate(xf, idxdf,xdloc,nullptr,&scatter2d), "Problem while creating  scatter2d", comm)
   DO(VecScatterBegin(scatter2d, xf, xdloc, INSERT_VALUES,  SCATTER_FORWARD),"Problem while start scattering xf in xdloc",comm);
   DO(VecScatterEnd(scatter2d, xf, xdloc, INSERT_VALUES,  SCATTER_FORWARD),"Problem while end scattering xf in xdloc",comm);
   DO(VecScatterDestroy(&scatter2d),"Problem while releasing scatter2d",comm);
   DO(VecRestoreLocalVector(xd,xdloc),"Problem while releasing xdloc",comm);

   if (!(state & DUMMY))
   {
      //if (1 && (i_seq == 1172 || i_seq == 1138 || i_seq == 1129))
      //{
      //   cout << "xd from xf" << endl;
      //   VecView(xd, PETSC_VIEWER_STDOUT_(univ));
      // }

      // add TS Dirichlet bc
      DO(MatMult(Aqd, xd, bq), "Problem while computing bq=Aqd.xd", comm);

      //if (1 && (i_seq == 1172 || i_seq == 1138 || i_seq == 1129))
      //{
      //   cout << "bq Dirichlet" << endl;
      //   VecView(bq, PETSC_VIEWER_STDOUT_(univ));

      //   cout << "bq0 " << endl;
      //   VecView(bq0, PETSC_VIEWER_STDOUT_(univ));
      //}

      // compute bq = bq0 - Aqd.xd
      DO(VecAYPX(bq, -1., bq0), "Problem while computing bq=  bq0 - bq", comm)

      //if (1 && (i_seq == 1172 || i_seq == 1138 || i_seq == 1129))
      //{
      //   cout << "bq full" << endl;
      //   VecView(bq, PETSC_VIEWER_STDOUT_(univ));
      //}

      // solve
      DO(KSPSolve(solver, bq, xq), "Problem while solving  q system", comm)

      //if (1 && (i_seq == 1172 || i_seq == 1138 || i_seq == 1129))
      //{
      //   cout << "xq" << endl;
      //   VecView(xq, PETSC_VIEWER_STDOUT_(univ));
      //}

      // KSPView(solver, PETSC_VIEWER_STDOUT_(univ));
   }

    // change state
    state |= PRBSOLVED;
}
void patchMPIDirect::setPETSETState() { state |= PETSET; }
bool patchMPIDirect::getPETSETState() { return state & PETSET; }
void patchMPIDirect::grabPatchSolution(Vec xf, MPI_Comm comm, MPI_Comm univ)
{
   assert(state&PRBSOLVED);
   //cout<<"================================================"<<endl;
   //std::println("Patch id {}: local rank {} and size {}, global rank {} and size {}", id, pidp, nbprocp, pid, nbproc);
   //std::println("dofs      {}: {}\nDiri_dofs {}: {}\nfree_dofs {}: {}", dofs.size(), dofs, Diri_dofs.size(), Diri_dofs, free_dofs.size(), free_dofs);


   // collect TS dirichlet
   DO(VecGetLocalVector(xd, xdloc), "Problem while getting  xdloc", comm)
   VecScatter scatter2d;
   DO(VecScatterCreate(xf, idxdf,xdloc,nullptr,&scatter2d), "Problem while creating  scatter2d", comm)
   DO(VecScatterBegin(scatter2d, xdloc, xf, INSERT_VALUES,  SCATTER_REVERSE),"Problem while start scattering xdloc in xf",comm);
   DO(VecScatterEnd(scatter2d, xdloc, xf, INSERT_VALUES,  SCATTER_REVERSE),"Problem while end scattering xdloc in xf",comm);
   DO(VecScatterDestroy(&scatter2d),"Problem while releasing scatter2d",comm);
   DO(VecRestoreLocalVector(xd,xdloc),"Problem while releasing xdloc",comm);
   
   // collect solution
   DO(VecISCopy(xf, idxqf,  SCATTER_FORWARD, xq), "Problem while copying xq in computef", comm)

}

}  // namespace twoscale

namespace twoscale_dolfinx
{
template <>
patchManager<twoscale::patchMPIDirect>::patchManager(std::vector<twoscale::patchMPIDirect *> &&patches_, std::int32_t nbdl_,
                                                     std::int32_t bs_d_, MPI_Comm comm_)
    : patches(std::move(patches_)), comm(comm_), nbdl(nbdl_), bs_d(bs_d_), dummy_patch(nullptr), computef(nullptr)
{
   // generate a dummy patch to satisfy petsc collective operation
   std::vector<PetscInt> dofs_, ddofs_, dofs_to_exchange_, ddofs_to_exchange_;
   std::vector<int> remotes_;
   dummy_patch = static_cast<void *>(new twoscale::patchMPIDirect(-1, -1, twoscale::patch::patchState::DUMMY, std::move(dofs_),
                                                              std::move(dofs_to_exchange_), std::move(ddofs_),
                                                              std::move(ddofs_to_exchange_), -1, std::move(remotes_)));
}
template <>
patchManager<twoscale::patchMPIDirect>::~patchManager()
{
   std::cout << "in ~patchManager" << std::endl;
   auto it = patches.begin();
   for (int i_seq = 0; i_seq < nb_totg; ++i_seq)
   {
      if (dist_com[i_seq] != MPI_COMM_NULL)
      {
         twoscale::patchMPIDirect *patch = *it;
         patch->clear(comm, dist_com[i_seq]);
         delete patch;
         ++it;
      }
      /*
       *
       * ??? work without that ?????
       * 
       *
      // no patch for this sequence but collective operation need to be done
      else
      {
         (static_cast<twoscale::patchMPIDirect *>(dummy_patch))->clear(comm, MPI_COMM_SELF);
      }
      */
   }
   if (computef != nullptr)
   {
      DO(VecDestroy(&computef), "Problem to destroy computef", comm)
      computef = nullptr;
   }

   if (dummy_patch) delete static_cast<twoscale::patchMPIDirect *>(dummy_patch);
}
template <>
patchManager<twoscale::patchMPIDirect>::patchManager(const patchManager<twoscale::patchMPIDirect> &other)
    : comm(other.comm), nbdl(other.nbdl), bs_d(other.bs_d), dummy_patch(nullptr)
{
   std::cout<<"in patchManager copy constructor"<<std::endl;
   assert(0); // need to be tested
   // generate a dummy patch to satisfy petsc collective operation
   // simple but a heap copy would make more sens as it alleviate this dirty copy of source  => TODO
   std::vector<PetscInt> dofs_, ddofs_, dofs_to_exchange_, ddofs_to_exchange_;
   std::vector<int> remotes_;
   dummy_patch = static_cast<void *>(new twoscale::patchMPIDirect(-1, -1, twoscale::patch::patchState::DUMMY, std::move(dofs_),
                                                              std::move(dofs_to_exchange_), std::move(ddofs_),
                                                              std::move(ddofs_to_exchange_), -1, std::move(remotes_)));
   // copy other patches into new instance stored in 'patches' 
   patches.clear();
   for (auto *p : other.patches) patches.push_back(new twoscale::patchMPIDirect(*p));
}
template <>
patchManager<twoscale::patchMPIDirect>::patchManager(patchManager<twoscale::patchMPIDirect> &&other)
    : patches(std::exchange(other.patches, {})),
      comm(std::move(other.comm)),
      nbdl(std::exchange(other.nbdl, 0)),
      bs_d(std::exchange(other.bs_d, 0)),
      nb_totg(std::exchange(other.nb_totg, 0)),
      dummy_patch(std::exchange(other.dummy_patch, nullptr)),
      computef(std::exchange(other.computef, nullptr))
{
   std::cout<<"in patchManager move constructor "<<other.patches.size()<<" "<<patches.size()<<std::endl;
}
template <>
void patchManager<twoscale::patchMPIDirect>::generateProblems(Mat Aff, Vec bf)
{
   // check comm coherant
   MPI_Comm comm_petsc;
   int rcomp;
   DO(PetscObjectGetComm((PetscObject)Aff, &comm_petsc),"Problem while retrieving Aff communicator",MPI_COMM_WORLD)
   MPI_Comm_compare(comm, comm_petsc, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT), "patchManager communicator must be at least congruent with Aff operators", comm)
   DO(PetscObjectGetComm((PetscObject)bf, &comm_petsc),"Problem while retrieving bf communicator",MPI_COMM_WORLD)
   MPI_Comm_compare(comm, comm_petsc, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT), "patchManager communicator must be at least congruent with bf operators", comm)

   // sort patches 
   sortPatches();

   //MatView(Aff, PETSC_VIEWER_STDOUT_(comm));
   //VecView(bf, PETSC_VIEWER_STDOUT_(comm));
   //cout << endl << "nbdl " << nbdl << endl;

   // create computef based on bf to exchange solution
   DO(VecDuplicate(bf, &computef), "Problem while creating  computef", comm)

   auto it = patches.begin();
   for (int i_seq = 0; i_seq < nb_totg; ++i_seq)
   {
      if (dist_com[i_seq] != MPI_COMM_NULL)
      {
         twoscale::patchMPIDirect *patch = *it;
         patch->i_seq=i_seq;
         //cout << endl << "i_seq " << i_seq << " generate" << endl;
         //patch->print();
         patch->initPatchForComm(comm, dist_com[i_seq]);
         patch->generateProblem(Aff, bf, computef, comm, dist_com[i_seq], nbdl, bs_d);
         ++it;
      }
      // no patch for this sequence but collective operation need to be done
      else
      {
         //cout << endl << "i_seq " << i_seq << " generate" << endl << "No patch" << endl;
         (static_cast<twoscale::patchMPIDirect *>(dummy_patch))->initPatchForComm(comm, MPI_COMM_SELF);
         (static_cast<twoscale::patchMPIDirect *>(dummy_patch))
             ->generateProblem(Aff, bf, computef, comm, MPI_COMM_SELF, nbdl, bs_d);
      }

      MPI_Barrier(comm);
   }
   
}
template <>
void patchManager<twoscale::patchMPIDirect>::solveProblems(Vec xf)
{
   // check comm coherant
   MPI_Comm comm_petsc;
   int rcomp;
   DO(PetscObjectGetComm((PetscObject)xf, &comm_petsc),"Problem while retrieving xf communicator",MPI_COMM_WORLD)
   MPI_Comm_compare(comm, comm_petsc, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT), "patchManager communicator must be at least congruent with xf operators", comm)

   //VecView(xf, PETSC_VIEWER_STDOUT_(comm));

   auto it = patches.begin();
   for (int i_seq = 0; i_seq < nb_totg; ++i_seq)
   {
      if (dist_com[i_seq] != MPI_COMM_NULL)
      {
         twoscale::patchMPIDirect *patch = *it;
         //cout<<endl<<"i_seq "<<i_seq<<" solve"<<endl;
         //patch->print();
         patch->solveProblem(xf, comm, dist_com[i_seq],  bs_d);
         ++it;
      }
      // no patch for this sequence but collective operation need to be done
      else
      {
         //cout << endl << "i_seq " << i_seq << " solve" << endl << "No patch" << endl;
         (static_cast<twoscale::patchMPIDirect *>(dummy_patch))->solveProblem(xf, comm, MPI_COMM_SELF, bs_d);
      }

      // TODO check that syncronization is mandatory. A priori no ....
      MPI_Barrier(comm);
   }
   
}
template <>
std::int32_t patchManager<twoscale::patchMPIDirect>::grabPatchSolution(std::int32_t seq, Vec xf)
{
   assert(seq<nb_totg);
   auto it = patches.begin();
   // durty !!! loop before seq to update it !!!! In future patches should be reshaped by adding dummy_patch to any sequence with null_com and
   // dist_com should be update with MPI_COMM_SELF instead of null comm or dist_com transfered to patches TODO
   int i_seq = 0;
   for (; i_seq < seq; ++i_seq)
      if (dist_com[i_seq] != MPI_COMM_NULL) ++it;

   std::int32_t n_id;
   if (dist_com[i_seq] != MPI_COMM_NULL)
   {
      twoscale::patchMPIDirect *patch = *it;
      //cout << endl << "i_seq " << i_seq << " grab" << endl;
      //patch->print();
      patch->grabPatchSolution(xf, comm, dist_com[i_seq]);
      n_id=patch->getSID();
   }
   // no patch for this sequence but collective operation need to be done
   else
   {
      //cout << endl << "i_seq " << i_seq << " grab" << endl << "No patch" << endl;
      (static_cast<twoscale::patchMPIDirect *>(dummy_patch))->grabPatchSolution(xf, comm, MPI_COMM_SELF);
      n_id=-1;
   }

   return n_id;
}
// local function used in sortPatches method
void setColor(twoscale::patchMPIDirect *patch, std::int64_t *trans_table, int proc_id)
{
   trans_table[0] = patch->getID();
   auto rm = patch->getRemote();
   for (const auto ro : rm)
      if (ro > proc_id) trans_table[ro - proc_id] = trans_table[0];
}
bool checkColor(twoscale::patchMPIDirect *patch, std::int64_t *trans_table, int proc_id)
{
   auto rm = patch->getRemote();
   for (const auto ro : rm)
   {
      if (ro > proc_id)
      {
         if (trans_table[ro - proc_id] > -1) return false;
      }
      else
         return false;
   }
   return true;
}

template <>
void patchManager<twoscale::patchMPIDirect>::sortPatches()
{
#if 0
   const bool debug=true;
#else
   const bool debug = false;
#endif
   auto nb_patch = patches.size();
   auto front = patches.begin();
   auto back = patches.end();
   int tag_trans = 1;
   int tag_max = 2;
   int tag_weight = 3;
   double ratiol = 0.;
   float max_w = 0.;
   float weight = 0.;
   int pid;
   MPI_Comm_rank(comm, &pid);
   int nbproc;
   MPI_Comm_size(comm, &nbproc);
   if (nb_patch)
   {
      ratiol = 1. / nb_patch;
      // sort distributed first and in decreasing weight order
      std::sort(patches.begin(), patches.end(), [](const twoscale::patchMPIDirect *p1, const twoscale::patchMPIDirect *p2) -> bool {
         const bool p1_dist = p1->isDist();
         const bool p2_dist = p2->isDist();
         if ((p1_dist && p2_dist) || (!p1_dist && !p2_dist))
         {
            if (p1->getWeight() > p2->getWeight())
               return true;
            else
               return false;
         }
         else
         {
            if (p1_dist) return true;
            return false;
         }
      });
      if (debug) show();
      // set front, back to trace distributed only
      front = patches.begin();
      back = patches.end();
      while (front < back && (*front)->isDist()) ++front;
      back = front;
      front = patches.begin();
   }
   // set l_font, l_back to trace local only
   auto l_front = back;
   auto l_back = patches.end();
   nb_dist = std::distance(front, back);
   assert(nb_dist > -1);

   if (debug)
   {
      if (front != back)
      {
         cout << "front";
         (*front)->printID();
      }
      if (front != back)
      {
         cout << " back";
         (*(back - 1))->printID();
      }
      if (l_front != l_back)
      {
         cout << " l_front";
         (*l_front)->printID();
      }
      if (l_front != l_back)
      {
         cout << " l_back";
         (*(l_back - 1))->printID();
      }
   }

   // finding first maximum number of distributed patch
   MPI_Allreduce(MPI_IN_PLACE, &nb_dist, 1, MPI_INT, MPI_MAX, comm);

   double ratiog = nb_dist;

   // color transfer container
   int enlarge = nb_dist, i_seq = 0, trans_size = nbproc - pid;
   std::int64_t color;
   vector<std::int64_t> trans_color;

   do
   {
      nb_dist = enlarge;

      // color transfer container resizing
      trans_color.resize(trans_size * nb_dist, -1);

      // loop to set sequence in trans_color
      for (; i_seq < nb_dist; ++i_seq)
      {
         if (debug)
         {
            cout << "iseq " << i_seq << endl;
            show();
         }
         std::int64_t *tc = &trans_color[i_seq * trans_size];

         // compute max_w in a ring fashion. reduce ?
         if (debug) cout << "max w 1:" << max_w << endl;
         if (pid < nbproc - 1)
         {
            // proc as a receiver in the ring
            MPI_Recv((void *)&max_w, 1, MPI_FLOAT, pid + 1, tag_max, comm, MPI_STATUS_IGNORE);
         }
         else
         {
            max_w = 0.;
         }
         if (debug) cout << "max w 2:" << max_w << endl;
         // proc as a sender in the ring
         if (pid > 0)
         {
            if (back - front)
               max_w = std::max((*std::max_element(front, back,
                                                   [](const twoscale::patchMPIDirect *p1, const twoscale::patchMPIDirect *p2) -> bool {
                                                      if (p1->getWeight() < p2->getWeight())
                                                         return true;
                                                      else
                                                         return false;
                                                   }))
                                    ->getWeight(),
                                max_w);
            // send maximum weight
            MPI_Send((void *)&max_w, 1, MPI_FLOAT, pid - 1, tag_max, comm);
         }
         if (debug)
         {
            cout << "max w 3:" << max_w << endl;
            cout << "weight 1:" << weight << endl;
         }

         // first proc of the ring make its choice
         if (!pid)
         {
            // choose a new distributed patch
            if (front < back)
            {
               setColor(*front, tc, pid);
               weight = (*front)->getWeight();
               if (debug)
               {
                  cout << "new chosen front";
                  (*front)->printID();
               }
               ++front;
            }
            else
            {
               // choose a new local patch not greater then max_w
               if (l_front < l_back)
               {
                  auto itf = std::find_if(l_front, l_back, [max_w](const twoscale::patchMPIDirect *p) -> bool {
                     if (p->getWeight() < max_w)
                        return true;
                     else
                        return false;
                  });

                  // no local patch fulfill condition => take the smallest
                  if (itf == l_back) --itf;
                  // put this patch in front and maintain decreasing order
                  twoscale::patchMPIDirect *tmp = *itf;
                  if (itf != l_front)
                  {
                     std::copy_backward(l_front, itf, itf + 1);
                     *l_front = tmp;
                  }
                  weight = tmp->getWeight();
                  // set color with this patch
                  *tc = tmp->getID();
                  if (debug)
                  {
                     cout << "new local front ";
                     tmp->printID();
                  }
                  ++l_front;
               }
               // no more patch to treat
               else
               {
                  weight = 0.;
                  *tc = MPI_UNDEFINED;
               }
            }
         }
         if (debug) cout << "weight 2:" << weight << endl;
         // proc as a receiver in the ring
         if (pid > 0)
         {
            MPI_Recv((void *)tc, trans_size, MPI_INT64_T, pid - 1, tag_trans, comm, MPI_STATUS_IGNORE);
            MPI_Recv((void *)&weight, 1, MPI_FLOAT, pid - 1, tag_weight, comm, MPI_STATUS_IGNORE);
            if (debug) cout << "weight 3:" << weight << endl;
            color = *tc;
            //  color is free find a patch to treat
            if (color < 0)
            {
               //  new distributed patch are available
               bool notfound = true;
               if (front < back)
               {
                  for (auto it = front; it != back; ++it)
                  {
                     if (checkColor((*it), tc, pid))
                     {
                        twoscale::patchMPIDirect *tmp = *it;
                        if (it != front)
                        {
                           std::copy_backward(front, it, it + 1);
                           *front = tmp;
                        }
                        setColor(tmp, tc, pid);
                        weight = std::max(tmp->getWeight(), weight);
                        if (debug)
                        {
                           cout << "new front ";
                           tmp->printID();
                        }
                        ++front;
                        notfound = false;
                        break;
                     }
                  }
               }

               if (notfound)
               {
                  // choose a new local patch not greater then weight
                  // Here choosing the weight as limit is:
                  // * conservative if weight<max_w; maybe next chosen patchs in the ring have bigger weight
                  // up to max_n and a bigger local front might have been better.
                  // * perfect if remaining choice are all with smaller or equal patch weight
                  // Due to ring traversal we may not know what will be future chosen weight chosen patch ... maybe
                  // with an other algo ...
                  // Choosing max(weight,max_n) force bigger local patch in all circumstances. Says if
                  // remaining choice are all with smaller weight then weight big local patch may be an error.
                  // To check
                  // note: max_n is the maximum for all remaining proc in the ring, not the maximum for all procs.
                  if (l_front < l_back)
                  {
                     // double mx= weight;
                     double mx = std::max(weight, max_w);
                     auto itf = std::find_if(l_front, l_back, [mx](const twoscale::patchMPIDirect *p) -> bool {
                        if (p->getWeight() < mx)
                           return true;
                        else
                           return false;
                     });

                     // no local patch fulfill condition => take the smallest
                     if (itf == l_back) --itf;
                     // put this patch in front and maintain decreasing order
                     twoscale::patchMPIDirect *tmp = *itf;
                     if (front != itf)
                     {
                        std::copy_backward(front, itf, itf + 1);
                        *front = tmp;
                     }
                     *tc = tmp->getID();
                     weight = std::max(weight, tmp->getWeight());
                     if (debug)
                     {
                        cout << "new local front introduced in seq ";
                        (*front)->printID();
                     }
                     ++back;
                     ++front;
                     ++l_front;
                  }
                  // no more local patch to treat
                  else
                  {
                     *tc = MPI_UNDEFINED;
                     if (debug) cout << "no front";
                  }
               }
            }
            else
            {
               // retrieve patch with id=color and set it first if not. It must be in
               // distributed patch section
               assert(front < back);
               bool found TS_MACRO_WARNUNUSEDTYPE = false;
               for (auto it = front; it != back; ++it)
               {
                  if ((*it)->getID() == color)
                  {
                     twoscale::patchMPIDirect *tmp = *it;
                     if (it != front)
                     {
                        std::copy_backward(front, it, it + 1);
                        *front = tmp;
                     }
                     // no need to set color again : previous proc done it
                     if (debug)
                     {
                        cout << "associate front ";
                        tmp->printID();
                     }
                     ++front;
                     found = true;
                     break;
                  }
               }
               assert(found);
            }
            if (debug) cout << "weight 4:" << weight << endl;
         }
         if (pid < nbproc - 1)
         {
            // send actual selection
            MPI_Send((void *)&tc[1], trans_size - 1, MPI_INT64_T, pid + 1, tag_trans, comm);
            MPI_Send((void *)&weight, 1, MPI_FLOAT, pid + 1, tag_weight, comm);
         }
      }

      // if dist patch remaining we need to enlarge imposed sequence
      if (front < back)
      {
         enlarge += back - front;
         if (debug) cout << "enlarge " << enlarge << endl;
      }

      // find maximum number of distributed patch taking into account enlargement
      MPI_Allreduce(MPI_IN_PLACE, &enlarge, 1, MPI_INT, MPI_MAX, comm);

   } while (enlarge > nb_dist);

   // dist patch should be exhausted
   assert(front == back);

   // communicator container resizing
   int nb_tot = (nb_dist > nb_patch) ? nb_dist : nb_patch;
   nb_totg = nb_tot;
   MPI_Allreduce(MPI_IN_PLACE, &nb_totg, 1, MPI_INT, MPI_MAX, comm);
   dist_com.resize(nb_totg, MPI_COMM_NULL);

   // loop to set communicator according to color
   for (int i_seq = 0; i_seq < nb_totg; ++i_seq)
   {
      // create spited communicator using patch id (i.e. enriched id) as color
      if (i_seq < nb_dist)
      {
         if (debug) cout << "proc " << pid << " i_seq " << i_seq << " color " << trans_color[i_seq * trans_size] << endl;
         MPI_Comm_split(comm, trans_color[i_seq * trans_size], 0, &dist_com[i_seq]);
      }
      else
      {
         // remaining patch are local ones and are treated locally
         if (i_seq < nb_tot) dist_com[i_seq] = MPI_COMM_SELF;
      }
   }

   // statistic
   cout << "=====================================================================================================================================" << endl;
   cout << "Patches sorting statistic" << endl;
   cout << "In this proc: " << endl;
   cout << " Total number of iterations: " << nb_tot << endl;
   cout << " Total number of sequences(number of patches computed with dependency): " << nb_dist << endl;
   cout << " Total number of patches:" << nb_patch << endl;
   int nb_inde = ((nb_dist < nb_patch) ? nb_patch - nb_dist : 0);
   cout << " Total number of patches independent:" << nb_inde << endl;
   cout << " Number of sequence not computing patches:" << ((nb_dist > nb_patch) ? nb_dist - nb_patch : 0) << endl;
   cout << "Local performance Ratio (nb of computing sequence/total nb of patch to compute; ideal 1): " << ratiol * nb_tot
        << endl;
   if (nbproc > 1)
   {
      assert(ratiog);  // normally mesh is balanced in a way to divide work which imply at least one patch is distributed
      cout << "Global performance Ratio (nb of computing sequence for distributed patch/ total nb of distributed patch to "
              "compute ) : "
           << (nb_dist * 1.) / ratiog << endl;
   }
   cout << "=====================================================================================================================================" << endl;
}
}  // namespace twoscale_dolfinx
