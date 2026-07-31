/*
 * Copyright (C) 2026 - Ecole Centrale de Nantes
 * Author: Alexis Salzman
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_PATCHMANAGER_IMP
#define TS_DOLFINX_PATCHMANAGER_IMP
#ifndef TS_DOLFINX_PATCHMANAGER
#error "Should not be included by hand"
#endif

#include <unordered_set>

#include "debug.h"

namespace twoscale
{
template <typename F>
void patchMPIDirect::PEtUpdate(Mat PEt, std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &enriched_dof_id,
                               std::shared_ptr<F> func, MPI_Comm comm, MPI_Comm univ, std::int32_t bs, PetscInt rstart,
                               PetscInt rend)
{
   // prerequisites
   // You can not call PEtUpdate if  patch problem were not solved (i.e. xq store a valid solution for this patch)
   assert(state&PRBSOLVED);
   // constant
   const PetscScalar zero(0.);

   //std::cout<<"================================================"<<std::endl;
   //std::println("Patch id {}: local rank {} and size {}, global rank {} and size {}", id, pidp, nbprocp, pid, nbproc);
   //std::println("dofs      {}: {}\nDiri_dofs {}: {}\nfree_dofs {}: {}", dofs.size(), dofs, Diri_dofs.size(), Diri_dofs, free_dofs.size(), free_dofs);
   

   // If not set transfert PEt rows into associated patches
   if(!(state&PETSET))
   {
      idr = -1;
      if (state & DUMMY)
      {
         if (!enrich) DO(VecCreateSeq(MPI_COMM_SELF, 0, &enrich), "problem while creating null vector enrich for dummy", comm)
         IS idxr;
         DO(ISCreateGeneral(comm, PEtJCols.size(), PEtJCols.data(), PETSC_USE_POINTER, &idxr),
            "problem while creating row idx (comm)", comm)
         // TODO
         // if (scatter2e!=null) VecScatterDestroy is not possible as it is a collective operation
         // transform scatter2e as vector of VecScatter so that for all patch only one VecScatter is present
         // and for dummy as many VecScatter are present as there is dummy in sequence.
         // For now by miracle overwriting scatter2e doesn't seems to make problems. Each dummy is associated with comm which is
         // stable but the forest is not the same in between sequence. The fact that dummy do not provide trees in the forest is
         // certainly what makes things working. Nevertheless at each sequence with dummy there is a memory leak as previous
         // scatter2e local instance is not cleaned !!
         std::cout << "dummy scatter2e  memory leak !!!!" << scatter2e << std::endl;
         DO(VecScatterCreate(computef, idxr, enrich, nullptr, &scatter2e), "Problem while creating  dummy scatter2e", comm)

         //std::cout << "idxr " << std::endl;
         //ISView(idxr, PETSC_VIEWER_STDOUT_(comm));

         DO(ISDestroy(&idxr), "problem while deleting row idx", comm)

      }
      else
      {
         const PetscInt *cols;
         const PetscScalar *vals;
         PetscInt ncols = 0;
         // find if this proc owns the enriched row(s) to update
         auto itf = enriched_dof_id.find(enriched_id);
         if (itf != enriched_dof_id.end())
         {
            auto &idt = itf->second.global_coarse_block_dof_idx;
            //std::print("id {} enriched_id {} concerne {}", id, enriched_id, idt);
            if (idt < rend && rstart <= idt) 
            {
               idr = idt;
               eliminated_coarse_encoding = itf->second.eliminated_coarse_encoding;
            }
         }

         // if this proc owns the  rows, get it and store current values in PEtXXCols vector.
         // This is done once here when patch is not in state PETSET and thus normaly PEt row
         // is expected to store initial computed coefficiant created in generateCoarseManager function.
         // Attention needed around that choice: does this may be wrong
         // with some combination of update ? in NL ? A prori as PEt is related
         // to scale jump calling coarse or patch manager routine will be ok. But
         // in future if some sort of scale jump update exist this part had to be checked
         //
         // Get also the bloc idx of the fine dof corresponding to coarse enriched node of this patch
         if (idr > -1)
         {
            if (eliminated_coarse_encoding)
            {
               //loop on component
               std::int32_t b = 0;
               for (; b < bs; ++b)
               {
                  if (eliminated_coarse_encoding ^ (1 << b))
                  {
                     // get row
                     DO(MatGetRow(PEt, idr + b, &ncols, &cols, &vals), "Problem while retrieving specific PEt row ", comm)
                     // copy
                     PEtJCols.resize(ncols);
                     std::ranges::copy(std::span<const PetscInt>(cols, ncols), PEtJCols.begin());
                     PEtValCols.resize(ncols);
                     std::ranges::copy(std::span<const PetscScalar>(vals, ncols), PEtValCols.begin());
                     // release row
                     DO(MatRestoreRow(PEt, idr + b, &ncols, &cols, &vals), "Problem while retrieving specific PEt row ", comm)
                     ncols = PEtJCols.size();
                     // the first available row is used: remaining rows are ignored
                     break;
                  }
               }
               // only if it is not the first component that is available pack 
               if (b)
               {
                  // pack : considering that same values holds for others rows of the block the retrieved one is packed to form
                  // the first component so that PEtxxCols is similar to selective_update==false
                  // Column are the sames only values needs to be shifted
                  PetscScalar *pv = &PEtValCols[-b];
                  for (std::int32_t i = b; i < ncols; i += bs)
                  {
                     pv[i] = PEtValCols[i];
                     PEtValCols[i] = zero;
                  }
               }
            }
            else
            {
               // Only first row of the block is stored considering that same values holds for others rows of the block (i.e.
               // components). For this others rows values are just in different column
               // std::print("get row {}", idr);
               // get row
               DO(MatGetRow(PEt, idr, &ncols, &cols, &vals), "Problem while retrieving specific PEt row ", comm)
               // copy
               PEtJCols.resize(ncols);
               std::ranges::copy(std::span<const PetscInt>(cols, ncols), PEtJCols.begin());
               PEtValCols.resize(ncols);
               std::ranges::copy(std::span<const PetscScalar>(vals, ncols), PEtValCols.begin());
               // release row
               DO(MatRestoreRow(PEt, idr, &ncols, &cols, &vals), "Problem while retrieving specific PEt row ", comm)
            }
            //std::print(" nb cols {} cols: {} vals: {}", PEtJCols.size(), PEtJCols, PEtValCols);
            //std::cout << std::endl;
            // find idx of fine dof
            auto f = std::lower_bound(PEtJCols.begin(), PEtJCols.end(), itf->second.global_fine_block_dof_idx);
            assert(f != PEtJCols.end());
            idx_dof_f_enriched = f - PEtJCols.begin();
            assert(idx_dof_f_enriched < PEtJCols.size());
            assert(idx_dof_f_enriched % bs < 1);
            idx_dof_f_enriched /= bs;
         }
         // create vector to store patch solution in row order
         DO(VecCreateSeq(MPI_COMM_SELF, PEtJCols.size(), &enrich), "problem while creating vector enrich", comm)

         // generate the scatter Context to grab patch solution in row order
         IS idxr;
         DO(ISCreateGeneral(comm, PEtJCols.size(), PEtJCols.data(), PETSC_USE_POINTER, &idxr),
            "problem while creating row idx (comm)", comm)
         DO(VecScatterCreate(computef, idxr, enrich, nullptr, &scatter2e), "Problem while creating  scatter2e", comm)

         //std::cout << "idxr " << std::endl;
         //ISView(idxr, PETSC_VIEWER_STDOUT_(comm));

         DO(ISDestroy(&idxr), "problem while deleting row idx", comm)

         // PETJcols not needed anymore only block J are used to set PEt rows: compute block version
         assert(PEtJCols.size() % bs < 1);
         for (std::int32_t i = 0, k = 0; i < PEtJCols.size(); i += bs)
         {
            assert(PEtJCols[i] % bs < 1);
            PEtJCols[k++] = PEtJCols[i] / bs;
         }
         PEtJCols.resize(PEtJCols.size() / bs);

         // change state
         state |= PETSET;
      }
   }

   // transfert patch solution to computef buffer
   DO(VecISCopy(computef, idxqf,  SCATTER_FORWARD, xq), "Problem while copying xq in computef", comm)

   //std::cout<<"xq"<<std::endl;
   //VecView(xq, PETSC_VIEWER_STDOUT_(univ));
   //std::cout<<"computef "<<std::endl;
   //VecView(computef, PETSC_VIEWER_STDOUT_(comm));

   // collect patch solution in enrich
   DO(VecScatterBegin(scatter2e, computef, enrich, INSERT_VALUES,  SCATTER_FORWARD),"Problem while start scattering computef in enrich",comm)
   DO(VecScatterEnd(scatter2e, computef, enrich, INSERT_VALUES,  SCATTER_FORWARD),"Problem while end scattering computef in enrich",comm)

   //std::cout << "enrich " << std::endl;
   //VecView(enrich, PETSC_VIEWER_STDOUT_(MPI_COMM_SELF));


   // compute enriched function and store it in PEt row
   if (idr > -1)
   {
      std::int32_t ncols = PEtValCols.size();
      // prepare memory for enriched function computation
      // all bs row to change have the same number of column
      std::vector<PetscScalar> compute(bs * ncols, 0);

      // call enrichment function functor to obtain enrichment function from collected patch solution
      PetscScalar *enrich_array = nullptr;
      DO(VecGetArray(enrich, &enrich_array),"Problem while getting enrich array",comm)
      func->operator()(std::span<const PetscScalar>(enrich_array, ncols), idx_dof_f_enriched, compute);
      //if (1 && (i_seq == 1172 || i_seq == 1138 || i_seq == 1129))
      //{
      //   std::println("idx_dof_f_enriched   {}", idx_dof_f_enriched);
      //   std::println("field   {}", std::span<const PetscScalar>(enrich_array, ncols));
      //   std::println("enrich f{}", compute);
      //}
      DO(VecRestoreArray(enrich, &enrich_array), "Problem while getting enrich array", comm)

      // Compute the product of the coarse interpolation coefficients by the enrichment function value.
      // Use the fact that coefficient are the same for block (i.e. same value in diagonal terms of each block)
      // Fine field is explicitly blocked while coarse enriched field is implicitly blocked (not anymore the case for MatNest)
      // compute store the row major full block (i.e. all rows and columns)
      for (std::int32_t i = 0; i < ncols; i += bs)
      {
         auto &val = PEtValCols[i];
         for (std::int32_t b = 0; b < bs; ++b)
         {
            auto k=b*ncols;
            auto v = compute[i + b];
            compute[i+b]=zero;
            compute[k + i + b] = v * val;
         }
      }

      //std::println("new row {}\ncolsb {}\nvalsb {}\nidrb {} ncols {}", compute,PEtJCols,PEtValCols,idr,ncols);

      if (eliminated_coarse_encoding)
      {
         // set new row values only for non eliminated rows
         PetscInt idrb = idr;
         for (std::int32_t b = 0; b < bs; ++b)
         {
            if (eliminated_coarse_encoding ^ (1 << b))
               DO(MatSetValuesBlocked(PEt, 1, &idrb, PEtJCols.size(), PEtJCols.data(), compute.data() + b * ncols, INSERT_VALUES),
                  "Problem while setting new row to PEt", comm)
            ++idrb;
         }
      }
      else
      {
         // set new row values for full block
#ifdef TWOSCALE_PETSC_MATNEST
         assert(!(idr % bs));
         PetscInt idrb = idr / bs;
         DO(MatSetValuesBlocked(PEt, 1, &idrb, PEtJCols.size(), PEtJCols.data(), compute.data(), INSERT_VALUES),
            "Problem while setting new row to PEt", comm)
#else
         PetscInt idrb = idr;
         for (std::int32_t b = 0; b < bs; ++b)
         {
            DO(MatSetValuesBlocked(PEt, 1, &idrb, PEtJCols.size(), PEtJCols.data(), compute.data() + b * ncols, INSERT_VALUES),
               "Problem while setting new row to PEt", comm)
            ++idrb;
         }
#endif
      }
   }
}
}  // namespace twoscale
namespace twoscale_dolfinx
{
template <typename P>
patchManager<P>::patchManager(std::vector<P *> &&patches_, std::int32_t nbdl_, std::int32_t bs_d_, MPI_Comm comm_)
    : patches(std::forward<std::vector<P *>>(patches_)), comm(comm_), nbdl(nbdl_), bs_d(bs_d_)
{
}
template <typename P>
patchManager<P>::~patchManager()
{
}
template <typename P>
void patchManager<P>::sortPatches()
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename P>
void patchManager<P>::show() const
{
   std::cout << "Patches Id:";
   for (auto p : patches) std::cout << " " << p->getID();
   std::cout << std::endl;
}
template <typename P>
void patchManager<P>::generateProblems(Mat Aff, Vec bf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename P>
void patchManager<P>::solveProblems(Vec xf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename P>
template <typename F>
void patchManager<P>::PEtUpdate(Mat PEt, std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &enriched_dof_id,
                                std::shared_ptr<F> func)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
}
template <typename P>
int patchManager<P>::numberOfSequence()
{
   return nb_totg;
}
template <typename P>
std::int32_t patchManager<P>::grabPatchSolution(std::int32_t seq, Vec xf)
{
   std::cout << "To implement" << __FILE__ << " " << __LINE__ << std::endl;
   MPI_Abort(MPI_COMM_WORLD, -1);
   return 0;
}
template <>
patchManager<twoscale::patchMPIDirect>::patchManager(const patchManager<twoscale::patchMPIDirect> &other);
template <>
patchManager<twoscale::patchMPIDirect>::patchManager(patchManager<twoscale::patchMPIDirect> &&other);
template <>
patchManager<twoscale::patchMPIDirect>::~patchManager();
template <>
patchManager<twoscale::patchMPIDirect>::patchManager(std::vector<twoscale::patchMPIDirect *> &&patches_, std::int32_t nbdl_,
                                                     std::int32_t bs_d_, MPI_Comm comm_);
template <>
void patchManager<twoscale::patchMPIDirect>::generateProblems(Mat Aff, Vec bf);
template <>
void patchManager<twoscale::patchMPIDirect>::solveProblems(Vec xf);
template <>
std::int32_t patchManager<twoscale::patchMPIDirect>::grabPatchSolution(std::int32_t seq, Vec xf);
template <>
void patchManager<twoscale::patchMPIDirect>::sortPatches();

template <>
template <typename F>
void patchManager<twoscale::patchMPIDirect>::PEtUpdate(Mat PEt, std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &enriched_dof_id,
                                                       std::shared_ptr<F> func)
{
   // check comm coherant
   MPI_Comm comm_petsc;
   int rcomp;
   DO(PetscObjectGetComm((PetscObject)PEt, &comm_petsc),"Problem while retrieving PEt communicator",MPI_COMM_WORLD)
   MPI_Comm_compare(comm, comm_petsc, &rcomp);
   CHECKTS((rcomp != MPI_CONGRUENT && rcomp != MPI_IDENT), "patchManager communicator must be at least congruent with PEt operator", comm)

   //std::println("PEt matrix");
   //MatView(PEt, PETSC_VIEWER_STDOUT_(comm));

   // get row ownership
   // same for all patches treatment
   PetscInt rstart, rend;
   DO(MatGetOwnershipRange(PEt, &rstart, &rend), "Problem while retrieving PEt owner range", comm)

   // By looking if dummy patch is in PETSET mode or not the LAST or INTERMEDIATE strategy can be
   // chosen. If PETSET is set for this dummy patch it means that at least on call to this function
   // has been done and thus all PEt coefficiant are transfered into patches PEtXXCols vectors:
   // * INTERMEDIATE is mandatory in the loop when this method is called for the first time and
   //   PEt coefficiant are transfered into PEtXXCols using MatGetRow which requires that PEt
   //   is in FINAL assembly mode
   // * LAST can be use in next call to this method as MatGetRow in not called anymore. The final
   //   assembly is donne at the end of the loop merging all modified rows per all sequence at once 
   //
   bool do_last=(static_cast<twoscale::patchMPIDirect *>(dummy_patch))->getPETSETState();

   auto it = patches.begin();
   for (int i_seq = 0; i_seq < nb_totg; ++i_seq)
   {
      if (dist_com[i_seq] != MPI_COMM_NULL)
      {
         twoscale::patchMPIDirect *patch = *it;
         //std::cout<<std::endl<<"i_seq "<<i_seq<<" PEt"<<std::endl;
         //patch->print();
         patch->PEtUpdate(PEt, enriched_dof_id, func, comm, dist_com[i_seq], bs_d, rstart, rend);
         ++it;
      }
      // no patch for this sequence but collective operation need to be done
      // As dummy is call many times it state is controled from this method
      else
      {
         //std::cout << std::endl << "i_seq " << i_seq << " PEt" << std::endl << "No patch" << std::endl;
         (static_cast<twoscale::patchMPIDirect *>(dummy_patch))
             ->PEtUpdate(PEt, enriched_dof_id, func, comm, MPI_COMM_SELF, bs_d, rstart, rend);
      }

      MPI_Barrier(comm);

      // INTERMEDIATE strategy
      // considering that for this sequence more then one patch may have update rows of PEt a full 
      // assembly is required before updating new rows as MatGetRow is called
      if (!do_last)
      {
         DO(MatAssemblyBegin(PEt, MAT_FINAL_ASSEMBLY), "PEt begin assembly operator", comm)
         DO(MatAssemblyEnd(PEt, MAT_FINAL_ASSEMBLY), "PEt end assembly operator", comm)
      }

   }

   // if LAST strategy is used, after the loop full assembly is required
   if (do_last)
   {
      DO(MatAssemblyBegin(PEt, MAT_FINAL_ASSEMBLY), "PEt begin assembly operator", comm)
      DO(MatAssemblyEnd(PEt, MAT_FINAL_ASSEMBLY), "PEt end assembly operator", comm)
   }

   // Force dummy state to PETSET after the loop so that in next call to this method all patch are in same state
   // and collective operation remain synchronized.
   // Logically a test should look if PETSET is not set for dummy and set it only in this case 
   // But always set it, certainnely cost the same.
   (static_cast<twoscale::patchMPIDirect *>(dummy_patch))->setPETSETState();

#ifdef TWOSCALE_OUTPUT_MATVECT
   PetscViewer PET_viewer;
   DO(PetscViewerBinaryOpen(comm, "PEt.mat", FILE_MODE_WRITE, &PET_viewer), "openning file PEt.mat fails", comm)
   DO(PetscViewerBinarySetUseMPIIO(PET_viewer, PETSC_TRUE), "switching to MPIIO viewer fails", comm)
   DO(MatView(PEt, PET_viewer),"save PEt.mat fails",comm)
   DO(PetscViewerDestroy(&PET_viewer), "Destroy PET_viewer fails", comm)
#endif
   //std::println("PEt matrix updated");
   //MatView(PEt, PETSC_VIEWER_STDOUT_(comm));
}

template <typename P, std::floating_point U>
twoscale_dolfinx::patchManager<P> generatePatchManager(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &sj,
                                                       std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space)
{
   spdlog::info("Compute patches from fine Matrix and vector");

   // ===================
   // check coherance
   // ===================
   // assert((sj.needs_mpc) ? (mpc != nullptr) : (mpc == nullptr));
   // TODO: mesh dim etc

   // ================
   // collect mpi info
   // ================
   MPI_Comm comm = sj.getCoarseMesh()->comm();
   int pid = dolfinx::MPI::rank(comm);
   int nbproc = dolfinx::MPI::size(comm);

   auto enriched = sj.getEnriched();
   auto extra_enriched = sj.getExtraEnriched();
   std::vector<P *> patches;
   patches.reserve(enriched.size() + extra_enriched.size());

   // ======================================
   // collect coarse topological information
   // ======================================
   auto cdomain = sj.getCoarseMesh();
   auto topoc = cdomain->topology();
   const int dim = topoc->dim();
   const int dimm1 = dim - 1;
   const int dimp2 = dim + 2;
   auto adj = topoc->connectivity(0, dim);
   auto adjcf = topoc->connectivity(dim,dimm1);
   auto adjfc = topoc->connectivity(dimm1,dim);
   auto adjcv = topoc->connectivity(dim,0);
   auto idxmapc0 = topoc->index_map(0);
   std::int32_t nbncl = idxmapc0->size_local();
   auto idx_2_dest_rank = idxmapc0->index_to_dest_ranks();
   auto owners = idxmapc0->owners();
   auto idxmapc1 = topoc->index_map(dimm1);
   auto idxf_2_dest_rank = idxmapc1->index_to_dest_ranks();
   std::int32_t nbfcl = idxmapc1->size_local();
   std::int32_t nbfc = nbfcl + idxmapc1->num_ghosts();


   // ======================================
   // create surrounding face identification
   // ======================================
   std::vector<std::int32_t> faces_extrasupport;
   std::vector<std::int64_t> exchangeg;
   auto support = sj.getSupport();
   auto surround = sj.getSurroundingCells();
   {
      std::unordered_set<std::int32_t> faces_extrasupport_;
      auto addf = [&adjcf, &faces_extrasupport_](const std::int32_t &cell) {
         auto f = adjcf->links(cell);
         faces_extrasupport_.insert(f.begin(), f.end());
      };
      std::ranges::for_each(support, addf);
      std::ranges::for_each(surround, addf);
      faces_extrasupport.insert(faces_extrasupport.end(), faces_extrasupport_.begin(), faces_extrasupport_.end());
      std::ranges::sort(faces_extrasupport);
   }
   {
      auto sub_idxmapc1 = dolfinx::common::create_sub_index_map(*idxmapc1, std::span<const std::int32_t>(faces_extrasupport),
                                                                dolfinx::common::IndexMapOrder::any, true);
      std::int32_t nbfscl = sub_idxmapc1.first.size_local();
      std::int32_t nbfsc = nbfscl + sub_idxmapc1.first.num_ghosts();
      assert(nbfsc <= nbfc);
      faces_extrasupport.resize(nbfc);
      std::ranges::fill(faces_extrasupport, -1);
      std::int32_t k = -1;
      // loop on all coarse face of interest in sub index map order to set index maping
      for (auto f : sub_idxmapc1.second) faces_extrasupport[f] = ++k;
      // container to store tag per face that will be exchanged
      assert(dimp2<6);
      exchangeg.resize(nbfsc*dimp2);
      {
         std::vector<std::int32_t> exchange(nbfsc * dimp2, -1);
         // functor to update localy tag per face
         auto addt = [&adjcv, &adjcf, &adjfc, &exchange, &dimp2, &faces_extrasupport, &idxf_2_dest_rank,
                      &nbfcl](const std::int32_t &cell) {
            auto v = adjcv->links(cell);
            auto faces = adjcf->links(cell);
            for (auto f : faces)
            {
               // skip face that are local (not ghost nor owned by this proc for any other proc) and connected only to one cell:
               //  ===> part boundary
               // No node attached to them as for all patch related to them they are just free face for the patch
               // problem (no TS Dirichlet)
               if (f < nbfcl && (idxf_2_dest_rank.links(f)).size() < 1 && adjfc->links(f).size() < 2) continue;

               auto rf = faces_extrasupport[f];
               std::span<std::int32_t> face_node(exchange.data() + rf * dimp2, dimp2);
               for (auto node : v)
               {
                  // if node not present add it
                  auto itf = std::ranges::find(face_node, node);
                  if (itf == face_node.end())
                  {
                     auto itf = std::ranges::find(face_node, -1);
                     assert(itf != face_node.end());
                     *itf = node;
                  }
                  // if node present its a double on the face:
                  // 2 cells of the same patch connected to this face thus this face is not on boundary of the patch
                  // thus it is removed
                  else
                     *itf = -1;
               }
            }
         };
         std::ranges::for_each(support, addt);
         std::ranges::for_each(surround, addt);
         idxmapc0->local_to_global(std::span<const std::int32_t>(exchange), std::span<std::int64_t>(exchangeg));
         std::ranges::transform(exchangeg, exchange, exchangeg.begin(),
                                [](const std::int64_t &i, const std::int32_t &j) -> std::int64_t {
                                   if (j < 0)
                                      return -1;
                                   else
                                      return i;
                                });
         
      }
      // ghost to local exchange with local update
      dolfinx::common::Scatterer coarse_sub_face_scatterer(sub_idxmapc1.first, dimp2);
      coarse_sub_face_scatterer.scatter_rev(std::span<std::int64_t>(exchangeg.begin(), exchangeg.begin() + nbfscl * dimp2),
                                            std::span<const std::int64_t>(exchangeg.begin() + nbfscl * dimp2, exchangeg.end()),
                                            [&exchangeg, &dimp2](const std::int64_t &i, const std::int64_t &j) {
                                               std::int32_t rf = (&i - exchangeg.data()) / dimp2;
                                               std::span<std::int64_t> face_node(exchangeg.data() + rf * dimp2, dimp2);
                                               if (j>-1)
                                               {
                                                  // if j not present add it
                                                  auto itf = std::ranges::find(face_node, j);
                                                  if (itf == face_node.end())
                                                  {
                                                     auto itf = std::ranges::find(face_node, -1);
                                                     assert(itf != face_node.end());
                                                     *itf = j;
                                                  }
                                                  // if j present its a double on the face:
                                                  // 2 cells of the same patch connected to this face thus this face is not on
                                                  // boundary of the patch thus it is removed
                                                  else
                                                     *itf = -1;
                                               }
                                               return i;
                                            });
      // local updated now imposed to ghost
      coarse_sub_face_scatterer.scatter_fwd(std::span<const std::int64_t>(exchangeg.begin(), exchangeg.begin() + nbfscl * dimp2),
                                            std::span<std::int64_t>(exchangeg.begin() + nbfscl*dimp2, exchangeg.end()));
   }

   // ==============================
   // collect fine space information
   // ==============================
   auto dofmapf = fine_space->dofmap();
   auto idxmapf = dofmapf->index_map;
   auto bs_f = dofmapf->index_map_bs();
   std::int32_t nbdfl = idxmapf->size_local();
   auto ownersf = idxmapf->owners();
   auto dof_idx_2_dest_rank = idxmapf->index_to_dest_ranks();


   auto &layout = dofmapf->element_dof_layout();
   const auto &closure = layout.entity_closure_dofs_all();


   // ====================================
   // collect fine topological information
   // ====================================
   auto fdomain = sj.getFineMesh();
   auto topof = fdomain->topology();
   auto adjcff = topof->connectivity(dim,dimm1);

   // ============================
   // create enriched node patches
   // ============================
   std::uint16_t state;
   auto gen = [&dofmapf, &idxmapf, &ownersf, &dof_idx_2_dest_rank, &idx_2_dest_rank, &nbncl, &owners, &sj, &adj, &adjcf, &adjcff,
               &state, &patches, &pid, &idxmapc0, &faces_extrasupport, &exchangeg, &dimm1, &dimp2, &closure,
               &nbdfl](const std::int32_t &node) {
      int owner;
      std::int64_t nodeg;
      std::vector<int> remotes;
      std::unordered_set<std::int32_t> u_dofs;
      std::unordered_map<std::int32_t, int> u_dofs_g;
      std::unordered_set<std::int32_t> u_ddofs;
      std::unordered_map<std::int32_t, int> u_ddofs_g;
      std::unordered_set<std::int32_t> u_faces;
      auto rem = idx_2_dest_rank.links(node);
      if (rem.size() > 0) remotes.assign(rem.begin(), rem.end());
      idxmapc0->local_to_global(std::span<const std::int32_t>(&node, 1), std::span<std::int64_t>(&nodeg, 1));

      if (node < nbncl)
      {
         state = (remotes.size() > 0) ? twoscale::patch::patchState::DIST : twoscale::patch::patchState::LOCAL;
         owner = pid;
      }
      else
      {
         state = twoscale::patch::patchState::DIST;
         owner = owners[node - nbncl];
      }
      // collect childs' faces
      auto cells_support = adj->links(node);
      for (auto cell_id : cells_support)
      {
         auto faces = adjcf->links(cell_id);
         for (auto f : faces)
         {
            auto rf = faces_extrasupport[f];
            assert(rf > -1);
            std::span<std::int64_t> face_node(exchangeg.data() + rf * dimp2, dimp2);
            auto itf = std::ranges::find(face_node, nodeg);
            // if face is on boundary of patch
            if (itf != face_node.end())
            {
               // First ask for standard childs for this face
               auto faces_child = sj.getFaceChilds(f);
               u_faces.insert(faces_child.begin(), faces_child.end());
               // In general case if getFaceChilds return nothing (faces_child.size()<1) it means that it is
               // a coarse face of surrounding element and thus getSurroundingFaceChilds needs to be used.
               // But in some special case surrounding element are connected by 2 faces to support. In this rare cases
               // depending on investigate patch master face are either boundary of the patch or internal to it.
               // In this particular case faces_child.size()>0 but in fact it is the coarse master face that correspond
               // to patch boundary thus even if getFaceChilds return faces we must look to getSurroundingFaceChilds !
               // Thus a test is impossible to reduce  getSurroundingFaceChilds usage.
               // Get now equivalent of coarse face
               faces_child=sj.getSurroundingFaceChilds(f);
               u_faces.insert(faces_child.begin(), faces_child.end());
            }
         }
      }
      auto feed = [&nbdfl, &ownersf, &remotes, &pid, &dof_idx_2_dest_rank](
                      const std::int32_t &x, std::unordered_set<std::int32_t> &u_d,
                      std::unordered_map<std::int32_t, int> &u_d_g) {
         // local dof kept
         if (x < nbdfl) u_d.insert(x);
         // ghost dof need to be treated
         else
         {
            // skip work if already treated
            if ((u_d_g.find(x) == u_d_g.end()) && (u_d.find(x) == u_d.end()))
            {
               int ownerf = ownersf[x - nbdfl];
               // ghost dof owned by one remote proc of the patch  must receve the information
               // that it is part of this patch. It may not have been activated in the owner process.
               // Store with its destination.
               if (std::ranges::find(remotes, ownerf) != remotes.end())
               {
                  u_d_g.insert(std::make_pair(x, ownerf));
               }
               // ghost dof is owned by a proc not related to the patch (i.e. having no element of the patch)
               else
               {
                  auto remotex = dof_idx_2_dest_rank.links(x);
                  // many remotes exist in addition to ownerf so one must be chosen to store x
                  if (remotex.size() > 1)
                  {
                     // init candidat to this proc
                     int target = pid;
                     int dist = abs(pid - ownerf);
                     // loop on candidate
                     for (auto r : remotex)
                     {
                        // exclude ownerf and not in remotes candidate
                        if (r != ownerf && std::ranges::find(remotes, r) != remotes.end())
                        {
                           int ndist = abs(r - ownerf);
                           // if candidat is "closer" to ownerf compaires to previous candidate it wins
                           if (ndist < dist)
                           {
                              dist = ndist;
                              target = r;
                           }
                           // if candidat is at same distance as ownerf compaires to previous candidate only one must be chosen:
                           // the smallest
                           else if (!(ndist>dist))
                           {
                              if (r < target) target = r;
                           }
                        }
                     }
                     // if target is the proc itself then this proc store x
                     if (target == pid) u_d.insert(x);
                     // x must be send to chossen target normaly unique across process (closest algo give the same everywhere)
                     else
                        u_d_g.insert(std::make_pair(x, target));
                  }
                  // only one remote exist (the ownerf) so this proc store x
                  else
                  {
                     assert(remotex[0] == ownerf);
                     u_d.insert(x);
                  }
               }
            }
         }
      };
      auto feed1 = std::bind(feed, std::placeholders::_1, std::ref(u_dofs), std::ref(u_dofs_g));
      auto feed2 = std::bind(feed, std::placeholders::_1, std::ref(u_ddofs), std::ref(u_ddofs_g));
      // collect childs' dof
      for (auto cell_id : cells_support)
      {
         // all dofs
         auto cells_child = sj.getChildren(cell_id);
         for (auto cc : cells_child)
         {
            // auto vf = dofmapf->cell_dofs(cc);
            auto v = dofmapf->cell_dofs(cc);

            // add to containers
            std::ranges::for_each(v, feed1 );

            // TS Dirichlet dofs : dof related to face surrounding patch
            auto faces = adjcff->links(cc);
            int k = 0;
            for (auto f : faces)
            {
               if (u_faces.find(f) != u_faces.end())
               {
                  for (auto loc_idx : closure[dimm1][k])
                  {

                     auto &vc=v[loc_idx];
                     feed2(vc);
                  }
               }
               ++k;
            }
         }
      }


      std::vector<PetscInt> dofs_;
      {
         std::vector<std::int64_t> dofsg;
         {
            // change into a vector
            std::vector<std::int32_t> dofs(u_dofs.begin(), u_dofs.end());
            u_dofs.clear();
            // pass in global numbering
            dofsg.resize(dofs.size());
            idxmapf->local_to_global(std::span<const std::int32_t>(dofs), std::span<std::int64_t>(dofsg));
         }
         // pass in petsc int
         dofs_.resize(dofsg.size());
         std::ranges::transform(dofsg, dofs_.begin(),
                                [](const std::int64_t &x) -> PetscInt { return static_cast<const PetscInt &>(x); });
         // sort
         std::ranges::sort(dofs_);
      }
      std::vector<PetscInt> dofs_to_exchange_;
      {
         dofs_to_exchange_.reserve(u_dofs_g.size()*2);
         
         std::int64_t dofg;
         for (auto & p : u_dofs_g)
         {
            idxmapf->local_to_global(std::span<const std::int32_t>(&p.first,1), std::span<std::int64_t>(&dofg,1));
            dofs_to_exchange_.push_back(static_cast<PetscInt >(dofg));
            dofs_to_exchange_.push_back(static_cast<PetscInt >(p.second));
         }
         u_dofs_g.clear();
      }
      std::vector<PetscInt> ddofs_to_exchange_;
      {
         ddofs_to_exchange_.reserve(u_ddofs_g.size()*2);
         
         std::int64_t dofg;
         for (auto & p : u_ddofs_g)
         {
            idxmapf->local_to_global(std::span<const std::int32_t>(&p.first,1), std::span<std::int64_t>(&dofg,1));
            ddofs_to_exchange_.push_back(static_cast<PetscInt >(dofg));
            ddofs_to_exchange_.push_back(static_cast<PetscInt >(p.second));
         }
         u_ddofs_g.clear();

      }
      std::vector<PetscInt> ddofs_;
      {
         std::vector<std::int64_t> ddofsg;
         {
            // change into a vector
            std::vector<std::int32_t> ddofs(u_ddofs.begin(), u_ddofs.end());
            u_ddofs.clear();
            // pass in global numbering
            ddofsg.resize(ddofs.size());
            idxmapf->local_to_global(std::span<const std::int32_t>(ddofs), std::span<std::int64_t>(ddofsg));
         }
         // pass in petsc int
         ddofs_.resize(ddofsg.size());
         std::ranges::transform(ddofsg, ddofs_.begin(),
                                [](const std::int64_t &x) -> PetscInt { return static_cast<const PetscInt &>(x); });
         // sort
         std::ranges::sort(ddofs_);
      }

      //std::println("dofs patch        {}: {}\ndofs derichlet    {}: {}\ndofs_to_exchange  {}: {}\nddofs_to_exchange {}: {}", dofs_.size(), dofs_, ddofs_.size(), ddofs_, dofs_to_exchange_.size(), dofs_to_exchange_, ddofs_to_exchange_.size(), ddofs_to_exchange_);

      // TODO : dof list of ts Dirichlet bc:
      //             * dofs connected to an element outside the patch (may be remote)   OK done
      //             * free face of the part should be exclude   OK done
      //             * global Dirichlet bc should be handeled correctly TODO
      // TODO : note clear if MPC should'nt be treated some how. For now zero eliminated slaves row and column are retrieved
      // like any other in sub matrix creation process. The question is how the right hand side should be treated ?
      //
      // mpc and ts Dirichlet ? Hum normaly by construction slaves can't be ts Dirichlet but master can. Heu not sure about this
      // last one !!!! If true a priori not an issue but ?? rhs master already get eliminated slave contribution so eliminating them
      // should be ok ?
      patches.push_back(new P(node, nodeg, state, std::move(dofs_), std::move(dofs_to_exchange_), std::move(ddofs_),
                              std::move(ddofs_to_exchange_), owner, std::move(remotes)));
   };
   for (auto node : enriched) gen(node);
   for (auto node : extra_enriched) gen(node);

   return twoscale_dolfinx::patchManager<P>(std::move(patches), nbdfl, bs_f, comm);
}
}  // namespace twoscale_dolfinx
#endif
