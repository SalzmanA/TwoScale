/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_DOLFINX_PATCHMANAGER
#define TS_DOLFINX_PATCHMANAGER
#ifdef TWOSCALE_DOLFINX
#include "dolfinx/mesh/Mesh.h"
#include <dolfinx/fem/Function.h>
#else
#error "need dolfinx implementation as no other is available"
#endif

#include "mpi.h"
#include "util.h"

#ifdef HAS_PETSC
#include <petscmat.h>
#include <petscksp.h>
#else
#error "PETSc is required. No alternative for now"
#endif
namespace twoscale
{
template <typename M>
class scaleJump;

class patch
{
  public:
   enum patchState
   {
      LOCAL = 1u,
      DIST = 2u,
      DUMMY = 4u,
      INIT = 8u,
      PRBSET = 16u,
      PRBSOLVED = 32u,
      PETSET = 64u,
      GLOBAL_WEIGHT = 32768u,
      UNKNOWN
   };
   patch(std::int32_t enriched_id_, std::int64_t id_, std::uint16_t state_, std::vector<PetscInt> &&dofs_,
         std::vector<PetscInt> &&dofs_to_exchange_, std::vector<PetscInt> &&Diri_dofs_,
         std::vector<PetscInt> &&Diri_dofs_to_exchange_, int owner_, std::vector<int> &&remotes_);
   void print() const;
   void printID() const;
   /// Function that gives patch id 
   std::int64_t getID() const;
   /// Function that gives secondary patch id (an id related to patch id some how but not necessarily equals to it) 
   std::int32_t getSID() const;
   std::span<const int> getRemote() const;
   void initPatchForComm(MPI_Comm comm, MPI_Comm univ);

   bool isDist() const;
   float getWeight() const;
   int i_seq;

  protected:
   std::vector<PetscInt> dofs;
   std::vector<PetscInt> dofs_to_exchange;
   std::vector<PetscInt> Diri_dofs;
   std::vector<PetscInt> Diri_dofs_to_exchange;
   std::vector<PetscInt> free_dofs;
   std::vector<int> remotes;
   std::int64_t id;
   float weight;
   std::int32_t enriched_id;
   int owner;
   std::uint16_t state;
   int pid, nbproc;
   int pidp, nbprocp;
};
/// No openMP only MPI (i.e. each patchs can't be treated in parrallel by multiple threads).
/// Linear system resolution use parallel direct solver
/// Local patch use sequential resolution
/// Distributed patch use parrallel resolution
class patchMPIDirect : public patch
{
  public:
   patchMPIDirect(std::int32_t enriched_id_, std::int64_t id_, std::uint16_t state_, std::vector<PetscInt> &&dofs_,
                  std::vector<PetscInt> &&dofs_to_exchange_, std::vector<PetscInt> &&Diri_dofs_,
                  std::vector<PetscInt> &&Diri_dofs_to_exchange_, int owner_, std::vector<int> &&remotes_);
   ~patchMPIDirect();
   /// Function that generate patch system of equation using given fine scale system
   void generateProblem(Mat Aff, Vec bf, Vec computf_, MPI_Comm comm, MPI_Comm univ, std::int32_t nbdl, std::int32_t bs);
   /// Function that update \f$PE^t\f$ terms relate to this patch (object) based on current solution of this patch system
   /// The first time this method is called the \f$PE^t\f$ rows associated to this patch are copied in internal storage so that
   /// during next calls updating can be done using this storage as \f$PE^t\f$ no longer contains interpolation coefficients but
   /// resulting terms from last call.
   /// @tparam F Type representing functor that transform patch solution into enrichment function
   /// @param[in,out] PEt Operator to update.
   /// @param[in] enriched_dof_id Used only in the first call, this container gives way to identify row(s) of \f$PE^t\f$ related to
   /// this patch and the fine dofs related to enriched node
   /// @param[in] func Functor to transform field solution of this patch problem into enrichment function
   /// @param[in] comm General communicator on which coarse/fine meshes are distributed
   /// @param[in] univ Communicator on which this patch is distributed
   /// @param[in] bs Block size supposed to be the same for coarse and fine field 
   /// @param[in] rstart First row of \f$PE^t\f$ stored in this proc
   /// @param[in] rend Past the last row of \f$PE^t\f$ stored in this proc
   template <typename F>
   void PEtUpdate(Mat PEt, std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &enriched_dof_id,
                  std::shared_ptr<F> func, MPI_Comm comm, MPI_Comm univ, std::int32_t bs, PetscInt rstart, PetscInt rend);
   /// Solve problem for this patch
   /// @param[in] xf Petsc Vector storing fine field solution approximation (from last TwoScale Loop for example)
   /// @param[in] comm General communicator on which coarse/fine meshes are distributed
   /// @param[in] univ Communicator on which this patch is distributed
   /// @param[in] bs Block size supposed to be the same for coarse and fine field 
   void solveProblem(Vec xf, MPI_Comm comm, MPI_Comm univ, std::int32_t bs);
   void setPETSETState();
   bool getPETSETState();
   void grabPatchSolution(Vec xf, MPI_Comm comm, MPI_Comm univ);
   void clear(MPI_Comm comm, MPI_Comm univ);

  private:
   IS idxqf,idxdf;
   Mat Aqq, Aqd;
   Mat *submat[2];
   Vec bq0,bq,xd,xdloc,xq,computef,enrich;
   KSP solver;
   PetscInt idr;
   std::vector<PetscInt> PEtJCols;
   std::vector<PetscScalar> PEtValCols;
   std::int32_t idx_dof_f_enriched;
   std::int64_t eliminated_coarse_encoding;
   VecScatter scatter2e;
};
/// variant to implement
/// patchMPIMagma use Magma or any GPU solver for patch resolution. All patch must use GPU
/// Distributed patch needs to gather the Matrix on 1 proc/GPU to launch computation on one device.
/// From experience in eXlibris using parallel direct solver on distributed patch and Magma resolution on local patch alleviate
/// the possibility of increasing scale jump and get gains.
class patchMPIMagma : public patch
{
};
/// variant to implement
/// patchOMPDirect use OpenMP to compute local patch in multi-threaded mode using a thread safe direct solver like Pardiso and
/// usual parallel direct solver for distributed patch.
/// From experience in eXlibris this hybrid computation did not show high gain but implementation might has been poorly done.
class patchOMPDirect : public patch
{
};
}  // namespace twoscale

namespace twoscale_dolfinx
{
/// @tparam P The type of patch managed (e.g. patchMPIDirect, ...)
/// @note P could be suppressed as pointer to base do the job but it may help
/// to do some specialization for some patch type in some methods  
template <typename P>
class patchManager
{
  public:
   /// Constructor bassed on patches provided as argument
   /// @param[in] patches_ vector of pointers to patches of type P. The new instance acquire the control on them.
   /// @param[in] nbdl Number of block dofs local to this process
   /// @param[in] bs_d Block size associated to dofs
   /// @param[in] comm_ Communicator associated to fine/coarse mesh distribution
   /// @warning This constructor is not intended to be called by the user who must use twoscale_dolfinx::generatePatchManager function instead.
   patchManager(std::vector<P *> &&patches_, std::int32_t nbdl, std::int32_t bs_d, MPI_Comm comm_);

   /// Destructor required to avoid internal dummy_patch pointer to be unfreed
   ~patchManager();

   /// copy constructor required to avoid internal dummy_patch pointer to be linked in between instances
   /// @param[in]  other Instance copied to created one new instance
   patchManager(const patchManager<P> &other);

   /// move constructor required to capture properly internal dummy_patch pointer. On output the instance provided as argument
   /// must not be used anymore
   /// @param[in,out]  other Instance moved to created one new instance
   patchManager(patchManager<P> &&other);

   /// Generates all patches system from fine scale system
   /// @param[in] Aff Fine scale matrix
   /// @param[in] bf Fine scale rhs
   void generateProblems(Mat Aff, Vec bf);

   /// Solve all patches system using a fine scale field approximation
   /// @param xf[in] Vector providing field values to be imposed as Dirichlet boundary condition for all patches during their
   /// resolution
   void solveProblems(Vec xf);

   /// Update PEt operator with current patches solution.The PEt matrix is the operator to transfers fine scale field into coarse
   /// enriched field.
   ///
   /// During its first call the rows (corresponding to a coarse node enriched) of PEt  are transferred to their associated patch
   /// instance.  The process owner of the row (PETSc sens) hold in the associated patch the initial values of PEt witch are
   /// simple interpolation coefficients. These coefficients are multiplied by the associated enriched function values to generate
   /// the new PEt terms. After the first call PEt do not hold anymore the interpolation coefficients but their product with the
   /// last enrichment function computed so far. In next call interpolation coefficients comes from  patches instances themselves.
   ///
   /// The enriched function in each patch is obtained by applying the func functor to the current patch solutions stored in the
   /// patch instance.
   ///
   /// @warning Has the patches solution are used in this function it implies that
   /// function patchManager::solveProblems had to be called before this function to have a correct enriched function.
   ///
   /// @tparam F Functor type
   /// @todo F functor should be defined in a concept
   /// @note This function could be split in 2 steps/function. More clear in term of API/user point of view. This may be
   /// mandatory in the future if scaleJumps is updated and PEt is modified on the fly.
   ///
   /// @param[in,out] PEt  operator to transfers fine scale field into coarse enriched field.
   /// @param[in] enriched_dof_id A map that connect indexes. The key of the map is the FEniCSx global topological index of the
   /// coarse node enriched (it correspond to one of the ids of a patch). The data associated to the key is a twoscale_dolfinx::enrichedDofIDs
   /// corresponding to block row and column of PEt. The row is related to enriched coarse dof and the column to the dof at fine
   /// scale equivalent to enriched coarse dof. This map is only used during the first call.
   /// @param[in] func functor applied to patch solution to obtain enriched function from it
   template <typename F>
   void PEtUpdate(Mat PEt, std::unordered_map<std::int32_t, twoscale_dolfinx::enrichedDofIDs> &enriched_dof_id, std::shared_ptr<F> func);

   /// Function that gives the number of sequence needed for patches treatement
   /// @return number of sequence used to compute all distributed patches
   /// @warning Should not be called before at least one call to generateProblem to provide correct information
   int numberOfSequence();


   /// Function to copy in xf all patches solutions related to sequence seq
   /// @param [in] seq Sequence id to collect. It correspond to a set of independant patche computed at the same time.
   /// @param [in,out] xf A PETSc vector provided by user and filled with solutions of patches computed by sequence seq.
   /// @warning Only for debugging.
   std::int32_t grabPatchSolution(std::int32_t seq, Vec xf);

  private:
   std::vector<P *> patches;
   MPI_Comm comm;
   std::int32_t nbdl, bs_d;

   // specific to distributed patch
   int nb_dist, nb_totg;
   std::vector<MPI_Comm> dist_com;

   // dummy patch
   void *dummy_patch;

   // vector used has a buffer to collect solutions from the patches
   // of one sequence, all at once. This buffer can then be used to 
   // retrieves these solution for PEt rows update
   // This buffer can be used for all sequences and thus is placed in
   // this class and provided to patches.
   // Thanks to sequence construction, patch solution from all patches 
   // in a sequence cannot overlap (i.e. being on a common dof in computef)
   // because only internal dofs are collected (has PEt column are null for
   // surrounding dofs). In a sequence, patches are guaranteed to be part of 
   // different process, thus this eliminate the case of overlapping patches.
   Vec computef;

   void sortPatches();
   void show() const;
};


/// Function to generate a patchManager instance from scaleJump instance and dolfinx::fem::Function
/// @tparam P The type of patch managed (e.g. patchMPIDirect, ...)
/// @tparam U The mesh geometry/space scalar type (float or double).
///
/// @param[in] sj The scaleJump object describing the fine and coarse scale and there relationship.
/// @param[in] fine_space The discretization space at fine scale level
///
//template <typename P, dolfinx::scalar T, std::floating_point U = dolfinx::scalar_value_type_t<T>>
template <typename P,  std::floating_point U >
twoscale_dolfinx::patchManager<P> generatePatchManager(const twoscale::scaleJump<dolfinx::mesh::Mesh<U>> &sj,
                                                       std::shared_ptr<const dolfinx::fem::FunctionSpace<U>> fine_space);

}  // namespace twoscale_dolfinx

#include "patchManager_imp.h"
#endif
