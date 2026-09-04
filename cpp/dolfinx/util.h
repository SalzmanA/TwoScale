/*
 * Copyright (C) 2026 - Alexis Salzman, Ecole Centrale de Nantes
 *
 * SPDX-License-Identifier:    LGPL-3.0-or-later
 */ 

#ifndef TS_UTILS
#define TS_UTILS

#include <unordered_map>
#include <mpi.h>
#include <basix/mdspan.hpp>
#ifdef HAS_PETSC
#include <petscsystypes.h>
#include <petscversion.h>
#endif

#ifdef __GNUC__
#define TS_MACRO_WARNUNUSEDTYPE  __attribute__((unused))
#else
#define TS_MACRO_WARNUNUSEDTYPE
#endif


#define CHECKTS(ierr, msg, comm)                     \
   if (ierr)                                         \
   {                                                 \
      std::cout << msg << ": " << ierr << std::endl; \
      MPI_Abort(comm, -1);                           \
   }
#ifdef HAS_PETSC
#if PETSC_VERSION_GE(3, 25, 0)
#define CHECK(ierr, msg, comm)                                                         \
   if (ierr)                                                                           \
   {                                                                                   \
      const char *desc;                                                                \
      const char *extr;                                                                \
      PetscErrorMessage(ierr, &desc, &extr);                                           \
      std::cout << msg << ": " << desc << " (" << ierr << ") / " << extr << std::endl; \
      MPI_Abort(comm, -ierr);                                                          \
   }
#else
#define CHECK(ierr, msg, comm)                                                         \
   if (ierr)                                                                           \
   {                                                                                   \
      const char *desc;                                                                \
      char *extr;                                                                      \
      PetscErrorMessage(ierr, &desc, &extr);                                           \
      std::cout << msg << ": " << desc << " (" << ierr << ") / " << extr << std::endl; \
      MPI_Abort(comm, -ierr);                                                          \
   }
#endif
#else
#define CHECK(ierr, msg, comm)                                                         \
   if (ierr)                                                                           \
   {                                                                                   \
      std::cout << msg << ": " <<  ierr << std::endl;                                  \
      MPI_Abort(comm, -ierr);                                                          \
   }
#endif
#define DO(cmd, msg, comm)       \
   {                             \
      PetscErrorCode ierr = cmd; \
      CHECK(ierr, msg, comm)     \
   }

namespace twoscale
{
template <typename T, typename F, typename W>
void sendVectMPI3(
    std::unordered_map<int, std::vector<T>> &send_buff, MPI_Datatype TMPI, int tag, MPI_Comm univ, F func, W work = []() {})
{
   // send info
   int nb_send_message = send_buff.size(), nb_send = 0;
   std::vector<MPI_Request> request_to(nb_send_message, MPI_REQUEST_NULL);
   for (auto &send : send_buff)
      MPI_Issend((void *)send.second.data(), send.second.size(), TMPI, send.first, tag, univ, &request_to[nb_send++]);
   assert(nb_send == nb_send_message);

   // somme work during comunication: 
   work();

   // loop to receive info and wait for all data been sent
   bool do_test = false;
   int flag, from;
   MPI_Status status;
   MPI_Request barrier_request(MPI_REQUEST_NULL);
   do
   {
      // probing pending message without blocking to be able to check barrier achievement (end of loop)
      MPI_Iprobe(MPI_ANY_SOURCE, tag, univ, &flag, &status);

      // there is a pending message : get it and unpack
      if (flag)
      {
         // get size and then message
         assert(tag == status.MPI_TAG);
         from = status.MPI_SOURCE;
         int count = 0;
         MPI_Get_count(&status, TMPI, &count);
         std::vector<T>recv_buff(count);
         MPI_Recv(recv_buff.data(), count, TMPI, from, tag, univ, &status);

         // apply functor
         func(recv_buff, from);
      }

      // Does all sended mesages are arrived ? we test ... we don't want to wait for that as we want to loop
      // so it is a test i.e. non blocking
      // We do this test only if we haven't already entered barrier
      if (!do_test)
      {
         MPI_Testany(nb_send_message, &request_to[0], &from, &flag, MPI_STATUS_IGNORE);
         if (flag && from == MPI_UNDEFINED)
         {
            // This process has sent all its messages and all receiving process have treated them
            // so it enters the barrier. We know that it is finished in receiving process
            // because we use a synchronous send. When all process of the communicator will
            // have enter the barrier we will know that all message sent have been received
            // by all target process and we will not need to loop anymore to wait for nothing.
            MPI_Ibarrier(univ, &barrier_request);

            // we can now test if barrier request is achieved and this avoid also to reenter barrier many time
            do_test = true;
         }
      }

      // reset flag if no test is done to loop again
      flag = 0;

      // This test is done only if at least current proc have entered the barrier. Otherwise it's useless 
      // If all proc enter the barrier flag will be set to a non zero which will finish the loop
      if (do_test) MPI_Test(&barrier_request, &flag, MPI_STATUS_IGNORE);

   } while (!flag);
}
// hash function for std::pair<T1,T2>
template <typename T1, typename T2>
class hashPair
{
  public:
   std::size_t operator()(const std::pair<T1, T2> &key) const
   {
      return std::hash<std::string>{}(std::to_string(key.first) + " " + std::to_string(key.second));
   }
};
// equal function for std::pair<T1,T2>
template <typename T1, typename T2>
class equalPair
{
  public:
   bool operator()(const std::pair<T1, T2> &key1, const std::pair<T1, T2> &key2) const
   {
      if (key1.first != key2.first) return false;
      if (key1.second != key2.second) return false;
      return true;
   }
};
}

namespace twoscale_dolfinx
{
/// multi dimensional span type shortcut to basix mdspan implementation declaration
/// @tparam T Type of the data stored in the array warped by mdspan
/// @tparam D integer describing number of dimension of the mdspan
template <typename T, std::size_t D>
using mdspan_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, D>>;
/// Utility class to store enriched dofs indexes and their associated fine scale dofs indexes, both related to matrix \f$PE^t\f$
/// Only the block index is stored: for coarse it is an implicit block as mixed space have bs=1. For fine it is an explicit block
/// Eliminates row by Dirichlet boundary are encoded in eliminated_coarse_encoding
class enrichedDofIDs
{
  public:
#ifdef HAS_PETSC
   /// enriched dof block index at coarse scale in global numbering (PETSc \f$PE^t\f$ row numbering)
   PetscInt global_coarse_block_dof_idx;
   /// dof block index at fine scale corresponding to global_coarse_block_dof_idx in global numbering (PETSc \f$PE^t\f$ column
   /// numbering) This give in a patch a way to identify dofs that can be used for shift enrichment function for example
   PetscInt global_fine_block_dof_idx;
#else
#error "to be implemented: class enrichedDofIDs"
#endif
   /// encoding using basic binary encoding:
   ///
   /// * each component of a block are encode on one different bit in increasing power of 2 (component 0 -> \f$2^0\f$, component 1
   /// -> \f$2^1\f$, ....)
   /// * a bit is set to 1 if eliminated 0 otherwise
   ///
   /// Thus eliminated_coarse_encoding is null if the full block is retained. And it is non null if at leas one component is
   /// eliminated
   std::int64_t eliminated_coarse_encoding;
};

/// Function that tells if library is using or no the nested matrix strategy
bool useNest();

/// Function encapsulating a simple scatter fwd operation
/// It use *MPI neighbourhood collective communication*
/// @param[in] l Buffer containing local data to send
/// @param[out] g Buffer containing local ghost data to be updated
/// @tparam S Type of the scatterer object
/// @tparam T Type of the data to be exchanged
template <typename S, typename T>
void scatter_fwd(S &scatterer, std::span<const T> l, std::span<T> g)
{
   MPI_Request request = MPI_REQUEST_NULL;
   auto &idxl = scatterer.local_indices();
   auto nb_idxl = idxl.size();
   auto &idxr = scatterer.remote_indices();
   auto nb_idxr = idxr.size();
   std::vector<T> bufferA(nb_idxl, 0);
   std::vector<T> bufferB(nb_idxr, 0);
   // pack
   for (std::size_t i = 0; i < nb_idxl; ++i) bufferA[i] = l[idxl[i]];

   // exchange
   scatterer.scatter_fwd_begin(bufferA.data(), bufferB.data(), request);
   scatterer.scatter_end(request);

   // unpack
   for (std::size_t i = 0; i < nb_idxr; ++i) g[idxr[i]] = bufferB[i];
}
/// Function encapsulating a simple scatter rev operation
/// It use *MPI neighbourhood collective communication*
/// @param[out] l Buffer containing local data to be updated
/// @param[in] g Buffer containing local ghost data to send
/// @param[in] op binary operator that takes current local value and received corresponding value as argument. It returns local
/// value to set.
/// @tparam S Type of the scatterer object
/// @tparam T Type of the data to be exchanged
/// @tparam S Type of the binary operator argument
template <typename S, typename T, typename BinaryOp>
void scatter_rev(S &scatterer, std::span<T> l, std::span<const T> g, BinaryOp op)
{
   MPI_Request request = MPI_REQUEST_NULL;
   auto &idxl = scatterer.local_indices();
   auto nb_idxl = idxl.size();
   auto &idxr = scatterer.remote_indices();
   auto nb_idxr = idxr.size();
   std::vector<T> bufferA(nb_idxr, 0);
   std::vector<T> bufferB(nb_idxl, 0);
   // pack
   for (std::size_t i = 0; i < nb_idxr; ++i) bufferA[i] = g[idxr[i]];

   // exchange
   scatterer.scatter_rev_begin(bufferA.data(), bufferB.data(), request);
   scatterer.scatter_end(request);

   // unpack
   for (std::size_t i = 0; i < nb_idxl; ++i)
   {
      auto &li = l[idxl[i]];
      // apply operator
      li = op(li, bufferB[i]);
   }
}
/// Function encapsulating a scatter rev followed by fwd operation
/// It use *MPI neighbourhood collective communication*
/// @param[in,out] l Buffer containing local data to update and send
/// @param[in,out] g Buffer containing local ghost data to send and update
/// @param[in] op binary operator that takes current local value and received corresponding value as argument. It returns local
/// value to set.
/// @tparam S Type of the scatterer object
/// @tparam T Type of the data to be excanged
/// @tparam S Type of the binary operator argument
template <typename S, typename T, typename BinaryOp>
void scatter_rev_fwd(S &scatterer, std::span<T> l, std::span<T> g, BinaryOp op)
{
   MPI_Request request = MPI_REQUEST_NULL;
   auto &idxl = scatterer.local_indices();
   auto nb_idxl = idxl.size();
   auto &idxr = scatterer.remote_indices();
   auto nb_idxr = idxr.size();
   std::vector<T> bufferA(nb_idxr, 0);
   std::vector<T> bufferB(nb_idxl, 0);
   // pack
   for (std::size_t i = 0; i < nb_idxr; ++i) bufferA[i] = g[idxr[i]];

   // exchange
   scatterer.scatter_rev_begin(bufferA.data(), bufferB.data(), request);
   scatterer.scatter_end(request);

   // unpack/pack
   for (std::size_t i = 0; i < nb_idxl; ++i)
   {
      auto &ri = bufferB[i];
      auto &li = l[idxl[i]];
      // apply operator and unpack
      li = op(li, ri);
      // pack
      ri = li;
   }

   // exchange
   scatterer.scatter_fwd_begin(bufferB.data(), bufferA.data(), request);
   scatterer.scatter_end(request);

   // unpack
   for (std::size_t i = 0; i < nb_idxr; ++i) g[idxr[i]] = bufferA[i];

}
}  // namespace twoscale_dolfinx
#endif
