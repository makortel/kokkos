/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_TBB_HPP
#define KOKKOS_TBB_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_TBB)

#include <Kokkos_Core_fwd.hpp>

#include <Kokkos_HostSpace.hpp>
#include <cstddef>
#include <iosfwd>
#include <cassert>

#ifdef KOKKOS_ENABLE_HBWSPACE
#include <Kokkos_HBWSpace.hpp>
#endif

#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_ScratchSpace.hpp>
#include <Kokkos_TaskScheduler.hpp>
#include <impl/Kokkos_FunctorAdapter.hpp>
#include <impl/Kokkos_FunctorAnalysis.hpp>
#include <impl/Kokkos_Profiling_Interface.hpp>
#include <impl/Kokkos_Tags.hpp>
#include <impl/Kokkos_TaskQueue.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

#include <tbb/task_scheduler_init.h>
#include <tbb/task.h>
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_scan.h>
#include <tbb/blocked_range.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>


namespace Kokkos {
namespace Impl {
class thread_buffer {
  static constexpr std::size_t m_cache_line_size = 64;

  std::size_t m_num_threads;
  std::size_t m_size_per_thread;
  std::size_t m_size_total;
  char *m_data;

  void pad_to_cache_line(std::size_t &size) {
    size = ((size + m_cache_line_size - 1) / m_cache_line_size) *
           m_cache_line_size;
  }

public:
  thread_buffer()
      : m_num_threads(0), m_size_per_thread(0), m_size_total(0),
        m_data(nullptr) {}
  thread_buffer(const std::size_t num_threads,
                const std::size_t size_per_thread) {
    resize(num_threads, size_per_thread);
  }
  ~thread_buffer() { delete[] m_data; }

  thread_buffer(const thread_buffer &) = delete;
  thread_buffer(thread_buffer &&) = delete;
  thread_buffer &operator=(const thread_buffer &) = delete;
  thread_buffer &operator=(thread_buffer) = delete;

  void resize(const std::size_t num_threads,
              const std::size_t size_per_thread) {
    m_num_threads = num_threads;
    m_size_per_thread = size_per_thread;

    pad_to_cache_line(m_size_per_thread);

    std::size_t size_total_new = m_num_threads * m_size_per_thread;

    if (m_size_total < size_total_new) {
      delete[] m_data;
      m_data = new char[size_total_new];
      m_size_total = size_total_new;
    }
  }

  char *get(std::size_t thread_num) {
    assert(thread_num < m_num_threads);
    if (m_data == nullptr) {
      return nullptr;
    }
    return &m_data[thread_num * m_size_per_thread];
  }

  std::size_t size_per_thread() const noexcept { return m_size_per_thread; }
  std::size_t size_total() const noexcept { return m_size_total; }
};
} // namespace Impl

namespace Experimental {
class TBB {
private:
  static bool m_tbb_initialized;
  static std::unique_ptr<tbb::task_scheduler_init> m_scheduler;

public:
  using execution_space = TBB;
  using memory_space = HostSpace;
  using device_type = Kokkos::Device<execution_space, memory_space>;
  using array_layout = LayoutRight;
  using size_type = memory_space::size_type;
  using scratch_memory_space = ScratchMemorySpace<TBB>;

  TBB() noexcept {}
  static void print_configuration(std::ostream &,
                                  const bool /* verbose */ = false) {
    std::cout << "TBB backend" << std::endl;
  }

  static bool in_parallel(TBB const & = TBB()) noexcept { return false; }
  static void impl_static_fence(TBB const & = TBB())
        noexcept {
    }

  #ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  static void fence(TBB const & = TBB()) {
  #else
  void fence() const {
  #endif
    impl_static_fence();
  }

  static bool is_asynchronous(TBB const & = TBB()) noexcept {
    return false;
  }

  static std::vector<TBB> partition(...) {
    Kokkos::abort("Kokkos::Experimental::TBB::partition_master: can't partition an TBB "
                  "instance\n");
    return std::vector<TBB>();
  }

  template <typename F>
  static void partition_master(F const &f, int requested_num_partitions = 0,
                               int requested_partition_size = 0) {
    if (requested_num_partitions > 1) {
      Kokkos::abort("Kokkos::Experimental::TBB::partition_master: can't partition an "
                    "TBB instance\n");
    }
  }

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  static bool is_initialized() noexcept {
    return impl_is_initialized();
  }

  inline
    static int thread_pool_size() noexcept {
    return impl_thread_pool_size();
  }

  static int hardware_thread_id() noexcept {
    return impl_hardware_thread_id();
  }

#endif

  static int concurrency();
  static void impl_initialize(int thread_count);
  static void impl_initialize();
  static bool impl_is_initialized() noexcept;
  static void impl_finalize();

  static int impl_thread_pool_size() noexcept {
    if(tbb::this_task_arena::current_thread_index() == tbb::task_arena::not_initialized) {
      return 0;
    }
    return tbb::this_task_arena::max_concurrency();
  }

  static int impl_thread_pool_rank() noexcept {
    auto id = tbb::this_task_arena::current_thread_index();
    if(id == tbb::task_arena::not_initialized) {
      return 0;
    }
    //TBB index starts at 0
    return id+1;
  }

  static int impl_thread_pool_size(int depth) {
    if (depth == 0) {
      return impl_thread_pool_size();
    } else {
      return 1;
    }
  }

  static int impl_max_hardware_threads() noexcept {
    return tbb::task_scheduler_init::default_num_threads();
  }

  static int impl_hardware_thread_id() noexcept {
    //This is probably the wrong thing to use as a thread_index is
    // not stable
    return tbb::this_task_arena::current_thread_index();
  }

  static constexpr const char *name() noexcept { return "TBB"; }
};
} // namespace Experimental

} // namespace Kokkos

namespace Kokkos {
namespace Impl {
template <>
struct MemorySpaceAccess<Kokkos::Experimental::TBB::memory_space,
                         Kokkos::Experimental::TBB::scratch_memory_space> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = false };
};

template <>
struct VerifyExecutionCanAccessMemorySpace<
    Kokkos::Experimental::TBB::memory_space,
    Kokkos::Experimental::TBB::scratch_memory_space> {
  enum { value = true };
  inline static void verify(void) {}
  inline static void verify(const void *) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Experimental {
template <> class UniqueToken<TBB, UniqueTokenScope::Instance> {
public:
  using execution_space = TBB;
  using size_type = int;
  UniqueToken(execution_space const & = execution_space()) noexcept {}

  int size() const noexcept { return TBB::impl_max_hardware_threads(); }
  int acquire() const noexcept { return TBB::impl_hardware_thread_id(); }
  void release(int) const noexcept {}
};

template <> class UniqueToken<TBB, UniqueTokenScope::Global> {
public:
  using execution_space = TBB;
  using size_type = int;
  UniqueToken(execution_space const & = execution_space()) noexcept {}

  int size() const noexcept { return TBB::impl_max_hardware_threads(); }
  int acquire() const noexcept { return TBB::impl_hardware_thread_id(); }
  void release(int) const noexcept {}
};
} // namespace Experimental
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

struct TBBTeamMember {
public:
  using execution_space = Kokkos::Experimental::TBB;
  using scratch_memory_space =
      Kokkos::ScratchMemorySpace<Kokkos::Experimental::TBB>;

private:
  scratch_memory_space m_team_shared;
  std::size_t m_team_shared_size;

  int m_league_size;
  int m_league_rank;
  int m_team_size;
  int m_team_rank;

public:
  KOKKOS_INLINE_FUNCTION
  const scratch_memory_space &team_shmem() const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space &team_scratch(const int) const {
    return m_team_shared.set_team_thread_mode(0, 1, 0);
  }

  KOKKOS_INLINE_FUNCTION
  const execution_space::scratch_memory_space &thread_scratch(const int) const {
    return m_team_shared.set_team_thread_mode(0, team_size(), team_rank());
  }

  KOKKOS_INLINE_FUNCTION int league_rank() const noexcept {
    return m_league_rank;
  }

  KOKKOS_INLINE_FUNCTION int league_size() const noexcept {
    return m_league_size;
  }

  KOKKOS_INLINE_FUNCTION int team_rank() const noexcept { return m_team_rank; }
  KOKKOS_INLINE_FUNCTION int team_size() const noexcept { return m_team_size; }

  template <class... Properties>
  constexpr KOKKOS_INLINE_FUNCTION
  TBBTeamMember(const TeamPolicyInternal<Kokkos::Experimental::TBB,
                                         Properties...> &policy,
                const int team_rank, const int league_rank, void *scratch,
                int scratch_size) noexcept
      : m_team_shared(scratch, scratch_size, scratch, scratch_size),
        m_team_shared_size(scratch_size), m_league_size(policy.league_size()),
        m_league_rank(league_rank), m_team_size(policy.team_size()),
        m_team_rank(team_rank) {}

  KOKKOS_INLINE_FUNCTION
  void team_barrier() const {}

  template <class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(ValueType &, const int &) const {
    static_assert(std::is_trivially_default_constructible<ValueType>(),
                  "Only trivial constructible types can be broadcasted");
  }

  template <class Closure, class ValueType>
  KOKKOS_INLINE_FUNCTION void team_broadcast(const Closure &, ValueType &,
                                             const int &) const {
    static_assert(std::is_trivially_default_constructible<ValueType>(),
                  "Only trivial constructible types can be broadcasted");
  }

  template <class ValueType, class JoinOp>
  KOKKOS_INLINE_FUNCTION ValueType team_reduce(const ValueType &value,
                                               const JoinOp &) const {
    return value;
  }

  template <class ReducerType>
  KOKKOS_INLINE_FUNCTION
      typename std::enable_if<is_reducer<ReducerType>::value>::type
      team_reduce(const ReducerType &reducer) const {}

  template <typename Type>
  KOKKOS_INLINE_FUNCTION Type
  team_scan(const Type &value, Type *const global_accum = nullptr) const {
    if (global_accum) {
      Kokkos::atomic_fetch_add(global_accum, value);
    }

    return 0;
  }
};

template <class... Properties>
class TeamPolicyInternal<Kokkos::Experimental::TBB, Properties...>
    : public PolicyTraits<Properties...> {
  using traits = PolicyTraits<Properties...>;

  int m_league_size;
  int m_team_size;
  std::size_t m_team_scratch_size[2];
  std::size_t m_thread_scratch_size[2];
  int m_chunk_size;

public:
  using member_type = TBBTeamMember;

  // NOTE: Max size is 1 for simplicity. In most cases more than 1 is not
  // necessary on CPU. Implement later if there is a need.
  template <class FunctorType>
  inline static int team_size_max(const FunctorType &) {
    return 1;
  }

  template <class FunctorType>
  inline static int team_size_recommended(const FunctorType &) {
    return 1;
  }

  template <class FunctorType>
  inline static int team_size_recommended(const FunctorType &, const int &) {
    return 1;
  }

  template <class FunctorType>
  int team_size_max(const FunctorType &, const ParallelForTag &) const {
    return 1;
  }

  template <class FunctorType>
  int team_size_max(const FunctorType &, const ParallelReduceTag &) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_recommended(const FunctorType &, const ParallelForTag &) const {
    return 1;
  }
  template <class FunctorType>
  int team_size_recommended(const FunctorType &,
                            const ParallelReduceTag &) const {
    return 1;
  }

private:
  inline void init(const int league_size_request, const int team_size_request) {
    m_league_size = league_size_request;
    const int max_team_size = 1; // TODO: Can't use team_size_max(...) because
                                 // it requires a functor as argument.
    m_team_size =
        team_size_request > max_team_size ? max_team_size : team_size_request;

    if (m_chunk_size > 0) {
      if (!Impl::is_integral_power_of_two(m_chunk_size))
        Kokkos::abort("TeamPolicy blocking granularity must be power of two");
    } else {
      int new_chunk_size = 1;
      while (new_chunk_size * 4 * Kokkos::Experimental::TBB::concurrency() <
             m_league_size) {
        new_chunk_size *= 2;
      }

      if (new_chunk_size < 128) {
        new_chunk_size = 1;
        while ((new_chunk_size * Kokkos::Experimental::TBB::concurrency() <
                m_league_size) &&
               (new_chunk_size < 128))
          new_chunk_size *= 2;
      }

      m_chunk_size = new_chunk_size;
    }
  }

public:
  inline int team_size() const { return m_team_size; }
  inline int league_size() const { return m_league_size; }

  inline size_t scratch_size(const int &level, int team_size_ = -1) const {
    if (team_size_ < 0) {
      team_size_ = m_team_size;
    }
    return m_team_scratch_size[level] +
           team_size_ * m_thread_scratch_size[level];
  }

public:
  template <class ExecSpace, class... OtherProperties>
  friend class TeamPolicyInternal;

  template <class... OtherProperties>
  TeamPolicyInternal(
      const TeamPolicyInternal<Kokkos::Experimental::TBB, OtherProperties...> &p) {
    m_league_size = p.m_league_size;
    m_team_size = p.m_team_size;
    m_team_scratch_size[0] = p.m_team_scratch_size[0];
    m_thread_scratch_size[0] = p.m_thread_scratch_size[0];
    m_team_scratch_size[1] = p.m_team_scratch_size[1];
    m_thread_scratch_size[1] = p.m_thread_scratch_size[1];
    m_chunk_size = p.m_chunk_size;
  }

  TeamPolicyInternal(const typename traits::execution_space &,
                     int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(const typename traits::execution_space &,
                     int league_size_request,
                     const Kokkos::AUTO_t &team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, 1);
  }

  TeamPolicyInternal(int league_size_request, int team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, team_size_request);
  }

  TeamPolicyInternal(int league_size_request,
                     const Kokkos::AUTO_t &team_size_request,
                     int /* vector_length_request */ = 1)
      : m_team_scratch_size{0, 0}, m_thread_scratch_size{0, 0},
        m_chunk_size(0) {
    init(league_size_request, 1);
  }

  inline int chunk_size() const { return m_chunk_size; }
#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal set_chunk_size(typename traits::index_type chunk_size_) const {
    TeamPolicyInternal p = *this;
    p.m_chunk_size = chunk_size_;
    return p;
  }

  inline TeamPolicyInternal set_scratch_size(const int& level, const PerTeamValue& per_team) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    return p;
  }

  inline TeamPolicyInternal set_scratch_size(const int& level, const PerThreadValue& per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  }

  inline TeamPolicyInternal set_scratch_size(const int& level, const PerTeamValue& per_team, const PerThreadValue& per_thread) const {
    TeamPolicyInternal p = *this;
    p.m_team_scratch_size[level] = per_team.value;
    p.m_thread_scratch_size[level] = per_thread.value;
    return p;
  }
#else
  inline TeamPolicyInternal &
  set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  inline TeamPolicyInternal &set_scratch_size(const int &level,
                                              const PerTeamValue &per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  inline TeamPolicyInternal &
  set_scratch_size(const int &level, const PerThreadValue &per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  inline TeamPolicyInternal &
  set_scratch_size(const int &level, const PerTeamValue &per_team,
                   const PerThreadValue &per_thread) {
    m_team_scratch_size[level] = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }
#endif

#ifdef KOKKOS_ENABLE_DEPRECATED_CODE
protected:

  /** \brief set chunk_size to a discrete value*/
  inline TeamPolicyInternal internal_set_chunk_size(typename traits::index_type chunk_size_) {
    m_chunk_size = chunk_size_;
    return *this;
  }

  /** \brief set per team scratch size for a specific level of the scratch hierarchy */
  inline TeamPolicyInternal internal_set_scratch_size(const int& level, const PerTeamValue& per_team) {
    m_team_scratch_size[level] = per_team.value;
    return *this;
  }

  /** \brief set per thread scratch size for a specific level of the scratch hierarchy */
  inline TeamPolicyInternal internal_set_scratch_size(const int& level, const PerThreadValue& per_thread) {
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }

  /** \brief set per thread and per team scratch size for a specific level of the scratch hierarchy */
  inline TeamPolicyInternal internal_set_scratch_size(const int& level, const PerTeamValue& per_team, const PerThreadValue& per_thread) {
    m_team_scratch_size[level] = per_team.value;
    m_thread_scratch_size[level] = per_thread.value;
    return *this;
  }
#endif

};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::RangePolicy<Traits...>,
                  Kokkos::Experimental::TBB> {
private:
  using Policy = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member = typename Policy::member_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  static typename std::enable_if<std::is_same<TagType, void>::value>::type
  execute_functor(const FunctorType &functor, const Member i) {
    functor(i);
  }

  template <class TagType>
  static typename std::enable_if<!std::is_same<TagType, void>::value>::type
  execute_functor(const FunctorType &functor, const Member i) {
    const TagType t{};
    functor(t, i);
  }

  template <class TagType>
  static typename std::enable_if<std::is_same<TagType, void>::value>::type
  execute_functor_range(const FunctorType &functor, const Member i_begin,
                        const Member i_end) {
    for (Member i = i_begin; i < i_end; ++i) {
      functor(i);
    }
  }

  template <class TagType>
  static typename std::enable_if<!std::is_same<TagType, void>::value>::type
  execute_functor_range(const FunctorType &functor, const Member i_begin,
                        const Member i_end) {
    const TagType t{};
    for (Member i = i_begin; i < i_end; ++i) {
      functor(t, i);
    }
  }

public:
  void execute() const { 
    tbb::this_task_arena::isolate([this]{
        using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;
        tbb::parallel_for(RangeType(m_policy.begin(),
                                    m_policy.end(),
                                    m_policy.chunk_size()),
                          [this](const RangeType& r) {
                            execute_functor_range<WorkTag>(m_functor, r.begin(),r.end());
                          });
      });
  }


  inline ParallelFor(const FunctorType &arg_functor, Policy arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class... Traits>
class ParallelFor<FunctorType, Kokkos::MDRangePolicy<Traits...>,
                  Kokkos::Experimental::TBB> {
private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy = typename MDRangePolicy::impl_range_policy;
  using WorkTag = typename MDRangePolicy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member = typename Policy::member_type;
  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, void>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;

public:
  void execute() const { 
    tbb::this_task_arena::isolate([this]{
        using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;
        tbb::parallel_for(RangeType(m_policy.begin(),
                                    m_policy.end(),
                                    m_policy.chunk_size()),
                          [this](const RangeType& r) {
                            for(auto i = r.begin(); i != r.end(); ++i) {
                              iterate_type(m_mdr_policy, m_functor)(i);
                            }
                          });
      });
  }

  inline ParallelFor(const FunctorType &arg_functor, MDRangePolicy arg_policy)
      : m_functor(arg_functor), m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {
template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::RangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::TBB> {
private:
  using Policy = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member = typename Policy::member_type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;
  using ValueOps = Kokkos::Impl::FunctorValueOps<ReducerTypeFwd, WorkTagFwd>;
  using value_type = typename Analysis::value_type;
  using pointer_type = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;

  const FunctorType m_functor;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor(const FunctorType &functor, const Member i,
                      reference_type update) {
    functor(i, update);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor(const FunctorType &functor, const Member i,
                      reference_type update) {
    const TagType t{};
    functor(t, i, update);
  }

  template <class TagType>
  inline typename std::enable_if<std::is_same<TagType, void>::value>::type
  execute_functor_range(reference_type update, const Member i_begin,
                        const Member i_end) const {
    for (Member i = i_begin; i < i_end; ++i) {
      m_functor(i, update);
    }
  }

  template <class TagType>
  inline typename std::enable_if<!std::is_same<TagType, void>::value>::type
  execute_functor_range(reference_type update, const Member i_begin,
                        const Member i_end) const {
    const TagType t{};

    for (Member i = i_begin; i < i_end; ++i) {
      m_functor(t, i, update);
    }
  }

  class value_type_wrapper {
  private:
    std::size_t m_value_size;
    char *m_value_buffer;

  public:
    value_type_wrapper() : m_value_size(0), m_value_buffer(nullptr) {}

    value_type_wrapper(const std::size_t value_size)
        : m_value_size(value_size), m_value_buffer(new char[m_value_size]) {}

    value_type_wrapper(const value_type_wrapper &other)
        : m_value_size(0), m_value_buffer(nullptr) {
      if (this != &other) {
        m_value_buffer = new char[other.m_value_size];
        m_value_size = other.m_value_size;

        std::copy(other.m_value_buffer, other.m_value_buffer + m_value_size,
                  m_value_buffer);
      }
    }

    ~value_type_wrapper() { delete[] m_value_buffer; }

    value_type_wrapper(value_type_wrapper &&other)
        : m_value_size(0), m_value_buffer(nullptr) {
      if (this != &other) {
        m_value_buffer = other.m_value_buffer;
        m_value_size = other.m_value_size;

        other.m_value_buffer = nullptr;
        other.m_value_size = 0;
      }
    }

    value_type_wrapper &operator=(const value_type_wrapper &other) {
      if (this != &other) {
        delete[] m_value_buffer;
        m_value_buffer = new char[other.m_value_size];
        m_value_size = other.m_value_size;

        std::copy(other.m_value_buffer, other.m_value_buffer + m_value_size,
                  m_value_buffer);
      }

      return *this;
    }

    value_type_wrapper &operator=(value_type_wrapper &&other) {
      if (this != &other) {
        delete[] m_value_buffer;
        m_value_buffer = other.m_value_buffer;
        m_value_size = other.m_value_size;

        other.m_value_buffer = nullptr;
        other.m_value_size = 0;
      }

      return *this;
    }

    pointer_type pointer() const {
      return reinterpret_cast<pointer_type>(m_value_buffer);
    }

    reference_type reference() const {
      return ValueOps::reference(
          reinterpret_cast<pointer_type>(m_value_buffer));
    }
  };

public:
  void execute() const {
    std::size_t value_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

    value_type_wrapper final_value(value_size);
    value_type_wrapper identity(value_size);

    ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                    final_value.pointer());
    ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                    identity.pointer());
    
    using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;
    tbb::this_task_arena::isolate([this, &final_value,&identity]{
        final_value = tbb::parallel_reduce(RangeType(m_policy.begin(),
                                                     m_policy.end(),
                                                     m_policy.chunk_size()),
                                           identity,
                                           [this](const RangeType& r, value_type_wrapper const &init) {
                                             auto update = init;
                                             execute_functor_range<WorkTag>(update.reference(), r.begin(), r.end());
                                             return update;
                                           },
                                           [this](value_type_wrapper a,
                                                  value_type_wrapper const &b)  {
                                             ValueJoin::join(
                                                             ReducerConditional::select(m_functor, m_reducer),
                                                             a.pointer(), b.pointer());
                                             return a;
                                           }
                                           );
      }); //isolate

    pointer_type final_value_ptr = final_value.pointer();

    Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
        ReducerConditional::select(m_functor, m_reducer), final_value_ptr);

    if (m_result_ptr != nullptr) {
      const int n = Analysis::value_count(
          ReducerConditional::select(m_functor, m_reducer));

      for (int j = 0; j < n; ++j) {
        m_result_ptr[j] = final_value_ptr[j];
      }
    }
    
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType &arg_functor, Policy arg_policy,
      const ViewType &arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = NULL)
      : m_functor(arg_functor), m_policy(arg_policy), m_reducer(InvalidType()),
        m_result_ptr(arg_view.data()) {}

  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor), m_policy(arg_policy), m_reducer(reducer),
        m_result_ptr(reducer.view().data()) {}
};

template <class FunctorType, class ReducerType, class... Traits>
class ParallelReduce<FunctorType, Kokkos::MDRangePolicy<Traits...>, ReducerType,
                     Kokkos::Experimental::TBB> {
private:
  using MDRangePolicy = Kokkos::MDRangePolicy<Traits...>;
  using Policy = typename MDRangePolicy::impl_range_policy;
  using WorkTag = typename MDRangePolicy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member = typename Policy::member_type;
  using Analysis = FunctorAnalysis<FunctorPatternInterface::REDUCE,
                                   MDRangePolicy, FunctorType>;
  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;
  using ValueOps = Kokkos::Impl::FunctorValueOps<ReducerTypeFwd, WorkTagFwd>;
  using pointer_type = typename Analysis::pointer_type;
  using value_type = typename Analysis::value_type;
  using reference_type = typename Analysis::reference_type;
  using iterate_type =
      typename Kokkos::Impl::HostIterateTile<MDRangePolicy, FunctorType,
                                             WorkTag, reference_type>;

  const FunctorType m_functor;
  const MDRangePolicy m_mdr_policy;
  const Policy m_policy;
  const ReducerType m_reducer;
  const pointer_type m_result_ptr;

  using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;

  class SingleBody {
    FunctorType const& m_functor;
    ReducerType const& m_reducer;
    const MDRangePolicy m_mdr_policy;
    value_type m_sum;
    
  public:
    SingleBody(FunctorType const& functor,
               ReducerType const& reducer,
               const MDRangePolicy& policy) :
      m_functor(functor),
      m_reducer(reducer),
      m_mdr_policy(policy) {
      ValueInit::init(ReducerConditional::select(m_functor, m_reducer),&m_sum);
    }
    SingleBody( SingleBody& iOther, tbb::split) : SingleBody(iOther.m_functor, iOther.m_reducer, iOther.m_mdr_policy) {}
    
    void operator()(const RangeType& r) {
      for(auto i = r.begin(); i != r.end(); ++i) {
        iterate_type(m_mdr_policy, m_functor, m_sum)(i);
      }
    }
    
    void join(SingleBody& rhs) {
      ValueJoin::join(ReducerConditional::select(m_functor, m_reducer),
                      &m_sum,
                      &rhs.m_sum);
    }

    value_type& sum() { return m_sum; }
  };
  
  class ArrayBody {
    FunctorType const& m_functor;
    ReducerType const& m_reducer;
    const MDRangePolicy m_mdr_policy;
    value_type* m_sum;
    size_t m_size;
    
  public:
    ArrayBody() = delete;
    ArrayBody( ArrayBody const& ) = delete;

    ArrayBody(FunctorType const& functor,
              ReducerType const& reducer,
              const MDRangePolicy& policy,
              size_t size) :
      m_functor(functor),
      m_reducer(reducer),
      m_mdr_policy(policy),
      m_sum( new value_type[size] ),
      m_size(size) {
      ValueInit::init(ReducerConditional::select(m_functor, m_reducer),m_sum);
    }
    ArrayBody( ArrayBody& iOther, tbb::split) : ArrayBody(iOther.m_functor, iOther.m_reducer, iOther.m_mdr_policy, iOther.m_size) {
    }
    
    ~ArrayBody() {
      delete [] m_sum;
    }

    void operator()(const RangeType& r) {
      reference_type update = ValueOps::reference(m_sum);
      for(auto i = r.begin(); i != r.end(); ++i) {
        iterate_type(m_mdr_policy, m_functor, update )(i);
      }
    }
    
    void join(ArrayBody& rhs) {
      auto old = *m_sum;
      ValueJoin::join(ReducerConditional::select(m_functor, m_reducer),
                      m_sum,
                      rhs.m_sum);
    }
    
    pointer_type sum() { return m_sum; }
  };
  
  void executeImpl(SingleBody const*) const {
    const int n = Analysis::value_count(
                                        ReducerConditional::select(m_functor, m_reducer));
      SingleBody body(m_functor, m_reducer, m_mdr_policy);
      
      tbb::this_task_arena::isolate([this,&body]{
          tbb::parallel_reduce(RangeType(m_policy.begin(),
                                         m_policy.end(),
                                         m_policy.chunk_size()),
                               body);
          
        }
        );

      Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
                                                                    ReducerConditional::select(m_functor, m_reducer), &body.sum() );
      
      if (m_result_ptr != nullptr) {
        m_result_ptr[0] == body.sum();
      }
  }

  void executeImpl(ArrayBody const* ) const {
    const int n = Analysis::value_count(
                                        ReducerConditional::select(m_functor, m_reducer));
      ArrayBody body(m_functor, m_reducer, m_mdr_policy, n);
      
      tbb::this_task_arena::isolate([this,&body]{
          tbb::parallel_reduce(RangeType(m_policy.begin(),
                                         m_policy.end(),
                                         m_policy.chunk_size()),
                               body);
          
        }
        );
      
      Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
                                                                    ReducerConditional::select(m_functor, m_reducer), body.sum() );
      
      if (m_result_ptr != nullptr) {
        for(size_t i=0; i< n; ++i) {
          m_result_ptr[i] = body.sum()[i];
        }
      }
  }

public:
  
  void execute() const {
    executeImpl(static_cast<ArrayBody const*>(nullptr));;
  }

  template <class ViewType>
  inline ParallelReduce(
      const FunctorType &arg_functor, MDRangePolicy arg_policy,
      const ViewType &arg_view,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = NULL)
      : m_functor(arg_functor), m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(InvalidType()), m_result_ptr(arg_view.data()) {}

  inline ParallelReduce(const FunctorType &arg_functor,
                        MDRangePolicy arg_policy, const ReducerType &reducer)
      : m_functor(arg_functor), m_mdr_policy(arg_policy),
        m_policy(Policy(0, m_mdr_policy.m_num_tiles).set_chunk_size(1)),
        m_reducer(reducer), m_result_ptr(reducer.view().data()) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {

template <class FunctorType, class... Traits>
class ParallelScan<FunctorType, Kokkos::RangePolicy<Traits...>,
                   Kokkos::Experimental::TBB> {
private:
  using Policy = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member = typename Policy::member_type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;
  using pointer_type = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;
  using value_type = typename Analysis::value_type;

  const FunctorType m_functor;
  const Policy m_policy;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Member i_begin,
                            const Member i_end, reference_type update,
                            const bool final) {
    for (Member i = i_begin; i < i_end; ++i) {
      functor(i, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Member i_begin,
                            const Member i_end, reference_type update,
                            const bool final) {
    const TagType t{};
    for (Member i = i_begin; i < i_end; ++i) {
      functor(t, i, update, final);
    }
  }

  using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;

  class Body {

    value_type m_sum; 
    FunctorType const& m_functor;
  public:
    Body(FunctorType const& iFunc): m_functor(iFunc) {
      ValueInit::init(m_functor, reinterpret_cast<pointer_type>(&m_sum) );
    }
    template<typename Tag>
    void operator()( const RangeType& r, Tag ) {
      execute_functor_range<WorkTag>(m_functor, r.begin(), r.end(), m_sum, Tag::is_final_scan());
    }
    
    Body( Body& b, tbb::split ) : m_functor(b.m_functor) {
      ValueInit::init(m_functor, reinterpret_cast<pointer_type>(&m_sum) );
    }
    void reverse_join( Body& a ) { ValueJoin::join(m_functor, &m_sum, &a.m_sum); }
    void assign( Body& b ) { m_sum = b.m_sum; }
  
  };
public:
  void execute() const { 

    Body body(m_functor);

    tbb::this_task_arena::isolate([this,&body]{
        tbb::parallel_scan(RangeType(m_policy.begin(),
                                     m_policy.end(),
                                     m_policy.chunk_size()),
                           body);
      });
  }

  inline ParallelScan(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy) {}
};

template <class FunctorType, class ReturnType, class... Traits>
class ParallelScanWithTotal<FunctorType, Kokkos::RangePolicy<Traits...>,
                            ReturnType, Kokkos::Experimental::TBB> {
private:
  using Policy = Kokkos::RangePolicy<Traits...>;
  using WorkTag = typename Policy::work_tag;
  using WorkRange = typename Policy::WorkRange;
  using Member = typename Policy::member_type;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::SCAN, Policy, FunctorType>;
  using ValueInit = Kokkos::Impl::FunctorValueInit<FunctorType, WorkTag>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<FunctorType, WorkTag>;
  using ValueOps = Kokkos::Impl::FunctorValueOps<FunctorType, WorkTag>;
  using pointer_type = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;
  using value_type = typename Analysis::value_type;

  const FunctorType m_functor;
  const Policy m_policy;
  ReturnType &m_returnvalue;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Member i_begin,
                            const Member i_end, reference_type update,
                            const bool final) {
    for (Member i = i_begin; i < i_end; ++i) {
      functor(i, update, final);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Member i_begin,
                            const Member i_end, reference_type update,
                            const bool final) {
    const TagType t{};
    for (Member i = i_begin; i < i_end; ++i) {
      functor(t, i, update, final);
    }
  }

  using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;

  class Body {

    value_type m_sum; 
    FunctorType const& m_functor;
  public:
    Body(FunctorType const& iFunc): m_functor(iFunc) {
      ValueInit::init(m_functor, reinterpret_cast<pointer_type>(&m_sum) );
    }
    template<typename Tag>
    void operator()( const RangeType& r, Tag ) {
      execute_functor_range<WorkTag>(m_functor, r.begin(), r.end(), m_sum, Tag::is_final_scan());
    }
    
    Body( Body& b, tbb::split ) : m_functor(b.m_functor) {
      ValueInit::init(m_functor, reinterpret_cast<pointer_type>(&m_sum) );
    }
    void reverse_join( Body& a ) { ValueJoin::join(m_functor, &m_sum, &a.m_sum); }
    void assign( Body& b ) { m_sum = b.m_sum; }
  
    value_type const& sum() const { return m_sum;}
  
  };

public:
  void execute() const { 
    Body body(m_functor);

    using RangeType = tbb::blocked_range<decltype(m_policy.begin())>;
    tbb::this_task_arena::isolate([this,&body]{
        tbb::parallel_scan(RangeType(m_policy.begin(),
                                     m_policy.end(),
                                     m_policy.chunk_size()),
                           body);
      });
    m_returnvalue = body.sum();
  }

  inline ParallelScanWithTotal(const FunctorType &arg_functor,
                               const Policy &arg_policy,
                               ReturnType &arg_returnvalue)
      : m_functor(arg_functor), m_policy(arg_policy),
        m_returnvalue(arg_returnvalue) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {
namespace Impl {
template <class FunctorType, class... Properties>
class ParallelFor<FunctorType, Kokkos::TeamPolicy<Properties...>,
                  Kokkos::Experimental::TBB> {
private:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::TBB, Properties...>;
  using WorkTag = typename Policy::work_tag;
  using Member = typename Policy::member_type;
  using memory_space = Kokkos::HostSpace;

  const FunctorType m_functor;
  const Policy m_policy;
  const int m_league;
  const std::size_t m_shared;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor(const FunctorType &functor, const Policy &policy,
                      const int league_rank, char *local_buffer,
                      const std::size_t local_buffer_size) {
    functor(Member(policy, 0, league_rank, local_buffer, local_buffer_size));
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor(const FunctorType &functor, const Policy &policy,
                      const int league_rank, char *local_buffer,
                      const std::size_t local_buffer_size) {
    const TagType t{};
    functor(t, Member(policy, 0, league_rank, local_buffer, local_buffer_size));
  }

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Policy &policy,
                            const int league_rank_begin,
                            const int league_rank_end, char *local_buffer,
                            const std::size_t local_buffer_size) {
    for (int league_rank = league_rank_begin; league_rank < league_rank_end;
         ++league_rank) {
      functor(Member(policy, 0, league_rank, local_buffer, local_buffer_size));
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Policy &policy,
                            const int league_rank_begin,
                            const int league_rank_end, char *local_buffer,
                            const std::size_t local_buffer_size) {
    const TagType t{};
    for (int league_rank = league_rank_begin; league_rank < league_rank_end;
         ++league_rank) {
      functor(t,
              Member(policy, 0, league_rank, local_buffer, local_buffer_size));
    }
  }

public:
  void execute() const { 
    constexpr size_t kMaxLocalBuffer = 1024;

    using RangeType = tbb::blocked_range<decltype(m_policy.league_size())>;

    if(kMaxLocalBuffer >= m_shared) {
      tbb::this_task_arena::isolate([this]{
          tbb::parallel_for(RangeType(0,
                                      m_policy.league_size(),
                                      m_policy.chunk_size()),
                            [this](const RangeType& r) {
                              //make sure memory is aligned to at least what a ptr needs
                              std::array<intptr_t, kMaxLocalBuffer/sizeof(intptr_t)>  buffer;
                              for(auto league_rank = r.begin(); league_rank != r.end(); ++league_rank) {
                                execute_functor<WorkTag>(
                                                         m_functor, m_policy, league_rank,
                                                         reinterpret_cast<char*>(buffer.data()),
                                                         m_shared);
                              }
                            });
        });
    } else {
      tbb::this_task_arena::isolate([this]{
          tbb::parallel_for(RangeType(0,
                                      m_policy.league_size(),
                                      m_policy.chunk_size()),
                            [this](const RangeType& r) {
                              std::unique_ptr<char[]> buffer( new char[m_shared] );
                              for(auto league_rank = r.begin(); league_rank != r.end(); ++league_rank) {
                                execute_functor<WorkTag>(
                                                         m_functor, m_policy, league_rank,
                                                         buffer.get(),
                                                         m_shared);
                              }
                            });
        });
    }
    
 }

  ParallelFor(const FunctorType &arg_functor, const Policy &arg_policy)
      : m_functor(arg_functor), m_policy(arg_policy),
        m_league(arg_policy.league_size()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                     arg_functor, arg_policy.team_size())) {}
};

template <class FunctorType, class ReducerType, class... Properties>
class ParallelReduce<FunctorType, Kokkos::TeamPolicy<Properties...>,
                     ReducerType, Kokkos::Experimental::TBB> {
private:
  using Policy = TeamPolicyInternal<Kokkos::Experimental::TBB, Properties...>;
  using Analysis =
      FunctorAnalysis<FunctorPatternInterface::REDUCE, Policy, FunctorType>;
  using Member = typename Policy::member_type;
  using WorkTag = typename Policy::work_tag;
  using ReducerConditional =
      Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                         FunctorType, ReducerType>;
  using ReducerTypeFwd = typename ReducerConditional::type;
  using WorkTagFwd =
      typename Kokkos::Impl::if_c<std::is_same<InvalidType, ReducerType>::value,
                                  WorkTag, void>::type;
  using ValueInit = Kokkos::Impl::FunctorValueInit<ReducerTypeFwd, WorkTagFwd>;
  using ValueJoin = Kokkos::Impl::FunctorValueJoin<ReducerTypeFwd, WorkTagFwd>;
  using ValueOps = Kokkos::Impl::FunctorValueOps<ReducerTypeFwd, WorkTagFwd>;
  using pointer_type = typename Analysis::pointer_type;
  using reference_type = typename Analysis::reference_type;
  using value_type = typename Analysis::value_type;

  const FunctorType m_functor;
  const int m_league;
  const Policy m_policy;
  const ReducerType m_reducer;
  pointer_type m_result_ptr;
  const std::size_t m_shared;

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor(const FunctorType &functor, const Policy &policy,
                      const int league_rank, char *local_buffer,
                      const std::size_t local_buffer_size,
                      reference_type update) {
    functor(Member(policy, 0, league_rank, local_buffer, local_buffer_size),
            update);
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor(const FunctorType &functor, const Policy &policy,
                      const int league_rank, char *local_buffer,
                      const std::size_t local_buffer_size,
                      reference_type update) {
    const TagType t{};
    functor(t, Member(policy, 0, league_rank, local_buffer, local_buffer_size),
            update);
  }

  template <class TagType>
  inline static
      typename std::enable_if<std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Policy &policy,
                            const int league_rank_begin,
                            const int league_rank_end, char *local_buffer,
                            const std::size_t local_buffer_size,
                            reference_type update) {
    for (int league_rank = league_rank_begin; league_rank < league_rank_end;
         ++league_rank) {
      functor(Member(policy, 0, league_rank, local_buffer, local_buffer_size),
              update);
    }
  }

  template <class TagType>
  inline static
      typename std::enable_if<!std::is_same<TagType, void>::value>::type
      execute_functor_range(const FunctorType &functor, const Policy &policy,
                            const int league_rank_begin,
                            const int league_rank_end, char *local_buffer,
                            const std::size_t local_buffer_size,
                            reference_type update) {
    const TagType t{};
    for (int league_rank = league_rank_begin; league_rank < league_rank_end;
         ++league_rank) {
      functor(t,
              Member(policy, 0, league_rank, local_buffer, local_buffer_size),
              update);
    }
  }

public:
    void execute() const {
      const int num_worker_threads = Kokkos::Experimental::TBB::concurrency();
      const std::size_t value_size =
        Analysis::value_size(ReducerConditional::select(m_functor, m_reducer));

      thread_buffer buffer;
      buffer.resize(num_worker_threads, value_size + m_shared);
      

      tbb::this_task_arena::isolate([this, value_size, &buffer, num_worker_threads]{
      
          tbb::parallel_for(0, num_worker_threads, [this, &buffer](std::size_t t) {
              ValueInit::init(ReducerConditional::select(m_functor, m_reducer),
                              reinterpret_cast<pointer_type>(buffer.get(t)));
            });

          using RangeType = tbb::blocked_range<decltype(m_policy.league_size())>;
          tbb::parallel_for(RangeType(0,
                                      m_policy.league_size(),
                                      m_policy.chunk_size()),
                            [this,&buffer, value_size](const RangeType& r) {
                              auto t = Kokkos::Experimental::TBB::impl_hardware_thread_id();
                              reference_type update = ValueOps::reference(
                                                                          reinterpret_cast<pointer_type>(buffer.get(
                                                                                                                  t)));
                            for(auto league_rank = r.begin(); league_rank != r.end(); ++league_rank) {
                              execute_functor<WorkTag>(m_functor, m_policy, league_rank,
                                                       buffer.get(t) + value_size, m_shared,
                                                       update);
                            }
                            });
        });//isolate
      
      const pointer_type ptr = reinterpret_cast<pointer_type>(buffer.get(0));
      for (int t = 1; t < num_worker_threads; ++t) {
        ValueJoin::join(ReducerConditional::select(m_functor, m_reducer), ptr,
                        reinterpret_cast<pointer_type>(buffer.get(t)));
      }
      
      Kokkos::Impl::FunctorFinal<ReducerTypeFwd, WorkTagFwd>::final(
                                                                    ReducerConditional::select(m_functor, m_reducer), ptr);
      
      if (m_result_ptr) {
        const int n = Analysis::value_count(
                                            ReducerConditional::select(m_functor, m_reducer));
        
        for (int j = 0; j < n; ++j) {
          m_result_ptr[j] = ptr[j];
        }
      }
      
    }

  template <class ViewType>
  ParallelReduce(
      const FunctorType &arg_functor, const Policy &arg_policy,
      const ViewType &arg_result,
      typename std::enable_if<Kokkos::is_view<ViewType>::value &&
                                  !Kokkos::is_reducer_type<ReducerType>::value,
                              void *>::type = NULL)
      : m_functor(arg_functor), m_league(arg_policy.league_size()),
        m_policy(arg_policy), m_reducer(InvalidType()),
        m_result_ptr(arg_result.data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                                                          m_functor, arg_policy.team_size())) {}


  inline ParallelReduce(const FunctorType &arg_functor, Policy arg_policy,
                        const ReducerType &reducer)
      : m_functor(arg_functor), m_league(arg_policy.league_size()),
        m_policy(arg_policy), m_reducer(reducer),
        m_result_ptr(reducer.view().data()),
        m_shared(arg_policy.scratch_size(0) + arg_policy.scratch_size(1) +
                 FunctorTeamShmemSize<FunctorType>::value(
                                                          arg_functor, arg_policy.team_size())) {}
};
} // namespace Impl
} // namespace Kokkos

namespace Kokkos {

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>
    TeamThreadRange(const Impl::TBBTeamMember &thread, const iType &count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type, Impl::TBBTeamMember>
TeamThreadRange(const Impl::TBBTeamMember &thread, const iType1 &i_begin,
                const iType2 &i_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>(
      thread, iType(i_begin), iType(i_end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>
    TeamVectorRange(const Impl::TBBTeamMember &thread, const iType &count) {
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>(
      thread, count);
}

template <typename iType1, typename iType2>
KOKKOS_INLINE_FUNCTION Impl::TeamThreadRangeBoundariesStruct<
    typename std::common_type<iType1, iType2>::type, Impl::TBBTeamMember>
TeamVectorRange(const Impl::TBBTeamMember &thread, const iType1 &i_begin,
                const iType2 &i_end) {
  using iType = typename std::common_type<iType1, iType2>::type;
  return Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>(
      thread, iType(i_begin), iType(i_end));
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>
    ThreadVectorRange(const Impl::TBBTeamMember &thread, const iType &count) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>(
      thread, count);
}

template <typename iType>
KOKKOS_INLINE_FUNCTION
    Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>
    ThreadVectorRange(const Impl::TBBTeamMember &thread, const iType &i_begin,
                      const iType &i_end) {
  return Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>(
      thread, i_begin, i_end);
}

KOKKOS_INLINE_FUNCTION
Impl::ThreadSingleStruct<Impl::TBBTeamMember>
PerTeam(const Impl::TBBTeamMember &thread) {
  return Impl::ThreadSingleStruct<Impl::TBBTeamMember>(thread);
}

KOKKOS_INLINE_FUNCTION
Impl::VectorSingleStruct<Impl::TBBTeamMember>
PerThread(const Impl::TBBTeamMember &thread) {
  return Impl::VectorSingleStruct<Impl::TBBTeamMember>(thread);
}

/** \brief  Inter-thread parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team.
 * This functionality requires C++11 support.*/
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const Lambda &lambda) {
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment)
    lambda(i);
}

/** \brief  Inter-thread vector parallel_reduce. Executes lambda(iType i,
 * ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all threads of the the calling thread team
 * and a summation of val is performed and put into result. This functionality
 * requires C++11 support.*/
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const Lambda &lambda, ValueType &result) {
  result = ValueType();
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, result);
  }
}

/** \brief  Intra-thread vector parallel_for. Executes lambda(iType i) for each
 * i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread.
 * This functionality requires C++11 support.*/
template <typename iType, class Lambda>
KOKKOS_INLINE_FUNCTION void parallel_for(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const Lambda &lambda) {
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i);
  }
}

/** \brief  Intra-thread vector parallel_reduce. Executes lambda(iType i,
 * ValueType & val) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes of the the calling thread
 * and a summation of val is performed and put into result. This functionality
 * requires C++11 support.*/
template <typename iType, class Lambda, typename ValueType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const Lambda &lambda, ValueType &result) {
  result = ValueType();
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, result);
  }
}

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const Lambda &lambda, const ReducerType &reducer) {
  reducer.init(reducer.reference());
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, reducer.reference());
  }
}

template <typename iType, class Lambda, typename ReducerType>
KOKKOS_INLINE_FUNCTION void parallel_reduce(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const Lambda &lambda, const ReducerType &reducer) {
  reducer.init(reducer.reference());
#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, reducer.reference());
  }
}

template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    Impl::TeamThreadRangeBoundariesStruct<iType, Impl::TBBTeamMember> const
        &loop_boundaries,
    const FunctorType &lambda) {
  using value_type = typename Kokkos::Impl::FunctorAnalysis<
      Kokkos::Impl::FunctorPatternInterface::SCAN, void,
      FunctorType>::value_type;

  value_type scan_val = value_type();

  // Intra-member scan
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, scan_val, false);
  }

  // 'scan_val' output is the exclusive prefix sum
  scan_val = loop_boundaries.thread.team_scan(scan_val);

  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, scan_val, true);
  }
}

/** \brief  Intra-thread vector parallel exclusive prefix sum. Executes
 * lambda(iType i, ValueType & val, bool final) for each i=0..N-1.
 *
 * The range i=0..N-1 is mapped to all vector lanes in the thread and a scan
 * operation is performed. Depending on the target execution space the operator
 * might be called twice: once with final=false and once with final=true. When
 * final==true val contains the prefix sum value. The contribution of this "i"
 * needs to be added to val no matter whether final==true or not. In a serial
 * execution (i.e. team_size==1) the operator is only called once with
 * final==true. Scan_val will be set to the final sum value over all vector
 * lanes. This functionality requires C++11 support.*/
template <typename iType, class FunctorType>
KOKKOS_INLINE_FUNCTION void parallel_scan(
    const Impl::ThreadVectorRangeBoundariesStruct<iType, Impl::TBBTeamMember>
        &loop_boundaries,
    const FunctorType &lambda) {
  using ValueTraits = Kokkos::Impl::FunctorValueTraits<FunctorType, void>;
  using value_type = typename ValueTraits::value_type;

  value_type scan_val = value_type();

#ifdef KOKKOS_ENABLE_PRAGMA_IVDEP
#pragma ivdep
#endif
  for (iType i = loop_boundaries.start; i < loop_boundaries.end;
       i += loop_boundaries.increment) {
    lambda(i, scan_val, true);
  }
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::VectorSingleStruct<Impl::TBBTeamMember> &single_struct,
       const FunctorType &lambda) {
  lambda();
}

template <class FunctorType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::ThreadSingleStruct<Impl::TBBTeamMember> &single_struct,
       const FunctorType &lambda) {
  lambda();
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::VectorSingleStruct<Impl::TBBTeamMember> &single_struct,
       const FunctorType &lambda, ValueType &val) {
  lambda(val);
}

template <class FunctorType, class ValueType>
KOKKOS_INLINE_FUNCTION void
single(const Impl::ThreadSingleStruct<Impl::TBBTeamMember> &single_struct,
       const FunctorType &lambda, ValueType &val) {
  lambda(val);
}

} // namespace Kokkos

#include <TBB/Kokkos_TBB_Task.hpp>

#endif /* #if defined( KOKKOS_ENABLE_TBB ) */
#endif /* #ifndef KOKKOS_TBB_HPP */
