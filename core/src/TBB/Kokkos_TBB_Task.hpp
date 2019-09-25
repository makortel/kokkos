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

#ifndef KOKKOS_TBB_TASK_HPP
#define KOKKOS_TBB_TASK_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_TBB) && defined(KOKKOS_ENABLE_TASKDAG)

#include <Kokkos_TaskScheduler_fwd.hpp>

#include <Kokkos_TBB.hpp>

#include <type_traits>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

  template<class SchedulerType, class QueueType>
  class TBBTask : public tbb::task {
    using scheduler_type = SchedulerType;
    using task_base_type = typename scheduler_type::task_base_type;

  public:
    TBBTask(OptionalRef<task_base_type> taskToRun,
         tbb::empty_task* iWaitingTask,
         scheduler_type iScheduler):
      task(std::move(taskToRun)),
      waitingTask(iWaitingTask),
      scheduler(iScheduler) {
    }

    tbb::task* execute() {
      const int num_worker_threads = Kokkos::Experimental::TBB::concurrency();

      // NOTE: This implementation has been simplified based on the
      // assumption that team_size = 1. The TBB backend currently only
      // supports a team size of 1.
      std::size_t t = Kokkos::Experimental::TBB::impl_hardware_thread_id();

      intptr_t buffer[512/sizeof(intptr_t)];
      TBBTeamMember member(TeamPolicyInternal<Kokkos::Experimental::TBB>(
                                                                         Kokkos::Experimental::TBB(), num_worker_threads, 1),
                           0, t, buffer, 512);

      using member_type =
        TaskTeamMemberAdapter<Kokkos::Impl::TBBTeamMember, scheduler_type>;
      
      member_type single_exec(scheduler, member);
      member_type &team_exec = single_exec;
      
      auto &team_scheduler = team_exec.scheduler();
      task->as_runnable_task().run(single_exec);

      auto &queue = scheduler.queue();

      //This is where any tasks waiting on this one will be scheduled

      //NOTE: does team_scheduler.team_scheduler_info have to be the same as the 
      // value used when calling queue.pop_ready_task? If so, then can't do it this way
      queue.complete((*std::move(task)).as_runnable_task(),
                     team_scheduler.team_scheduler_info());

      
      if(!queue.is_done()) {
        bool moreTasks = true;
        do {
          auto current_task =
            queue.pop_ready_task(team_scheduler.team_scheduler_info());
        
          if (current_task) {
            KOKKOS_ASSERT(current_task->is_single_runnable() ||
                          current_task->is_team_runnable());
            waitingTask->increment_ref_count();
            auto newTask = new(waitingTask->allocate_child()) TBBTask<scheduler_type, QueueType>(std::move(current_task),
                                                                                 waitingTask,
                                                                                 scheduler);
            tbb::task::spawn(*newTask);
          } else {
            moreTasks = false;
          }
        } while(moreTasks);
      }
      return nullptr;
    }
  private:
    OptionalRef<task_base_type> task;
    tbb::empty_task* waitingTask;
    scheduler_type scheduler;
  };

template <class QueueType>
class TaskQueueSpecialization<
    SimpleTaskScheduler<Kokkos::Experimental::TBB, QueueType>> {
public:
  using execution_space = Kokkos::Experimental::TBB;
  using scheduler_type =
      SimpleTaskScheduler<Kokkos::Experimental::TBB, QueueType>;
  using member_type =
      TaskTeamMemberAdapter<Kokkos::Impl::TBBTeamMember, scheduler_type>;
  using memory_space = Kokkos::HostSpace;

  static void execute(scheduler_type const &scheduler) {

    tbb::this_task_arena::isolate([scheduler]{

        // NOTE: We create an instance so that we can use dispatch_execute_task.
        // This is not necessarily the most efficient, but can be improved later.
        TaskQueueSpecialization<scheduler_type> task_queue;
        task_queue.scheduler = &scheduler;
        
        
        const int num_worker_threads = Kokkos::Experimental::TBB::concurrency();
        
        auto &queue = scheduler.queue();
        
        //setup the task used for waiting.
        auto waitingTask = new (tbb::task::allocate_root()) tbb::empty_task();
        waitingTask->increment_ref_count(); //waiting tasks need a ref count > 0
        
        bool moreTasks = true;
        
        TBBTeamMember member(TeamPolicyInternal<Kokkos::Experimental::TBB>(
                                                                           Kokkos::Experimental::TBB(), num_worker_threads, 1),
                             0, 0 /*say we are thread 0*/, nullptr, 0);
        
        
        member_type single_exec(scheduler, member);
        member_type &team_exec = single_exec;
        
        auto &team_scheduler = team_exec.scheduler();
        
        do {
          auto current_task =
            queue.pop_ready_task(team_scheduler.team_scheduler_info());
          
          if (current_task) {
            KOKKOS_ASSERT(current_task->is_single_runnable() ||
                          current_task->is_team_runnable());
            waitingTask->increment_ref_count();          
            auto newTask = new(waitingTask->allocate_child()) TBBTask<scheduler_type, QueueType>(std::move(current_task),
                                                                                                 waitingTask,
                                                                                                 scheduler);
            tbb::task::spawn(*newTask);
          } else {
            moreTasks = false;
          }
        } while(moreTasks);
        
        
        waitingTask->wait_for_all();

        tbb::task::destroy(*waitingTask);    
      });
  }

  static uint32_t get_max_team_count(execution_space const &espace) {
    return static_cast<uint32_t>(espace.concurrency());
  }

  template <typename TaskType>
  static void get_function_pointer(typename TaskType::function_type &ptr,
                                   typename TaskType::destroy_type &dtor) {
    ptr = TaskType::apply;
    dtor = TaskType::destroy;
  }

private:
  const scheduler_type *scheduler;
};


  template<class SchedulerType, class QueueType, class TaskBase>
  class TBBTaskForConstrained : public tbb::task {

  public:

    using scheduler_type = SchedulerType;
    using task_base_type = TaskBase;

    TBBTaskForConstrained(task_base_type* taskToRun,
         tbb::empty_task* iWaitingTask,
         scheduler_type iScheduler):
      task(taskToRun),
      waitingTask(iWaitingTask),
      scheduler(iScheduler) {
    }

    tbb::task* execute() {
      auto nextTask = TaskQueueSpecializationConstrained<SchedulerType>::execute_task_and_get_next(task, scheduler, waitingTask);
      if(nullptr != nextTask) {
        waitingTask->increment_ref_count();
        auto newTask = new(waitingTask->allocate_child()) TBBTaskForConstrained<scheduler_type, QueueType, TaskBase>(nextTask,
                                                                                                                     waitingTask,
                                                                                                                     scheduler);
        return newTask;
      }
      return nullptr;
    }
  private:
    task_base_type* task;
    tbb::empty_task* waitingTask;
    scheduler_type scheduler;
  };

template <class Scheduler>
class TaskQueueSpecializationConstrained<
    Scheduler, typename std::enable_if<
                   std::is_same<typename Scheduler::execution_space,
                                Kokkos::Experimental::TBB>::value>::type> {
public:
  using execution_space = Kokkos::Experimental::TBB;
  using scheduler_type = Scheduler;
  using member_type =
      TaskTeamMemberAdapter<Kokkos::Impl::TBBTeamMember, scheduler_type>;
  using memory_space = Kokkos::HostSpace;

  static void
  iff_single_thread_recursive_execute(scheduler_type const &scheduler) {
    using task_base_type = typename scheduler_type::task_base;
    using queue_type = typename scheduler_type::queue_type;

    if (1 == Kokkos::Experimental::TBB::concurrency()) {
      task_base_type *const end = (task_base_type *)task_base_type::EndTag;
      task_base_type *task = end;

      TBBTeamMember member(TeamPolicyInternal<Kokkos::Experimental::TBB>(
                               Kokkos::Experimental::TBB(), 1, 1),
                           0, 0, nullptr, 0);
      member_type single_exec(scheduler, member);

      do {
        task = end;

        // Loop by priority and then type
        for (int i = 0; i < queue_type::NumQueue && end == task; ++i) {
          for (int j = 0; j < 2 && end == task; ++j) {
            task =
                queue_type::pop_ready_task(&scheduler.m_queue->m_ready[i][j]);
          }
        }

        if (end == task)
          break;

        (*task->m_apply)(task, &single_exec);

        scheduler.m_queue->complete(task);

      } while (true);
    }
  }

  
  static void execute(scheduler_type const &scheduler) {
    tbb::this_task_arena::isolate([&scheduler]{

        using queue_type = typename scheduler_type::queue_type;
        
        // NOTE: We create an instance so that we can use dispatch_execute_task.
        // This is not necessarily the most efficient, but can be improved later.
        TaskQueueSpecializationConstrained<scheduler_type> task_queue;
        task_queue.scheduler = &scheduler;
        
        //setup the task used for waiting.
        auto waitingTask = new (tbb::task::allocate_root()) tbb::empty_task();
        waitingTask->increment_ref_count(); //waiting tasks need a ref count > 0
        
        const int num_worker_threads = Kokkos::Experimental::TBB::concurrency();
        TBBTeamMember member(TeamPolicyInternal<Kokkos::Experimental::TBB>(
                                                                           Kokkos::Experimental::TBB(), num_worker_threads, 1),
                             0, 0 /*say we are thread 0*/, nullptr, 0);
        
        
        
        auto &queue = scheduler.queue();
        queue.initialize_team_queues(num_worker_threads);
        
        
        using task_base_type = typename scheduler_type::task_base;
        task_base_type *const end = (task_base_type *)task_base_type::EndTag;
        constexpr task_base_type const * no_more_tasks_sentinel = nullptr;
        task_base_type* nextTaskToRun = end;
        
        member_type single_exec(scheduler, member);
        member_type &team_exec = single_exec;
        
        auto &team_queue = team_exec.scheduler().queue();
        
        if (*((volatile int *)&team_queue.m_ready_count) > 0) {
          //pull all tasks from our queue
          
          task_base_type* task =  end;
          do {
            task = end;
            for (int i = 0; i < queue_type::NumQueue && end == task; ++i) {
              for (int j = 0; j < 2 && end == task; ++j) {
                task = queue_type::pop_ready_task(&team_queue.m_ready[i][j]);
                if(nextTaskToRun == end) {
                  nextTaskToRun = task;
                }
              }
            }
            if(task != end and task != no_more_tasks_sentinel and task != nextTaskToRun) {
              waitingTask->increment_ref_count();
              auto newTask = new(waitingTask->allocate_child()) TBBTaskForConstrained<scheduler_type, queue_type, task_base_type>(task,
                                                                                                                                  waitingTask,
                                                                                                                                  scheduler);
              tbb::task::spawn(*newTask);            
            }
          } while(task != end and task != no_more_tasks_sentinel);
        }
        
        if(nextTaskToRun != end and nextTaskToRun != no_more_tasks_sentinel) {
          waitingTask->increment_ref_count();
          auto newTask = new(waitingTask->allocate_child()) TBBTaskForConstrained<scheduler_type, queue_type, task_base_type>(nextTaskToRun,
                                                                                                                              waitingTask,
                                                                                                                              scheduler);
          tbb::task::spawn(*newTask);
        }
        waitingTask->wait_for_all();
        tbb::task::destroy(*waitingTask);    
      });
  }

  template<typename TaskType>
  static TaskType* execute_task_and_get_next(TaskType *task, scheduler_type scheduler, tbb::empty_task* waitingTask) {
    using queue_type = typename scheduler_type::queue_type;
    const int num_worker_threads = Kokkos::Experimental::TBB::concurrency();
    
    // NOTE: This implementation has been simplified based on the
    // assumption that team_size = 1. The TBB backend currently only
    // supports a team size of 1.
    std::size_t t = Kokkos::Experimental::TBB::impl_hardware_thread_id();
    
    intptr_t buffer[512/sizeof(intptr_t)];
    TBBTeamMember member(TeamPolicyInternal<Kokkos::Experimental::TBB>(
                                                                       Kokkos::Experimental::TBB(), num_worker_threads, 1),
                         0, t, buffer, 512);
    
    using member_type =
      TaskTeamMemberAdapter<Kokkos::Impl::TBBTeamMember, scheduler_type>;
    
    member_type single_exec(scheduler, member);
    member_type &team_exec = single_exec;
    
    (*task->m_apply)(task, &single_exec);

    auto &team_scheduler = team_exec.scheduler();
    auto &team_queue = scheduler.queue();
    
    //This is where any tasks waiting on this one will be scheduled
    team_queue.complete(task);
    
    TaskType *const end = (TaskType *)TaskType::EndTag;
    constexpr TaskType const * no_more_tasks_sentinel = nullptr;
    auto* nextTaskToRun = end;
    if (*((volatile int *)&team_queue.m_ready_count) > 0) {
      //pull all tasks from our queue
      
      auto* task =  end;
      do {
        task = end;
        for (int i = 0; i < queue_type::NumQueue && end == task; ++i) {
          for (int j = 0; j < 2 && end == task; ++j) {
            task = queue_type::pop_ready_task(&team_queue.m_ready[i][j]);
            if(nextTaskToRun == end) {
              nextTaskToRun = task;
            }
          }
        }
        if(task != end and task != no_more_tasks_sentinel and task != nextTaskToRun) {
          waitingTask->increment_ref_count();
          auto newTask = new(waitingTask->allocate_child()) TBBTaskForConstrained<scheduler_type, queue_type, TaskType>(task,
                                                                                                                       waitingTask,
                                                                                                                       scheduler);
          tbb::task::spawn(*newTask);            
        }
      } while(task != end and task != no_more_tasks_sentinel);
    } else {
      nextTaskToRun = team_queue.attempt_to_steal_task();
    }
    return nextTaskToRun == end ? static_cast<TaskType*>(nullptr) : nextTaskToRun;
      
  }

  template <typename TaskType>
  static void get_function_pointer(typename TaskType::function_type &ptr,
                                   typename TaskType::destroy_type &dtor) {
    ptr = TaskType::apply;
    dtor = TaskType::destroy;
  }

private:
  const scheduler_type *scheduler;
};

extern template class TaskQueue<
    Kokkos::Experimental::TBB,
    typename Kokkos::Experimental::TBB::memory_space>;


} // namespace Impl
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif /* #if defined( KOKKOS_ENABLE_TASKDAG ) */
#endif /* #ifndef KOKKOS_TBB_TASK_HPP */
