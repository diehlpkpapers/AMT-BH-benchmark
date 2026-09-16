#pragma once

#include <optional>
#include <unistd.h>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"


namespace ityr::ito {


/**
 * @class callstack
 * @brief Manages a call stack with support for direct memory acces on destributed systems using MPI.
 * 
 * In a distributed task scheduler, where threads or tasks are migrated between processes,
 * the stack contents must be transferred accordingly.
 * The `callstack` class encapsulates the management, access and allocation to this stack region,
 * providing facilities for mapping physical memory to virtual memory, and enabling direct memory
 * copying between ranks in a distributed environment using MPI.
 *
 *
 * Memory Reservation:
 * The constructor reserves a virtual memory region (`vm_`) and backs it with physical memory (`pm_`),
 * allowing each process to maintain its own stack region.
 *
 * Remote Access:
 * Through the MPI window (`win_`), the stack region can be read or written by other processes
 * (e.g., for work-stealing or thread migration).
 *
 * Stack Copy:
 * The `direct_copy_from` method enables direct copying of a stack region from another process (rank)
 * into the local stack.
 *
 * Helper Functions:
 * Methods such as `top()`, `bottom()`, and `size()` provide access to the stack boundaries and its size.
 *
 *
 */
class callstack {
public:
  callstack(std::size_t size)
    : vm_(common::reserve_same_vm_coll(size, common::get_page_size())),
      pm_(init_stack_pm()),
      win_(common::topology::mpicomm(), reinterpret_cast<std::byte*>(vm_.addr()), vm_.size()) {
    ityr::common::verbose("begin address = %p size = %u parameter size: %u", vm_.addr(), vm_.size(), size);
  }

  void* top() const { return vm_.addr(); }
  void* bottom() const { return reinterpret_cast<std::byte*>(vm_.addr()) + vm_.size(); }
  std::size_t size() const { return vm_.size(); }

  static void print_stack_pointer2(std::size_t size) {
    void* p = NULL;
    ityr::common::verbose("direct copy stack pointer [%p, %p)", (void*)&p, ((std::byte*)&p - size));
  }

  void direct_copy_from(void*                    addr,
                        std::size_t              size,
                        common::topology::rank_t target_rank) const {
    ITYR_CHECK(target_rank != common::topology::my_rank());
    ITYR_CHECK(target_rank < common::topology::n_ranks());
    ITYR_CHECK(vm_.addr() <= addr);
    ITYR_CHECK(reinterpret_cast<std::byte*>(addr) + size <= reinterpret_cast<std::byte*>(vm_.addr()) + vm_.size());

    auto target_disp = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(vm_.addr());
    common::mpi_get(reinterpret_cast<std::byte*>(addr), size, target_rank, target_disp, win_.win());
  }

private:
  static std::string stack_shmem_name(int rank) {
    std::stringstream ss;
    ss << "/ityr_ito_stack_" << rank << "_" << getuid();
    return ss.str();
  }

  common::physical_mem init_stack_pm() {
    common::physical_mem pm(stack_shmem_name(common::topology::my_rank()), vm_.size(), true);
    pm.map_to_vm(vm_.addr(), vm_.size(), 0);
    ityr::common::verbose("callstack mapped from physical mam to virtual mem at address [%p, %p)", vm_.addr(), (uint8_t*) vm_.addr() + vm_.size());
    return pm;
  }

  common::virtual_mem                vm_;
  common::physical_mem               pm_;
  common::mpi_win_manager<std::byte> win_;
};

}
