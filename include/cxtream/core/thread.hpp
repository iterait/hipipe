/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Thread Multithreading.

#ifndef CXTREAM_CORE_THREAD_HPP
#define CXTREAM_CORE_THREAD_HPP

#include <boost/asio.hpp>
#include <boost/thread/thread.hpp>

#include <cmath>
#include <functional>
#include <future>
#include <thread>
#include <experimental/tuple>

namespace cxtream {

/// \ingroup Thread
/// \brief A simple thread pool class.
///
/// This class manages the given number of threads across which
/// it automatically distributes the given tasks.
class thread_pool {
private:
    boost::asio::io_service service_;
    boost::asio::io_service::work work_{service_};
    boost::thread_group threads_;

public:

    /// Spawn the given number of threads and start processing the queue.
    ///
    /// \param n_threads The number of threads to be spawned.
    thread_pool(unsigned n_threads = std::thread::hardware_concurrency())
    {
        // always use at least a single thread
        n_threads = std::max(1u, n_threads);
        // populate the thread pool
        for (unsigned i = 0; i < n_threads; ++i) {
            threads_.create_thread([this]() { return this->service_.run(); });
        }
    }

    /// Enqueue a function for processing.
    ///
    /// \code
    ///     thread_pool tp{3};
    ///     auto fun = [](int i) { return i + 1; };
    ///     std::future<int> f1 = tp.enqueue(fun, 10);
    ///     std::future<int> f2 = tp.enqueue(fun, 11);
    ///     assert(f1.get() == 11);
    ///     assert(f2.get() == 12);
    /// \endcode
    ///
    /// \param fun The function to be executed.
    /// \param args Parameters for the function. Note that they are taken by value.
    /// \returns An std::future corresponding to the result of the function call.
    template<typename Fun, typename... Args>
    std::future<std::result_of_t<Fun(Args...)>> enqueue(Fun fun, Args... args)
    {
        using Ret = std::result_of_t<Fun(Args...)>;
        std::packaged_task<Ret(Args...)> task{fun};
        std::future<Ret> future = task.get_future();
        // Packaged task is non-copyable, so ASIO's post() method cannot handle it.
        // So we build a shared_ptr of the task and post a lambda
        // dereferencing and running the task stored in the pointer.
        auto shared_task = std::make_shared<std::packaged_task<Ret(Args...)>>(std::move(task));
        auto shared_args = std::make_shared<std::tuple<Args...>>(std::move(args)...);
        auto asio_task = [task = std::move(shared_task), args = std::move(shared_args)]() {
            return std::experimental::apply(std::move(*task), std::move(*args));
        };
        service_.post(std::move(asio_task));
        return future;
    }
};

/// \ingroup Thread
/// \brief Global thread pool object.
///
/// Prefer to use this object instead of spawning a new thread pool.
static thread_pool global_thread_pool;

} // namespace cxtream
#endif
