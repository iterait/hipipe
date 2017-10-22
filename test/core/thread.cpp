/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE thread_test

#include <cxtream/core/thread.hpp>

#include <boost/test/unit_test.hpp>

#include <future>
#include <memory>
#include <thread>

using namespace std::chrono_literals;

BOOST_AUTO_TEST_CASE(test_enqueue)
{
    cxtream::thread_pool tp{2};
    std::vector<std::thread::id> ids(3);
    auto task = [&ids](std::shared_ptr<int> ptr) {
        // save the id of this thread
        ids[*ptr] = std::this_thread::get_id();
        // clone the given shared_ptr
        std::shared_ptr<int> new_ptr{ptr};
        // sleep so that tests have enough time to evaluate
        std::this_thread::sleep_for(50ms);
        return *new_ptr;
    };
    std::shared_ptr<int> ptr1 = std::make_shared<int>(0);
    std::shared_ptr<int> ptr2 = std::make_shared<int>(1);
    std::shared_ptr<int> ptr3 = std::make_shared<int>(2);

    // run the tasks
    std::future<int> f1 = tp.enqueue(task, ptr1);
    std::future<int> f2 = tp.enqueue(task, ptr2);
    std::future<int> f3 = tp.enqueue(task, ptr3);

    // test that the processes run in parallel
    std::this_thread::sleep_for(20ms);
    // one reference to the pointer is held by this scope,
    // other two are held by the task
    BOOST_TEST(ptr1.use_count() == 3);
    BOOST_TEST(ptr2.use_count() == 3);
    // the third task should not be running yet, so
    // one reference is held in this scope and one in the
    // task queue 
    BOOST_TEST(ptr3.use_count() == 2);
    // now wait until the first two threads finish
    BOOST_TEST(f1.get() == 0);
    BOOST_TEST(f2.get() == 1);
    // after reading their results, the third task should be running
    std::this_thread::sleep_for(20ms);
    BOOST_TEST(ptr3.use_count() == 3);
    // and wait until it is finished
    BOOST_TEST(f3.get() == 2);
    
    // make sure that exactly 2 threads were spawned
    // (exactly one pair has to be the same)
    BOOST_CHECK((ids[0] == ids[1]) ^ (ids[1] == ids[2]) ^ (ids[0] == ids[2]));
}

BOOST_AUTO_TEST_CASE(test_enqueue_move)
{
    // test that thread_pool.enqueue accepts move-only arguments
    cxtream::thread_pool tp{2};
    auto task = [](std::unique_ptr<int> ptr) { return *ptr; };
    std::future<int> f = tp.enqueue(std::move(task), std::make_unique<int>(10));
    BOOST_TEST(f.get() == 10);
}

BOOST_AUTO_TEST_CASE(test_graceful_destruction)
{
    std::vector<std::future<int>> futures;
    auto task = [](int i) {
        std::this_thread::sleep_for(50ms);
        return i;
    };

    {
        cxtream::thread_pool tp{2};
        for (int i = 0; i < 5; ++i) futures.push_back(tp.enqueue(task, i));
        // at the end of this block, the thread pool is destroyed
    }

    for (int i = 0; i < 5; ++i) {
        BOOST_CHECK(futures[i].wait_for(0ms) == std::future_status::ready);
        BOOST_TEST(futures[i].get() == i);
    }
}
