/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE buffer_view_test

#include "../common.hpp"

#include <cxtream/core/stream/buffer.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <memory>
#include <vector>

using namespace cxtream::stream;
using namespace std::chrono_literals;

void test_use_count(const std::vector<std::shared_ptr<int>>& ptrs, const std::vector<int>& desired)
{
    for (unsigned i = 0; i < ptrs.size(); ++i) {
        BOOST_TEST(ptrs[i].use_count() == desired[i]);
    }
}

BOOST_AUTO_TEST_CASE(test_simple_traverse)
{
    // check simple traverse
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto rng1 = buffer(data, 2);
    auto rng2 = data | buffer(2);

    auto it1 = ranges::begin(rng1);
    static_assert(std::is_same<const int&, decltype(*it1)>{});
    BOOST_TEST(*it1 == 1);

    auto it2 = ranges::begin(rng2);
    static_assert(std::is_same<const int&, decltype(*it2)>{});
    BOOST_TEST(*it2 == 1);

    test_ranges_equal(rng1, data);
    test_ranges_equal(rng2, data);
}

BOOST_AUTO_TEST_CASE(test_move_only_data)
{
    // check move only type buffering
    auto rng = ranges::view::iota(1, 6)
      | ranges::view::transform([](int i) {
          return std::make_unique<int>(i);
        })
      | buffer(2)
      | ranges::view::indirect;

    test_ranges_equal(rng, ranges::view::iota(1, 6));
}

BOOST_AUTO_TEST_CASE(test_check_if_buffered)
{
    // check if it is really buffer
    std::vector<std::shared_ptr<int>> data;
    for (int i = 0; i < 5; ++i) data.emplace_back(std::make_shared<int>(i));
    auto rng = buffer(data, 2);
    BOOST_TEST(rng.size() == data.size());

    // iterate through and check not-yet visited elements' use count
    test_use_count(data, {1, 1, 1, 1, 1});
    auto it = ranges::begin(rng);
    BOOST_CHECK(it != ranges::end(rng));
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {2, 2, 1, 1, 1});
    ++it;
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 2, 2, 1, 1});
    ++it;
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 1, 2, 2, 1});
    ++it;
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 1, 1, 2, 2});
    ++it;
    BOOST_CHECK(it != ranges::end(rng));
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 1, 1, 1, 2});
    ++it;
    BOOST_CHECK(it == ranges::end(rng));
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 1, 1, 1, 1});

    // iterate with two iterators at once
    auto it2 = ranges::begin(rng);
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {2, 2, 1, 1, 1});
    auto it3 = ranges::begin(rng);
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {3, 3, 1, 1, 1});
    ++it2;
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {2, 3, 2, 1, 1});
    static_assert(std::is_same<const std::shared_ptr<int>&, decltype(*it)>{});

    // check values
    test_ranges_equal(rng | ranges::view::indirect, ranges::view::iota(0, 5));
}

BOOST_AUTO_TEST_CASE(test_buffer_whole_range)
{
    // check infinite size buffer
    std::vector<std::shared_ptr<int>> data;
    for (int i = 0; i < 5; ++i) data.emplace_back(std::make_shared<int>(i));
    auto rng = data | buffer;
    BOOST_TEST(rng.size() == data.size());

    // iterate through and check not-yet visited elements' use count
    test_use_count(data, {1, 1, 1, 1, 1});
    auto it = ranges::begin(rng);
    BOOST_CHECK(it != ranges::end(rng));
    std::this_thread::sleep_for(40ms);
    test_use_count(data, {2, 2, 2, 2, 2});
    ++it;
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 2, 2, 2, 2});
    ++it;
    ++it;
    ++it;
    ++it;
    BOOST_CHECK(it == ranges::end(rng));
    std::this_thread::sleep_for(20ms);
    test_use_count(data, {1, 1, 1, 1, 1});
}
