/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE buffer_view_test

#include "common.hpp"

#include <hipipe/core/stream/buffer.hpp>
#include <hipipe/core/stream/transform.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/transform.hpp>

#include <memory>
#include <vector>

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
    auto rng1 = hipipe::stream::buffer(data, 2);
    auto rng2 = data | hipipe::stream::buffer(2);

    auto it1 = ranges::begin(rng1);
    static_assert(std::is_same<int&&, decltype(*it1)>{});
    BOOST_TEST(*it1 == 1);

    auto it2 = ranges::begin(rng2);
    static_assert(std::is_same<int&&, decltype(*it2)>{});
    BOOST_TEST(*it2 == 1);

    BOOST_TEST(ranges::to_vector(rng1) == data);
    BOOST_TEST(ranges::to_vector(rng2) == data);
}


BOOST_AUTO_TEST_CASE(test_move_only_data)
{
    // check move only type buffering
    auto rng = ranges::view::iota(1, 6)
      | ranges::view::transform([](int i) {
          return std::make_unique<int>(i);
        })
      | hipipe::stream::buffer(2)
      | ranges::view::indirect;

    BOOST_TEST(ranges::to_vector(rng) == ranges::to_vector(ranges::view::iota(1, 6)));
}


BOOST_AUTO_TEST_CASE(test_check_if_buffered)
{
    // check if it is really buffer
    std::vector<std::shared_ptr<int>> data;
    for (int i = 0; i < 5; ++i) data.emplace_back(std::make_shared<int>(i));
    auto rng = hipipe::stream::buffer(data, 2);
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
    static_assert(std::is_same<std::shared_ptr<int>&&, decltype(*it)>{});

    BOOST_TEST(ranges::to_vector(ranges::view::indirect(rng)) ==
               ranges::to_vector(ranges::view::iota(0, 5)));
}


BOOST_AUTO_TEST_CASE(test_buffer_whole_range)
{
    // check infinite size buffer
    std::vector<std::shared_ptr<int>> data;
    for (int i = 0; i < 5; ++i) data.emplace_back(std::make_shared<int>(i));
    auto rng = data | hipipe::stream::buffer;
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


BOOST_AUTO_TEST_CASE(test_buffer_transformed_stream)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>();
    batch1.extract<Int>().push_back(3);
    batch1.extract<Int>().push_back(7);
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    batch1.extract<Unique>().push_back(std::make_unique<int>(1));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>();
    batch2.extract<Int>().push_back(1);
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::view::move
      | hipipe::stream::transform(from<Int, Unique>, to<Int>,
          [](int i, std::unique_ptr<int>& p) -> int {
            return i + *p;
        })
      | hipipe::stream::buffer(3);

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>().size() == 2);
    BOOST_TEST(stream.at(0).extract<Int>().at(0)  == 3 + 5);
    BOOST_TEST(stream.at(0).extract<Int>().at(1)  == 7 + 1);
    BOOST_TEST(stream.at(0).extract<Unique>().size() == 2);
    BOOST_TEST(*stream.at(0).extract<Unique>().at(0) == 5);
    BOOST_TEST(*stream.at(0).extract<Unique>().at(1) == 1);
    BOOST_TEST(stream.at(1).extract<Int>().size() == 1);
    BOOST_TEST(stream.at(1).extract<Int>().at(0)  == 1 + 2);
    BOOST_TEST(stream.at(1).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(1).extract<Unique>().at(0) == 2);
}
