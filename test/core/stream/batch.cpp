/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE batch_view_test

#include "../common.hpp"

#include <cxtream/core/stream/batch.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>

#include <limits>
#include <vector>
#include <memory>

using namespace cxtream::stream;

auto generate_batched_data(std::vector<int> batch_sizes)
{
    std::vector<std::tuple<Unique, Shared>> data;
    int counter = 0;
    for (std::size_t i = 0; i < batch_sizes.size(); ++i) {
        std::vector<std::unique_ptr<int>> unique_data;
        std::vector<std::shared_ptr<int>> shared_data;
        for (int j = 0; j < batch_sizes[i]; ++j) {
            unique_data.emplace_back(std::make_unique<int>(counter));
            shared_data.emplace_back(std::make_shared<int>(counter));
            ++counter;
        }
        data.emplace_back(std::make_tuple(std::move(unique_data), std::move(shared_data)));
    }
    return data;
}

auto generate_regular_batched_data(int batches, int batch_size)
{
    return generate_batched_data(std::vector<int>(batches, batch_size));
}

template<typename Data>
void check_20_elems_batch_size_2(Data data)
{
    auto rng = data | ranges::view::move | batch(3);
    using tuple_type = decltype(*ranges::begin(rng));
    static_assert(std::is_same<std::tuple<Unique, Shared>&, tuple_type>{});
    using first_type = decltype(std::get<0>(*ranges::begin(rng)).value()[0]);
    static_assert(std::is_same<std::unique_ptr<int>&, first_type>{});
    using second_type = decltype(std::get<1>(*ranges::begin(rng)).value()[0]);
    static_assert(std::is_same<std::shared_ptr<int>&, second_type>{});

    // iterate through batches
    std::vector<int> result_unique;
    std::vector<int> result_shared;
    int batch_n = 0;
    int n = 0;
    for (auto&& tuple : rng) {
        // the last batch should be smaller
        if (batch_n == 6) {
            BOOST_TEST(std::get<0>(tuple).value().size() == 2U);
            BOOST_TEST(std::get<1>(tuple).value().size() == 2U);
        }
        else {
            BOOST_TEST(std::get<0>(tuple).value().size() == 3U);
            BOOST_TEST(std::get<1>(tuple).value().size() == 3U);
        }
        BOOST_CHECK(is_same_batch_size(tuple));

        // iterate through batch values
        for (auto& elem : std::get<0>(tuple).value()) {
            result_unique.push_back(*elem);
            ++n;
        }
        for (auto& elem : std::get<1>(tuple).value()) {
            result_shared.push_back(*elem);
        }
        ++batch_n;
    }
    BOOST_TEST(batch_n == 7);
    BOOST_TEST(n == 20);

    auto desired = ranges::view::iota(0, 20);
    test_ranges_equal(result_unique, desired);
    test_ranges_equal(result_shared, desired);
}

BOOST_AUTO_TEST_CASE(test_batch_larger_batches)
{
    // batch out of larger batches
    auto data = generate_regular_batched_data(3, 4);

    auto rng = data | ranges::view::move | batch(1);
    using tuple_type = decltype(*ranges::begin(rng));
    static_assert(std::is_same<std::tuple<Unique, Shared>&, tuple_type>{});
    using first_type = decltype(std::get<0>(*ranges::begin(rng)).value()[0]);
    static_assert(std::is_same<std::unique_ptr<int>&, first_type>{});
    using second_type = decltype(std::get<1>(*ranges::begin(rng)).value()[0]);
    static_assert(std::is_same<std::shared_ptr<int>&, second_type>{});

    // iterate through batches
    std::vector<int> result_unique;
    std::vector<int> result_shared;
    int n = 0;
    for (auto&& tuple : rng) {
        // the batch should be only a single element
        BOOST_TEST(std::get<0>(tuple).value().size() == 1U);
        BOOST_TEST(std::get<1>(tuple).value().size() == 1U);
        // remember the values
        result_unique.push_back(*(std::get<0>(tuple).value()[0]));
        result_shared.push_back(*(std::get<1>(tuple).value()[0]));
        ++n;
    }
    BOOST_TEST(n == 12);

    auto desired = ranges::view::iota(0, 12);
    test_ranges_equal(result_unique, desired);
    test_ranges_equal(result_shared, desired);
}

BOOST_AUTO_TEST_CASE(test_batch_smaller_batches)
{
    // batch out of smaller batches
    check_20_elems_batch_size_2(generate_regular_batched_data(10, 2));
}

BOOST_AUTO_TEST_CASE(test_batch_irregular_batches)
{
    // batch out of iregularly sized batches
    check_20_elems_batch_size_2(generate_batched_data({0, 1, 2, 0, 5, 2, 0, 1, 7, 0, 2, 0, 0}));
}

BOOST_AUTO_TEST_CASE(test_batch_empty_batches)
{
    // batch out of empty batches
    auto data = generate_batched_data({0, 0, 0, 0});
    auto rng = data | ranges::view::move | batch(1);
    BOOST_CHECK(rng.begin() == rng.end());
}

BOOST_AUTO_TEST_CASE(test_batch_empty_stream)
{
    // batch out of empty range
    auto data = generate_batched_data({});
    auto rng = data | ranges::view::move | batch(1);
    BOOST_CHECK(rng.begin() == rng.end());
}

BOOST_AUTO_TEST_CASE(test_infinite_batch)
{
    auto data = generate_regular_batched_data(3, 4);
    // make batch of infinite size (no parameter given)
    auto rng = data | ranges::view::move | batch(std::numeric_limits<std::size_t>::max());
    auto rng_it = rng.begin();
    auto result = std::move(*rng_it);
    auto result_unique = std::get<0>(result).value() | ranges::view::indirect;
    auto result_shared = std::get<1>(result).value() | ranges::view::indirect;
    BOOST_CHECK(++rng_it == rng.end());
    BOOST_TEST(result_unique.size() == 12);
    BOOST_TEST(result_shared.size() == 12);

    auto desired = ranges::view::iota(0, 12);
    test_ranges_equal(result_unique, desired);
    test_ranges_equal(result_shared, desired);
}
