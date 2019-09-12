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
#define BOOST_TEST_MODULE batch_view_test

#include "common.hpp"

#include <hipipe/core/stream/rebatch.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>

#include <limits>


auto generate_batched_data(std::vector<int> batch_sizes)
{
    int counter = 0;
    std::vector<hipipe::stream::batch_t> data;
    for (std::size_t i = 0; i < batch_sizes.size(); ++i) {
        hipipe::stream::batch_t batch;
        batch.insert_or_assign<Unique>();
        batch.insert_or_assign<Int>();
        for (int j = 0; j < batch_sizes[i]; ++j) {
            batch.extract<Unique>().push_back(std::make_unique<int>(counter));
            batch.extract<Int>().push_back(counter);
            ++counter;
        }
        data.push_back(std::move(batch));
    }
    return data;
}


auto generate_regular_batched_data(int batches, int batch_size)
{
    return generate_batched_data(std::vector<int>(batches, batch_size));
}


void check_20_elems_batch_size_2(std::vector<hipipe::stream::batch_t> data)
{
    std::vector<hipipe::stream::batch_t> stream = data
      | rgv::move
      | hipipe::stream::rebatch(3)
      | rg::to_vector;

    // iterate through batches
    std::vector<int> result_unique;
    std::vector<int> result_int;
    int batch_n = 0;
    int n = 0;
    for (const hipipe::stream::batch_t& batch : stream) {
        // the last batch should be smaller
        if (batch_n == 6) {
            BOOST_TEST(batch.batch_size() == 2U);
            BOOST_TEST(batch.extract<Int>().size() == 2U);
            BOOST_TEST(batch.extract<Unique>().size() == 2U);
        } else {
            BOOST_TEST(batch.batch_size() == 3U);
            BOOST_TEST(batch.extract<Int>().size() == 3U);
            BOOST_TEST(batch.extract<Unique>().size() == 3U);
        }

        // iterate through batch values
        for (const int& elem : batch.extract<Int>()) {
            result_int.push_back(elem);
            ++n;
        }
        for (const std::unique_ptr<int>& elem : batch.extract<Unique>()) {
            result_unique.push_back(*elem);
        }
        ++batch_n;
    }
    BOOST_TEST(batch_n == 7);
    BOOST_TEST(n == 20);

    std::vector<int> desired = rg::to_vector(rgv::iota(0, 20));
    BOOST_TEST(result_unique == desired, boost::test_tools::per_element());
    BOOST_TEST(result_int == desired);
}


BOOST_AUTO_TEST_CASE(test_batch_larger_batches)
{
    // batch out of larger batches
    std::vector<hipipe::stream::batch_t> data = generate_regular_batched_data(3, 4);

    std::vector<hipipe::stream::batch_t> stream = data
      | rgv::move
      | hipipe::stream::rebatch(1)
      | rg::to_vector;

    // iterate through batches
    std::vector<int> result_unique;
    std::vector<int> result_int;
    int n = 0;
    for (hipipe::stream::batch_t& batch : stream) {
        // the batch should be only a single element
        BOOST_TEST(batch.batch_size() == 1U);
        BOOST_TEST(batch.extract<Int>().size() == 1U);
        BOOST_TEST(batch.extract<Unique>().size() == 1U);
        // remember the values
        result_unique.push_back(*batch.extract<Unique>().at(0));
        result_int.push_back(batch.extract<Int>().at(0));
        ++n;
    }
    BOOST_TEST(n == 12);

    std::vector<int> desired = rg::to_vector(rgv::iota(0, 12));
    BOOST_TEST(result_unique == desired);
    BOOST_TEST(result_int == desired);
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
    std::vector<hipipe::stream::batch_t> data = generate_batched_data({0, 0, 0, 0});
    std::vector<hipipe::stream::batch_t> stream = data
      | rgv::move
      | hipipe::stream::rebatch(1)
      | rg::to_vector;
    BOOST_TEST(stream.empty());
}


BOOST_AUTO_TEST_CASE(test_batch_empty_stream)
{
    // batch out of empty range
    std::vector<hipipe::stream::batch_t> data = generate_batched_data({});
    std::vector<hipipe::stream::batch_t> stream = data
      | rgv::move
      | hipipe::stream::rebatch(1)
      | rg::to_vector;
    BOOST_TEST(stream.empty());
}


BOOST_AUTO_TEST_CASE(test_infinite_batch)
{
    std::vector<hipipe::stream::batch_t> data = generate_regular_batched_data(3, 4);
    // make batch of infinite size
    auto stream = data
      | rgv::move
      | hipipe::stream::rebatch(std::numeric_limits<std::size_t>::max());
    auto stream_it = rg::begin(stream);
    static_assert(std::is_same_v<hipipe::stream::batch_t&&, decltype(*stream_it)>);
    hipipe::stream::batch_t result = *stream_it;
    std::vector<int> result_unique = result.extract<Unique>()
      | rgv::indirect | rg::to_vector;
    std::vector<int> result_int = result.extract<Int>();
    BOOST_CHECK(++stream_it == stream.end());
    BOOST_TEST(result_unique.size() == 12);
    BOOST_TEST(result_int.size() == 12);

    std::vector<int> desired = rg::to_vector(rgv::iota(0, 12));
    BOOST_TEST(result_unique == desired);
    BOOST_TEST(result_int == desired);
}
