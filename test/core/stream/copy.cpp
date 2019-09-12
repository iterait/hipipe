/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE column_copy_test

#include "common.hpp"

#include <hipipe/core/stream/copy.hpp>

HIPIPE_DEFINE_COLUMN(Int2, int)
HIPIPE_DEFINE_COLUMN(Int3, int)
HIPIPE_DEFINE_COLUMN(Int4, int)
HIPIPE_DEFINE_COLUMN(Long, long)


BOOST_AUTO_TEST_CASE(test_simple_copy)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::copy;
    using hipipe::stream::transform;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | copy(from<Int>, to<Int2>)
      | transform(from<Int>, to<Int>, [](int a) {return a+1;})
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).size() == 2);
    BOOST_TEST(stream.at(0).contains<Int>());
    BOOST_TEST(stream.at(0).contains<Int2>());
    BOOST_TEST(stream.at(0).extract<Int>() == (std::vector<int>{4}));
    BOOST_TEST(stream.at(0).extract<Int2>() == (std::vector<int>{3}));
    BOOST_TEST(stream.at(1).size() == 2);
    BOOST_TEST(stream.at(1).contains<Int>());
    BOOST_TEST(stream.at(1).contains<Int2>());
    BOOST_TEST(stream.at(1).extract<Int>() == (std::vector<int>{2}));
    BOOST_TEST(stream.at(1).extract<Int2>() == (std::vector<int>{1}));
}


BOOST_AUTO_TEST_CASE(test_multi_copy)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::copy;
    using hipipe::stream::transform;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | copy(from<Int>, to<Int2>)
      | transform(from<Int2>, to<Int2>, [](int a) {return a+1;})
      | copy(from<Int, Int2>, to<Int3, Int4>)
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).size() == 4);
    BOOST_TEST(stream.at(0).contains<Int>());
    BOOST_TEST(stream.at(0).contains<Int2>());
    BOOST_TEST(stream.at(0).contains<Int3>());
    BOOST_TEST(stream.at(0).contains<Int4>());
    BOOST_TEST(stream.at(0).extract<Int>() == (std::vector<int>{3}));
    BOOST_TEST(stream.at(0).extract<Int2>() == (std::vector<int>{4}));
    BOOST_TEST(stream.at(0).extract<Int3>() == (std::vector<int>{3}));
    BOOST_TEST(stream.at(0).extract<Int4>() == (std::vector<int>{4}));
    BOOST_TEST(stream.at(1).size() == 4);
    BOOST_TEST(stream.at(1).contains<Int>());
    BOOST_TEST(stream.at(1).contains<Int2>());
    BOOST_TEST(stream.at(1).contains<Int3>());
    BOOST_TEST(stream.at(1).contains<Int4>());
    BOOST_TEST(stream.at(1).extract<Int>() == (std::vector<int>{1}));
    BOOST_TEST(stream.at(1).extract<Int2>() == (std::vector<int>{2}));
    BOOST_TEST(stream.at(1).extract<Int3>() == (std::vector<int>{1}));
    BOOST_TEST(stream.at(1).extract<Int4>() == (std::vector<int>{2}));
}


BOOST_AUTO_TEST_CASE(test_constructible_copy)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::copy;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | copy(from<Int>, to<Long>)
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).size() == 2);
    BOOST_TEST(stream.at(0).contains<Int>());
    BOOST_TEST(stream.at(0).contains<Long>());
    BOOST_TEST(stream.at(0).extract<Int>() == (std::vector<int>{3}));
    BOOST_TEST(stream.at(0).extract<Long>() == (std::vector<long>{3L}));
}
