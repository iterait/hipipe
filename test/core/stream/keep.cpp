/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE column_keep_test

#include "common.hpp"

#include <hipipe/core/stream/keep.hpp>

HIPIPE_DEFINE_COLUMN(Unique2, std::unique_ptr<int>)


BOOST_AUTO_TEST_CASE(test_keep_int_column)
{
    using hipipe::stream::batch_t;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    batch1.insert_or_assign<Double>(5.);
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    batch2.insert_or_assign<Double>(2.);
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | hipipe::stream::keep<Int>
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).size() == 1);
    BOOST_TEST(!stream.at(0).contains<Double>());
    BOOST_TEST(stream.at(0).contains<Int>());
    BOOST_TEST(stream.at(0).extract<Int>() == (std::vector<int>{3}));
    BOOST_TEST(stream.at(1).size() == 1);
    BOOST_TEST(!stream.at(1).contains<Double>());
    BOOST_TEST(stream.at(1).contains<Int>());
    BOOST_TEST(stream.at(1).extract<Int>() == (std::vector<int>{1}));

}


BOOST_AUTO_TEST_CASE(test_keep_unique_column)
{
    using hipipe::stream::batch_t;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(1));
    batch1.insert_or_assign<Unique2>();
    batch1.extract<Unique2>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    batch2.insert_or_assign<Unique2>();
    batch2.extract<Unique2>().push_back(std::make_unique<int>(6));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | hipipe::stream::keep<Unique>
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).size() == 1);
    BOOST_TEST(stream.at(0).contains<Unique>());
    BOOST_TEST(!stream.at(0).contains<Unique2>());
    BOOST_TEST(stream.at(0).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(0).extract<Unique>().at(0) == 1);
    BOOST_TEST(stream.at(1).size() == 1);
    BOOST_TEST(stream.at(1).contains<Unique>());
    BOOST_TEST(!stream.at(1).contains<Unique2>());
    BOOST_TEST(stream.at(1).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(1).extract<Unique>().at(0) == 2);
}


BOOST_AUTO_TEST_CASE(test_keep_multiple_columns)
{
    using hipipe::stream::batch_t;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>();
    batch1.extract<Int>().push_back(7);
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(1));
    batch1.insert_or_assign<Unique2>();
    batch1.extract<Unique2>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>();
    batch2.extract<Int>().push_back(4);
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    batch2.insert_or_assign<Unique2>();
    batch2.extract<Unique2>().push_back(std::make_unique<int>(6));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | hipipe::stream::keep<Int, Unique>
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).size() == 2);
    BOOST_TEST(stream.at(0).contains<Int>());
    BOOST_TEST(stream.at(0).contains<Unique>());
    BOOST_TEST(!stream.at(0).contains<Unique2>());
    BOOST_TEST(stream.at(0).extract<Int>().size() == 1);
    BOOST_TEST(stream.at(0).extract<Int>().at(0) == 7);
    BOOST_TEST(stream.at(0).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(0).extract<Unique>().at(0) == 1);
    BOOST_TEST(stream.at(1).size() == 2);
    BOOST_TEST(stream.at(1).contains<Int>());
    BOOST_TEST(stream.at(1).contains<Unique>());
    BOOST_TEST(!stream.at(1).contains<Unique2>());
    BOOST_TEST(stream.at(1).extract<Int>().size() == 1);
    BOOST_TEST(stream.at(1).extract<Int>().at(0) == 4);
    BOOST_TEST(stream.at(1).extract<Unique>().size() == 1);
    BOOST_TEST(*stream.at(1).extract<Unique>().at(0) == 2);
}
