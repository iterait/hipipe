/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner, Jana Horecka
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE stream_batch_t_test

#include "common.hpp"

#include <hipipe/core/stream/batch_t.hpp>


BOOST_AUTO_TEST_CASE(test_batch_insert_assign_extract)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    batch.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    BOOST_TEST(batch.extract<Int>() == std::vector<int>({0, 1, 2}));
    batch.insert_or_assign<Int>(std::vector<int>{2});
    BOOST_TEST(batch.extract<Int>() == std::vector<int>({2}));
    batch.insert_or_assign<Int>(std::vector<int>{});
    BOOST_TEST(batch.extract<Int>() == std::vector<int>({}));
    BOOST_CHECK_THROW(batch.extract<Double>(), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(test_batch_const_insert_assign_extract)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    batch.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    const batch_t& c_batch = batch;
    BOOST_TEST(c_batch.extract<Int>() == std::vector<int>({0, 1, 2}));
}


BOOST_AUTO_TEST_CASE(test_batch_insert_assign_extract_move_only)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    batch.insert_or_assign<Unique>();
    batch.extract<Unique>().push_back(std::make_unique<int>(1));
    batch.extract<Unique>().push_back(std::make_unique<int>(2));
    BOOST_TEST(*batch.extract<Unique>().at(0) == 1);
    BOOST_TEST(*batch.extract<Unique>().at(1) == 2);
    batch.insert_or_assign<Unique>();
    BOOST_CHECK_THROW(*batch.extract<Unique>().at(0), std::out_of_range);
}    


BOOST_AUTO_TEST_CASE(test_batch_get_size)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    BOOST_TEST(batch.size() == 0);
    batch.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    batch.insert_or_assign<Double>(std::vector<double>{0., 1.});
    BOOST_TEST(batch.size() == 2);
}


BOOST_AUTO_TEST_CASE(test_batch_contains_column)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    batch.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    BOOST_TEST(batch.contains<Int>() == true);
    BOOST_TEST(batch.contains<Double>() == false);
}


BOOST_AUTO_TEST_CASE(test_batch_erase_column)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    batch.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    BOOST_TEST(batch.size() == 1);
    batch.erase<Int>();
    BOOST_TEST(batch.size() == 0);
    BOOST_CHECK_THROW(batch.erase<Int>(), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(test_batch_get_batch_size)
{
    using hipipe::stream::batch_t;

    batch_t batch;
    BOOST_TEST(batch.batch_size() == 0);
    batch.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    batch.insert_or_assign<Double>(std::vector<double>{0., 1., 2.});
    BOOST_TEST(batch.batch_size() == 3);
    batch.insert_or_assign<Double>(std::vector<double>{0.});
    BOOST_CHECK_THROW(batch.batch_size(), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(test_batch_take)
{
    using hipipe::stream::batch_t;

    batch_t batch1;
    batch1.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    batch1.insert_or_assign<Double>(std::vector<double>{0., 1., 2.});
    batch_t batch2 = batch1.take(2);
    BOOST_TEST(batch1.extract<Int>() == std::vector<int>({2}));
    BOOST_TEST(batch1.extract<Double>() == std::vector<double>({2.}));
    BOOST_TEST(batch2.extract<Int>() == std::vector<int>({0, 1}));
    BOOST_TEST(batch2.extract<Double>() == std::vector<double>({0., 1.}));
    batch_t batch3 = batch1.take(1);
    BOOST_TEST(batch1.extract<Int>() == std::vector<int>({}));
    BOOST_TEST(batch1.extract<Double>() == std::vector<double>({}));
    BOOST_TEST(batch3.extract<Int>() == std::vector<int>({2}));
    BOOST_TEST(batch3.extract<Double>() == std::vector<double>({2.}));
}


BOOST_AUTO_TEST_CASE(test_batch_take_move_only)
{
    using hipipe::stream::batch_t;

    batch_t batch1;
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(1));
    batch1.extract<Unique>().push_back(std::make_unique<int>(2));
    batch_t batch2 = batch1.take(1);
    BOOST_TEST(batch1.batch_size() == 1);
    BOOST_TEST(batch2.batch_size() == 1);
    BOOST_TEST(*batch1.extract<Unique>().at(0) == 2);
    BOOST_TEST(*batch2.extract<Unique>().at(0) == 1);
}


BOOST_AUTO_TEST_CASE(test_batch_take_throws_error)
{
    using hipipe::stream::batch_t;

    batch_t batch1;
    batch1.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    batch1.insert_or_assign<Double>(std::vector<double>{0., 1., 2.});
    BOOST_CHECK_THROW(batch1.take(4), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(test_batch_push_back)
{
    using hipipe::stream::batch_t;

    batch_t batch1;
    batch1.insert_or_assign<Int>(std::vector<int>{0, 1, 2});
    batch_t batch2;
    batch2.insert_or_assign<Int>(std::vector<int>{3, 4});
    batch2.insert_or_assign<Double>(std::vector<double>{0., 1., 2.});
    batch_t batch3;
    batch1.push_back(std::move(batch2));
    BOOST_TEST(batch1.extract<Int>() == std::vector<int>({0, 1, 2, 3, 4}));
    BOOST_TEST(batch1.extract<Double>() == std::vector<double>({0., 1., 2.}));
    batch1.push_back(std::move(batch3));
    BOOST_TEST(batch1.extract<Int>() == std::vector<int>({0, 1, 2, 3, 4}));
    BOOST_TEST(batch1.extract<Double>() == std::vector<double>({0., 1., 2.}));
}


BOOST_AUTO_TEST_CASE(test_batch_push_back_move_only)
{
    using hipipe::stream::batch_t;

    batch_t batch1;
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(1));
    batch1.extract<Unique>().push_back(std::make_unique<int>(2));
    batch_t batch2;
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(3));
    batch1.push_back(std::move(batch2));
    BOOST_TEST(batch1.batch_size() == 3);
    BOOST_TEST(batch2.batch_size() == 0);
    BOOST_TEST(*batch1.extract<Unique>().at(0) == 1);
    BOOST_TEST(*batch1.extract<Unique>().at(1) == 2);
    BOOST_TEST(*batch1.extract<Unique>().at(2) == 3);
}
