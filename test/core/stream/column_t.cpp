/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE stream_column_t_test

#include "common.hpp"

#include <hipipe/core/stream/column_t.hpp>

#include <boost/test/unit_test.hpp>

// TODO add some tests!
// Don't forget:
// - Read doc/mainpage.md and doc/installation.md.
// - See doc/installation.md for instructions how to run one or all unit tests.
// - Test all the public functions.
// - Fix the docstrings if they are wrong.
// - common.hpp contains multiple predefined columns.
// - Look into other tests if unsure how to proceed, e.g. for_each.cpp.
// - Test whether move-only (e.g., std::unique_ptr) values work as well.
// - Creating move-only data can sometimes be tricky,
//   see e.g. for_each.cpp:test_for_each_move_only.
// - Add yourself to the list of authors.
// - Remove this comment.
// - Make PR targeting into branch `runtime`.

BOOST_AUTO_TEST_CASE(test_extract_column)
{
    using hipipe::stream::abstract_column;

    std::unique_ptr<Int> col = std::make_unique<Int>();
    col->data().assign({1, 2, 3});
    BOOST_TEST(col->extract<Int>() == std::vector<int>({1, 2, 3}));

    std::unique_ptr<abstract_column> ab_col = std::move(col);
    BOOST_TEST(ab_col->extract<Int>() == std::vector<int>({1, 2, 3}));
}


BOOST_AUTO_TEST_CASE(test_extract_move_only_column)
{
    using hipipe::stream::abstract_column;

    std::unique_ptr<Unique> col = std::make_unique<Unique>();
    col->data().push_back(std::make_unique<int>(1));
    col->data().push_back(std::make_unique<int>(2));
    col->data().push_back(std::make_unique<int>(3));
    BOOST_TEST(*(col->extract<Unique>().at(0)) == 1);
    BOOST_TEST(*(col->extract<Unique>().at(1)) == 2);
    BOOST_TEST(*(col->extract<Unique>().at(2)) == 3);

    std::unique_ptr<abstract_column> ab_col = std::move(col);
    BOOST_TEST(*(ab_col->extract<Unique>().at(0)) == 1);
    BOOST_TEST(*(ab_col->extract<Unique>().at(1)) == 2);
    BOOST_TEST(*(ab_col->extract<Unique>().at(2)) == 3);
}


BOOST_AUTO_TEST_CASE(test_extract_throws_error)
{
    using hipipe::stream::abstract_column;

    std::unique_ptr<Int> col = std::make_unique<Int>();
    col->data().assign({1, 2, 3});
    BOOST_CHECK_THROW(col->extract<Double>(), std::runtime_error);

    std::unique_ptr<abstract_column> ab_col = std::move(col);
    BOOST_CHECK_THROW(ab_col->extract<Double>(), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(test_get_column_size)
{
    Int col1;
    col1.data().assign({1, 2, 3});
    Unique col2;
    BOOST_TEST(col1.size() == 3);
    BOOST_TEST(col2.size() == 0);
}


BOOST_AUTO_TEST_CASE(test_take_column)
{
    using hipipe::stream::abstract_column;
    
    std::unique_ptr<Int> col1 = std::make_unique<Int>();
    col1->data().assign({1, 2, 3, 4, 5});
    std::unique_ptr<abstract_column> col2 = col1->take(2);
    BOOST_TEST(col1->extract<Int>() == std::vector<int>({3, 4, 5}));
    BOOST_TEST(col2->extract<Int>() == std::vector<int>({1, 2}));
}


BOOST_AUTO_TEST_CASE(test_take_move_only_column)
{
    using hipipe::stream::abstract_column;
    
    std::unique_ptr<Unique> col1 = std::make_unique<Unique>();
    col1->data().push_back(std::make_unique<int>(1));
    col1->data().push_back(std::make_unique<int>(2));
    col1->data().push_back(std::make_unique<int>(3));
    std::unique_ptr<abstract_column> col2 = col1->take(2);
    
    BOOST_TEST(col1->size() == 1);
    BOOST_TEST(col2->size() == 2);
    BOOST_TEST(*(col1->extract<Unique>().at(0)) == 3);
    BOOST_TEST(*(col2->extract<Unique>().at(0)) == 1);
    BOOST_TEST(*(col2->extract<Unique>().at(1)) == 2);
}


BOOST_AUTO_TEST_CASE(test_take_throws_error)
{
    using hipipe::stream::abstract_column;

    std::unique_ptr<Int> col1 = std::make_unique<Int>();
    col1->data().assign({1, 2, 3, 4, 5});
    BOOST_CHECK_THROW(col1->take(10), std::runtime_error);
}


BOOST_AUTO_TEST_CASE(test_push_back_column)
{
    using hipipe::stream::abstract_column;
    
    std::unique_ptr<Int> col1 = std::make_unique<Int>();
    col1->data().assign({1, 2, 3, 4, 5});
    std::unique_ptr<Int> col2 = std::make_unique<Int>();
    col2->data().assign({});
    std::unique_ptr<Int> col3 = std::make_unique<Int>();
    col3->data().assign({6, 7});

    col1->push_back(std::move(col2));
    BOOST_TEST(col1->extract<Int>() == std::vector<int>({1, 2, 3, 4, 5}));
    col1->push_back(std::move(col3));
    BOOST_TEST(col1->extract<Int>() == std::vector<int>({1, 2, 3, 4, 5, 6, 7}));
}


BOOST_AUTO_TEST_CASE(test_push_back_only_column)
{
    using hipipe::stream::abstract_column;
    
    std::unique_ptr<Unique> col1 = std::make_unique<Unique>();
    col1->data().push_back(std::make_unique<int>(1));
    col1->data().push_back(std::make_unique<int>(2));
    std::unique_ptr<Unique> col2 = std::make_unique<Unique>();
    col2->data().push_back(std::make_unique<int>(3));
    col1->push_back(std::move(col2));

    BOOST_TEST(col1->size() == 3);
    BOOST_TEST(*(col1->extract<Unique>().at(0)) == 1);
    BOOST_TEST(*(col1->extract<Unique>().at(1)) == 2);
    BOOST_TEST(*(col1->extract<Unique>().at(2)) == 3);
}