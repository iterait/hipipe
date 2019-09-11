/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

// The tests for stream::transform are split to multiple
// files to speed up compilation in case of multiple CPUs.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE transform3_test

#include "common.hpp"

#include <hipipe/core/stream/transform.hpp>


BOOST_AUTO_TEST_CASE(test_dim2_move_only_mutable)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<IntVec>();
    batch1.extract<IntVec>().resize(2);
    batch1.extract<IntVec>().at(0).push_back(2);
    batch1.extract<IntVec>().at(0).push_back(5);
    batch1.extract<IntVec>().at(1).push_back(4);
    batch1.extract<IntVec>().at(1).push_back(9);
    batch1.insert_or_assign<UniqueVec>();
    batch1.extract<UniqueVec>().resize(3);
    batch1.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(6));
    batch1.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(3));
    batch1.extract<UniqueVec>().at(1).push_back(std::make_unique<int>(7));
    batch1.extract<UniqueVec>().at(1).push_back(std::make_unique<int>(4));
    batch1.extract<UniqueVec>().at(2).push_back(std::make_unique<int>(2));
    batch1.extract<UniqueVec>().at(2).push_back(std::make_unique<int>(1));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<IntVec>();
    batch2.extract<IntVec>().resize(1);
    batch2.extract<IntVec>().at(0).push_back(8);
    batch2.extract<IntVec>().at(0).push_back(9);
    batch2.insert_or_assign<UniqueVec>();
    batch2.extract<UniqueVec>().resize(1);
    batch2.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(2));
    batch2.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(8));
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::transform(from<UniqueVec>, to<UniqueVec>,
          [i = 4](std::unique_ptr<int>&) mutable -> std::unique_ptr<int> {
              return std::make_unique<int>(i++);
        }, dim<2>)
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);

    // batch 1 //

    // int vector
    BOOST_TEST(stream.at(0).extract<IntVec>().size() == 2);
    BOOST_TEST(stream.at(0).extract<IntVec>().at(0)  == (std::vector<int>{2, 5}));
    BOOST_TEST(stream.at(0).extract<IntVec>().at(1)  == (std::vector<int>{4, 9}));

    // unique vector
    BOOST_TEST(stream.at(0).extract<UniqueVec>().size() == 3);
    // example 1
    BOOST_TEST(stream.at(0).extract<UniqueVec>().at(0).size()  == 2);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(0).at(0)  == 4);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(0).at(1)  == 5);
    // example 2
    BOOST_TEST(stream.at(0).extract<UniqueVec>().at(1).size()  == 2);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(1).at(0)  == 6);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(1).at(1)  == 7);
    // example 3
    BOOST_TEST(stream.at(0).extract<UniqueVec>().at(2).size()  == 2);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(2).at(0)  == 8);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(2).at(1)  == 9);

    // batch 2 //

    // int vector
    BOOST_TEST(stream.at(1).extract<IntVec>().size() == 1);
    BOOST_TEST(stream.at(1).extract<IntVec>().at(0)  == (std::vector<int>{8, 9}));
    // unique vector
    BOOST_TEST(stream.at(1).extract<UniqueVec>().size() == 1);
    // example 1
    BOOST_TEST(stream.at(1).extract<UniqueVec>().at(0).size()  == 2);
    BOOST_TEST(*stream.at(1).extract<UniqueVec>().at(0).at(0)  == 10);
    BOOST_TEST(*stream.at(1).extract<UniqueVec>().at(0).at(1)  == 11);
}
