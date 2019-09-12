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
#define BOOST_TEST_MODULE transform2_test

#include "common.hpp"

#include <hipipe/core/stream/transform.hpp>


BOOST_AUTO_TEST_CASE(test_dim2_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_move_only_data_2d();

    std::vector<batch_t> stream = data
      | rgv::move
      | hipipe::stream::transform(from<UniqueVec>, to<UniqueVec>,
          [](std::unique_ptr<int>& ptr) -> std::unique_ptr<int> {
              return std::make_unique<int>(*ptr + 1);
        }, dim<2>)
      | rg::to_vector;

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
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(0).at(0)  == 7);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(0).at(1)  == 4);
    // example 2
    BOOST_TEST(stream.at(0).extract<UniqueVec>().at(1).size()  == 2);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(1).at(0)  == 8);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(1).at(1)  == 5);
    // example 3
    BOOST_TEST(stream.at(0).extract<UniqueVec>().at(2).size()  == 2);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(2).at(0)  == 3);
    BOOST_TEST(*stream.at(0).extract<UniqueVec>().at(2).at(1)  == 2);

    // batch 2 //

    // int vector
    BOOST_TEST(stream.at(1).extract<IntVec>().size() == 1);
    BOOST_TEST(stream.at(1).extract<IntVec>().at(0)  == (std::vector<int>{8, 9}));
    // unique vector
    BOOST_TEST(stream.at(1).extract<UniqueVec>().size() == 1);
    // example 1
    BOOST_TEST(stream.at(1).extract<UniqueVec>().at(0).size()  == 2);
    BOOST_TEST(*stream.at(1).extract<UniqueVec>().at(0).at(0)  == 3);
    BOOST_TEST(*stream.at(1).extract<UniqueVec>().at(0).at(1)  == 9);
}
