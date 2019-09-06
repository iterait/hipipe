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
#define BOOST_TEST_MODULE unpack_test

#include "common.hpp"

#include <hipipe/core/stream/unpack.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>

#include <memory>
#include <tuple>
#include <vector>


std::vector<hipipe::stream::batch_t> generate_data()
{
    using hipipe::stream::batch_t;
    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>();
    batch1.extract<Int>().push_back(3);
    batch1.extract<Int>().push_back(1);
    batch1.insert_or_assign<IntVec>();
    batch1.extract<IntVec>().push_back({1, 4});
    batch1.extract<IntVec>().push_back({8, 2});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>();
    batch2.extract<Int>().push_back(7);
    batch2.insert_or_assign<IntVec>();
    batch2.extract<IntVec>().push_back({2, 5});
    data.push_back(std::move(batch2));
    return data;
}

BOOST_AUTO_TEST_CASE(test_unpack_dim0)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_data();
    std::vector<std::vector<int>> unp_int;
    std::vector<std::vector<std::vector<int>>> unp_intvec;
    std::tie(unp_int, unp_intvec) =
      hipipe::stream::unpack(ranges::views::move(data), from<Int, IntVec>, dim<0>);
    BOOST_TEST(unp_int == (std::vector<std::vector<int>>{{3, 1}, {7}}));
    BOOST_TEST(unp_intvec == (std::vector<std::vector<std::vector<int>>>{
      {{1, 4}, {8, 2}}, {{2, 5}}
    }));
}


BOOST_AUTO_TEST_CASE(test_unpack_dim1)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_data();
    std::vector<int> unp_int;
    std::vector<std::vector<int>> unp_intvec;
    std::tie(unp_int, unp_intvec) =
      hipipe::stream::unpack(ranges::views::move(data), from<Int, IntVec>);
    BOOST_TEST(unp_int == (std::vector<int>{3, 1, 7}));
    BOOST_TEST(unp_intvec == (std::vector<std::vector<int>>{{1, 4}, {8, 2}, {2, 5}}));
}


BOOST_AUTO_TEST_CASE(test_unpack_dim2)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_data();
    std::vector<int> unp_intvec;
    unp_intvec = hipipe::stream::unpack(ranges::views::move(data), from<IntVec>, dim<2>);
    BOOST_TEST(unp_intvec == (std::vector<int>{1, 4, 8, 2, 2, 5}));
}


BOOST_AUTO_TEST_CASE(test_unpack_dim2_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::dim;

    std::vector<batch_t> data = generate_move_only_data_2d();
    std::vector<std::unique_ptr<int>> unp_uniquevec;
    unp_uniquevec = hipipe::stream::unpack(ranges::views::move(data), from<UniqueVec>, dim<2>);
    std::vector<int> values = ranges::views::indirect(unp_uniquevec);
    BOOST_TEST(values == (std::vector<int>{6, 3, 7, 4, 2, 1, 2, 8}));
}
