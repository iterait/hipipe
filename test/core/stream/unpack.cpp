/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE unpack_test

#include "../common.hpp"

#include <cxtream/core/stream/create.hpp>
#include <cxtream/core/stream/unpack.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>

#include <memory>
#include <tuple>
#include <vector>

using namespace cxtream::stream;

std::vector<std::tuple<int, std::vector<int>>> generate_data()
{
    return {{3, {1, 4}}, {1, {8, 2}}, {7, {2, 5}}};
}

BOOST_AUTO_TEST_CASE(test_unpack_dim0)
{
    auto data = generate_data();
    auto rng = data | create<Int, IntVec>(2);
    std::vector<std::vector<int>> unp_int;
    std::vector<std::vector<std::vector<int>>> unp_intvec;
    std::tie(unp_int, unp_intvec) = unpack(rng, from<Int, IntVec>, dim<0>);
    test_ranges_equal(unp_int, std::vector<std::vector<int>>{{3, 1}, {7}});
    test_ranges_equal(unp_intvec, std::vector<std::vector<std::vector<int>>>{
      {{1, 4}, {8, 2}}, {{2, 5}}
    });
}

BOOST_AUTO_TEST_CASE(test_unpack_dim1)
{
    auto data = generate_data();
    auto data_original = data;
    auto rng = data | create<Int, IntVec>(2);
    std::vector<int> unp_int;
    std::vector<std::vector<int>> unp_intvec;
    std::tie(unp_int, unp_intvec) = unpack(rng, from<Int, IntVec>);
    BOOST_CHECK(data == data_original);
    test_ranges_equal(unp_int, std::vector<int>{3, 1, 7});
    test_ranges_equal(unp_intvec, std::vector<std::vector<int>>{{1, 4}, {8, 2}, {2, 5}});
}

BOOST_AUTO_TEST_CASE(test_unpack_dim2)
{
    const auto data = generate_data();
    auto rng = data | create<Int, IntVec>(2);
    std::vector<int> unp_intvec;
    unp_intvec = unpack(rng, from<IntVec>, dim<2>);
    test_ranges_equal(unp_intvec, std::vector<int>{1, 4, 8, 2, 2, 5});
}

BOOST_AUTO_TEST_CASE(test_unpack_dim2_move_only)
{
    auto data = generate_move_only_data();
    auto rng = data | ranges::view::move | create<Int, UniqueVec>(2);

    std::vector<std::unique_ptr<int>> unp_uniquevec;
    unp_uniquevec = unpack(rng, from<UniqueVec>, dim<2>);
    std::vector<int> values = unp_uniquevec | ranges::view::indirect;
    test_ranges_equal(values, std::vector<int>{1, 4, 8, 2, 2, 5});
}
