/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

// The tests for stream::transform are split to multiple
// files to speed up compilation in case of multiple CPUs.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE transform2_test

#include "transform.hpp"

using namespace cxtream::stream;

BOOST_AUTO_TEST_CASE(test_dim2_move_only)
{
    auto data = generate_move_only_data();

    auto rng = data
      | ranges::view::move
      | create<Int, UniqueVec>(2)
      | transform(from<UniqueVec>, to<UniqueVec>, [](std::unique_ptr<int>& ptr) {
            return std::make_unique<int>(*ptr + 1);
        }, dim<2>)
      | drop<Int>
      | unique_vec_to_int_vec();

    std::vector<std::vector<std::vector<int>>> generated = unpack(rng, from<IntVec>, dim<0>);
    std::vector<std::vector<std::vector<int>>> desired = {{{2, 5}, {9, 3}}, {{3, 6}}};
    BOOST_CHECK(generated == desired);
}
