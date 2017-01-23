/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef TEST_STREAM_TRANSFORM_HPP
#define TEST_STREAM_TRANSFORM_HPP

#include "../common.hpp"

#include <cxtream/core/stream/create.hpp>
#include <cxtream/core/stream/drop.hpp>
#include <cxtream/core/stream/transform.hpp>
#include <cxtream/core/stream/unpack.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/to_container.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>
#include <range/v3/view/zip.hpp>

#include <memory>
#include <random>
#include <tuple>
#include <vector>

// test with a seeded random generator
std::mt19937 prng{1000033};

auto unique_vec_to_int_vec()
{
    using namespace cxtream::stream;
    return
        transform(from<UniqueVec>, to<IntVec>, [](auto&& ptrs) {
            return ptrs | ranges::view::indirect;
        }, dim<1>)
      | drop<UniqueVec>;
}

#endif
