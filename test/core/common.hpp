/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/build_config.hpp>

#include <boost/test/unit_test.hpp>

#include <range/v3/core.hpp>

namespace rg = ranges;
namespace rga = ranges::actions;
namespace rgv = ranges::views;

template<typename Rng1, typename Rng2>
void test_ranges_equal(Rng1&& rng1, Rng2&& rng2)
{
    // using this function, ranges with different
    // begin() and end() types can be compared
    auto it1 = rg::begin(rng1);
    auto it2 = rg::begin(rng2);
    while (it1 != rg::end(rng1) && it2 != rg::end(rng2)) {
        BOOST_TEST(*it1 == *it2);
        ++it1;
        ++it2;
    }
    BOOST_CHECK(it1 == rg::end(rng1));
    BOOST_CHECK(it2 == rg::end(rng2));
}
