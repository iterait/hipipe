/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE stream_generate_test

#include "../common.hpp"

#include <cxtream/core/stream/generate.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

using namespace cxtream::stream;
using namespace cxtream::utility;

CXTREAM_DEFINE_COLUMN(IntVec3d, std::vector<std::vector<int>>)
CXTREAM_DEFINE_COLUMN(Generated, std::vector<int>)
CXTREAM_DEFINE_COLUMN(Generated3d, std::vector<std::vector<std::vector<int>>>)

BOOST_AUTO_TEST_CASE(test_simple)
{
    std::vector<std::vector<std::vector<int>>> batch2 =
      {{{-1}, {-1}, {}}, {{-1}, {-1}}};
    std::vector<std::vector<std::vector<int>>> batch4 =
      {{{-1}, {-1}}, {}, {{-1}, {-1}}, {}};
    std::vector<std::tuple<IntVec3d>> data = {batch2, batch4};

    auto stream = data
      | generate(from<IntVec3d>, to<Generated>, [i = 0]() mutable {
            return i++;
        }, 1)
      | generate(from<IntVec3d>, to<Generated3d>, [i = 0]() mutable {
            return std::vector<int>{i++};
        }, 1);

    int batch_i = 0;
    for (auto batch : stream) {
        auto generated = std::get<Generated>(batch).value();
        auto generated3d = std::get<Generated3d>(batch).value();
        // check the contents
        switch (batch_i) {
        case 0: BOOST_CHECK((generated == Generated::batch_type
                                            {{0, 0, 0}, {1, 1}}));
                BOOST_CHECK((generated3d == Generated3d::batch_type
                                              {{{{0}}, {{0}}, {}}, {{{1}}, {{1}}}}));
                break;
        case 1: BOOST_CHECK((generated == Generated::batch_type
                                            {{2, 2}, {}, {4, 4}, {}}));
                BOOST_CHECK((generated3d == Generated3d::batch_type
                                              {{{{2}}, {{2}}}, {}, {{{4}}, {{4}}}, {}}));
                break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
        ++batch_i;
    }
}
