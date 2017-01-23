/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE stream_random_fill_test

#include "../common.hpp"

#include <cxtream/core/stream/create.hpp>
#include <cxtream/core/stream/random_fill.hpp>
#include <cxtream/core/utility/vector.hpp>

#include <range/v3/action/sort.hpp>
#include <range/v3/view/unique.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

using namespace cxtream::stream;
using namespace cxtream::utility;

CXTREAM_DEFINE_COLUMN(IntVec2d, std::vector<std::vector<int>>)
CXTREAM_DEFINE_COLUMN(Random, std::vector<double>)

template<typename Vector2d>
void check(Vector2d vec, std::vector<long> unique, long unique_total)
{
    for (std::size_t i = 0; i < vec.size(); ++i) {
        vec.at(i) |= ranges::action::sort;
        auto n_unique = ranges::distance(vec.at(i) | ranges::view::unique);
        BOOST_TEST(n_unique == unique.at(i));
    }

    std::vector<double> all_vals = flat_view(vec);
    all_vals |= ranges::action::sort;
    auto n_unique = ranges::distance(all_vals | ranges::view::unique);
    BOOST_TEST(n_unique == unique_total);
}

BOOST_AUTO_TEST_CASE(test_simple)
{
    std::mt19937 gen{1000003};
    std::uniform_real_distribution<> dist{0, 1};
    std::vector<std::vector<std::vector<int>>> batch2 =
      {{{}, {}}, {{}, {}, {}}};
    std::vector<std::vector<std::vector<int>>> batch4 =
      {{{}, {}}, {}, {{}, {}}, {}};
    std::vector<std::tuple<IntVec2d>> data = {batch2, batch4};

    auto stream = data
      | random_fill(from<IntVec2d>, to<Random>, 1, dist, gen);

    int batch_i = 0;
    std::vector<std::vector<double>> all_random;
    for (auto batch : stream) {
        auto random = std::get<Random>(batch).value();
        switch (batch_i) {
        case 0: check(random, {1, 1}, 2); break;
        case 1: check(random, {1, 0, 1, 0}, 2); break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
        all_random.insert(all_random.end(), random.begin(), random.end());
        ++batch_i;
    }
    check(all_random, {1, 1, 1, 0, 1, 0}, 4);
}
