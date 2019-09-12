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
#define BOOST_TEST_MODULE stream_random_fill_test

#include "common.hpp"

#include <hipipe/core/stream/random_fill.hpp>
#include <hipipe/core/utility/ndim.hpp>

#include <range/v3/action/sort.hpp>
#include <range/v3/view/move.hpp>
#include <range/v3/view/unique.hpp>

#include <vector>


// Check that the given 2D vector has the given number of unique values in each subvector.
void check(std::vector<std::vector<double>> vec, std::vector<long> unique, long unique_total)
{
    for (std::size_t i = 0; i < vec.size(); ++i) {
        vec.at(i) |= rga::sort;
        long n_unique = ranges::distance(vec.at(i) | rgv::unique);
        BOOST_TEST(n_unique == unique.at(i));
    }

    std::vector<double> all_vals = ranges::to_vector(hipipe::utility::flat_view(vec));
    all_vals |= rga::sort;
    long n_unique = ranges::distance(all_vals | rgv::unique);
    BOOST_TEST(n_unique == unique_total);
}


BOOST_AUTO_TEST_CASE(test_simple)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    HIPIPE_DEFINE_COLUMN(IntVec2d, std::vector<std::vector<int>>)
    HIPIPE_DEFINE_COLUMN(Random, std::vector<double>)

    std::mt19937 gen{1000003};
    std::uniform_real_distribution<> dist{0, 1};

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<IntVec2d>();
    batch1.extract<IntVec2d>().push_back(IntVec2d::example_type{{}, {}    });
    batch1.extract<IntVec2d>().push_back(IntVec2d::example_type{{}, {}, {}});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<IntVec2d>();
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{{}, {}    });
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{          });
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{{}, {}    });
    batch2.extract<IntVec2d>().push_back(IntVec2d::example_type{          });
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | rgv::move
      | random_fill(from<IntVec2d>, to<Random>, 1, dist, gen)
      | ranges::to_vector;

    std::vector<std::vector<double>> all_random;
    for (std::size_t i = 0; i < stream.size(); ++i) {
        std::vector<std::vector<double>> random = stream.at(i).extract<Random>();
        all_random.insert(all_random.end(), random.begin(), random.end());
        switch (i) {
        case 0: check(random, {1, 1}, 2); break;
        case 1: check(random, {1, 0, 1, 0}, 2); break;
        default: BOOST_FAIL("Only two batches should be provided");
        }
    }
    check(all_random, {1, 1, 1, 0, 1, 0}, 4);
}
