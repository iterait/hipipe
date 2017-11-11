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
#define BOOST_TEST_MODULE transform4_test

#include "transform.hpp"

#include <cxtream/core/stream/random_fill.hpp>

using namespace cxtream::stream;

BOOST_AUTO_TEST_CASE(test_conditional_simple)
{
    CXTREAM_DEFINE_COLUMN(dogs, int)
    CXTREAM_DEFINE_COLUMN(do_trans, int)
    std::vector<int> data_int = {3, 1, 5, 7, 2, 13};
    std::vector<int> data_cond = {true, true, false, false, true, false};
    auto rng = ranges::view::zip(data_int, data_cond)
      | create<dogs, do_trans>()
      | transform(from<dogs>, to<dogs>, cond<do_trans>, [](int dog) { return -1; });

    std::vector<int> generated = unpack(rng, from<dogs>, dim<1>);
    test_ranges_equal(generated, std::vector<int>{-1, -1, 5, 7, -1, 13});
}

BOOST_AUTO_TEST_CASE(test_conditional_with_random_fill)
{
    CXTREAM_DEFINE_COLUMN(dogs, int)
    CXTREAM_DEFINE_COLUMN(do_trans, char)
    const std::vector<int> data_int = {3, 1, 5, 7, 2, 13};
    std::bernoulli_distribution dist{0.5};
    auto rng = data_int
      | create<dogs>()
      | random_fill(from<dogs>, to<do_trans>, 1, dist, prng)
      | transform(from<dogs>, to<dogs>, cond<do_trans>, [](int dog) { return dog - 1; });

    std::vector<int> generated = unpack(rng, from<dogs>, dim<1>);
    long n_transformed = 0;
    BOOST_TEST(generated.size() == 6);
    for (std::size_t i = 0; i < generated.size(); ++i) {
        // check that they differ by one and count the number of actually transformed examples
        if (data_int[i] != generated[i]) {
            BOOST_TEST(data_int[i] - 1 == generated[i]);
            ++n_transformed;
        }
    }
    BOOST_TEST(n_transformed >= 2);
    BOOST_TEST(n_transformed <= 4);
}

BOOST_AUTO_TEST_CASE(test_probabilistic_simple)
{
    CXTREAM_DEFINE_COLUMN(dogs, int)
    std::vector<int> data = {3, 1, 5, 7, 2, 13};
    auto rng = data
      | create<dogs>()
      | transform(from<dogs>, to<dogs>, 1.0, [](int dog) { return 1; }, prng)
      | transform(from<dogs>, to<dogs>, 0.5, [](int dog) { return 2; }, prng)
      | transform(from<dogs>, to<dogs>, 0.0, [](int dog) { return 3; }, prng);

    std::vector<int> generated = unpack(rng, from<dogs>, dim<1>);
    BOOST_CHECK(generated.size() == 6);
    long number1 = ranges::count(generated, 1);
    long number2 = ranges::count(generated, 2);
    long number3 = ranges::count(generated, 3);
    BOOST_TEST(number1 >= 1);
    BOOST_TEST(number1 <= 5);
    BOOST_TEST(number1 == 6 - number2);
    BOOST_TEST(number3 == 0);
}
