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
#define BOOST_TEST_MODULE transform4_test

#include "common.hpp"

#include <hipipe/core/stream/random_fill.hpp>
#include <hipipe/core/stream/transform.hpp>


BOOST_AUTO_TEST_CASE(test_conditional_simple)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;
    using hipipe::stream::cond;

    HIPIPE_DEFINE_COLUMN(dogs, int)
    HIPIPE_DEFINE_COLUMN(do_trans, int)

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<dogs>(std::vector<int>{3, 1, 5});
    batch1.insert_or_assign<do_trans>(std::vector<int>{true, true, false});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<dogs>(std::vector<int>{7, 2, 13});
    batch2.insert_or_assign<do_trans>(std::vector<int>{false, true, false});
    data.push_back(std::move(batch2));

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::transform(from<dogs>, to<dogs>, cond<do_trans>,
          [](int dog) { return -1; }
        )
      | ranges::to_vector;

    BOOST_TEST(stream.size() == 2);
    BOOST_TEST(stream.at(0).extract<dogs>() == (std::vector<int>{-1, -1,  5}));
    BOOST_TEST(stream.at(1).extract<dogs>() == (std::vector<int>{ 7, -1, 13}));
}


BOOST_AUTO_TEST_CASE(test_conditional_with_random_fill)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::cond;

    HIPIPE_DEFINE_COLUMN(dogs, int)
    HIPIPE_DEFINE_COLUMN(do_trans, char)

    std::vector<std::vector<int>> expect_dogs = {{3, 1, 5}, {7, 2, 13}};

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<dogs>(expect_dogs.at(0));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<dogs>(expect_dogs.at(1));
    data.push_back(std::move(batch2));

    std::mt19937 prng{1000003};
    std::bernoulli_distribution dist{0.5};

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::random_fill(from<dogs>, to<do_trans>, 1, dist, prng)
      | hipipe::stream::transform(from<dogs>, to<dogs>, cond<do_trans>,
          [](int dog) { return dog - 1; }
        )
      | ranges::to_vector;


    long n_transformed = 0;
    BOOST_TEST(stream.size() == 2);
    for (std::size_t i = 0; i < stream.size(); ++i) {
        BOOST_TEST(stream.at(i).extract<dogs>().size() == 3);
        for (std::size_t j = 0; j < stream.at(i).extract<dogs>().size(); ++j) {
            // check that they differ by one and count the number of actually transformed examples
            if (expect_dogs.at(i).at(j) != stream.at(i).extract<dogs>().at(j)) {
                BOOST_TEST(expect_dogs.at(i).at(j) - 1 == stream.at(i).extract<dogs>().at(j));
                ++n_transformed;
            }
        }
    }

    BOOST_TEST(n_transformed >= 2);
    BOOST_TEST(n_transformed <= 4);
}


BOOST_AUTO_TEST_CASE(test_probabilistic_simple)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;

    HIPIPE_DEFINE_COLUMN(dogs, int)

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<dogs>(std::vector<int>{3, 1, 5});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<dogs>(std::vector<int>{7, 2, 13});
    data.push_back(std::move(batch2));

    std::mt19937 prng{1000003};

    std::vector<batch_t> stream = data
      | ranges::views::move
      | hipipe::stream::transform(from<dogs>, to<dogs>, 1.0, [](int dog) { return 1; }, prng)
      | hipipe::stream::transform(from<dogs>, to<dogs>, 0.5, [](int dog) { return 2; }, prng)
      | hipipe::stream::transform(from<dogs>, to<dogs>, 0.0, [](int dog) { return 3; }, prng)
      | ranges::to_vector;

    BOOST_CHECK(stream.size() == 2);
    long number1 = ranges::count(stream.at(0).extract<dogs>(), 1) +
                   ranges::count(stream.at(1).extract<dogs>(), 1);
    long number2 = ranges::count(stream.at(0).extract<dogs>(), 2) +
                   ranges::count(stream.at(1).extract<dogs>(), 2);
    long number3 = ranges::count(stream.at(0).extract<dogs>(), 3) +
                   ranges::count(stream.at(1).extract<dogs>(), 3);
    BOOST_TEST(number1 >= 1);
    BOOST_TEST(number1 <= 5);
    BOOST_TEST(number1 == 6 - number2);
    BOOST_TEST(number3 == 0);
}
