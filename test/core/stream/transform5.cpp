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
#define BOOST_TEST_MODULE transform5_test

#include "transform.hpp"

#include <hipipe/core/utility/vector.hpp>


BOOST_AUTO_TEST_CASE(test_probabilistic_dim2_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::dim;
    using hipipe::stream::cond;

    std::vector<batch_t> data = generate_move_only_data_2d();
    std::mt19937 prng{1000003};

    std::vector<batch_t> stream = data
      | ranges::view::move
      // create IntVec column
      | hipipe::stream::transform(from<UniqueVec>, to<IntVec>,
         [](std::unique_ptr<int>&) -> int {
             return 7;
         }, dim<2>)
      // probabilistically transform a single column to a different column
      // fills UniqueVec with number 18
      | hipipe::stream::transform(from<IntVec>, to<UniqueVec>, 1.0,
          [](int&) -> std::unique_ptr<int> {
              return std::make_unique<int>(18);
          }, prng, dim<2>)
      // probabilistically transform two columns to two columns
      // replaces approx half of the 18 to 19 in UniqueVec
      | hipipe::stream::transform(from<UniqueVec, IntVec>, to<IntVec, UniqueVec>, 0.5,
          [](std::unique_ptr<int>& ptr, int val) -> std::tuple<int, std::unique_ptr<int>> {
              return std::make_tuple(val, std::make_unique<int>(19));
          }, prng, dim<2>)
      // probabilistically transform two columns to one column
      // again, replaces approx half of the 18 to 19 in UniqueVec
      | hipipe::stream::transform(from<IntVec, UniqueVec>, to<UniqueVec>, 0.5,
          [](int, std::unique_ptr<int>& ptr) -> std::unique_ptr<int> {
              return std::make_unique<int>(19);
          }, prng, dim<2>)
      // convert UniqueVec to IntVec (just for convenience)
      | hipipe::stream::transform(from<UniqueVec>, to<IntVec>,
          [](std::unique_ptr<int>& ptr) -> int {
              return *ptr;
          }, dim<2>);

    BOOST_CHECK(stream.size() == 2);
    BOOST_CHECK(stream.at(0).extract<IntVec>().size()       == 3);
    BOOST_CHECK(stream.at(0).extract<IntVec>().at(0).size() == 2);
    BOOST_CHECK(stream.at(0).extract<IntVec>().at(1).size() == 2);
    BOOST_CHECK(stream.at(0).extract<IntVec>().at(2).size() == 2);
    BOOST_CHECK(stream.at(1).extract<IntVec>().size()       == 1);
    BOOST_CHECK(stream.at(1).extract<IntVec>().at(0).size() == 2);

    long number18 =
      ranges::count(hipipe::utility::flat_view(stream.at(0).extract<IntVec>()), 18) +
      ranges::count(hipipe::utility::flat_view(stream.at(1).extract<IntVec>()), 18);
    long number19 =
      ranges::count(hipipe::utility::flat_view(stream.at(0).extract<IntVec>()), 19) +
      ranges::count(hipipe::utility::flat_view(stream.at(1).extract<IntVec>()), 19);

    BOOST_TEST(number19 >= 4);
    BOOST_TEST(number19 == 8 - number18);
}
