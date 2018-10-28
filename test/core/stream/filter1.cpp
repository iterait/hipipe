/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

// The tests for stream::filter are split to multiple
// files to speed up compilation in case of multiple CPUs.
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE filter1_test

#include "common.hpp"

#include <hipipe/core/stream/filter.hpp>
#include <hipipe/core/stream/for_each.hpp>

#include <range/v3/to_container.hpp>

BOOST_AUTO_TEST_CASE(test_dim0)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2, batch3;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::batch_type{3, 1});
    batch1.insert_or_assign<Double>(Double::batch_type{5., 2.});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::batch_type{7, 8});
    batch2.insert_or_assign<Double>(Double::batch_type{3., 1.});
    data.push_back(std::move(batch2));
    batch3.insert_or_assign<Int>(Int::batch_type{2, 6});
    batch3.insert_or_assign<Double>(Double::batch_type{4., 5.});
    data.push_back(std::move(batch3));

    std::size_t i = 0;
    data
      | ranges::view::move
      // for dim0, `from` is ignored anyway
      | hipipe::stream::filter(from<Int, Double>, by<Double>,
          [](const std::vector<double>& v) { return v.at(0) > 3.; }, dim<0>)
      | hipipe::stream::for_each(from<Int, Double>, [&i](auto& ints, auto& doubles) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3, 1}));
                    BOOST_TEST(doubles == (std::vector<double>{5., 2.}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{2, 6}));
                    BOOST_TEST(doubles == (std::vector<double>{4., 5.}));
                    break;
            default:
                    BOOST_FAIL("Only two batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}


BOOST_AUTO_TEST_CASE(test_dim0_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));

    std::size_t i = 0;
    data
      | ranges::view::move
      // for dim0, `from` is ignored anyway
      | hipipe::stream::filter(from<>, by<Unique>,
          [](const std::vector<std::unique_ptr<int>>& v) { return *(v.at(0)) >= 3; }, dim<0>)
      | hipipe::stream::for_each(from<Int, Unique>, [&i](auto& ints, auto& uniques) {
            switch (i++) {
            case 0: BOOST_TEST(ints == (std::vector<int>{3}));
                    BOOST_TEST(uniques.size() == 1);
                    BOOST_TEST(*(uniques.at(0)) == 5);
                    break;
            default:
                    BOOST_FAIL("Only one batch should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 1);
}


BOOST_AUTO_TEST_CASE(test_dim1)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::batch_type{3, 1});
    batch1.insert_or_assign<Double>(Double::batch_type{5., 2.});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::batch_type{2, 6});
    batch2.insert_or_assign<Double>(Double::batch_type{4., 5.});
    data.push_back(std::move(batch2));

    std::size_t i = 0;
    data
      | ranges::view::move
      | hipipe::stream::filter(from<Int, Double>, by<Double>,
          [](double v) { return v >= 5.; })
      | hipipe::stream::for_each(from<Int, Double>, [&i](auto& ints, auto& doubles) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3}));
                    BOOST_TEST(doubles == (std::vector<double>{5.}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{6}));
                    BOOST_TEST(doubles == (std::vector<double>{5.}));
                    break;
            default:
                    BOOST_FAIL("Only two batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}


BOOST_AUTO_TEST_CASE(test_dim1_partial)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::batch_type{3, 1});
    batch1.insert_or_assign<Double>(Double::batch_type{5., 2.});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::batch_type{2, 6});
    batch2.insert_or_assign<Double>(Double::batch_type{4., 5.});
    data.push_back(std::move(batch2));

    std::size_t i = 0;
    data
      | ranges::view::move
      | hipipe::stream::filter(from<Double>, by<Double>,
          [](double v) { return v >= 5.; })
      | hipipe::stream::for_each(from<Int, Double>, [&i](auto& a, auto& b) {
            switch (i++) {
            case 0: BOOST_TEST(a == (std::vector<int>{3, 1}));
                    BOOST_TEST(b == (std::vector<double>{5.}));
                    break;
            case 1: BOOST_TEST(a == (std::vector<int>{2, 6}));
                    BOOST_TEST(b == (std::vector<double>{5.}));
                    break;
            default:
                    BOOST_FAIL("Only two batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}


BOOST_AUTO_TEST_CASE(test_dim1_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2, batch3, batch4;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(3);
    batch1.insert_or_assign<Unique>();
    batch1.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(1);
    batch2.insert_or_assign<Unique>();
    batch2.extract<Unique>().push_back(std::make_unique<int>(2));
    data.push_back(std::move(batch2));
    batch3.insert_or_assign<Int>(2);
    batch3.insert_or_assign<Unique>();
    batch3.extract<Unique>().push_back(std::make_unique<int>(4));
    data.push_back(std::move(batch3));
    batch4.insert_or_assign<Int>(6);
    batch4.insert_or_assign<Unique>();
    batch4.extract<Unique>().push_back(std::make_unique<int>(5));
    data.push_back(std::move(batch4));

    std::size_t i = 0;
    data
      | ranges::view::move
      | hipipe::stream::filter(from<Unique>, by<Unique>,
          [](auto& ptr) { return *ptr >= 5.; })
      | hipipe::stream::for_each(from<Int, Unique>, [&i](auto& a, auto& b) {
            switch (i++) {
            case 0: BOOST_TEST(a == (std::vector<int>{3}));
                    BOOST_TEST(b.size() == 1);
                    BOOST_TEST(*(b.at(0)) == 5.);
                    break;
            case 1: BOOST_TEST(a == (std::vector<int>{1}));
                    BOOST_TEST(b.size() == 0);
                    break;
            case 2: BOOST_TEST(a == (std::vector<int>{2}));
                    BOOST_TEST(b.size() == 0);
                    break;
            case 3: BOOST_TEST(a == (std::vector<int>{6}));
                    BOOST_TEST(b.size() == 1);
                    BOOST_TEST(*(b.at(0)) == 5.);
                    break;
            default:
                    BOOST_FAIL("Only four batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 4);
}


BOOST_AUTO_TEST_CASE(test_dim2)
{
    HIPIPE_DEFINE_COLUMN(IntVec1, std::vector<int>)
    HIPIPE_DEFINE_COLUMN(IntVec2, std::vector<int>)

    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<IntVec1>(IntVec1::batch_type{{3, 2}, {1, 5}});
    batch1.insert_or_assign<IntVec2>(IntVec2::batch_type{{1, 5}, {2, 4}});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<IntVec1>(IntVec1::batch_type{{2, 4}, {6, 4}});
    batch2.insert_or_assign<IntVec2>(IntVec2::batch_type{{7, 1}, {3, 5}});
    data.push_back(std::move(batch2));

    std::size_t i = 0;
    auto generated = data
      | ranges::view::move
      | hipipe::stream::filter(from<IntVec1, IntVec2>, by<IntVec2>,
          [](int v) { return v >= 4; }, dim<2>)
      | hipipe::stream::for_each(from<IntVec1, IntVec2>, [&i](auto& iv1, auto& iv2) {
            switch (i++) {
            case 0: BOOST_TEST(iv1 == (std::vector<std::vector<int>>{{2}, {5}}));
                    BOOST_TEST(iv2 == (std::vector<std::vector<int>>{{5}, {4}}));
                    break;
            case 1: BOOST_TEST(iv1 == (std::vector<std::vector<int>>{{2}, {4}}));
                    BOOST_TEST(iv2 == (std::vector<std::vector<int>>{{7}, {5}}));
                    break;
            default:
                    BOOST_FAIL("Only two batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}
