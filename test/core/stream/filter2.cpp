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
#define BOOST_TEST_MODULE filter2_test

#include "common.hpp"

#include <hipipe/core/stream/filter.hpp>
#include <hipipe/core/stream/for_each.hpp>

#include <range/v3/to_container.hpp>


BOOST_AUTO_TEST_CASE(test_mutable)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::batch_type{3, 1});
    batch1.insert_or_assign<IntVec>(IntVec::batch_type{{1, 5}, {2, 4}});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::batch_type{2, 6});
    batch2.insert_or_assign<IntVec>(IntVec::batch_type{{7, 1}, {3, 5}});
    data.push_back(std::move(batch2));

    std::size_t i = 0;
    auto generated = data
      | ranges::view::move
      | hipipe::stream::filter(from<Int, IntVec>, by<Int>,
          [i = 0](int v) mutable { return i++ % 2 == 0; })
      | hipipe::stream::for_each(from<Int, IntVec>, [&i](auto& ints, auto& intvecs) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3}));
                    BOOST_TEST(intvecs == (std::vector<std::vector<int>>{{1, 5}}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{2}));
                    BOOST_TEST(intvecs == (std::vector<std::vector<int>>{{7, 1}}));
                    break;
            default:
                    BOOST_FAIL("Only two batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}


BOOST_AUTO_TEST_CASE(test_dim2_partial)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2;
    std::vector<batch_t> data;
    batch1.insert_or_assign<Int>(Int::batch_type{3, 1});
    batch1.insert_or_assign<IntVec>(IntVec::batch_type{{1, 5}, {2, 4}});
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<Int>(Int::batch_type{2, 6});
    batch2.insert_or_assign<IntVec>(IntVec::batch_type{{7, 1}, {3, 5}});
    data.push_back(std::move(batch2));

    std::size_t i = 0;
    auto generated = data
      | ranges::view::move
      | hipipe::stream::filter(from<IntVec>, by<IntVec>, [](int v) { return v >= 4; }, dim<2>)
      | hipipe::stream::for_each(from<Int, IntVec>, [&i](auto& ints, auto& intvecs) {
            switch (i++) {
            case 0: BOOST_TEST(ints    == (std::vector<int>{3, 1}));
                    BOOST_TEST(intvecs == (std::vector<std::vector<int>>{{5}, {4}}));
                    break;
            case 1: BOOST_TEST(ints    == (std::vector<int>{2, 6}));
                    BOOST_TEST(intvecs == (std::vector<std::vector<int>>{{7}, {5}}));
                    break;
            default:
                    BOOST_FAIL("Only two batches should be provided.");
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 2);
}


BOOST_AUTO_TEST_CASE(test_dim2_move_only)
{
    using hipipe::stream::batch_t;
    using hipipe::stream::from;
    using hipipe::stream::to;
    using hipipe::stream::by;
    using hipipe::stream::dim;

    batch_t batch1, batch2, batch3;
    std::vector<batch_t> data;
    batch1.insert_or_assign<UniqueVec>(UniqueVec::example_type{});
    batch1.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(5));
    batch1.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(3));
    data.push_back(std::move(batch1));
    batch2.insert_or_assign<UniqueVec>(UniqueVec::example_type{});
    batch2.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(2));
    batch2.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(4));
    data.push_back(std::move(batch2));
    batch3.insert_or_assign<UniqueVec>(UniqueVec::example_type{});
    batch3.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(1));
    batch3.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(6));
    data.push_back(std::move(batch3));

    std::size_t i = 0;
    data
      | ranges::view::move
      | hipipe::stream::filter(from<UniqueVec>, by<UniqueVec>,
          [](auto& ptr) { return *ptr >= 4.; }, dim<2>)
      | hipipe::stream::for_each(from<UniqueVec>, [&i](auto& unique_vec) {
            switch (i++) {
            BOOST_TEST(unique_vec.size() == 1);
            BOOST_TEST(unique_vec.at(0).size() == 1);
            case 0: BOOST_TEST(*(unique_vec.at(0).at(0)) == 5);
                    break;
            case 1: BOOST_TEST(*(unique_vec.at(0).at(0)) == 4);
                    break;
            case 2: BOOST_TEST(*(unique_vec.at(0).at(0)) == 6);
                    break;
            }
        }, dim<0>)
      | ranges::to_vector;
    BOOST_TEST(i == 3);
}
