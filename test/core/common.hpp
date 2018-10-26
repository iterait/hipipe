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

#include <hipipe/core/stream/column.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/algorithm/count.hpp>
#include <range/v3/view/move.hpp>

#include <memory>
#include <vector>


// Define some columns to be used throughout the tests.
HIPIPE_DEFINE_COLUMN(Int, int)
HIPIPE_DEFINE_COLUMN(Double, double)
HIPIPE_DEFINE_COLUMN(Unique, std::unique_ptr<int>)
HIPIPE_DEFINE_COLUMN(UniqueVec, std::vector<std::unique_ptr<int>>)
HIPIPE_DEFINE_COLUMN(IntVec, std::vector<int>)


// Generate a fixed stream of two batches with columns IntVec and UniqueVec.
std::vector<hipipe::stream::batch_t> generate_move_only_data_2d()
{
    hipipe::stream::batch_t batch1, batch2;
    std::vector<hipipe::stream::batch_t> data;
    batch1.insert<IntVec>();
    batch1.extract<IntVec>().resize(2);
    batch1.extract<IntVec>().at(0).push_back(2);
    batch1.extract<IntVec>().at(0).push_back(5);
    batch1.extract<IntVec>().at(1).push_back(4);
    batch1.extract<IntVec>().at(1).push_back(9);
    batch1.insert<UniqueVec>();
    batch1.extract<UniqueVec>().resize(3);
    batch1.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(6));
    batch1.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(3));
    batch1.extract<UniqueVec>().at(1).push_back(std::make_unique<int>(7));
    batch1.extract<UniqueVec>().at(1).push_back(std::make_unique<int>(4));
    batch1.extract<UniqueVec>().at(2).push_back(std::make_unique<int>(2));
    batch1.extract<UniqueVec>().at(2).push_back(std::make_unique<int>(1));
    data.push_back(std::move(batch1));
    batch2.insert<IntVec>();
    batch2.extract<IntVec>().resize(1);
    batch2.extract<IntVec>().at(0).push_back(8);
    batch2.extract<IntVec>().at(0).push_back(9);
    batch2.insert<UniqueVec>();
    batch2.extract<UniqueVec>().resize(1);
    batch2.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(2));
    batch2.extract<UniqueVec>().at(0).push_back(std::make_unique<int>(8));
    data.push_back(std::move(batch2));
    return data;
}


template<typename Rng1, typename Rng2>
void test_ranges_equal(Rng1&& rng1, Rng2&& rng2)
{
    // using this function, ranges with different
    // begin() and end() types can be compared
    auto it1 = ranges::begin(rng1);
    auto it2 = ranges::begin(rng2);
    while (it1 != ranges::end(rng1) && it2 != ranges::end(rng2)) {
        BOOST_TEST(*it1 == *it2);
        ++it1;
        ++it2;
    }
    BOOST_CHECK(it1 == ranges::end(rng1));
    BOOST_CHECK(it2 == ranges::end(rng2));
}
