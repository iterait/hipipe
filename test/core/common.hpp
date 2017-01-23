/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef TEST_COMMON_HPP
#define TEST_COMMON_HPP

#include <cxtream/core/stream/column.hpp>
#include <cxtream/core/utility/tuple.hpp>

#include <boost/test/unit_test.hpp>

#include <memory>
#include <ostream>
#include <vector>

// make the tuple print visible for boost test
// this is forbidden by the standard (simple workarounds?)
namespace std { using cxtream::utility::operator<<; }

CXTREAM_DEFINE_COLUMN(Int, int)
CXTREAM_DEFINE_COLUMN(Double, double)
CXTREAM_DEFINE_COLUMN(Unique, std::unique_ptr<int>)
CXTREAM_DEFINE_COLUMN(Shared, std::shared_ptr<int>)
CXTREAM_DEFINE_COLUMN(UniqueVec, std::vector<std::unique_ptr<int>>)
CXTREAM_DEFINE_COLUMN(IntVec, std::vector<int>)

std::vector<std::tuple<int, std::vector<std::unique_ptr<int>>>> generate_move_only_data()
{
    std::vector<std::tuple<int, std::vector<std::unique_ptr<int>>>> data;
    std::vector<std::unique_ptr<int>> unique_data1;
    unique_data1.push_back(std::make_unique<int>(1));
    unique_data1.push_back(std::make_unique<int>(4));
    data.emplace_back(std::make_tuple(3, std::move(unique_data1)));
    std::vector<std::unique_ptr<int>> unique_data2;
    unique_data2.push_back(std::make_unique<int>(8));
    unique_data2.push_back(std::make_unique<int>(2));
    data.emplace_back(std::make_tuple(3, std::move(unique_data2)));
    std::vector<std::unique_ptr<int>> unique_data3;
    unique_data3.push_back(std::make_unique<int>(2));
    unique_data3.push_back(std::make_unique<int>(5));
    data.emplace_back(std::make_tuple(3, std::move(unique_data3)));
    return data;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec)
{
    out << "{";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        out << vec[i];
        if (i + 1 != vec.size()) out << ", ";
    }
    out << "}";
    return out;
}

bool operator==(const Int& lhs, const Int& rhs)
{ return lhs.value() == rhs.value(); }
bool operator==(const Double& lhs, const Double& rhs)
{ return lhs.value() == rhs.value(); }
std::ostream& operator<<(std::ostream& out, const Int& rhs)
{ return out << rhs.value(); }
std::ostream& operator<<(std::ostream& out, const Double& rhs)
{ return out << rhs.value(); }

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

#endif
