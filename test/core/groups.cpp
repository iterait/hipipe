/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE groups_test

#include "common.hpp"

#include <cxtream/core/groups.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/action/sort.hpp>
#include <range/v3/view/filter.hpp>

#include <random>
#include <vector>

using namespace cxtream;

// test with a seeded random generator
std::mt19937 prng{1000003};

std::size_t n_groups(const std::vector<std::size_t>& groups, std::size_t group)
{
    return
      (groups
         | ranges::view::filter([group](std::size_t l) { return l == group; }) 
         | ranges::to_vector
      ).size();
}

BOOST_AUTO_TEST_CASE(test_generate_groups)
{
    std::vector<std::size_t> groups = generate_groups(10, {1.5, 1.5}, prng);
    BOOST_TEST(groups.size() == 10UL);
    BOOST_TEST(n_groups(groups, 0) == 5UL);
    BOOST_TEST(n_groups(groups, 1) == 5UL);
  
    auto sorted_groups = groups;
    sorted_groups |= ranges::action::sort;
    BOOST_CHECK(sorted_groups != groups);
}

BOOST_AUTO_TEST_CASE(test_generate_groups_zero_ratio)
{
    std::vector<std::size_t> groups = generate_groups(10, {1.5, 0, 1.5}, prng);
    BOOST_TEST(groups.size() == 10UL);
    BOOST_TEST(n_groups(groups, 0) == 5UL);
    BOOST_TEST(n_groups(groups, 1) == 0UL);
    BOOST_TEST(n_groups(groups, 2) == 5UL);
  
    auto sorted_groups = groups;
    sorted_groups |= ranges::action::sort;
    BOOST_CHECK(sorted_groups != groups);
}

BOOST_AUTO_TEST_CASE(test_generate_groups_not_divisible)
{
    std::vector<std::size_t> groups = generate_groups(11, {0, 8, 3, 3, 0}, prng);
    BOOST_TEST(groups.size() == 11UL);
    BOOST_TEST(n_groups(groups, 0) == 0UL);

    // the first non-zero one gets round(11 / 14 * 8) = 6
    BOOST_TEST(n_groups(groups, 1) == 6UL);
    // the second non-zero one gets round(11 / 14 * 3) = 2
    BOOST_TEST(n_groups(groups, 2) == 2UL);
    // the third non-zero one gets the rest = 3
    BOOST_TEST(n_groups(groups, 3) == 3UL);

    BOOST_TEST(n_groups(groups, 4) == 0UL);
}

BOOST_AUTO_TEST_CASE(test_generate_many_groups)
{
    std::vector<std::vector<std::size_t>> groups =
      generate_groups(2, 20, {0.3, 0.3}, {0.2, 0.2}, prng);

    BOOST_TEST(groups.size() == 2UL);
    BOOST_TEST(groups[0].size() == 20UL);
    BOOST_TEST(groups[1].size() == 20UL);

    // check that the groups have correct ratio
    for (std::size_t j = 0; j < 2; ++j) {
        BOOST_TEST(n_groups(groups[j], 0) == 6UL);
        BOOST_TEST(n_groups(groups[j], 1) == 6UL);
        BOOST_TEST(n_groups(groups[j], 2) == 4UL);
        BOOST_TEST(n_groups(groups[j], 3) == 4UL);
    }

    // check that all the fixed groups (i.e., 2+) are the same
    for (std::size_t i = 0; i < 10; ++i) {
        if (groups[0][i] >= 2UL) BOOST_TEST(groups[0][i] == groups[0][i]);
    }

    // check that the groups differ
    BOOST_CHECK(groups[0] != groups[1]);
}

BOOST_AUTO_TEST_CASE(test_generate_many_groups_zero_ratio)
{
    std::vector<std::vector<std::size_t>> groups =
      generate_groups(2, 20, {0.3, 0, 0, 0.3}, {0.2, 0, 0, 0.2}, prng);

    BOOST_TEST(groups.size() == 2UL);
    BOOST_TEST(groups[0].size() == 20UL);
    BOOST_TEST(groups[1].size() == 20UL);

    // check that the groups have correct ratio
    for (std::size_t j = 0; j < 2; ++j) {
        BOOST_TEST(n_groups(groups[j], 0) == 6UL);
        BOOST_TEST(n_groups(groups[j], 3) == 6UL);
        BOOST_TEST(n_groups(groups[j], 4) == 4UL);
        BOOST_TEST(n_groups(groups[j], 7) == 4UL);
    }

    // check that all the fixed groups (i.e., 4+) are the same
    for (std::size_t i = 0; i < 10; ++i) {
        if (groups[0][i] >= 4UL) BOOST_TEST(groups[0][i] == groups[0][i]);
    }

    // check that the groups differ
    BOOST_CHECK(groups[0] != groups[1]);
}
