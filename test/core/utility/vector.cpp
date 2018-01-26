/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE utility_vector_test

#include "../common.hpp"

#include <cxtream/core/utility/vector.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/action/sort.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/unique.hpp>

#include <list>
#include <memory>
#include <random>
#include <vector>

using namespace cxtream::utility;

BOOST_AUTO_TEST_CASE(test_ndims)
{
    BOOST_TEST(ndims<std::vector<int>>{} == 1);
    BOOST_TEST(ndims<std::vector<std::vector<int>>>{} == 2);
    BOOST_TEST(ndims<std::list<std::vector<std::list<int>>>>{} == 3);
}

BOOST_AUTO_TEST_CASE(test_ndim_type)
{
    static_assert(std::is_same<int,
      ndim_type<std::vector<int>>::type>{});
    static_assert(std::is_same<char,
      ndim_type<std::vector<std::list<char>>>::type>{});
    static_assert(std::is_same<double,
      ndim_type<std::list<std::vector<std::vector<double>>>>::type>{});
}

BOOST_AUTO_TEST_CASE(test_ndim_type_cutoff)
{
    static_assert(std::is_same<std::vector<std::list<char>>,
      ndim_type<std::vector<std::list<char>>, 0>::type>{});
    static_assert(std::is_same<std::list<char>,
      ndim_type<std::vector<std::list<char>>, 1>::type>{});
    static_assert(std::is_same<std::vector<double>,
      ndim_type<std::list<std::vector<std::vector<double>>>, 2>::type>{});
}

BOOST_AUTO_TEST_CASE(test_ndim_size_cutoff)
{
    const std::vector<int> vec = {};
    const std::vector<int> vec5 = {0, 0, 0, 0, 0};
    std::vector<std::vector<int>> vec3231 = {{0, 0}, {0, 0, 0}, {0}};
    const std::list<std::vector<std::vector<int>>> vec3302 = {
      {{0, 0, 0, 0}, {0, 0, 0}, {0, 0}},
      {},
      {{0}, {}}
    };
    std::vector<std::list<std::vector<int>>> vec200 = {{}, {}};

    test_ranges_equal(ndim_size<1>(vec),
      std::vector<std::vector<long>>{{0}});
    test_ranges_equal(ndim_size<1>(vec5),
      std::vector<std::vector<long>>{{5}});
    test_ranges_equal(ndim_size<1>(vec3231),
      std::vector<std::vector<long>>{{3}});
    test_ranges_equal(ndim_size<2>(vec3302),
      std::vector<std::vector<long>>{{3}, {3, 0, 2}});
    test_ranges_equal(ndim_size<3>(vec200),
      std::vector<std::vector<long>>{{2}, {0, 0}, {}});
}

BOOST_AUTO_TEST_CASE(test_ndim_size_default)
{
    // if the ndims<> function works correctly, this test is only necessary
    // to check whether the code compiles fine
    std::vector<std::vector<int>> vec3231 = {{0, 0}, {0, 0, 0}, {0}};
    test_ranges_equal(ndim_size(vec3231), std::vector<std::vector<long>>{{3}, {2, 3, 1}});
}

BOOST_AUTO_TEST_CASE(test_shape_cutoff)
{
    const std::vector<int> vec5 = {0, 0, 0, 0, 0};
    std::vector<std::list<int>> vec23 = {{0, 0, 0}, {0, 0, 0}};
    const std::list<std::vector<std::vector<int>>> vec234 = {
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}},
      {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}}
    };
    std::vector<std::vector<std::vector<int>>> vec200 = {{}, {}};

    test_ranges_equal(shape<1>(vec5),  std::vector<long>{5});
    test_ranges_equal(shape<1>(vec23), std::vector<long>{2});
    test_ranges_equal(shape<2>(vec234), std::vector<long>{2, 3});
    test_ranges_equal(shape<3>(vec200), std::vector<long>{2, 0, 0});
}

BOOST_AUTO_TEST_CASE(test_shape)
{
    // if the ndims<> function works correctly, this test is only necessary
    // to check whether the code compiles fine
    std::vector<std::list<int>> vec23 = {{0, 0, 0}, {0, 0, 0}};
    test_ranges_equal(shape(vec23), std::vector<long>{2, 3});
}

BOOST_AUTO_TEST_CASE(test_ndim_resize)
{
    std::vector<std::vector<long>> vec5_size    = {{5}};
    std::vector<std::vector<long>> vec3231_size = {{3}, {2, 3, 1}};
    std::vector<std::vector<long>> vec3302_size = {{3}, {3, 0, 2}, {4, 3, 2, 1, 0}};
    std::vector<std::vector<long>> vec200_size  = {{2}, {0, 0}, {}};

    std::vector<int> vec5_desired = {7, 8, 9, 1, 1};
    std::vector<std::vector<int>> vec3231_desired = {{1, 1}, {3, 4, 2}, {2}};
    std::vector<std::vector<std::vector<int>>> vec3302_desired = {
      {{0, 0, 0, 0}, {0, 0, 0}, {0, 0}},
      {},
      {{0}, {}}
    };
    std::vector<std::vector<std::vector<int>>> vec200_desired = {{}, {}};

    std::vector<int> vec5 = {7, 8, 9};
    std::vector<std::vector<int>> vec3231 = {{1, 1, 1}, {3, 4}};
    std::vector<std::vector<std::vector<int>>> vec3302;
    std::vector<std::vector<std::vector<int>>> vec200;

    BOOST_CHECK(ndim_resize(vec5, vec5_size, 1) == vec5_desired);
    BOOST_CHECK(ndim_resize(vec3231, vec3231_size, 2) == vec3231_desired);
    BOOST_CHECK(ndim_resize(vec3302, vec3302_size) == vec3302_desired);
    BOOST_CHECK(ndim_resize(vec200, vec200_size, 3) == vec200_desired);
}

BOOST_AUTO_TEST_CASE(test_flatten)
{
    const std::vector<std::list<std::vector<int>>> vec = {
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
      {{10, 11, 12}, {13, 14, 15}}
    };
    test_ranges_equal(flat_view(vec), ranges::view::iota(1, 16));
}

BOOST_AUTO_TEST_CASE(test_flatten_identity)
{
    const std::vector<int> vec = ranges::view::iota(1, 16);
    test_ranges_equal(flat_view(vec), ranges::view::iota(1, 16));
}

BOOST_AUTO_TEST_CASE(test_flatten_cutoff_dim1)
{
    const std::list<std::vector<std::vector<int>>> vec = {
      {{1, 2}, {3, 4}, {5}},
      {{6}, {7, 8}}
    };
    std::vector<std::vector<std::vector<int>>> retrieved = flat_view<1>(vec);
    test_ranges_equal(retrieved, vec);
}

BOOST_AUTO_TEST_CASE(test_flatten_cutoff_dim2)
{
    const std::vector<std::vector<std::list<int>>> vec = {
      {{1, 2}, {3, 4}, {5}},
      {{6}, {7, 8}}
    };
    const std::vector<std::list<int>> desired = {
      {{1, 2}, {3, 4}, {5}, {6}, {7, 8}}
    };
    std::vector<std::list<int>> retrieved = flat_view<2>(vec);
    BOOST_CHECK(retrieved == desired);
}

BOOST_AUTO_TEST_CASE(test_flatten_empty)
{
    std::list<std::vector<std::vector<int>>> vec = {
      {{}, {}, {}},
      {}
    };
    test_ranges_equal(flat_view(vec), std::vector<int>{});
    std::vector<int> vec2 = {};
    test_ranges_equal(flat_view(vec2), std::vector<int>{});
}

BOOST_AUTO_TEST_CASE(test_flatten_move_only)
{
    std::vector<std::list<std::unique_ptr<int>>> vec;
    std::list<std::unique_ptr<int>> inner;
    inner.push_back(std::make_unique<int>(1));
    inner.push_back(std::make_unique<int>(2));
    vec.push_back(std::move(inner));
    inner = std::list<std::unique_ptr<int>>{};
    inner.push_back(std::make_unique<int>(3));
    inner.push_back(std::make_unique<int>(4));
    vec.push_back(std::move(inner));

    test_ranges_equal(flat_view(vec) | ranges::view::indirect, ranges::view::iota(1, 5));
}

BOOST_AUTO_TEST_CASE(test_reshape_1d)
{
    std::vector<std::list<std::vector<int>>> vec = {
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
      {{10, 11, 12}, {13, 14, 15}}
    };
    std::vector<int> rvec = reshaped_view<1>(vec, {15});
    BOOST_TEST(rvec.size() == 15);
    test_ranges_equal(rvec, ranges::view::iota(1, 16));
}

BOOST_AUTO_TEST_CASE(test_reshape_2d)
{
    const std::list<std::vector<std::vector<int>>> vec = {
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
      {{10, 11, 12}, {13, 14, 15}}
    };
    std::vector<std::vector<int>> rvec = reshaped_view<2>(vec, {3, 5});
    BOOST_TEST(rvec.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(rvec[i].size() == 5);
        test_ranges_equal(rvec[i], ranges::view::iota(5 * i + 1, 5 * i + 6));
    }
}

BOOST_AUTO_TEST_CASE(test_reshape_3d)
{
    std::vector<std::vector<std::list<int>>> vec = {
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
      {{10, 11, 12}, {13, 14, 15}}
    };
    std::vector<std::vector<std::vector<int>>> rvec = reshaped_view<3>(vec, {3, 1, 5});
    BOOST_TEST(rvec.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(rvec[i].size() == 1);
        test_ranges_equal(rvec[i][0], ranges::view::iota(5 * i + 1, 5 * i + 6));
    }
}

BOOST_AUTO_TEST_CASE(test_reshape_auto_dimension)
{
    const std::list<std::vector<std::vector<int>>> vec = {
      {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
      {{10, 11, 12}, {13, 14, 15}}
    };
    std::vector<std::vector<int>> rvec1 = reshaped_view<2>(vec, {3, -1});
    std::vector<std::vector<int>> rvec2 = reshaped_view<2>(vec, {-1, 5});
    BOOST_TEST(rvec1.size() == 3);
    BOOST_TEST(rvec2.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(rvec1[i].size() == 5);
        BOOST_TEST(rvec2[i].size() == 5);
        test_ranges_equal(rvec1[i], ranges::view::iota(5 * i + 1, 5 * i + 6));
        test_ranges_equal(rvec2[i], ranges::view::iota(5 * i + 1, 5 * i + 6));
    }
}

BOOST_AUTO_TEST_CASE(test_reshape_move_only)
{
    std::list<std::vector<std::unique_ptr<int>>> vec;
    std::vector<std::unique_ptr<int>> inner;
    inner.push_back(std::make_unique<int>(1));
    inner.push_back(std::make_unique<int>(2));
    vec.push_back(std::move(inner));
    inner = std::vector<std::unique_ptr<int>>{};
    inner.push_back(std::make_unique<int>(3));
    inner.push_back(std::make_unique<int>(4));
    vec.push_back(std::move(inner));

    auto rvec = reshaped_view<2>(vec, {1, 4});
    test_ranges_equal(*ranges::begin(rvec) | ranges::view::indirect, ranges::view::iota(1, 5));
}

BOOST_AUTO_TEST_CASE(test_generate)
{
    // This test is rather simple, since most of the functionality is
    // tested in random_fill test.
    struct gen {
        int i = 0;
        int operator()() { return i++; };
    };
    using DataType = std::vector<std::vector<std::vector<int>>>;
    DataType data = {{{-1, -1, -1}, {-1}}, {{}, {}}, {{-1}, {-1, -1}}};
    generate(data, gen{}, 0);
    BOOST_CHECK((data == DataType{{{0, 0, 0}, {0}}, {{}, {}}, {{0}, {0, 0}}}));
    generate(data, gen{}, 1);
    BOOST_CHECK((data == DataType{{{0, 0, 0}, {0}}, {{}, {}}, {{2}, {2, 2}}}));
    generate(data, gen{}, 2);
    BOOST_CHECK((data == DataType{{{0, 0, 0}, {1}}, {{}, {}}, {{4}, {5, 5}}}));
    generate(data, gen{}, 3);
    BOOST_CHECK((data == DataType{{{0, 1, 2}, {3}}, {{}, {}}, {{4}, {5, 6}}}));
}

BOOST_AUTO_TEST_CASE(test_random_fill_1d)
{
    std::mt19937 gen{1000003};
    std::uniform_int_distribution<long> dist{0, 1000000000};
    std::vector<long> vec(10);

    random_fill(vec, dist, gen, 0);
    BOOST_TEST(vec.size() == 10);
    vec |= ranges::action::sort;
    auto n_unique = ranges::distance(vec | ranges::view::unique);
    BOOST_TEST(n_unique == 1);

    random_fill(vec, dist, gen, 5);  // any number larger than 0 should suffice
    BOOST_TEST(vec.size() == 10);
    vec |= ranges::action::sort;
    n_unique = ranges::distance(vec | ranges::view::unique);
    BOOST_TEST(n_unique == 10);
}

BOOST_AUTO_TEST_CASE(test_random_fill_2d)
{
    std::mt19937 gen{1000003};
    std::uniform_real_distribution<> dist{0, 1};
    std::vector<std::vector<double>> vec = {std::vector<double>(10),
                                            std::vector<double>(5)};

    auto check = [](auto vec, std::vector<long> unique, long unique_total) {
        for (std::size_t i = 0; i < vec.size(); ++i) {
            vec[i] |= ranges::action::sort;
            auto n_unique = ranges::distance(vec[i] | ranges::view::unique);
            BOOST_TEST(n_unique == unique[i]);
        }

        std::vector<double> all_vals = flat_view(vec);
        all_vals |= ranges::action::sort;
        auto n_unique = ranges::distance(all_vals | ranges::view::unique);
        BOOST_TEST(n_unique == unique_total);
    };

    random_fill(vec, dist, gen, 0);
    check(vec, {1, 1}, 1);
    random_fill(vec, dist, gen, 1);
    check(vec, {1, 1}, 2);
    random_fill(vec, dist, gen, 2);
    check(vec, {10, 5}, 15);
}

BOOST_AUTO_TEST_CASE(test_random_fill_3d)
{
    std::mt19937 gen{1000003};
    std::uniform_real_distribution<> dist{0, 1};
    std::vector<std::vector<std::vector<double>>> vec =
      {{{0, 0, 0}, {0, 0}, {0}}, {{0}, {0, 0}}};

    auto check = [](auto vec, std::vector<std::vector<long>> unique, long unique_total) {
        for (std::size_t i = 0; i < vec.size(); ++i) {
            for (std::size_t j = 0; j < vec[i].size(); ++j) {
                vec[i][j] |= ranges::action::sort;
                auto n_unique = ranges::distance(vec[i][j] | ranges::view::unique);
                BOOST_TEST(n_unique == unique[i][j]);
            }
        }

        std::vector<double> all_vals = flat_view(vec);
        all_vals |= ranges::action::sort;
        auto n_unique = ranges::distance(all_vals | ranges::view::unique);
        BOOST_TEST(n_unique == unique_total);
    };

    random_fill(vec, dist, gen, 0);
    check(vec, {{1, 1, 1}, {1, 1}}, 1);
    random_fill(vec, dist, gen, 1);
    check(vec, {{1, 1, 1}, {1, 1}}, 2);
    random_fill(vec, dist, gen, 2);
    check(vec, {{1, 1, 1}, {1, 1}}, 5);
    random_fill(vec, dist, gen, 3);
    check(vec, {{3, 2, 1}, {1, 2}}, 9);
}

BOOST_AUTO_TEST_CASE(test_same_size)
{
    const std::vector<int> v1 = {1, 2, 3};
    const std::vector<bool> v2 = {true, false, true};
    const std::vector<char> v3 = {'a', 'b'};
    const std::vector<double> v4 = {};
    BOOST_TEST(same_size(std::tuple<>{}));
    BOOST_TEST(same_size(std::tie(v4)));
    BOOST_TEST(same_size(std::tie(v1, v2)));
    BOOST_TEST(!same_size(std::tie(v1, v3)));
    BOOST_TEST(!same_size(std::tie(v1, v4)));
    BOOST_TEST(same_size(std::tie(v3, v3, v3)));
    BOOST_TEST(same_size(std::tie(v1, v2, v1)));
    BOOST_TEST(!same_size(std::tie(v1, v2, v3)));
    BOOST_TEST(!same_size(std::tie(v1, v2, v3, v4)));
}
