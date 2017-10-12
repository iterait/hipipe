/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE utility_tuple_test

#include "../common.hpp"

#include <cxtream/core/utility/tuple.hpp>

#include <boost/test/unit_test.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/indirect.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/zip.hpp>

#include <memory>
#include <vector>

using namespace cxtream::utility;

// make the tuple print visible for boost test
// this is forbidden by the standard (simple workarounds?)
namespace std { using cxtream::utility::operator<<; }

BOOST_AUTO_TEST_CASE(test_variadic_find)
{
    static_assert(variadic_find<int, int, double, double>::value == 0);
    static_assert(variadic_find<double, int, double, double>::value == 1);
    static_assert(variadic_find<float, int, double, float>::value == 2);
}

BOOST_AUTO_TEST_CASE(test_plus_index_sequence)
{
    static_assert(std::is_same<decltype(plus<2>(std::index_sequence<1, 3, 4>{})),
                                                std::index_sequence<3, 5, 6>>{});
    static_assert(std::is_same<decltype(plus<0>(std::index_sequence<1, 3, 4>{})),
                                                std::index_sequence<1, 3, 4>>{});
    static_assert(std::is_same<decltype(plus<1>(std::index_sequence<>{})),
                                                std::index_sequence<>>{});
}

BOOST_AUTO_TEST_CASE(test_make_offset_index_sequence)
{
    static_assert(std::is_same<decltype(make_offset_index_sequence<3, 4>()),
                               std::index_sequence<3, 4, 5, 6>>{});
    static_assert(std::is_same<decltype(make_offset_index_sequence<3, 0>()),
                               std::index_sequence<>>{});
}

BOOST_AUTO_TEST_CASE(test_tuple_contains)
{
    // tuple_contains
    static_assert(tuple_contains<int, std::tuple<>>{} == false);
    static_assert(tuple_contains<int, std::tuple<int>>{} == true);
    static_assert(tuple_contains<int, std::tuple<float>>{} == false);
    static_assert(tuple_contains<int, std::tuple<bool, float, int>>{} == true);
    static_assert(tuple_contains<int, std::tuple<bool, float, char>>{} == false);
    static_assert(tuple_contains<int, std::tuple<const int, const float>>{} == true);
    static_assert(tuple_contains<const int, std::tuple<const int, const float>>{} == true);
}

BOOST_AUTO_TEST_CASE(test_tuple_type_view)
{
    // type_view
    auto t1 = std::make_tuple(0, 5., 'c');
    auto t2 = tuple_type_view<double, int>(t1);
    auto t3 = tuple_type_view<char, int>(t1);
    static_assert(std::is_same<std::tuple<char&, int&>, decltype(t3)>{});
    BOOST_TEST(t2 == std::make_tuple(5., 0));
    BOOST_TEST(t2 == std::make_tuple(5., 0));
    BOOST_TEST(t3 == std::make_tuple('c', 0));
}

BOOST_AUTO_TEST_CASE(test_tuple_type_view_writethrough)
{
    // type_view writethrough
    auto t1 = std::make_tuple(0, 5., 'c');
    auto t2 = tuple_type_view<double, int>(t1);
    std::get<int&>(t2) = 1;
    BOOST_TEST(std::get<int>(t1) == 1);

    // double writethrough
    auto t3 = tuple_type_view<double&>(t2);
    std::get<double&>(t3) = 3.;
    BOOST_TEST(std::get<double>(t1) == 3);
}

BOOST_AUTO_TEST_CASE(test_tuple_index_view)
{
    auto t1 = std::make_tuple(0, 5., 'c');
    auto t2 = tuple_index_view<1, 0>(t1);
    auto t3 = tuple_index_view(t1, std::index_sequence<2, 0>{});
    static_assert(std::is_same<std::tuple<char&, int&>, decltype(t3)>{});
    BOOST_TEST(t2 == std::make_tuple(5., 0));
    BOOST_TEST(t2 == std::make_tuple(5., 0));
    BOOST_TEST(t3 == std::make_tuple('c', 0));
}

BOOST_AUTO_TEST_CASE(test_tuple_index_view_writethrough)
{
    // index_view writethrough
    auto t1 = std::make_tuple(0, 5., 'c');
    auto t2 = tuple_index_view<1, 0>(t1);
    std::get<int&>(t2) = 1;
    BOOST_TEST(std::get<int>(t1) == 1);

    // double writethrough
    auto t3 = tuple_index_view<0>(t2);
    std::get<double&>(t3) = 3.;
    BOOST_TEST(std::get<double>(t1) == 3);
}

BOOST_AUTO_TEST_CASE(test_tuple_cat_unique_single)
{
    // cat_unique add existing type
    auto t1 = std::make_tuple(0, '1');
    auto t2 = tuple_cat_unique(t1, std::make_tuple(1));
    static_assert(std::is_same<std::tuple<int, char>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(0, '1'));
}

BOOST_AUTO_TEST_CASE(test_tuple_cat_unique_multiple)
{
    // cat_unique add multiple existing types
    auto t1 = std::make_tuple(0, '1');
    auto t2 = tuple_cat_unique(std::move(t1), std::make_tuple(2, '3'));
    BOOST_TEST(t2 == std::make_tuple(0, '1'));
}

BOOST_AUTO_TEST_CASE(test_tuple_cat_unique_mix_rvalues)
{
    // cat_unique add mix of existing and nonexisting types - rvalue version
    auto t1 = std::make_tuple(0, '2');
    auto t2 = tuple_cat_unique(t1, std::make_tuple('4', 5., 4, 5));
    BOOST_TEST(t2 == std::make_tuple(0, '2', 5.));
}

BOOST_AUTO_TEST_CASE(test_tuple_cat_unique_mix_lvalues)
{
    // cat_unique add mix of existing and nonexisting types - lvalue version
    auto a = 5;
    auto b = '4';
    auto c = 5.;
    auto t1 = std::make_tuple(0, '2');
    auto t2 = tuple_cat_unique(std::move(t1), std::make_tuple(b, std::ref(c), 4, a));
    // references are decayed to values
    static_assert(std::is_same<std::tuple<int, char, double>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(0, '2', 5.));
}

BOOST_AUTO_TEST_CASE(test_tuple_cat_unique_move_only)
{
    // cat_unique move-only
    auto t1 = std::make_tuple(std::unique_ptr<int>{},
                              std::unique_ptr<double>{});
    auto t2 = std::make_tuple(std::unique_ptr<double>{},
                              std::unique_ptr<bool>{},
                              std::unique_ptr<int>{});
    auto t3 = tuple_cat_unique(std::move(t1), std::move(t2));
    static_assert(std::is_same<std::tuple<std::unique_ptr<int>,
                                          std::unique_ptr<double>,
                                          std::unique_ptr<bool>>,
                               decltype(t3)>{});
}

BOOST_AUTO_TEST_CASE(test_tuple_cat_unique_const)
{
    // cat_unique const
    double d = 6.;
    const auto t1 = std::make_tuple(4, std::cref(d), 7);
    const auto t2 = std::make_tuple(1., true, 8.);
    auto t3 = tuple_cat_unique(t1, t2);
    // references are decayed to values
    static_assert(std::is_same<std::tuple<int, double, bool>, decltype(t3)>{});
    BOOST_TEST(t3 == std::make_tuple(4, 6., true));
}

BOOST_AUTO_TEST_CASE(test_tuple_transform)
{
    // tuple_transform
    auto t1 = std::make_tuple(0, 10L, 5.);
    auto t2 = tuple_transform(t1, [](const auto& v) { return v + 1; });
    static_assert(std::is_same<std::tuple<int, long, double>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(0 + 1, 10L + 1, 5. + 1));
}

BOOST_AUTO_TEST_CASE(test_tuple_transform_type_change)
{
    // tuple_transform with type change
    auto t1 = std::make_tuple(0, 'a', 5.);
    auto t2 = tuple_transform(t1, [](auto v) { return 10; });
    static_assert(std::is_same<std::tuple<int, int, int>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(10, 10, 10));
}

BOOST_AUTO_TEST_CASE(test_tuple_transform_empty)
{
    // tuple_transform empty
    auto t1 = std::tuple<>{};
    auto t2 = tuple_transform(t1, [](auto v) { return v + 1; });
    static_assert(std::is_same<std::tuple<>, decltype(t2)>{});
    BOOST_TEST(t2 == std::tuple<>{});
}

BOOST_AUTO_TEST_CASE(test_tuple_transform_move_only)
{
    // tuple_transform move-only
    auto t1 = std::make_tuple(std::unique_ptr<int>{}, std::unique_ptr<double>{});
    auto t2 = tuple_transform(std::move(t1), [](auto v) { return v; });
    static_assert(
      std::is_same<std::tuple<std::unique_ptr<int>, std::unique_ptr<double>>, decltype(t2)>{});
}

BOOST_AUTO_TEST_CASE(test_tuple_transform_mutable)
{
    // tuple_transform for mutable functions
    // beware, the order of application is unspecified
    std::tuple<std::unique_ptr<int>, std::unique_ptr<double>> t1{};

    int called = 0;
    struct Fun {
        int& called_;

        std::unique_ptr<int> operator()(std::unique_ptr<int>& ptr)
        {
            called_++;
            return std::make_unique<int>(0);
        }
        std::unique_ptr<double> operator()(std::unique_ptr<double>& ptr)
        {
            called_++;
            return std::make_unique<double>(1);
        }
    } fun{called};

    auto t2 = tuple_transform(t1, fun);
    static_assert(
      std::is_same<std::tuple<std::unique_ptr<int>, std::unique_ptr<double>>, decltype(t2)>{});
    BOOST_TEST(called == 2);

    auto t3 = tuple_transform(t2, [](const auto& ptr) { return *ptr; });
    static_assert(std::is_same<std::tuple<int, double>, decltype(t3)>{});
    BOOST_TEST(t3 == std::make_tuple(0, 1.));
}

BOOST_AUTO_TEST_CASE(test_tuple_for_each)
{
    // tuple_for_each
    auto t1 = std::make_tuple(std::make_unique<int>(5), std::make_unique<double>(2.));

    tuple_for_each(t1, [](auto& ptr) {
        ptr = std::make_unique<std::remove_reference_t<decltype(*ptr)>>(*ptr + 1);
    });

    auto t2 = tuple_transform(t1, [](const auto &ptr) { return *ptr; });
    static_assert(std::is_same<std::tuple<int, double>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(6, 3.));
}

BOOST_AUTO_TEST_CASE(test_tuple_for_each_order)
{
    // test whether tuple_for_each keeps the application order left-to-right
    auto tuple = std::make_tuple(1, 2, 3);
    std::vector<int> tuple_clone;

    tuple_for_each(tuple, [&tuple_clone](int i) { tuple_clone.push_back(i); });
    std::vector<int> desired = std::vector<int>{1, 2, 3};
    BOOST_TEST(tuple_clone == desired);
}

BOOST_AUTO_TEST_CASE(test_tuple_for_each_mutable)
{
    // tuple_for_each
    // assign to pointers increasing values
    std::tuple<std::unique_ptr<int>, std::unique_ptr<double>> t1{};

    struct {
        int called = 0;
        void operator()(std::unique_ptr<int>& ptr)
        {
            ptr.reset(new int(called++));
        }
        void operator()(std::unique_ptr<double>& ptr)
        {
            ptr.reset(new double(called++));
        }
    } fun;

    fun = tuple_for_each(t1, fun);
    BOOST_TEST(fun.called == 2);

    auto t2 = tuple_transform(t1, [](const auto& ptr) { return *ptr; });
    static_assert(std::is_same<std::tuple<int, double>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(0, 1.));
}

BOOST_AUTO_TEST_CASE(test_tuple_remove)
{
    // tuple_remove
    auto t1 = std::make_tuple(5, 10L, 'a');
    auto t2 = tuple_remove<long>(t1);
    static_assert(std::is_same<std::tuple<int, char>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(5, 'a'));
}

BOOST_AUTO_TEST_CASE(test_tuple_remove_not_contained)
{
    // tuple_remove not contained
    auto t1 = std::make_tuple(2L, 5L, true);
    auto t2 = tuple_remove<int>(std::move(t1));
    static_assert(std::is_same<std::tuple<long, long, bool>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(2L, 5L, true));
}

BOOST_AUTO_TEST_CASE(test_tuple_remove_choose_one_remove_multi)
{
    // tuple_remove multiple fields of the same type
    auto t1 = std::make_tuple(2L, 5L, true);
    auto t2 = tuple_remove<long>(std::move(t1));
    static_assert(std::is_same<std::tuple<bool>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(true));
}

BOOST_AUTO_TEST_CASE(test_tuple_remove_choose_multi_remove_multi)
{
    // tuple_remove multiple fields of multiple types
    auto t1 = std::make_tuple(2L, 5L, true, 'a', 3);
    auto t2 = tuple_remove<long, char, bool>(t1);
    static_assert(std::is_same<std::tuple<int>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(3));
}

BOOST_AUTO_TEST_CASE(test_tuple_remove_references)
{
    // tuple_remove references
    long l1 = 5;
    long l2 = 10L;
    bool b = true;
    auto t1 = std::make_tuple(std::ref(l1), std::ref(l2), std::ref(b));
    auto t2 = tuple_remove<long>(std::move(t1));
    // references are converted to values
    static_assert(std::is_same<std::tuple<bool>, decltype(t2)>{});
    BOOST_TEST(t2 == std::make_tuple(true));
}

BOOST_AUTO_TEST_CASE(test_tuple_remove_move_only)
{
    // tuple_remove move-only
    auto t1 = std::make_tuple(std::unique_ptr<int>{},
                              std::unique_ptr<bool>{},
                              std::unique_ptr<double>{});
    auto t2 = tuple_remove<std::unique_ptr<bool>>(std::move(t1));
    static_assert(std::is_same<std::tuple<std::unique_ptr<int>,
                                          std::unique_ptr<double>>,
                               decltype(t2)>{});
}

BOOST_AUTO_TEST_CASE(test_unzip)
{
    // unzip
    std::vector<std::tuple<int, double>> data{};
    data.emplace_back(1, 5.);
    data.emplace_back(2, 6.);
    data.emplace_back(3, 7.);
    auto data_orig = data;

    std::vector<int> va;
    std::vector<double> vb;
    std::tie(va, vb) = unzip(data);

    std::vector<int> va_desired{1, 2, 3};
    std::vector<double> vb_desired{5., 6., 7.};
    BOOST_TEST(va == va_desired);
    BOOST_TEST(vb == vb_desired);
    BOOST_CHECK(data == data_orig);
}

BOOST_AUTO_TEST_CASE(test_unzip_move_only)
{
    // unzip move only
    std::vector<std::tuple<int, std::unique_ptr<int>>> data{};
    data.emplace_back(1, std::make_unique<int>(5));
    data.emplace_back(2, std::make_unique<int>(6));
    data.emplace_back(3, std::make_unique<int>(7));

    std::vector<int> va;
    std::vector<std::unique_ptr<int>> vb;
    std::tie(va, vb) = unzip(std::move(data));

    std::vector<int> va_desired{1, 2, 3};
    BOOST_TEST(va == va_desired);
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(*vb[i] == (int)i + 5);
    }
}

BOOST_AUTO_TEST_CASE(test_unzip_keeps_valid_containers)
{
    // test that unzip does not move elements out of containers
    std::vector<std::tuple<int, std::shared_ptr<int>>> data{};
    data.emplace_back(1, std::make_shared<int>(5));
    data.emplace_back(2, std::make_shared<int>(6));
    data.emplace_back(3, std::make_shared<int>(7));

    std::vector<int> va;
    std::vector<std::shared_ptr<int>> vb;
    std::tie(va, vb) = unzip(data);

    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(*vb[i] == (int)i + 5);
        BOOST_TEST(vb[i].use_count() == 2);
    }
}

BOOST_AUTO_TEST_CASE(test_unzip_view)
{
    // test that unzip works on views and that it does not move data out
    std::vector<std::tuple<int, std::shared_ptr<int>>> data{};
    data.emplace_back(1, std::make_shared<int>(5));
    data.emplace_back(2, std::make_shared<int>(6));
    data.emplace_back(3, std::make_shared<int>(7));

    std::vector<int> va1;
    std::vector<std::shared_ptr<int>> vb1;
    std::tie(va1, vb1) = unzip(ranges::view::all(data));
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(vb1[i].use_count() == 2);
    }

    std::vector<int> va2;
    std::vector<std::shared_ptr<int>> vb2;
    // test unzip on lvalue view
    auto view = ranges::view::zip(ranges::view::iota(0, 3), vb1);
    std::tie(va2, vb2) = unzip(view);
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(va2[i] == (int)i);
        BOOST_TEST(vb2[i].use_count() == 3);
    }
}

BOOST_AUTO_TEST_CASE(test_unzip_if)
{
    std::vector<std::tuple<int, double>> data{};
    data.emplace_back(1, 5.);
    data.emplace_back(2, 6.);
    data.emplace_back(3, 7.);

    std::vector<int> va;
    std::vector<double> vb;
    std::tie(va, vb) = unzip_if<true>(data);

    std::vector<int> va_desired{1, 2, 3};
    std::vector<double> vb_desired{5., 6., 7.};
    BOOST_TEST(va == va_desired);
    BOOST_TEST(vb == vb_desired);

    std::vector<int> vc = unzip_if<false>(va);
    BOOST_TEST(vc == va);
}

BOOST_AUTO_TEST_CASE(test_unzip_if_move_only)
{
    std::vector<std::tuple<int, std::unique_ptr<int>>> data{};
    data.emplace_back(1, std::make_unique<int>(5));
    data.emplace_back(2, std::make_unique<int>(6));
    data.emplace_back(3, std::make_unique<int>(7));

    std::vector<int> va;
    std::vector<std::unique_ptr<int>> vb;
    std::tie(va, vb) = unzip_if<true>(std::move(data));

    std::vector<int> va_desired{1, 2, 3};
    BOOST_TEST(va == va_desired);
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(*vb[i] == (int)i + 5);
    }

    std::vector<std::unique_ptr<int>> vc = unzip_if<false>(std::move(vb));
    for (std::size_t i = 0; i < 3; ++i) {
        BOOST_TEST(*vc[i] == (int)i + 5);
    }
}

BOOST_AUTO_TEST_CASE(test_range_to_tuple)
{
    std::vector<std::unique_ptr<int>> data;
    data.emplace_back(std::make_unique<int>(5));
    data.emplace_back(std::make_unique<int>(6));
    data.emplace_back(std::make_unique<int>(7));

    auto tpl = range_to_tuple<3>(std::move(data));
    auto tpl_values = tuple_transform(tpl, [](auto& ptr) { return *ptr; });

    std::tuple<int, int, int> tpl_desired{5, 6, 7};
    BOOST_TEST(tpl_values == tpl_desired);
}

BOOST_AUTO_TEST_CASE(test_times_with_index)
{
    auto tpl = std::make_tuple(1, 0.25, 'a');
    times_with_index<3>([&tpl](auto index) { ++std::get<index>(tpl); });
    auto tpl_desired = std::make_tuple(2, 1.25, 'b');
    BOOST_TEST(tpl == tpl_desired);
}

BOOST_AUTO_TEST_CASE(test_times_with_index_mutable)
{
    auto tpl = std::make_tuple(1, 0.25, 'a');
    int called = 0;
    times_with_index<3>([&tpl, &called](auto index) { std::get<index>(tpl) += ++called; });
    auto tpl_desired = std::make_tuple(2, 2.25, 'd');
    BOOST_TEST(tpl == tpl_desired);
}

BOOST_AUTO_TEST_CASE(test_tuple_for_each_with_index)
{
    // tuple_for_each
    auto tpl = std::make_tuple(1, 2.);
    tuple_for_each_with_index(tpl, [](auto& val, auto index) { val += index; });
    BOOST_TEST(tpl == std::make_tuple(1, 3.));
}

BOOST_AUTO_TEST_CASE(test_transform_with_index)
{
    auto tpl = std::make_tuple(1, 0.25, 'a');
    auto tpl2 =
      tuple_transform_with_index(tpl, [](auto&& elem, auto index) { return elem + index; });
    auto tpl2_desired = std::make_tuple(1, 1.25, 'c');
    BOOST_TEST(tpl2 == tpl2_desired);
}

BOOST_AUTO_TEST_CASE(test_transform_with_index_move_only)
{
    auto tpl = std::make_tuple(std::make_unique<int>(1), std::make_unique<int>(2));
    auto tpl2 = tuple_transform_with_index(std::move(tpl),
      [](auto ptr, auto index) { return std::make_unique<int>(*ptr + index); });
    auto tpl2_values = std::make_tuple(*std::get<0>(tpl2), *std::get<1>(tpl2));
    auto tpl2_desired = std::make_tuple(1, 3);
    BOOST_TEST(tpl2_values == tpl2_desired);
}
