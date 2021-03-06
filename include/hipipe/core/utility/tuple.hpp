/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Tuple Tuple and variadic template utilites.

#ifndef HIPIPE_CORE_TUPLE_UTILS_HPP
#define HIPIPE_CORE_TUPLE_UTILS_HPP

#include <range/v3/core.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/range/primitives.hpp>

#include <experimental/tuple>
#include <ostream>
#include <type_traits>
#include <vector>

namespace hipipe::utility {

namespace rg = ranges;

/// \ingroup Tuple
/// \brief Get the first index of a type in a variadic template list
///
/// The first template argument is the argument to be searched.
/// The rest of the arguments is the variadic template.
///
/// Example:
/// \code
///     variadic_find<int, int, double, double>::value == 0
///     variadic_find<double, int, double, double>::value == 1
///     variadic_find<float, int, double, float>::value == 2
/// \endcode
template<typename T1, typename T2, typename... Ts>
struct variadic_find : std::integral_constant<std::size_t, variadic_find<T1, Ts...>{}+1> {
};

template<typename T, typename... Ts>
struct variadic_find<T, T, Ts...> : std::integral_constant<std::size_t, 0> {
};

/// \ingroup Tuple
/// \brief Wrap variadic template pack in a tuple if there is more than one type.
///
/// Example:
/// \code
///     static_assert(std::is_same_v<maybe_tuple<int, double>, std::tuple<int, double>>);
///     static_assert(std::is_same_v<maybe_tuple<int>, int>);
///     static_assert(std::is_same_v<maybe_tuple<>, std::tuple<>>);
/// \endcode
template<std::size_t N, typename... Ts>
struct maybe_tuple_impl {
    using type = std::tuple<Ts...>;
};

template<typename T>
struct maybe_tuple_impl<1, T> {
    using type = T;
};

template<typename... Ts>
using maybe_tuple = typename maybe_tuple_impl<sizeof...(Ts), Ts...>::type;

/// \ingroup Tuple
/// \brief Add a number to all values in std::index_sequence.
///
/// Example:
/// \code
///     std::is_same<decltype(plus<2>(std::index_sequence<1, 3, 4>{})),
///                                   std::index_sequence<3, 5, 6>>;
/// \endcode
template<std::size_t Value, std::size_t... Is>
constexpr std::index_sequence<(Value + Is)...> plus(std::index_sequence<Is...>)
{
    return {};
}

/// \ingroup Tuple
/// \brief Make std::index_sequence with the given offset.
///
/// Example:
/// \code
///     std::is_same<decltype(make_offset_index_sequence<3, 4>()),
///                           std::index_sequence<3, 4, 5, 6>>;
/// \endcode
template <std::size_t Offset, std::size_t N>
using make_offset_index_sequence = decltype(plus<Offset>(std::make_index_sequence<N>{}));

// tuple_for_each //

namespace detail {

    template<typename Fun, typename Tuple, std::size_t... Is>
    constexpr Fun tuple_for_each_impl(Tuple&& tuple, Fun&& fun, std::index_sequence<Is...>)
    {
        (..., (std::invoke(fun, std::get<Is>(std::forward<Tuple>(tuple)))));
        return fun;
    }

}  // namespace detail

/// \ingroup Tuple
/// \brief Apply a function on each element of a tuple.
///
/// The order of application is from the first to the last element.
///
/// Example:
/// \code
///     auto tpl = std::make_tuple(5, 2.);
///     tuple_for_each(tpl, [](auto& val) { std::cout << val << '\n'; });
/// \endcode
///
/// \returns The function after application.
template<typename Tuple, typename Fun>
constexpr auto tuple_for_each(Tuple&& tuple, Fun&& fun)
{
    constexpr std::size_t tuple_size = std::tuple_size<std::decay_t<Tuple>>::value;
    return detail::tuple_for_each_impl(std::forward<Tuple>(tuple),
                                       std::forward<Fun>(fun),
                                       std::make_index_sequence<tuple_size>{});
}

// tuple_transform //

namespace detail {

    template<typename Tuple, typename Fun, std::size_t... Is>
    constexpr auto tuple_transform_impl(Tuple&& tuple, Fun&& fun, std::index_sequence<Is...>)
    {
        return std::make_tuple(std::invoke(fun, std::get<Is>(std::forward<Tuple>(tuple)))...);
    }

}  // end namespace detail

/// \ingroup Tuple
/// \brief Transform each element of a tuple.
///
/// The order of application is unspecified.
///
/// Example:
/// \code
///    auto t1 = std::make_tuple(0, 10L, 5.);
///    auto t2 = tuple_transform(t1, [](const auto &v) { return v + 1; });
///    static_assert(std::is_same<std::tuple<int, long, double>, decltype(t2)>{});
///    assert(t2 == std::make_tuple(0 + 1, 10L + 1, 5. + 1));
/// \endcode
///
/// \returns The transformed tuple.
template<typename Tuple, typename Fun>
constexpr auto tuple_transform(Tuple&& tuple, Fun&& fun)
{
    constexpr std::size_t tuple_size = std::tuple_size<std::decay_t<Tuple>>::value;
    return detail::tuple_transform_impl(std::forward<Tuple>(tuple),
                                        std::forward<Fun>(fun),
                                        std::make_index_sequence<tuple_size>{});
}

/// \ingroup Tuple
/// \brief Check whether a tuple contains a given type.
template<typename T, typename Tuple = void>
struct tuple_contains;

template<typename T, typename... Types>
struct tuple_contains<T, std::tuple<Types...>>
  : std::disjunction<std::is_same<std::decay_t<T>, std::decay_t<Types>>...> {
};

/// \ingroup Tuple
/// \brief Tuple pretty printing to std::ostream.
template<typename Tuple, size_t... Is>
std::ostream& tuple_print(std::ostream& out, const Tuple& tuple, std::index_sequence<Is...>)
{
    out << "(";
    (..., (out << (Is == 0 ? "" : ", ") << std::get<Is>(tuple)));
    out << ")";
    return out;
}

/// \ingroup Tuple
/// \brief Tuple pretty printing to std::ostream.
template<typename... Ts>
std::ostream& operator<<(std::ostream& out, const std::tuple<Ts...>& tuple)
{
    return utility::tuple_print(out, tuple, std::make_index_sequence<sizeof...(Ts)>{});
}

// unzip //

namespace detail {

    // wrap each type of a tuple in std::vector, i.e., make a tuple of empty vectors
    template<typename Tuple, std::size_t... Is>
    auto vectorize_tuple(std::index_sequence<Is...>)
    {
        return std::make_tuple(std::vector<std::tuple_element_t<Is, std::decay_t<Tuple>>>()...);
    }

    // push elements from the given tuple to the corresponding vectors in a tuple of vectors
    template<typename ToR, typename Tuple, std::size_t... Is>
    void push_unzipped(ToR& tuple_of_ranges, Tuple&& tuple, std::index_sequence<Is...>)
    {
        (..., (std::get<Is>(tuple_of_ranges).push_back(std::get<Is>(std::forward<Tuple>(tuple)))));
    }

    // if the size of the given range is known, return it, otherwise return 0
    CPP_template(class Rng)(requires rg::sized_range<Rng>)
    std::size_t safe_reserve_size(Rng&& rng)
    {
        return rg::size(rng);
    }
    CPP_template(class Rng)(requires (!rg::sized_range<Rng>))
    std::size_t safe_reserve_size(Rng&& rng)
    {
        return 0;
    }

    template<typename Rng>
    auto unzip_impl(Rng& range_of_tuples)
    {
        using tuple_type = rg::range_value_t<Rng>;
        constexpr auto tuple_size = std::tuple_size<tuple_type>{};
        constexpr auto indices = std::make_index_sequence<tuple_size>{};
        std::size_t reserve_size = detail::safe_reserve_size(range_of_tuples);

        auto tuple_of_ranges = detail::vectorize_tuple<tuple_type>(indices);
        utility::tuple_for_each(
          tuple_of_ranges, [reserve_size](auto& rng) { rng.reserve(reserve_size); });

        for (auto& v : range_of_tuples) {
            detail::push_unzipped(tuple_of_ranges, std::move(v), indices);
        }

        return tuple_of_ranges;
    }

}  // namespace detail

/// \ingroup Tuple
/// \brief Unzips a range of tuples to a tuple of std::vectors.
///
/// Example:
/// \code
///     std::vector<std::tuple<int, double>> data{};
///     data.emplace_back(1, 5.);
///     data.emplace_back(2, 6.);
///     data.emplace_back(3, 7.);
///
///     std::vector<int> va;
///     std::vector<double> vb;
///     std::tie(va, vb) = unzip(data);
/// \endcode
CPP_template(class Rng)(requires rg::range<Rng> && (!rg::view_<Rng>))
auto unzip(Rng range_of_tuples)
{
    // copy the given container and move elements out of it
    return detail::unzip_impl(range_of_tuples);
}

/// Specialization of unzip function for views.
CPP_template(class Rng)(requires rg::view_<Rng>)
auto unzip(Rng view_of_tuples)
{
    return utility::unzip(rg::to_vector(view_of_tuples));
}

// maybe unzip //

namespace detail {

    template<bool Enable>
    struct unzip_if_impl
    {
        template<typename Rng>
        static decltype(auto) impl(Rng&& rng)
        {
            return utility::unzip(std::forward<Rng>(rng));
        }
    };

    template<>
    struct unzip_if_impl<false>
    {
        template<typename Rng>
        static Rng&& impl(Rng&& rng)
        {
            return std::forward<Rng>(rng);
        }
    };

}  // namespace detail

/// \ingroup Tuple
/// \brief Unzips a range of tuples to a tuple of ranges if a constexpr condition holds.
///
/// This method is enabled or disabled by its first template parameter.
/// If disabled, it returns identity. If enabled, it returns the same
/// thing as unzip() would return.
///
/// Example:
/// \code
///     std::vector<std::tuple<int, double>> data{};
///     data.emplace_back(1, 5.);
///     data.emplace_back(2, 6.);
///     data.emplace_back(3, 7.);
///
///     std::vector<int> va;
///     std::vector<double> vb;
///     std::tie(va, vb) = unzip_if<true>(data);
///
///     std::vector<int> vc = unzip_if<false>(va);
/// \endcode
template<bool Enable, typename RangeT>
decltype(auto) unzip_if(RangeT&& range)
{
    return detail::unzip_if_impl<Enable>::impl(std::forward<RangeT>(range));
}

// maybe untuple //

namespace detail {

    template<std::size_t Size>
    struct maybe_untuple_impl
    {
        template<typename Tuple>
        static Tuple&& impl(Tuple&& tuple)
        {
            return std::forward<Tuple>(tuple);
        }
    };

    template<>
    struct maybe_untuple_impl<1>
    {
        template<typename Tuple>
        static decltype(auto) impl(Tuple&& tuple)
        {
            return std::get<0>(std::forward<Tuple>(tuple));
        }
    };

}  // namespace detail

/// \ingroup Tuple
/// \brief Extract a value from a tuple if the tuple contains only a single value.
///
/// If the tuple contains zero or more than one element, this is an identity.
///
/// Example:
/// \code
///     std::tuple<int, double> t1{1, 3.};
///     auto t2 = maybe_untuple(t1);
///     static_assert(std::is_same_v<decltype(t2), std::tuple<int, double>>);
///
///     std::tuple<int> t3{1};
///     auto t4 = maybe_untuple(t3);
///     static_assert(std::is_same_v<decltype(t4), int>);
///
///     int i = 1;
///     std::tuple<int&> t5{i};
///     auto& t6 = maybe_untuple(t5);
///     static_assert(std::is_same_v<decltype(t6), int&>);
///     t6 = 2;
///     BOOST_TEST(i == 2);
/// \endcode
template<typename Tuple>
decltype(auto) maybe_untuple(Tuple&& tuple)
{
    constexpr std::size_t tuple_size = std::tuple_size<std::decay_t<Tuple>>::value;
    return detail::maybe_untuple_impl<tuple_size>::impl(std::forward<Tuple>(tuple));
}

// times with index //

namespace detail {

    template<typename Fun, std::size_t... Is>
    constexpr Fun times_with_index_impl(Fun&& fun, std::index_sequence<Is...>)
    {
        (..., (std::invoke(fun, std::integral_constant<std::size_t, Is>{})));
        return fun;
    }

}  // namespace detail

/// \ingroup Tuple
/// \brief Repeat a function N times in compile time.
///
/// Example:
/// \code
///     auto tpl = std::make_tuple(1, 0.25, 'a');
///
///     times_with_index<3>([&tpl](auto index) {
///         ++std::get<index>(tpl);
///     });
///     assert(tpl == std::make_tuple(2, 1.25, 'b'));
/// \endcode
template<std::size_t N, typename Fun>
constexpr Fun times_with_index(Fun&& fun)
{
    return detail::times_with_index_impl(std::forward<Fun>(fun), std::make_index_sequence<N>{});
}

/// \ingroup Tuple
/// \brief Similar to tuple_for_each(), but with index available.
///
/// Example:
/// \code
///     auto tpl = std::make_tuple(1, 2.);
///
///     tuple_for_each_with_index(tpl, [](auto& val, auto index) {
///         val += index;
///     });
///
///     assert(tpl == std::make_tuple(1, 3.));
/// \endcode
template <typename Tuple, typename Fun>
constexpr auto tuple_for_each_with_index(Tuple&& tuple, Fun&& fun)
{
    return utility::times_with_index<std::tuple_size<std::decay_t<Tuple>>{}>(
      [&fun, &tuple](auto index) {
          std::invoke(fun, std::get<index>(tuple), index);
    });
}

// transform with index //

namespace detail {

    template <typename Fun, typename Tuple, std::size_t... Is>
    constexpr auto tuple_transform_with_index_impl(Tuple&& tuple, Fun&& fun,
                                                   std::index_sequence<Is...>)
    {
        return std::make_tuple(std::invoke(fun, std::get<Is>(std::forward<Tuple>(tuple)),
                                           std::integral_constant<std::size_t, Is>{})...);
    }

}  // namespace detail

/// \ingroup Tuple
/// \brief Similar to tuple_transform(), but with index available.
///
/// Example:
/// \code
///     auto tpl = std::make_tuple(1, 0.25, 'a');
///
///     auto tpl2 = tuple_transform_with_index(tpl, [](auto&& elem, auto index) {
///         return elem + index;
///     });
///
///     assert(tpl2 == std::make_tuple(1, 1.25, 'c'));
/// \endcode
template <typename Tuple, typename Fun>
constexpr auto tuple_transform_with_index(Tuple&& tuple, Fun&& fun)
{
    return detail::tuple_transform_with_index_impl(
      std::forward<Tuple>(tuple), std::forward<Fun>(fun),
      std::make_index_sequence<std::tuple_size<std::decay_t<Tuple>>{}>{});
}

}  // namespace hipipe::utility

#endif
