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

#include <hipipe/core/stream/template_arguments.hpp>
#include <hipipe/core/stream/transform.hpp>
#include <hipipe/core/utility/tuple.hpp>

#include <range/v3/view/filter.hpp>
#include <range/v3/view/move.hpp>
#include <range/v3/view/zip.hpp>

#include <functional>
#include <utility>

namespace hipipe::stream {

namespace detail {

    // Filter the stream using the given function.
    // This function wrapper is to be applied in one lower
    // dimension than the wrapped function itself.
    // This function wrapper is to be called in dimensions higher than 0.
    template<typename Fun, typename From, typename ByIdxs>
    struct wrap_filter_fun_for_transform;
    template<typename Fun, typename... FromTypes, std::size_t... ByIdxs>
    struct wrap_filter_fun_for_transform<Fun, from_t<FromTypes...>, std::index_sequence<ByIdxs...>>
    {
        Fun fun;

        // Properly zips/unzips the data and applies the filter function.
        utility::maybe_tuple<FromTypes...> operator()(FromTypes&... cols)
        {
            auto range_of_tuples =
                ranges::view::zip(cols...)
              | ranges::view::filter([this](const auto& tuple) -> bool {
                    return std::invoke(this->fun, std::get<ByIdxs>(tuple)...);
                })
              | ranges::view::move;
            return utility::maybe_untuple(utility::unzip(std::move(range_of_tuples)));
        }
    };

    // Helper function wrapper for dimension 0.
    // This wrapper takes a single tuple of columns as argument and
    // applies the stored function to a subset of columns selected by types.
    // The columns are projected to their value().
    template<typename Fun, typename... ByColumns>
    struct apply_filter_fun_to_columns
    {
        Fun fun;

        bool operator()(const batch_t& source)
        {
            std::tuple<const typename ByColumns::batch_type&...> slice_view{
                source.extract<ByColumns>()...
            };
            static_assert(std::is_invocable_r_v<
              bool, Fun&, const typename ByColumns::batch_type&...>,
              "hipipe::stream::filter: "
              "The function has to accept the selected `by<>` columns (specifically "
              "const ByColumns::batch_type&) and return a bool.");
            return std::apply(fun, std::move(slice_view));
        }
    };

    // Entry point for stream::filter.
    // For dimensions higher than 0, use stream::transform to Dim-1 and
    // wrap_filter_fun_for_transform wrapper.
    template<int Dim>
    struct filter_impl
    {
        template<typename... FromColumns, typename... ByColumns, typename Fun>
        static auto impl(from_t<FromColumns...> f, by_t<ByColumns...> b, Fun fun)
        {
            static_assert(sizeof...(ByColumns) <= sizeof...(FromColumns),
              "Cannot have more ByColumns than FromColumns.");
            static_assert(
              ((utility::ndims<typename FromColumns::batch_type>::value >= Dim) && ...) &&
              ((utility::ndims<typename ByColumns::batch_type>::value >= Dim) && ...),
              "hipipe::stream::filter: The dimension in which to apply the operation needs"
              " to be at most the lowest dimension of all the from<> and by<> columns.");

            detail::wrap_filter_fun_for_transform<
              Fun, from_t<utility::ndim_type_t<typename FromColumns::batch_type, Dim-1>...>,
              std::index_sequence<utility::variadic_find<ByColumns, FromColumns...>::value...>>
                fun_wrapper{std::move(fun)};

            return stream::transform(f, to<FromColumns...>, std::move(fun_wrapper), dim<Dim-1>);
        }
    };

    // Special case for batch filtering (Dim == 0).
    template<>
    struct filter_impl<0>
    {
        template<typename From, typename... ByColumns, typename Fun>
        static auto impl(From, by_t<ByColumns...>, Fun fun)
        {
            apply_filter_fun_to_columns<Fun, ByColumns...> fun_wrapper{std::move(fun)};
            return ranges::view::filter(std::move(fun_wrapper));
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Filter stream data.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(id, int)
///     HIPIPE_DEFINE_COLUMN(value, double)
///     std::vector<std::tuple<int, double>> data = {{3, 5.}, {1, 2.}};
///     auto rng = data
///       | create<id, value>()
///       | filter(from<id, value>, by<value>, [](double value) { return value > 3.; });
/// \endcode
///
/// \param f The columns to be filtered.
/// \param b The columns to be passed to the filtering function. Those have to be
///          a subset of f.
/// \param fun The filtering function returning a boolean.
/// \param d The dimension in which the function is applied. Choose 0 to filter
///          whole batches (in such a case, the f parameter is ignored).
template<typename... FromColumns, typename... ByColumns, typename Fun, int Dim = 1>
auto filter(from_t<FromColumns...> f,
            by_t<ByColumns...> b,
            Fun fun,
            dim_t<Dim> d = dim_t<1>{})
{
    static_assert(
      ((utility::ndims<typename FromColumns::batch_type>::value >= Dim) && ...) &&
      ((utility::ndims<typename ByColumns::batch_type>::value >= Dim) && ...),
      "hipipe::stream::filter: The dimension in which to apply the operation "
      " needs to be at most the lowest dimension of all the from<> and by<> columns.");
    // a bit of function type erasure to speed up compilation
    using FunT = std::function<
      bool(const utility::ndim_type_t<typename ByColumns::batch_type, Dim>&...)>;
    return detail::filter_impl<Dim>::impl(f, b, FunT{std::move(fun)});
}

}  // namespace hipipe::stream
