/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_FILTER_HPP
#define CXTREAM_CORE_STREAM_FILTER_HPP

#include <cxtream/core/stream/template_arguments.hpp>
#include <cxtream/core/stream/transform.hpp>
#include <cxtream/core/utility/tuple.hpp>

#include <range/v3/view/zip.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/move.hpp>
#include <range/v3/view/transform.hpp>

#include <functional>
#include <utility>

namespace cxtream::stream {

namespace detail {

    // returns a function which takes a single tuple as argument and
    // applies the given function to its subtuple
    // all the columns are projected to their value()
    template<typename... Types>
    struct filter_fun_for_subtuple_by_type
    {
        template<typename Fun>
        static constexpr auto impl(Fun fun)
        {
            return [fun = std::move(fun)](auto&& tuple) {
                auto proj = [](auto& column) { return std::ref(column.value()); };
                auto slice_view =
                  utility::tuple_transform(utility::tuple_type_view<Types...>(tuple), proj);
                return std::experimental::apply(fun, std::move(slice_view));
            };
        }
    };

    // similar to above, but
    // no projection is performed, the contents of columns in the given dimension
    // are presented to the function as is
    template<std::size_t... Idxs>
    struct filter_fun_for_subtuple_by_idx_no_proj
    {
        template<typename Fun>
        static constexpr auto impl(Fun fun)
        {
            return [fun = std::move(fun)](auto&& tuple) {
                auto slice_view = utility::tuple_index_view<Idxs...>(tuple);
                return std::experimental::apply(fun, std::move(slice_view));
            };
        }
    };


    // filter the stream using the given function in the selected dimension
    // recurse through dimensions
    template<int Dim, std::size_t... ByIdxs>
    struct wrap_filter_fun_for_dim
    {
        template<typename Fun>
        static constexpr auto impl(Fun fun)
        {
            return [fun = std::move(fun)](auto&& tuple_of_ranges) {
                auto range_of_tuples =
                  std::experimental::apply(ranges::view::zip,
                                           std::forward<decltype(tuple_of_ranges)>(tuple_of_ranges))
                  | ranges::view::transform(wrap_filter_fun_for_dim<Dim-1, ByIdxs...>::impl(fun));
                return utility::unzip(std::move(range_of_tuples));
            };
        }
    };

    // recursion bottom for the above, applies the function and filters
    template<std::size_t... ByIdxs>
    struct wrap_filter_fun_for_dim<1, ByIdxs...>
    {
        template<typename Fun>
        static constexpr auto impl(Fun fun)
        {
            return [fun = std::move(fun)](auto&& tuple_of_ranges) {
                auto range_of_tuples =
                  std::experimental::apply(ranges::view::zip,
                                           std::forward<decltype(tuple_of_ranges)>(tuple_of_ranges))
                  | ranges::view::filter(
                      filter_fun_for_subtuple_by_idx_no_proj<ByIdxs...>::impl(fun))
                  | ranges::view::move;
                return utility::unzip(std::move(range_of_tuples));
            };
        }
    };

    // entry point for filter
    // for dimension 0, just apply view::filter with the given function
    // for larger dimensions, use partial_transform until you reach the dimension to be filtered
    template<int Dim>
    struct filter_impl
    {
        template<typename... FromColumns, typename... ByColumns, typename Fun>
        static constexpr auto impl(from_t<FromColumns...> f, by_t<ByColumns...> b, Fun fun)
        {
            auto fun_wrapper =
              detail::wrap_filter_fun_for_dim<
                Dim,
                utility::variadic_find<ByColumns, FromColumns...>::value...>
                  ::impl(std::move(fun));
            return stream::partial_transform(f, to<FromColumns...>, std::move(fun_wrapper),
                                             [](auto& column) { return std::ref(column.value()); });
        }
    };

    template<>
    struct filter_impl<0>
    {
        template<typename... FromColumns, typename... ByColumns, typename Fun>
        static constexpr auto impl(from_t<FromColumns...> f, by_t<ByColumns...> b, Fun fun)
        {
            return ranges::view::filter(
              filter_fun_for_subtuple_by_type<ByColumns...>::impl(std::move(fun)));
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Filter stream data.
///
/// Example:
/// \code
///     CXTREAM_DEFINE_COLUMN(id, int)
///     CXTREAM_DEFINE_COLUMN(value, double)
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
constexpr auto filter(from_t<FromColumns...> f,
                      by_t<ByColumns...> b,
                      Fun fun,
                      dim_t<Dim> d = dim_t<1>{})
{
    return detail::filter_impl<Dim>::impl(f, b, std::move(fun));
}

}  // namespace cxtream::stream
#endif
