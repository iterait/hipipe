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

namespace hipipe::stream {

namespace detail {

    // Wrap the given function so that its return value is ignored
    // and so that it can be forwarded to stream::transform.
    template<typename Fun, typename... FromTypes>
    struct wrap_void_fun_for_transform {
        Fun fun;

        utility::maybe_tuple<FromTypes...> operator()(FromTypes&... args)
        {
            static_assert(std::is_invocable_v<Fun, FromTypes&...>,
              "hipipe::stream::for_each: "
              "Cannot apply the given function to the given `from<>` columns.");
            std::invoke(fun, args...);
            // we can force std::move here because the old
            // data are going to be ignored anyway
            return {std::move(args)...};
        }
    };

}  // namespace detail


/// \ingroup Stream
/// \brief Apply a function to a subset of stream columns.
///
/// The given function is applied to a subset of columns given by FromColumns.
/// The transformed range is the same as the input range, no elements are actually changed.
/// The function is applied lazily, i.e., only when the range is iterated.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(Int, int)
///     HIPIPE_DEFINE_COLUMN(Double, double)
///     std::vector<std::tuple<Int, Double>> data = {{3, 5.}, {1, 2.}};
///     auto rng = data
///       | for_each(from<Int, Double>, [](int& v, double& d) { std::cout << c + d; });
/// \endcode
///
/// \param f The columns to be exctracted out of the tuple of columns and passed to fun.
/// \param fun The function to be applied.
/// \param d The dimension in which the function is applied. Choose 0 for the function to
///          be applied to the whole batch.
template<typename... FromColumns, typename Fun, int Dim = 1>
auto for_each(from_t<FromColumns...> f, Fun fun, dim_t<Dim> d = dim_t<1>{})
{
    static_assert(
      ((utility::ndims<typename FromColumns::batch_type>::value >= Dim) && ...),
      "hipipe::stream::for_each: The dimension in which to apply the operation "
      " needs to be at most the lowest dimension of all the from<> columns.");
    // a bit of function type erasure to speed up compilation
    using FunT = std::function<
      void(utility::ndim_type_t<typename FromColumns::batch_type, Dim>&...)>;
    // wrap the function to be compatible with stream::transform
    detail::wrap_void_fun_for_transform<
      FunT, utility::ndim_type_t<typename FromColumns::batch_type, Dim>...>
        fun_wrapper{std::move(fun)};
    // apply the dummy transformation
    return stream::transform(f, to<FromColumns...>, std::move(fun_wrapper), d);
}

}  // namespace hipipe::stream
