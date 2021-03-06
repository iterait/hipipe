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

#include <hipipe/build_config.hpp>
#include <hipipe/core/stream/stream_t.hpp>
#include <hipipe/core/stream/template_arguments.hpp>
#include <hipipe/core/utility/ndim.hpp>
#include <hipipe/core/utility/random.hpp>
#include <hipipe/core/utility/tuple.hpp>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/any_view.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

#include <functional>
#include <utility>

namespace hipipe::stream {

namespace rg = ranges;
namespace rgv = ranges::views;

// partial transform //

namespace detail {

    // Implementation of partial_transform.
    template<typename Fun, typename From, typename To>
    struct partial_transform_impl;

    template<typename Fun, typename... FromTypes, typename... ToTypes>
    struct partial_transform_impl<Fun, from_t<FromTypes...>, to_t<ToTypes...>> {
        Fun fun;

        batch_t operator()(batch_t source)
        {
            // build the view of the selected source columns for the transformer
            std::tuple<typename FromTypes::data_type&...> slice_view{
                source.extract<FromTypes>()...
            };
            // process the transformer's result and convert it to the requested types
            static_assert(std::is_invocable_v<Fun&, decltype(slice_view)&&>,
              "hipipe::stream::partial_transform: "
              "Cannot apply the given function to the given `from<>` columns.");
            static_assert(std::is_invocable_r_v<
              std::tuple<typename ToTypes::data_type...>, Fun&, decltype(slice_view)&&>,
              "hipipe::stream::partial_transform: "
              "The function return type does not correspond to the tuple of the "
              "selected `to<>` columns.");
            std::tuple<typename ToTypes::data_type...> result =
              std::invoke(fun, std::move(slice_view));
            // convert the function results to the corresponding column(s)
            utility::times_with_index<sizeof...(ToTypes)>([&source, &result](auto i) {
                using Column = std::tuple_element_t<i, std::tuple<ToTypes...>>;
                source.insert_or_assign<Column>(std::move(std::get<i>(result)));
            });
            return source;
        }
    };

    class partial_transform_fn {
    public:
        template <typename... FromTypes, typename... ToTypes, typename Fun>
        forward_stream_t operator()(
          forward_stream_t rng, from_t<FromTypes...>, to_t<ToTypes...>, Fun fun) const
        {
            static_assert(sizeof...(ToTypes) > 0,
              "For non-transforming operations, please use stream::for_each.");

            detail::partial_transform_impl<Fun, from_t<FromTypes...>, to_t<ToTypes...>>
              trans_fun{std::move(fun)};

            return rgv::transform(std::move(rng), std::move(trans_fun));
        }

        template <typename... FromTypes, typename... ToTypes, typename Fun>
        auto operator()(
          from_t<FromTypes...> f, to_t<ToTypes...> t, Fun fun) const
        {
            return rg::make_view_closure(
              rg::bind_back(partial_transform_fn{}, f, t, std::move(fun)));
        }
    };

}  // namespace detail

// Transform a subset of columns in each batch.
//
// This transformer accepts a function that is applied on the chosen
// subset of source columns from the batch. The function should accept
// data_type of the chosen source columns as its parameters and return
// a tuple of data_type of the chosen target columns.
//
// This transformer is used internally by stream::transform and should not
// be used directly by the end user of the library.
inline rgv::view_closure<detail::partial_transform_fn> partial_transform{};

// transform //

namespace detail {
    /// Convert a tuple of containers to a tuple of containers of different type using rg::to.
    ///
    /// This basically calls `ranges::to<std::tuple_element<i, DestTuple>>(std::get<i>(rngs))`
    /// for every index `i`.
    template <typename DestTuple, typename SourceTuple>
    DestTuple convert_tuple_of_ranges(SourceTuple rngs)
    {
        static_assert(std::tuple_size_v<SourceTuple> == std::tuple_size_v<DestTuple>);
        return utility::tuple_transform_with_index(std::move(rngs), [](auto rng, auto index) {
            using DestType = std::tuple_element_t<index, DestTuple>;
            return rg::to<DestType>(rgv::move(rng));
        });
    }

    // Apply fun to each element in tuple of ranges in the given dimension.
    template<typename Fun, std::size_t Dim, typename From, typename To>
    struct wrap_fun_for_dim;

    template<typename Fun, std::size_t Dim, typename... FromTypes, typename... ToTypes>
    struct wrap_fun_for_dim<Fun, Dim, from_t<FromTypes...>, to_t<ToTypes...>> {
        Fun fun;
        using FunRef = decltype(std::ref(fun));

        utility::maybe_tuple<ToTypes...>
        operator()(std::tuple<FromTypes&...> tuple_of_ranges)
        {
            assert(utility::same_size(tuple_of_ranges));
            // build the function to be applied
            wrap_fun_for_dim<FunRef, Dim-1,
              from_t<rg::range_value_t<FromTypes>...>,
              to_t<rg::range_value_t<ToTypes>...>>
                fun_wrapper{std::ref(fun)};
            // transform
            auto trans_view_of_tuples =
              rgv::transform(
                std::apply(rgv::zip, std::move(tuple_of_ranges)),
                std::move(fun_wrapper));
            // unzip the result and convert the ranges to the desired types
            if constexpr (sizeof...(ToTypes) > 1) {
                auto trans_tuple_of_vectors =
                  utility::unzip(std::move(trans_view_of_tuples));
                return convert_tuple_of_ranges<std::tuple<ToTypes...>>(
                  std::move(trans_tuple_of_vectors));
            // result is only one range, no unzipping
            } else {
                return rg::to<ToTypes...>(std::move(trans_view_of_tuples));
            }
        }
    };

    template<typename Fun, typename... FromTypes, typename... ToTypes>
    struct wrap_fun_for_dim<Fun, 0, from_t<FromTypes...>, to_t<ToTypes...>> {
        Fun fun;

        utility::maybe_tuple<ToTypes...>
        operator()(std::tuple<FromTypes&...> tuple)
        {
            static_assert(std::is_invocable_v<Fun&, FromTypes&...>,
              "hipipe::stream::transform: "
              "Cannot call the given function on the selected from<> columns.");
            if constexpr(sizeof...(ToTypes) == 1) {
                static_assert(std::is_invocable_r_v<
                  ToTypes..., Fun&, FromTypes&...>,
                  "hipipe::stream::transform: "
                  "The function does not return the selected to<> column.");
            } else {
                static_assert(std::is_invocable_r_v<
                  std::tuple<ToTypes...>, Fun&, FromTypes&...>,
                  "hipipe::stream::transform: "
                  "The function does not return the tuple of the selected to<> columns.");
            }
            return std::apply(fun, std::move(tuple));
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Transform a subset of hipipe columns to a different subset of hipipe columns.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(id, int)
///     HIPIPE_DEFINE_COLUMN(value, double)
///     std::vector<std::tuple<int, double>> data = {{3, 5.}, {1, 2.}};
///     auto rng = data
///       | create<id, value>()
///       | transform(from<id>, to<value>, [](int id) { return id * 5. + 1.; });
/// \endcode
///
/// \param f The columns to be extracted out of the tuple of columns and passed to fun.
/// \param t The columns where the result will be saved. If the stream does not contain
///          the selected columns, they are added to the stream. This parameter can
///          overlap with the parameter f.
/// \param fun The function to be applied. The function should return the type represented
///            by the target column in the given dimension. If there are multiple target
///            columns, the function should return a tuple of the corresponding types.
/// \param d The dimension in which is the function applied. Choose 0 for the function to
///          be applied to the whole batch.
template<typename... FromColumns, typename... ToColumns, typename Fun, int Dim = 1>
auto transform(
  from_t<FromColumns...> f,
  to_t<ToColumns...> t,
  Fun fun,
  dim_t<Dim> d = dim_t<1>{})
{
    static_assert(
      ((utility::ndims<typename FromColumns::data_type>::value >= Dim) && ...) &&
      ((utility::ndims<typename ToColumns::data_type>::value >= Dim) && ...),
      "hipipe::stream::transform: The dimension in which to apply the operation needs"
      " to be at most the lowest dimension of all the from<> and to<> columns.");

    // a bit of function type erasure to speed up compilation
    using FunT = std::function<
      utility::maybe_tuple<utility::ndim_type_t<typename ToColumns::data_type, Dim>...>
      (utility::ndim_type_t<typename FromColumns::data_type, Dim>&...)>;
    // wrap the function to be applied in the appropriate dimension
    detail::wrap_fun_for_dim<
      FunT, Dim,
      from_t<typename FromColumns::data_type...>,
      to_t<typename ToColumns::data_type...>>
        fun_wrapper{std::move(fun)};

    return stream::partial_transform(f, t, std::move(fun_wrapper));
}

// conditional transform //

namespace detail {

    // wrap the function to be applied only on if the first argument evaluates to true
    template<typename Fun, typename FromIdxs, typename ToIdxs, typename From, typename To>
    struct wrap_fun_with_cond;

    template<typename Fun, std::size_t... FromIdxs, std::size_t... ToIdxs,
             typename CondCol, typename... Cols, typename... ToTypes>
    struct wrap_fun_with_cond<Fun,
                              std::index_sequence<FromIdxs...>,
                              std::index_sequence<ToIdxs...>,
                              from_t<CondCol, Cols...>, to_t<ToTypes...>> {
        Fun fun;

        utility::maybe_tuple<ToTypes...> operator()(CondCol& cond, Cols&... cols)
        {
            // make a tuple of all arguments, except for the condition
            std::tuple<Cols&...> args_view{cols...};
            // apply the function if the condition is true
            if (cond) {
                // the function is applied only on a subset of the arguments
                // representing FromColumns
                static_assert(std::is_invocable_v<Fun&,
                  std::tuple_element_t<FromIdxs, decltype(args_view)>...>,
                  "hipipe::stream::conditional_transform: "
                  "Cannot apply the given function to the given `from<>` columns.");
                static_assert(std::is_invocable_r_v<
                  std::tuple<ToTypes...>, Fun&,
                  std::tuple_element_t<FromIdxs, decltype(args_view)>...>,
                  "hipipe::stream::conditional_transform: "
                  "The function return type does not correspond to the selected `to<>` columns.");
                return std::invoke(fun, std::get<FromIdxs>(args_view)...);
            }
            // return the original arguments if the condition is false
            // only a subset of the arguments representing ToColumns is returned
            // note: We can force std::move in here, because
            // we are only copying data to themselves.
            return {std::move(std::get<ToIdxs>(args_view))...};
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Conditional transform of a subset of hipipe columns.
///
/// This function behaves the same as the original stream::transform(), but it accepts one extra
/// argument denoting a column of `true`/`false` values of the same shape as the columns to be
/// transformed. The transformation will only be applied on true values and it will be an identity
/// on false values.
///
/// Note that this can be very useful in combination with \ref stream::random_fill() and
/// [std::bernoulli_distribution](
/// http://en.cppreference.com/w/cpp/numeric/random/bernoulli_distribution).
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(dogs, int)
///     HIPIPE_DEFINE_COLUMN(do_trans, char)  // do not use bool here, vector<bool> is
///                                            // not a good OutputRange
///     std::vector<int> data_int = {3, 1, 5, 7};
///
///     // hardcoded usage
///     std::vector<int> data_cond = {true, true, false, false};
///     auto rng = ranges::views::zip(data_int, data_cond)
///       | create<dogs, do_trans>()
///       // this transforms only the first two examples and does nothing for the last two
///       | transform(from<dogs>, to<dogs>, cond<do_trans>, [](int dog) { return dog + 1; })
///       // this transformation reverts the previous one
///       | transform(from<dogs>, to<dogs>, cond<do_trans>, [](int dog) { return dog - 1; });
///
///     // random_fill usage
///     std::bernoulli_distribution dist{0.5};
///     auto rng2 = data_int
///       | create<dogs>()
///       | random_fill(from<dogs>, to<do_trans>, 1, dist, prng)
///       // the transformation of each example is performed with 50% probability
///       | transform(from<dogs>, to<dogs>, cond<do_trans>, [](int dog) { return dog + 1; })
///       // this transformation reverts the previous one
///       | transform(from<dogs>, to<dogs>, cond<do_trans>, [](int dog) { return dog - 1; });
/// \endcode
///
/// \param f The columns to be extracted out of the tuple of columns and passed to fun.
/// \param t The columns where the result will be saved. Those have to already exist
///          in the stream.
/// \param c The column of `true`/`false` values denoting whether the transformation should be
///          performed or not. For `false` values, the transformation is an identity
///          on the target columns.
/// \param fun The function to be applied. The function should return the type represented
///            by the selected column in the given dimension. If there are multiple target
///            columns, the function should return a tuple of the corresponding types.
/// \param d The dimension in which is the function applied. Choose 0 for the function to
///          be applied to the whole batch.
template<
  typename... FromColumns,
  typename... ToColumns,
  typename CondColumn,
  typename Fun,
  int Dim = 1>
auto transform(
  from_t<FromColumns...> f,
  to_t<ToColumns...> t,
  cond_t<CondColumn> c,
  Fun fun,
  dim_t<Dim> d = dim_t<1>{})
{
    // make index sequences for source and target columns when they
    // are concatenated in a single tuple
    constexpr std::size_t n_from = sizeof...(FromColumns);
    constexpr std::size_t n_to = sizeof...(ToColumns);
    using FromIdxs = std::make_index_sequence<n_from>;
    using ToIdxs = utility::make_offset_index_sequence<n_from, n_to>;

    static_assert(
      ((utility::ndims<typename FromColumns::data_type>::value >= Dim) && ...) &&
      ((utility::ndims<typename ToColumns::data_type>::value >= Dim) && ...) &&
      utility::ndims<typename CondColumn::data_type>::value >= Dim,
      "hipipe::stream::conditional_transform: The dimension in which to apply the operation needs"
      " to be at most the lowest dimension of all the from<>, to<> and cond<> columns.");

    // a bit of function type erasure to speed up compilation
    using FunT = std::function<
      utility::maybe_tuple<utility::ndim_type_t<typename ToColumns::data_type, Dim>...>
      (utility::ndim_type_t<typename FromColumns::data_type, Dim>&...)>;
    // wrap the function to be applied in the appropriate dimension using the condition column
    detail::wrap_fun_with_cond<
      FunT, FromIdxs, ToIdxs,
      from_t<utility::ndim_type_t<typename CondColumn::data_type, Dim>,
             utility::ndim_type_t<typename FromColumns::data_type, Dim>...,
             utility::ndim_type_t<typename ToColumns::data_type, Dim>...>,
      to_t<utility::ndim_type_t<typename ToColumns::data_type, Dim>...>>
      cond_fun{std::move(fun)};

    // transform from both, FromColumns and ToColumns into ToColumns
    // the wrapper function takes care of extracting the parameters for the original function
    return stream::transform(from_t<CondColumn, FromColumns..., ToColumns...>{},
                             t, std::move(cond_fun), d);
}

// probabilistic transform //

namespace detail {

    // wrap the function to be an identity if the dice roll fails
    template<typename Fun, typename Prng,
             typename FromIdxs, typename ToIdxs,
             typename From, typename To>
    struct wrap_fun_with_prob;

    template<typename Fun, typename Prng,
             std::size_t... FromIdxs, std::size_t... ToIdxs,
             typename... FromTypes, typename... ToTypes>
    struct wrap_fun_with_prob<Fun, Prng,
                              std::index_sequence<FromIdxs...>,
                              std::index_sequence<ToIdxs...>,
                              from_t<FromTypes...>, to_t<ToTypes...>> {
        Fun fun;
        std::reference_wrapper<Prng> prng;
        const double prob;

        utility::maybe_tuple<ToTypes...> operator()(FromTypes&... cols)
        {
            assert(prob >= 0. && prob <= 1.);
            std::uniform_real_distribution<> dis{0, 1};
            // make a tuple of all arguments
            std::tuple<FromTypes&...> args_view{cols...};
            // apply the function if the dice roll succeeds
            if (prob == 1. || (prob > 0. && dis(prng.get()) < prob)) {
                // the function is applied only on a subset of the arguments
                // representing FromColumns
                static_assert(std::is_invocable_v<Fun&,
                  std::tuple_element_t<FromIdxs, decltype(args_view)>...>,
                  "hipipe::stream::probabilistic_transform: "
                  "Cannot apply the given function to the given `from<>` columns.");
                static_assert(std::is_invocable_r_v<
                  std::tuple<ToTypes...>, Fun&,
                  std::tuple_element_t<FromIdxs, decltype(args_view)>...>,
                  "hipipe::stream::probabilistic_transform: "
                  "The function return type does not correspond to the selected `to<>` columns.");
                return std::invoke(fun, std::get<FromIdxs>(args_view)...);
            }
            // return the original arguments if the dice roll fails
            // only a subset of the arguments representing ToColumns is returned
            // note: We can force std::move in here, because
            // we are only copying data to themselves.
            return {std::move(std::get<ToIdxs>(args_view))...};
        }
    };

}  // namespace detail

/// \ingroup Stream
/// \brief Probabilistic transform of a subset of hipipe columns.
///
/// This function behaves the same as the original stream::transform(), but it accepts one extra
/// argument denoting the probability of transformation. If this probability is 0.0,
/// the transformer behaves as an identity. If it is 1.0, the transofrmation function
/// is always applied.
///
/// Example:
/// \code
///     HIPIPE_DEFINE_COLUMN(dogs, int)
///     std::vector<int> data = {3, 1, 5, 7};
///     auto rng = data
///       | create<dogs>()
///       // In 50% of the cases, the number of dogs increase,
///       // and in the other 50% of the cases, it stays the same.
///       | transform(from<dogs>, to<dogs>, 0.5, [](int dog) { return dog + 1; });
/// \endcode
///
/// \param f The columns to be extracted out of the tuple of columns and passed to fun.
/// \param t The columns where the result will be saved. Those have to already exist
///          in the stream.
/// \param prob The probability of transformation. If the dice roll fails, the transformer
///             applies an identity on the target columns.
/// \param fun The function to be applied. The function should return the type represented
///            by the selected column in the given dimension. If there are multiple target
///            columns, the function should return a tuple of the corresponding types.
/// \param prng The random generator to be used. Defaults to a thread_local
///             std::mt19937.
/// \param d The dimension in which is the function applied. Choose 0 for the function to
///          be applied to the whole batch.
template<
  typename... FromColumns,
  typename... ToColumns,
  typename Fun,
  typename Prng = std::mt19937,
  int Dim = 1>
auto transform(
  from_t<FromColumns...> f,
  to_t<ToColumns...> t,
  double prob,
  Fun fun,
  Prng& prng = utility::random_generator,
  dim_t<Dim> d = dim_t<1>{})
{
    // make index sequences for source and target columns when they
    // are concatenated in a single tuple
    constexpr std::size_t n_from = sizeof...(FromColumns);
    constexpr std::size_t n_to = sizeof...(ToColumns);
    using FromIdxs = std::make_index_sequence<n_from>;
    using ToIdxs = utility::make_offset_index_sequence<n_from, n_to>;

    static_assert(
      ((utility::ndims<typename FromColumns::data_type>::value >= Dim) && ...) &&
      ((utility::ndims<typename ToColumns::data_type>::value >= Dim) && ...),
      "hipipe::stream::probabilistic_transform: The dimension in which to apply the operation "
      " needs to be at most the lowest dimension of all the from<> and to<> columns.");

    // a bit of function type erasure to speed up compilation
    using FunT = std::function<
      utility::maybe_tuple<utility::ndim_type_t<typename ToColumns::data_type, Dim>...>
      (utility::ndim_type_t<typename FromColumns::data_type, Dim>&...)>;
    // wrap the function to be applied in the appropriate dimension with the given probabiliy
    detail::wrap_fun_with_prob<
      FunT, Prng, FromIdxs, ToIdxs,
      from_t<utility::ndim_type_t<typename FromColumns::data_type, Dim>...,
             utility::ndim_type_t<typename ToColumns::data_type, Dim>...>,
      to_t<utility::ndim_type_t<typename ToColumns::data_type, Dim>...>>
      prob_fun{std::move(fun), prng, prob};

    // transform from both, FromColumns and ToColumns into ToColumns
    // the wrapper function takes care of extracting the parameters for the original function
    return stream::transform(from_t<FromColumns..., ToColumns...>{}, t, std::move(prob_fun), d);
}

} // namespace hipipe::stream
