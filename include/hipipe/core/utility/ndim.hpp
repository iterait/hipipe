/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup NDim Multidimensional container utilities.

#pragma once

#include <hipipe/core/utility/random.hpp>

#include <range/v3/action/reverse.hpp>
#include <range/v3/algorithm/adjacent_find.hpp>
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/fill.hpp>
#include <range/v3/algorithm/find.hpp>
#include <range/v3/algorithm/max.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/numeric/partial_sum.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/chunk.hpp>
#include <range/v3/view/for_each.hpp>

#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

namespace hipipe::utility {

namespace rg = ranges;
namespace rga = ranges::actions;
namespace rgv = ranges::views;

/// \ingroup NDim
/// \brief Gets the number of dimensions of a multidimensional range.
///
/// Example:
/// \code
///     std::size_t rng_ndims = ndims<std::vector<std::list<int>>>{};
///     // rng_ndims == 2;
/// \endcode
template<typename Rng, typename PrevRng = void, bool IsRange = rg::range<Rng>>
struct ndims {
};

template<typename Rng, typename PrevRng>
struct ndims<Rng, PrevRng, false> : std::integral_constant<long, 0L> {
};

template<typename Rng, typename PrevRng>
struct ndims<Rng, PrevRng, true>
  : std::integral_constant<long, ndims<rg::range_value_t<Rng>, Rng>{} + 1> {
};

// For recursive ranges, such as std::filesystem::path, do not recurse further and
// consider it to be a scalar.
template<typename Rng>
struct ndims<Rng, Rng, true> : std::integral_constant<long, -1L> {
};

/// \ingroup NDim
/// \brief Detect whether the given type is a specialization of the given container template.
///
/// Example:
/// \code
///     static_assert(is_specialization<std::vector<int>, std::vector>);
///     static_assert(!is_specialization<std::vector<int>, std::deque>);
/// \endcode
template<typename, template<typename...> typename>
struct is_specialization : std::false_type {};

template<template<typename...> typename Template, typename... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

/// \ingroup NDim
/// \brief Fast declaration of a multidimensional container.
///
/// Example:
/// \code
///     ndim_container_t<double, 3> vec_3d;
///     static_assert(std::is_same<decltype(vec_3d),
///                                std::vector<std::vector<std::vector<double>>>
///                                >{});
///
///     ndim_container_t<int, 2, std::list> list_2d;
///     static_assert(std::is_same<decltype(list_2d),
///                                std::list<std::list<int>>
///                                >{});
/// \endcode
template<typename T, long Dims, template<typename...> typename Container = std::vector>
struct ndim_container {
    using type = Container<typename ndim_container<T, Dims-1, Container>::type>;
    static_assert(Dims >= 0, "The number of dimensions has to be non-negative.");
};

template<typename T, template<typename...> typename Container>
struct ndim_container<T, 0, Container> {
    using type = T;
};

template<typename T, long Dims, template<typename...> typename Container = std::vector>
using ndim_container_t = typename ndim_container<T, Dims, Container>::type;

/// \ingroup NDim
/// \brief Gets the innermost value_type of a multidimensional range.
///
/// The number of dimensions can be also provided manually.
///
/// Example:
/// \code
///     using rng_type = ndim_type_t<std::vector<std::list<int>>>;
///     // rng_type is int
///     using rng1_type = ndim_type<std::list<std::vector<int>>, 1>::type;
///     // rng1_type is std::vector<int>
/// \endcode
template<typename Rng, long Dim = -1L>
struct ndim_type
//// \cond
  : ndim_type<typename rg::range_value_t<Rng>, Dim - 1L> {
  static_assert(Dim > 0, "Dimension has to be positive, zero or -1.");
//// \endcond
};

template<typename T>
struct ndim_type<T, 0L> {
    using type = T;
};

template<typename Rng>
struct ndim_type<Rng, -1L>
  : ndim_type<Rng, ndims<Rng>{}> {
};

/// Template alias for quick access to ndim_type<>::type.
template<typename T, long Dim = -1L>
using ndim_type_t = typename ndim_type<T, Dim>::type;

// multidimensional range size //

namespace detail {

    template<typename Rng, long Dim, long NDims>
    struct ndim_size_impl {
        static void impl(const Rng& rng, std::vector<std::vector<long>>& size_out)
        {
            size_out[Dim-1].push_back(rg::size(rng));
            for (auto& subrng : rng) {
                ndim_size_impl<rg::range_value_t<Rng>, Dim+1, NDims>
                  ::impl(subrng, size_out);
            }
        }
    };

    template<typename Rng, long Dim>
    struct ndim_size_impl<Rng, Dim, Dim> {
        static void impl(const Rng& rng, std::vector<std::vector<long>>& size_out)
        {
            size_out[Dim-1].push_back(rg::size(rng));
        }
    };

}  // namespace detail

/// \ingroup NDim
/// \brief Calculates the size of a multidimensional range.
///
/// i-th element of the resulting vector are the sizes of the ranges in the i-th dimension.
///
/// Example:
/// \code
///     std::vector<std::list<int>> rng{{1, 2, 3}, {1}, {5, 6}, {7}};
///     std::vector<std::vector<long>> rng_size = ndim_size<2>(rng);
///     // rng_size == {{4}, {3, 1, 2, 1}};
///     rng_size = ndim_size(rng); // default number of dimensions is the depth of nested ranges
///     // rng_size == {{4}, {3, 1, 2, 1}};
///     rng_size = ndim_size<1>(rng);
///     // rng_size == {{4}};
/// \endcode
///
/// \param rng The multidimensional range whose size shall be calculated.
/// \tparam NDims The number of dimensions that should be considered.
/// \returns The sizes of the given range.
template<long NDims, typename Rng>
std::vector<std::vector<long>> ndim_size(const Rng& rng)
{
    static_assert(NDims > 0);
    std::vector<std::vector<long>> size_out(NDims);
    detail::ndim_size_impl<Rng, 1, NDims>::impl(rng, size_out);
    return size_out;
}

/// \ingroup NDim
/// \brief Calculates the size of a multidimensional range.
///
/// This overload automatically deduces the number of dimension using ndims<Rng>.
template<typename Rng>
std::vector<std::vector<long>> ndim_size(const Rng& rng)
{
    return utility::ndim_size<ndims<Rng>{}>(rng);
}

// multidimensional range resize //

namespace detail {

    template<long Dim, long NDims>
    struct ndim_resize_impl {
        template<typename Rng, typename ValT>
        static void impl(Rng& vec,
                         const std::vector<std::vector<long>>& vec_size,
                         std::vector<long>& vec_size_idx,
                         const ValT& val)
        {
            vec.resize(vec_size[Dim-1][vec_size_idx[Dim-1]++]);
            for (auto& subvec : vec) {
                ndim_resize_impl<Dim+1, NDims>::impl(subvec, vec_size, vec_size_idx, val);
            }
        }
    };

    template<long Dim>
    struct ndim_resize_impl<Dim, Dim> {
        template<typename Rng, typename ValT>
        static void impl(Rng& vec,
                         const std::vector<std::vector<long>>& vec_size,
                         std::vector<long>& vec_size_idx,
                         const ValT& val)
        {
            vec.resize(vec_size[Dim-1][vec_size_idx[Dim-1]++], val);
        }
    };

}  // namespace detail

/// \ingroup NDim
/// \brief Resizes a multidimensional range to the given size.
///
/// i-th element of the given size vector are the sizes of the ranges in the i-th dimension.
/// See ndim_size. The range has to support `.resize()` method up to the given dimension.
///
/// Example:
/// \code
///     std::vector<std::vector<int>> vec;
///     ndim_resize(vec, {{2}, {3, 1}}, 2);
///     // vec == {{2, 2, 2}, {2}};
/// \endcode
///
/// \param vec The range to be resized.
/// \param vec_size The requested size created by ndim_size.
/// \param val The value to pad with.
/// \tparam NDims The number of dimensions to be considered.
///               If omitted, it defaults to ndims<Rng> - ndims<ValT>.
/// \returns The reference to the given vector after resizing.
template<long NDims, typename Rng, typename ValT = ndim_type_t<Rng, NDims>>
Rng& ndim_resize(Rng& vec,
                 const std::vector<std::vector<long>>& vec_size,
                 ValT val = ValT{})
{
    // check that the size is valid
    assert(vec_size.size() == NDims);
    static_assert(NDims <= ndims<Rng>{} - ndims<ValT>{});
    for (std::size_t i = 1; i < vec_size.size(); ++i) {
        assert(vec_size[i].size() == rg::accumulate(vec_size[i-1], 0UL));
    }
    // build initial indices
    std::vector<long> vec_size_idx(vec_size.size());
    // recursively resize
    detail::ndim_resize_impl<1, NDims>::impl(vec, vec_size, vec_size_idx, val);
    return vec;
}

/// A specialization which automatically deduces the number of dimensions.
template<typename Rng, typename ValT = ndim_type_t<Rng>>
Rng& ndim_resize(Rng& vec,
                 const std::vector<std::vector<long>>& vec_size,
                 ValT val = ValT{})
{
    return ndim_resize<ndims<Rng>{}-ndims<ValT>{}>(vec, vec_size, std::move(val));
}

/// \ingroup NDim
/// \brief Pads a mutlidimensional range to a rectangular size.
///
/// The range has to support `.resize()` method up to the given dimension.
///
/// Example:
/// \code
///     std::vector<std::vector<int>> v1 = {{1, 2}, {3, 4, 5}, {}};
///     ndim_pad(v1, -1);
///     // v1 == {{1, 2, -1}, {3, 4, 5}, {-1, -1, -1}};
///     std::vector<std::list<std::vector<int>>> v2 = {{{1}, {2, 3}}, {{4, 5, 6}}};
///     ndim_pad<2>(vec, {-1, -2, -3, -4});  // pad only the first two dimensions
///     // v2 = {{{1}, {2, 3}}, {{4, 5, 6}, {-1, -2, -3, -4}}};
/// \endcode
///
/// \param vec The range to be padded.
/// \param val The value to pad with.
/// \tparam NDims The number of dimensions to be considered.
///               If omitted, it defaults to ndims<Rng> - ndims<ValT>.
/// \returns The reference to the given vector after padding.
template<long NDims, typename Rng, typename ValT = ndim_type_t<Rng, NDims>>
Rng& ndim_pad(Rng& vec, ValT val = ValT{})
{
    static_assert(NDims <= ndims<Rng>{} - ndims<ValT>{});
    std::vector<std::vector<long>> vec_size = ndim_size<NDims>(vec);
    // replace the sizes in each dimension with the max size in the same dimension
    for (std::size_t i = 1; i < vec_size.size(); ++i) {
        long max_size = rg::max(vec_size[i]);
        rg::fill(vec_size[i], max_size);
    }
    return utility::ndim_resize<NDims>(vec, vec_size, std::move(val));
}

/// A specialization which automatically deduces the number of dimensions.
template<typename Rng, typename ValT = ndim_type_t<Rng>>
Rng& ndim_pad(Rng& vec, ValT val = ValT{})
{
    return utility::ndim_pad<ndims<Rng>{}-ndims<ValT>{}>(vec, std::move(val));
}

// multidimensional range shape //

namespace detail {

    template<typename Rng, long Dim, long NDims>
    struct shape_impl {
        static void impl(const Rng& rng, std::vector<long>& shape)
        {
            shape[Dim-1] = rg::size(rng);
            if (rg::size(rng)) {
                shape_impl<typename rg::range_value_t<Rng>, Dim+1, NDims>
                  ::impl(*rg::begin(rng), shape);
            }
        }
    };

    template<typename Rng, long Dim>
    struct shape_impl<Rng, Dim, Dim> {
        static void impl(const Rng& rng, std::vector<long>& shape)
        {
            shape[Dim-1] = rg::size(rng);
        }
    };

    // this is just for assertion purposes - check that the entire vector has a rectangular shape
    template<typename Rng, long NDims>
    struct check_rectangular_shape {
        static bool all_same(std::vector<long>& vec)
        {
            return rg::adjacent_find(vec, std::not_equal_to<long>{}) == rg::end(vec);
        }

        static bool impl(const Rng& rng)
        {
            std::vector<std::vector<long>> size = utility::ndim_size<NDims>(rng);
            return rg::all_of(size, all_same);
        }
    };

}  // namespace detail

/// \ingroup NDim
/// \brief Calculates the shape of a multidimensional range.
///
/// \code
///     std::list<std::vector<int>> rng{{1, 2}, {3, 4}, {5, 6}, {5, 6}};
///     std::vector<long> rng_shape = shape<2>(rng);
///     // rng_shape == {4, 2};
///     rng_shape = shape<1>(rng);
///     // rng_shape == {4};
///     rng_shape = shape(rng);  // the number of dimensions defaults to ndims<Rng>
///     // rng_shape == {4, 2};
/// \endcode
///
/// \param rng The range whose shape shall be calculated. All the subranges
///            on the same dimension have to be of equal size.
/// \tparam NDims The number of dimensions that should be considered.
/// \returns The shape of the given range.
template<long NDims, typename Rng>
std::vector<long> shape(const Rng& rng)
{
    static_assert(NDims > 0);
    std::vector<long> shape(NDims);
    // the ndim_size is not used for efficiency in ndebug mode (only the 0-th element is inspected)
    detail::shape_impl<Rng, 1, NDims>::impl(rng, shape);
    assert((detail::check_rectangular_shape<Rng, NDims>::impl(rng)));
    return shape;
}

/// \ingroup NDim
/// \brief Calculates the shape of a multidimensional range.
///
/// This overload automatically deduces the number of dimension using ndims<Rng>.
template<typename Rng>
std::vector<long> shape(const Rng& rng)
{
    return utility::shape<ndims<Rng>{}>(rng);
}

// recursive range flatten //

namespace detail {

    // recursively join the vector
    template<long Dim>
    struct flat_view_impl {
        static auto impl()
        {
            return rgv::for_each([](auto& subrng) {
                return flat_view_impl<Dim-1>::impl()(subrng);
            });
        }
    };

    // for 0, return the original vector
    template<>
    struct flat_view_impl<1> {
        static auto impl()
        {
            return rgv::all;
        }
    };

}  // namespace detail

/// \ingroup NDim
/// \brief Makes a flat view out of a multidimensional range.
///
/// \code
///     std::vector<std::list<int>> vec{{1, 2}, {3}, {}, {4, 5, 6}};
///     std::vector<int> rvec = ranges::to_vector(flat_view(vec));
///     // rvec == {1, 2, 3, 4, 5, 6};
///
///     // the same with manually defined number of dimensions
///     std::vector<int> rvec = ranges::to_vector(flat_view<2>(vec));
///     // rvec == {1, 2, 3, 4, 5, 6};
/// \endcode
///
/// \param rng The range to be flattened.
/// \tparam NDims The number of dimensions that should be flattened into one.
/// \returns Flat view (input_range) of the given range.
template<long NDims, typename Rng>
auto flat_view(Rng& rng)
{
    static_assert(NDims > 0);
    return detail::flat_view_impl<NDims>::impl()(rng);
}

/// Const version of flat_view.
template<long NDims, typename Rng>
auto flat_view(const Rng& rng)
{
    static_assert(NDims > 0);
    return detail::flat_view_impl<NDims>::impl()(rng);
}

/// flat_view specialization which automatically deduced number of dimensions.
template<typename Rng>
auto flat_view(Rng&& rng)
{
    return utility::flat_view<ndims<Rng>{}>(std::forward<Rng>(rng));
}

// reshaped view of a multidimensional range //

namespace detail {

    template<long N>
    struct reshaped_view_impl_go {
        static auto impl(const std::shared_ptr<std::vector<long>>& shape_ptr)
        {
            return rgv::chunk((*shape_ptr)[N-2])
              | rgv::transform([shape_ptr](auto subview) {
                    return reshaped_view_impl_go<N-1>::impl(shape_ptr)(std::move(subview));
            });
        }
    };

    template<>
    struct reshaped_view_impl_go<1> {
        static auto impl(const std::shared_ptr<std::vector<long>>&)
        {
            return rgv::all;
        }
    };

    template<long N, typename Rng>
    auto reshaped_view_impl(Rng& vec, std::vector<long> shape)
    {
        assert(shape.size() == N);
        auto flat = flat_view(vec);

        // if -1 present in the shape list, deduce the dimension
        auto deduced_pos = rg::find(shape, -1);
        if (deduced_pos != shape.end()) {
            auto flat_size = rg::distance(flat);
            auto shape_prod = -rg::accumulate(shape, 1, std::multiplies<>{});
            assert(flat_size % shape_prod == 0);
            *deduced_pos = flat_size / shape_prod;
        }

        // check that all the requested dimenstions have positive size
        assert(rg::all_of(shape, [](long s) { return s > 0; }));
        // check that the user requests the same number of elements as there really is
        assert(rg::distance(flat) == rg::accumulate(shape, 1, std::multiplies<>{}));
        // calculate the cummulative product of the shape list in reverse order
        shape |= rga::reverse;
        rg::partial_sum(shape, shape, std::multiplies<>{});
        // the recursive chunks will share a single copy of the shape list (performance)
        auto shape_ptr = std::make_shared<std::vector<long>>(std::move(shape));
        return detail::reshaped_view_impl_go<N>::impl(shape_ptr)(std::move(flat));
    }

}  // namespace detail

/// \ingroup NDim
/// \brief Makes a view of a multidimensional range with a specific shape.
///
/// Usage:
/// \code
///     std::list<int> lst{1, 2, 3, 4, 5, 6};
///     std::vector<std::vector<int>> rlst = rg::to_vector(reshaped_view<2>(lst, {2, 3}));
///     // rlst == {{1, 2, 3}, {4, 5, 6}};
/// \endcode
///
/// \param rng The base range for the view.
/// \param shape The list of shapes. Those can contain a single -1, which denotes
///              that the dimension size shall be automatically deduced. All the other values
///              have to be positive.
/// \tparam N The number of dimensions. Has to be equal to shape.size().
/// \returns View (input_range) of the original range with the given shape.
template<long N, typename Rng>
auto reshaped_view(Rng& rng, std::vector<long> shape)
{
    return detail::reshaped_view_impl<N, Rng>(rng, std::move(shape));
}

/// Const version of reshaped_view.
template<long N, typename Rng>
auto reshaped_view(const Rng& rng, std::vector<long> shape)
{
    return detail::reshaped_view_impl<N, const Rng>(rng, std::move(shape));
}

// generate //

namespace detail {

    template<long Dim, long NDims>
    struct generate_impl {
        template<typename Rng, typename Gen>
        static void impl(Rng& rng, Gen& gen, long gendims)
        {
            if (Dim > gendims) {
                auto val = std::invoke(gen);
                for (auto& elem : flat_view<NDims-Dim+1>(rng)) elem = val;
            } else {
                for (auto& subrng : rng) {
                    detail::generate_impl<Dim+1, NDims>::impl(subrng, gen, gendims);
                }
            }
        }
    };

    template<long Dim>
    struct generate_impl<Dim, Dim> {
        template<typename Rng, typename Gen>
        static void impl(Rng& rng, Gen& gen, long gendims)
        {
            if (Dim > gendims) rg::fill(rng, std::invoke(gen));
            else for (auto& val : rng) val = std::invoke(gen);
        }
    };

}  // namespace detail

/// \ingroup NDim
/// \brief Fill a multidimensional range with values generated by a nullary function.
///
/// If the vector is multidimensional, the generator will be used only up to the
/// given dimension and the rest of the dimensions will be constant.
///
/// Example:
/// \code
///     struct gen {
///         int i = 0;
///         int operator()() { return i++; };
///     };
///     std::vector<std::vector<std::vector<int>>> data = {{{-1, -1, -1}, {-1}}, {{-1}, {-1, -1}}};
///     generate(data, gen{}, 0);
///     // data == {{{0, 0, 0}, {0}}, {{0}, {0, 0}}};
///     generate(data, gen{}, 1);
///     // data == {{{0, 0, 0}, {0}}, {{1}, {1, 1}}};
///     generate(data, gen{}, 2);
///     // data == {{{0, 0, 0}, {1}}, {{2}, {3, 3}}};
///     generate(data, gen{}, 3);
///     // data == {{{0, 1, 2}, {3}}, {{4}, {5, 6}}};
///
///     // Note that the generator is called with every element in the
///     // filled dimension, even if there are no actual elements in
///     // higher dimensions! Example:
///     data = {{{-1, -1, -1}, {-1}}, {{}, {}}, {{-1}, {-1, -1}}};
///     generate(data, gen{}, 1);
///     // data == {{{0, 0, 0}, {0}}, {{}, {}}, {{2}, {2, 2}}};
///     generate(data, gen{}, 2);
///     // data == {{{0, 0, 0}, {1}}, {{}, {}}, {{4}, {5, 5}}};
///     generate(data, gen{}, 3);
///     // data == {{{0, 1, 2}, {3}}, {{}, {}}, {{4}, {5, 6}}};
/// \endcode
///
/// \param rng The range to be filled.
/// \param gen The generator to be used.
/// \param gendims The generator will be used only for this number of dimension. The
///                rest of the dimensions will be filled by the last generated value.
///                Use std::numeric_limits<long>::max() to fill all the dimensions using
///                the generator.
/// \tparam NDims The number of dimensions to which will the fill routine recurse.
///               Elements after this dimension are considered to be units (even if
///               they are ranges). If omitted, it defaults to ndims<Rng> - ndims<gen()>.
template<long NDims, typename Rng, typename Gen>
void generate(Rng&& rng,
              Gen&& gen,
              long gendims = std::numeric_limits<long>::max())
{
    static_assert(NDims > 0);
    detail::generate_impl<1, NDims>::impl(rng, gen, gendims);
}

/// A specialization which automatically deduces the number of dimensions.
template<typename Rng, typename Gen>
void generate(Rng&& rng,
              Gen&& gen,
              long gendims = std::numeric_limits<long>::max())
{
    using GenT = std::result_of_t<Gen()>;
    detail::generate_impl<1, ndims<Rng>{}-ndims<GenT>{}>::impl(rng, gen, gendims);
}

/// \ingroup NDim
/// \brief Fill a multidimensional range with random values.
///
/// If the range is multidimensional, the random generator will be used only up to the
/// given dimension and the rest of the dimensions will be constant.
///
/// Note: This function internally uses \ref utility::generate().
///
/// Example:
/// \code
///     std::mt19937 gen = ...;
///     std::uniform_int_distribution dist = ...;
///     std::vector<std::vector<std::vector<int>>> data = {{{0, 0, 0}, {0}}, {{0}, {0, 0}}};
///     random_fill(data, dist, gen, 0);
///     // data == e.g., {{{4, 4, 4}, {4}}, {{4}, {4, 4}}};
///     random_fill(data, dist, gen, 1);
///     // data == e.g., {{{8, 8, 8}, {8}}, {{2}, {2, 2}}};
///     random_fill(data, dist, gen, 2);
///     // data == e.g., {{{8, 8, 8}, {6}}, {{7}, {3, 3}}};
///     random_fill(data, dist, gen, 3);
///     // data == e.g., {{{8, 2, 3}, {1}}, {{2}, {4, 7}}};
/// \endcode
///
/// \param rng The range to be filled.
/// \param dist The distribution to be used.
/// \param prng The random generator to be used.
/// \param gendims The random generator will be used only for this number of dimension. The
///                rest of the dimensions will be filled by the last generated value.
///                Use std::numeric_limits<long>::max() to randomly fill all the dimensions.
/// \tparam NDims The number of dimensions to which will the fill routine recurse.
///               Elements after this dimension are considered to be units (even if
///               they are ranges). If omitted, it defaults to ndims<Rng> - ndims<dist(prng)>.
template<long NDims,
         typename Rng,
         typename Prng = std::mt19937&,
         typename Dist = std::uniform_real_distribution<double>>
void random_fill(Rng&& rng,
                 Dist&& dist = Dist{0, 1},
                 Prng&& prng = utility::random_generator,
                 long gendims = std::numeric_limits<long>::max())
{
    auto gen = [&dist, &prng]() { return std::invoke(dist, prng); };
    utility::generate<NDims>(std::forward<Rng>(rng), std::move(gen), gendims);
}

/// A specialization which automatically deduces the number of dimensions.
template<typename Rng,
         typename Prng = std::mt19937&,
         typename Dist = std::uniform_real_distribution<double>>
void random_fill(Rng&& rng,
                 Dist&& dist = Dist{0, 1},
                 Prng&& prng = utility::random_generator,
                 long gendims = std::numeric_limits<long>::max())
{
    auto gen = [&dist, &prng]() { return std::invoke(dist, prng); };
    utility::generate(std::forward<Rng>(rng), std::move(gen), gendims);
}

namespace detail {

    struct same_size_impl {
        template<typename Rng, typename... Rngs>
        bool operator()(Rng&& rng, Rngs&&... rngs) const
        {
            return (... && (rg::size(rng) == rg::size(rngs)));
        }

        bool operator()() const
        {
            return true;
        }
    };

}  // namespace detail

/// \ingroup NDim
/// \brief Utility function which checks that all the ranges in a tuple have the same size.
///
/// Example:
/// \code
///     std::vector<int> v1 = {1, 2, 3};
///     std::vector<double> v2 = {1., 2., 3.};
///     std::vector<bool> v3 = {true};
///     assert(same_size(std::tie(v1, v2)) == true);
///     assert(same_size(std::tie(v1, v3)) == false);
/// \endcode
///
/// \param rngs A tuple of ranges.
template<typename Tuple>
bool same_size(Tuple&& rngs)
{
    return std::apply(detail::same_size_impl{}, rngs);
}

}  // namespace hipipe::utility
