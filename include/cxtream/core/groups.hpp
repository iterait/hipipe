/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Groups Data splitting.

#ifndef CXTREAM_CORE_GROUPS_HPP
#define CXTREAM_CORE_GROUPS_HPP

#include <cxtream/core/utility/random.hpp>

#include <range/v3/action/insert.hpp>
#include <range/v3/action/shuffle.hpp>
#include <range/v3/algorithm/all_of.hpp>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/concat.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/repeat_n.hpp>
#include <range/v3/view/take.hpp>

#include <vector>

namespace cxtream {

/// \ingroup Groups
/// \brief Randomly group data into multiple clusters with a given ratio.
///
/// Example:
/// \code
///     generate_groups(10, {2, 2, 6})  // == e.g. {1, 2, 2, 1, 0, 2, 2, 2, 0, 2}
/// \endcode
///
/// If the ratios do not exactly split the requested number of elements, the last
/// group with non-zero ratio gets all the remaining elements.
///
/// \param size The size of the data, i.e., the number of elements.
/// \param ratio Cluster size ratio. The ratios have to be non-negative and
///              the sum of ratios has to be positive.
/// \param gen The random generator to be used.
template<typename Prng = std::mt19937&>
std::vector<std::size_t> generate_groups(std::size_t size, std::vector<double> ratio,
                                         Prng&& gen = utility::random_generator)
{
    namespace view = ranges::view;

    // check all ratios non-negative
    assert(ranges::all_of(ratio, [](double d) { return d >= 0; }));

    // check positive ratio sum
    double ratio_sum = ranges::accumulate(ratio, 0.);
    assert(ratio_sum > 0);

    // remove trailing zeros
    ratio.erase(std::find_if(ratio.rbegin(), ratio.rend(), [](double r) { return r > 0; }).base(),
                ratio.end());

    // scale to [0, 1]
    for (double& r : ratio) r /= ratio_sum;

    std::vector<std::size_t> groups;
    groups.reserve(size);

    for (std::size_t i = 0; i < ratio.size(); ++i) {
        std::size_t count = std::lround(ratio[i] * size);
        // take all the remaining elements if this is the last non-zero group
        if (i + 1 == ratio.size()) count = size - groups.size();
        ranges::action::insert(groups, groups.end(), view::repeat_n(i, count));
    }

    ranges::action::shuffle(groups, gen);
    return groups;
}

/// \ingroup Groups
/// \brief Randomly group data into multiple clusters with a given ratio.
///
/// In this overload, multiple clusterings of the given size are generated. Some of the elements
/// are supposed to be fixed, i.e., to have the same group assigned in all the clusterings.
/// The rest of the data are volatile and their group may differ between the clusterings.
///
/// This function is convenient e.g., if you want to split the data into train/valid/test groups
/// multiple times (e.g., for ensemble training or x-validation) and you want to have the same test
/// group in all the splits.
///
/// Example:
/// \code
///     generate_groups(3, 5, {2, 1}, {2});
///     // == e.g. {{0, 2, 1, 2, 0},
///     //          {1, 2, 0, 2, 1},
///     //          {1, 2, 1, 2, 0}}
///     // note that group 2 is assigned equally in all the groupings
/// \endcode
///
/// \param n The number of different groupings.
/// \param size The size of the data, i.e., the number of elements.
/// \param volatile_ratio The ratio of volatile groups (i.e., groups that change between groupings).
/// \param fixed_ratio The ratio of groups that are assigned equally in all groupings.
/// \param gen The random generator to be used.
template<typename Prng = std::mt19937&>
std::vector<std::vector<std::size_t>>
generate_groups(std::size_t n, std::size_t size,
                const std::vector<double>& volatile_ratio,
                const std::vector<double>& fixed_ratio,
                Prng&& gen = utility::random_generator)
{
    namespace view = ranges::view;

    std::size_t volatile_size = volatile_ratio.size();
    auto full_ratio = view::concat(volatile_ratio, fixed_ratio);

    std::vector<std::vector<std::size_t>> all_groups;
    std::vector<std::size_t> initial_groups = generate_groups(size, full_ratio, gen);

    for (std::size_t i = 0; i < n; ++i) {
        auto groups = initial_groups;
        // select those groups, which are volatile (those will be replaced)
        auto groups_volatile =
          groups | view::filter([volatile_size](std::size_t l) { return l < volatile_size; });
        // count the number of volatile groups
        std::size_t volatile_count = ranges::distance(groups_volatile);
        // generate the replacement
        auto groups_volatile_new = generate_groups(volatile_count, volatile_ratio, gen);
        // replace
        ranges::copy(groups_volatile_new, groups_volatile.begin());
        // store
        all_groups.emplace_back(std::move(groups));
    }

    return all_groups;
}

}  // end namespace cxtream
#endif
