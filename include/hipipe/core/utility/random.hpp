/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Random Random generator utilites.

#ifndef HIPIPE_CORE_UTILITY_RANDOM_HPP
#define HIPIPE_CORE_UTILITY_RANDOM_HPP

#include <random>

namespace hipipe::utility {

/// \ingroup Random
/// \brief Thread local pseudo-random number generator seeded by std::random_device.
static thread_local std::mt19937 random_generator{std::random_device{}()};

}  // namespace hipipe::utility
#endif
