/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Filesystem Filesystem utilities.

#pragma once

#include <experimental/filesystem>
#include <string>

namespace hipipe::utility {


/// \ingroup Filesystem
/// \brief Create a temporary directory.
///
/// \param pattern Directory name pattern. All '\%' symbols in the pattern are
///                replaced by a random character from [0-9a-f].
std::experimental::filesystem::path create_temp_directory(const std::string &pattern);


}  // namespace hipipe::utility
