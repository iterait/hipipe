/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Base64 Base64 encoding.

#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace hipipe {

/// \ingroup Base64
/// \brief Decode base64 encoded string to a vector of bytes.
std::vector<std::uint8_t> base64_decode(const std::string& b64data);

/// \ingroup Base64
/// \brief Encode a vector of bytes to a base64 encoded string.
std::string base64_encode(const std::vector<std::uint8_t>& data);

} // end namespace hipipe
