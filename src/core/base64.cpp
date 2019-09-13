
/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <boost/algorithm/string.hpp>
#include <boost/archive/iterators/base64_from_binary.hpp>
#include <boost/archive/iterators/binary_from_base64.hpp>
#include <boost/archive/iterators/transform_width.hpp>

#include <hipipe/core/base64.hpp>

namespace hipipe {


std::vector<std::uint8_t> base64_decode(const std::string& b64data)
{
    namespace bai = boost::archive::iterators;
    using It =
      bai::transform_width<bai::binary_from_base64<std::string::const_iterator>, 8, 6>;
    // skip padding characters
    std::size_t len = b64data.size();
    while (len && b64data[len - 1] == '=') --len;
    return std::vector<std::uint8_t>(It(std::begin(b64data)), It(std::begin(b64data) + len));
}


std::string base64_encode(const std::vector<std::uint8_t>& data)
{
    namespace bai = boost::archive::iterators;
    using It =
      bai::base64_from_binary<bai::transform_width<std::vector<std::uint8_t>::const_iterator, 6, 8>>;
    std::string res(It(std::begin(data)), It(std::end(data)));
    return res.append((3 - data.size() % 3) % 3, '=');
}

} // end namespace hipipe
