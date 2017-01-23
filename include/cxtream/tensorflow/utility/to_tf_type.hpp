/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_TENSORFLOW_UTILITY_TO_TF_TYPE_HPP
#define CXTREAM_TENSORFLOW_UTILITY_TO_TF_TYPE_HPP

#include "tensorflow/core/framework/tensor.h"

namespace cxtream::tensorflow {

constexpr auto to_tf_type(bool)
{
    return ::tensorflow::DT_BOOL;
}

constexpr auto to_tf_type(float)
{
    return ::tensorflow::DT_FLOAT;
}

constexpr auto to_tf_type(double)
{
    return ::tensorflow::DT_DOUBLE;
}

constexpr auto to_tf_type(std::int8_t)
{
    return ::tensorflow::DT_INT8;
}

constexpr auto to_tf_type(std::int16_t)
{
    return ::tensorflow::DT_INT16;
}

constexpr auto to_tf_type(std::int32_t)
{
    return ::tensorflow::DT_INT32;
}

constexpr auto to_tf_type(std::int64_t)
{
    return ::tensorflow::DT_INT64;
}

constexpr auto to_tf_type(std::uint8_t)
{
    return ::tensorflow::DT_UINT8;
}

constexpr auto to_tf_type(std::uint16_t)
{
    return ::tensorflow::DT_UINT16;
}

auto to_tf_type(std::string)
{
    return ::tensorflow::DT_STRING;
}

}  // namespace cxtream::tensorflow
#endif
