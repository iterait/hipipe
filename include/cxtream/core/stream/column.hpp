/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_CORE_STREAM_COLUMN_HPP
#define CXTREAM_CORE_STREAM_COLUMN_HPP

#include <initializer_list>
#include <type_traits>
#include <vector>

namespace cxtream::stream {

/// \ingroup Stream
/// \brief Base class for cxtream columns.
///
/// Stores a vector of given types and provides convenient constructors.
template <typename T, bool = std::is_copy_constructible<T>{}>
class column_base {
private:
    std::vector<T> value_;

public:

    using batch_type = std::vector<T>;
    using example_type = T;

    // constructors //

    column_base() = default;
    column_base(T&& rhs)
    {
        value_.emplace_back(std::move(rhs));
    }
    column_base(const T& rhs)
      : value_{rhs}
    {}

    column_base(std::initializer_list<T> rhs)
      : value_{std::move(rhs)}
    {}

    column_base(std::vector<T>&& rhs)
      : value_{std::move(rhs)}
    {}

    column_base(const std::vector<T>& rhs)
      : value_{rhs}
    {}

    // conversion operators //

    operator std::vector<T>&() &
    {
        return value_;
    }

    operator std::vector<T>&&() &&
    {
        return std::move(value_);
    }

    // value accessors //

    std::vector<T>& value() { return value_; }
    const std::vector<T>& value() const { return value_; }
};

/// Specialization of column_base for non-copy-constructible types.
template <typename T>
struct column_base<T, false> : column_base<T, true> {
    using column_base<T, true>::column_base;

    column_base() = default;
    column_base(const T& rhs) = delete;
    column_base(const std::vector<T>& rhs) = delete;
};

}  // namespace cxtream::stream

/// \ingroup Stream
/// \brief Macro for fast column definition.
///
/// Under the hood, it creates a new type derived from column_base.
#define CXTREAM_DEFINE_COLUMN(col_name, col_type)               \
struct col_name : cxtream::stream::column_base<col_type> {      \
    using cxtream::stream::column_base<col_type>::column_base;  \
    static constexpr const char* name() { return #col_name; }   \
};

#endif
