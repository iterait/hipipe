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

#include <range/v3/view/any_view.hpp>

#include <cassert>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace hipipe::stream {

/// \ingroup Stream
/// \brief Abstract base class for HiPipe columns.
class abstract_column {

public:
    // typed value extractor //

    template<typename Column>
    void throw_check_extraction_type() const
    {
        if (typeid(*this) != typeid(Column)) {
            throw std::runtime_error{
              std::string{"Trying to extract column `"} + Column{}.name()
              + "` from a column of type `" + this->name() + "`."};
        }
    }

    // TODO add documentation
    template<typename Column>
    typename Column::batch_type& extract()
    {
        throw_check_extraction_type<Column>();
        return dynamic_cast<Column&>(*this).value();
    }

    template<typename Column>
    const typename Column::batch_type& extract() const
    {
        throw_check_extraction_type<Column>();
        return dynamic_cast<const Column&>(*this).value();
    }

    // name accessor

    virtual std::string name() const = 0;

    // virtual destrutor

    virtual ~abstract_column() = default;
};

/// \ingroup Stream
/// \brief Implementation stub of a column used in HIPIPE_DEFINE_COLUMN.
template <typename T>
class column_base : public abstract_column {
private:
    std::vector<T> value_;

public:

    // types //

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

    // value accessors //

    std::vector<T>& value() { return value_; }
    const std::vector<T>& value() const { return value_; }
};


class batch_t {
private:
    std::unordered_map<std::type_index, std::unique_ptr<abstract_column>> columns_;

public:

    // constructors //

    batch_t() = default;
    batch_t(const batch_t&) = delete;
    batch_t(batch_t&&) = default;

    // value extraction //

    template<typename Column>
    void throw_check_extraction_type() const
    {
        if (columns_.count(std::type_index{typeid(Column)}) == 0) {
            throw std::runtime_error{
              std::string{"Trying to retrieve column `"} + Column{}.name()
              + "`, but the batch contains no such column."};
        }
    }

    template<typename Column>
    typename Column::batch_type& extract()
    {
        throw_check_extraction_type<Column>();
        return columns_.at(std::type_index{typeid(Column)})->extract<Column>();
    }

    template<typename Column>
    const typename Column::batch_type& extract() const
    {
        throw_check_extraction_type<Column>();
        return columns_.at(std::type_index{typeid(Column)})->extract<Column>();
    }

    // raw column access //

    template<typename Column>
    Column& at()
    {
        throw_check_extraction_type<Column>();
        return *columns_.at(std::type_index{typeid(Column)});
    }

    template<typename Column>
    const Column& at() const
    {
        throw_check_extraction_type<Column>();
        return *columns_.at(std::type_index{typeid(Column)});
    }

    // column insertion/rewrite //

    // TODO rename to insert_or_assign or similar
    template<typename Column, typename... Args>
    void insert(Args&&... args)
    {
        static_assert(std::is_constructible_v<Column, Args&&...>,
          "Cannot cosntruct the given column from the provided arguments.");
        columns_[std::type_index{typeid(Column)}] =
          std::make_unique<Column>(std::forward<Args>(args)...);
    }

    // column check //

    template<typename Column>
    bool contains() const
    {
        return columns_.count(std::type_index{typeid(Column)});
    }
};

using stream_t = ranges::any_view<batch_t, ranges::category::forward>;

}  // namespace hipipe::stream

/// \ingroup Stream
/// \brief Macro for fast column definition.
///
/// Under the hood, it creates a new type derived from column_base.
#define HIPIPE_DEFINE_COLUMN(column_name, example_type)            \
struct column_name : hipipe::stream::column_base<example_type> {   \
    using hipipe::stream::column_base<example_type>::column_base;  \
    std::string name() const override { return #column_name; }     \
};
