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

#ifdef HIPIPE_BUILD_PYTHON
#include <hipipe/core/python/utility/ndim_vector_converter.hpp>
#endif

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
    void throw_check_contains() const
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
        throw_check_contains<Column>();
        return dynamic_cast<Column&>(*this).value();
    }

    template<typename Column>
    const typename Column::batch_type& extract() const
    {
        throw_check_contains<Column>();
        return dynamic_cast<const Column&>(*this).value();
    }

    // name accessor

    virtual std::string name() const = 0;

    // batch utilities //

    virtual std::size_t size() const = 0;

    virtual void push_back(std::unique_ptr<abstract_column> rhs) = 0;

    virtual std::unique_ptr<abstract_column> take(std::size_t n) = 0;

    // virtual destrutor

    virtual ~abstract_column() = default;

    // python conversion

    #ifdef HIPIPE_BUILD_PYTHON
    virtual boost::python::object to_python() = 0;
    #endif
};

/// \ingroup Stream
/// \brief Implementation stub of a column used in HIPIPE_DEFINE_COLUMN.
template <typename ExampleType, typename ColumnName>
class column_base : public abstract_column {
public:

    // types //

    using example_type = ExampleType;
    using batch_type = std::vector<example_type>;

private:

    batch_type value_;

public:

    // constructors //

    column_base() = default;

    column_base(example_type&& rhs)
    {
        value_.emplace_back(std::move(rhs));
    }

    column_base(const example_type& rhs)
      : value_{rhs}
    {}

    column_base(std::initializer_list<example_type> rhs)
      : value_{std::move(rhs)}
    {}

    column_base(std::vector<example_type>&& rhs)
      : value_{std::move(rhs)}
    {}

    column_base(const std::vector<example_type>& rhs)
      : value_{rhs}
    {}

    // batching utilities //

    std::size_t size() const override
    {
        // TODO rename value to e.g. data
        return value_.size();
    }

    /// \brief Steal the given number of examples from this column
    /// and create a new column consisting of them.
    ///
    /// \param n The number of examples to steal.
    std::unique_ptr<abstract_column> take(std::size_t n) override
    {
        if (n > value_.size()) {
            throw std::runtime_error{"hipipe: Attempting to take "
              + std::to_string(n) + " values out of column `" + name()
              + "` with " + std::to_string(size()) + " values."};

        }
        batch_type taken_examples(n);
        std::move(value_.begin(), value_.begin() + n, taken_examples.begin());
        // TODO linear complexity, batch_type should be a deque
        value_.erase(value_.begin(), value_.begin() + n);
        return std::make_unique<ColumnName>(std::move(taken_examples));
    }

    /// \brief Concatenate the examples from two columns.
    ///
    /// \param rhs The column whose examples will be appended.
    void push_back(std::unique_ptr<abstract_column> rhs) override
    {
        ColumnName& typed_rhs = dynamic_cast<ColumnName&>(*rhs);
        value_.reserve(value_.size() + typed_rhs.value_.size());
        for (example_type& example : typed_rhs.value_) {
            value_.push_back(std::move(example));
        }
    }

    // value accessors //

    batch_type& value() { return value_; }
    const batch_type& value() const { return value_; }

    // python converters

    /// \brief Convert the column value to a python object.
    ///
    /// The values (i.e, the batches) are converted to Python lists using to_python().
    /// If the batch is a multidimensional std::vector<std::vector<...>>, it
    /// is converted to multidimensional Python list.
    ///
    /// WARNING: The data are moved out of this object. Using them results
    /// in undefined behavior.
    #ifdef HIPIPE_BUILD_PYTHON
    boost::python::object to_python() override
    {
        return hipipe::python::utility::to_python(std::move(value_));
    }
    #endif
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
    void throw_check_contains() const
    {
        if (!columns_.count(std::type_index{typeid(Column)})) {
            throw std::runtime_error{
              std::string{"Trying to retrieve column `"} + Column{}.name()
              + "`, but the batch contains no such column."};
        }
    }

    template<typename Column>
    typename Column::batch_type& extract()
    {
        throw_check_contains<Column>();
        return columns_.at(std::type_index{typeid(Column)})->extract<Column>();
    }

    template<typename Column>
    const typename Column::batch_type& extract() const
    {
        throw_check_contains<Column>();
        return columns_.at(std::type_index{typeid(Column)})->extract<Column>();
    }

    // raw column access //

    template<typename Column>
    Column& at()
    {
        throw_check_contains<Column>();
        return *columns_.at(std::type_index{typeid(Column)});
    }

    template<typename Column>
    const Column& at() const
    {
        throw_check_contains<Column>();
        return *columns_.at(std::type_index{typeid(Column)});
    }

    // column insertion/rewrite //

    void insert(std::type_index key, std::unique_ptr<abstract_column> column)
    {
        columns_.emplace(std::move(key), std::move(column));
    }

    // TODO rename to insert_or_assign or similar
    template<typename Column, typename... Args>
    void insert(Args&&... args)
    {
        static_assert(std::is_constructible_v<Column, Args&&...>,
          "Cannot construct the given column from the provided arguments.");
        columns_[std::type_index{typeid(Column)}] =
          std::make_unique<Column>(std::forward<Args>(args)...);
    }

    // column check //

    std::size_t size() const
    {
        return columns_.size();
    }

    template<typename Column>
    bool contains() const
    {
        return columns_.count(std::type_index{typeid(Column)});
    }

    // column removal //

    template<typename Column>
    void erase()
    {
        throw_check_contains<Column>();
        columns_.erase(std::type_index{typeid(Column)});
    }

    // batching utilities //

    /// \brief Calculate the batch size.
    ///
    /// If the batch contains no columns, returns zero.
    ///
    /// \throws std::runtime_error If all the columns do not have the same size.
    std::size_t batch_size() const
    {
        if (columns_.empty()) return 0;
        std::size_t batch_size = columns_.begin()->second->size();
        for (auto it = ++columns_.begin(); it != columns_.end(); ++it) {
            if (it->second->size() != batch_size) {
                throw std::runtime_error{"hipipe: Canot deduce a batch size from a batch "
                  "with columns of different size (`" + it->second->name() + "`)."};
            }
            batch_size = it->second->size();
        }
        return batch_size;
    }

    /// \brief Steal the given number of examples from all the
    /// columns and create a new batch of them.
    ///
    /// \param n The number of examples to steal.
    batch_t take(std::size_t n)
    {
        batch_t new_batch;
        for (const auto& [key, col] : columns_) {
            new_batch.insert(key, col->take(n));
        }
        return new_batch;
    }

    /// \brief Concatenate the columns from two batches.
    ///
    /// If some of the pushed columns is not in this batch, it is inserted.
    ///
    /// \param rhs The batch whose columns will be stolen and appended.
    void push_back(batch_t rhs)
    {
        for (auto& [key, col] : rhs.columns_) {
            if (!columns_.count(key)) {
                columns_[key] = std::move(col);
            } else {
                columns_.at(key)->push_back(std::move(col));
            }
        }
    }

    /// \brief Convert all the columns into a Python `dict`.
    ///
    /// The dict is indexed by `column.name` and the value is `column.to_python()`.
    ///
    /// WARNING: The data are moved out of this objects and using this object further
    /// would result in an undefined behavior.
    // TODO conditional
    #ifdef HIPIPE_BUILD_PYTHON
    boost::python::dict to_python()
    {
        boost::python::dict res;
        for (auto it = columns_.begin(); it != columns_.end(); ++it) {
            res[it->second->name()] = it->second->to_python();
        }
        columns_.clear();
        return res;
    }
    #endif
};

using forward_stream_t = ranges::any_view<batch_t, ranges::category::forward>;
using input_stream_t = ranges::any_view<batch_t, ranges::category::input>;

}  // namespace hipipe::stream

/// \ingroup Stream
/// \brief Macro for fast column definition.
///
/// Under the hood, it creates a new type derived from column_base.
#define HIPIPE_DEFINE_COLUMN(column_name_, example_type_)                         \
struct column_name_ : hipipe::stream::column_base<example_type_, column_name_> {  \
    using hipipe::stream::column_base<example_type_, column_name_>::column_base;  \
    std::string name() const override { return #column_name_; }                   \
};
