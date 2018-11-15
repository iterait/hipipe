/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once

#include <hipipe/build_config.hpp>
#include <hipipe/core/stream/column_t.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

namespace hipipe::stream {

/// \ingroup Stream
/// \brief Container for multiple columns.
///
/// This is the value type of the stream.
class batch {
private:

    /// The stored columns.
    std::unordered_map<std::type_index, std::unique_ptr<abstract_column>> columns_;

    /// \brief Check whether the given column is present in the stream.
    ///
    /// \tparam Column The column to be retrieved.
    /// \throws std::runtime_error If the check fails.
    template<typename Column>
    void throw_check_contains() const
    {
        if (!columns_.count(std::type_index{typeid(Column)})) {
            throw std::runtime_error{
              std::string{"Trying to retrieve column `"} + Column{}.name()
              + "`, but the batch contains no such column."};
        }
    }

public:

    // constructors //

    batch() = default;
    batch(const batch&) = delete;
    batch(batch&&) = default;

    // value extraction //

    /// \brief Extract a reference to the stored data of the given column.
    ///
    /// Example:
    /// \code
    ///     HIPIPE_DEFINE_COLUMN(IntCol, int)
    ///     stream::batch b;
    ///     b.insert_or_assign<IntCol>(std::vector<int>{0, 1, 2});
    ///     b.extract<IntCol>() == std::vector<int>{0, 1, 2};
    /// \endcode
    ///
    /// \tparam Column The column whose data should be retrieved.
    /// \throws std::runtime_error If the batch does not contain the given column.
    template<typename Column>
    typename Column::data_type& extract()
    {
        throw_check_contains<Column>();
        return columns_.at(std::type_index{typeid(Column)})->extract<Column>();
    }

    /// Extract a const reference to the stored data of the given column.
    ///
    /// This is the same as previous, but returns a const reference.
    template<typename Column>
    const typename Column::data_type& extract() const
    {
        throw_check_contains<Column>();
        return columns_.at(std::type_index{typeid(Column)})->extract<Column>();
    }

    // column insertion/rewrite //

    /// \brief Insert a new column to the batch or overwrite an existing one.
    ///
    /// The parameters of this function are forwarded to the constructor of the given column.
    ///
    /// Example:
    /// \code
    ///     HIPIPE_DEFINE_COLUMN(IntCol, int)
    ///     stream::batch b;
    ///     b.insert_or_assign<IntCol>(std::vector<int>{0, 1, 2});
    ///     b.extract<IntCol>() == std::vector<int>{0, 1, 2};
    /// \endcode
    ///
    /// \tparam Column The column to be inserted.
    /// \param args The parameters that are forwarded to the constructor of the column.
    template<typename Column, typename... Args>
    void insert_or_assign(Args&&... args)
    {
        static_assert(std::is_constructible_v<Column, Args&&...>,
          "Cannot construct the given column from the provided arguments.");
        columns_.insert_or_assign(
          std::type_index{typeid(Column)},
          std::make_unique<Column>(std::forward<Args>(args)...));
    }

    // column check //

    /// \brief Get the number of columns in the batch.
    std::size_t size() const
    {
        return columns_.size();
    }

    /// \brief Check whether the given column is present in the batch.
    ///
    /// \tparam Column The column to be checked.
    template<typename Column>
    bool contains() const
    {
        return columns_.count(std::type_index{typeid(Column)});
    }

    // column removal //

    /// \brief Remove the given column from the batch.
    ///
    /// \tparam Column The column to be removed.
    /// \throws std::runtime_error If the columns is not in the batch.
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
    /// The behaviour of this function is equivalent to calling \ref column_base::take()
    /// of on all the columns.
    ///
    /// \param n The number of examples to steal.
    batch take(std::size_t n)
    {
        batch new_batch;
        for (const auto& [key, col] : columns_) {
            new_batch.columns_.insert_or_assign(key, col->take(n));
        }
        return new_batch;
    }

    /// \brief Concatenate the columns from two batches.
    ///
    /// If some of the pushed columns is not in this batch, it is inserted as a new column.
    ///
    /// The behaviour of this function is almost equivalent to calling
    /// \ref column_base::push_back() of on all the columns.
    ///
    /// \param rhs The batch whose columns will be stolen and appended.
    void push_back(batch rhs)
    {
        for (auto& [key, col] : rhs.columns_) {
            if (!columns_.count(key)) {
                columns_[key] = std::move(col);
            } else {
                columns_.at(key)->push_back(std::move(col));
            }
        }
    }

    /// \brief Convert all the columns to a Python `dict`.
    ///
    /// The dict is indexed by `column.name()` and the value is `column.to_python()`.
    ///
    /// WARNING: The data are moved out of this objects and using this object further
    /// will result in an undefined behavior.
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


/// Alias to the batch class.
using batch_t = batch;

}  // namespace hipipe::stream
