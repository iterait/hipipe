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

#ifdef HIPIPE_BUILD_PYTHON
#include <hipipe/core/python/utility/ndim_vector_converter.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#endif

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipipe::stream {

/// \ingroup Stream
/// \brief Abstract base class for HiPipe columns.
class abstract_column {
private:

    /// \brief Returns a runtime error informing the user that the extraction
    /// of the given column from this abstract column failed.
    template<typename Column>
    std::runtime_error extraction_error() const
    {
        return std::runtime_error{
          std::string{"Trying to extract column `"} + Column{}.name()
          + "` from a column of type `" + this->name() + "`."};
    }

public:
    // typed data extractor //

    /// Extract a reference to the stored data.
    ///
    /// Example:
    /// \code
    ///     HIPIPE_DEFINE_COLUMN(IntCol, int)
    ///     std::unique_ptr<IntCol> col = std::make_unique<IntCol>();
    ///     col->data().assign({1, 2, 3});
    ///     std::unique_ptr<abstract_column> ab_col = std::move(col);
    ///     ab_col->extract<IntCol>() == std::vector<int>({1, 2, 3});
    /// \endcode
    ///
    /// \tparam Column The column that is represented by this abstract column.
    /// \throws std::runtime_error If a column not corresponding to the stored one is requested.
    template<typename Column>
    typename Column::data_type& extract()
    {
        try {
            return dynamic_cast<Column&>(*this).data();
        } catch (const std::bad_cast&) {
            throw extraction_error<Column>();
        }
    }

    /// Extract a const reference to the stored data.
    ///
    /// The same as previous, but returns a const reference.
    template<typename Column>
    const typename Column::data_type& extract() const
    {
        try {
            return dynamic_cast<const Column&>(*this).data();
        } catch (const std::bad_cast&) {
            throw extraction_error<Column>();
        }
    }

    // name accessor //

    /// Retrieve the name of the stored column.
    ///
    /// This function is automatically overriden and returns the string corresponding
    /// to the first parameter of HIPIPE_DEFINE_COLUMN macro.
    virtual std::string name() const = 0;

    // batch utilities //

    /// Retrieve the number of examples stored in the column.
    virtual std::size_t size() const = 0;

    /// Concatenate the data of two columns.
    ///
    /// See the corresponding function in \ref column_base class for more info.
    virtual void push_back(std::unique_ptr<abstract_column> rhs) = 0;

    /// Steal the given number of examples and build a new column of them.
    ///
    /// See the corresponding function in \ref column_base class for more info.
    virtual std::unique_ptr<abstract_column> take(std::size_t n) = 0;

    // python conversion //

    #ifdef HIPIPE_BUILD_PYTHON
    /// Convert the column data to a python object.
    ///
    /// See the corresponding function in \ref column_base class for more info.
    virtual pybind11::object to_python() = 0;
    #endif

    // virtual destructor //

    virtual ~abstract_column() = default;
};


/// \ingroup Stream
/// \brief Implementation stub of a column defined by HIPIPE_DEFINE_COLUMN macro.
template <typename ColumnName, typename ExampleType>
class column_base : public abstract_column {
public:

    /// The type of a single example.
    using example_type = ExampleType;
    /// The type of multiple examples. This is what the column actually stores.
    using data_type = std::vector<example_type>;

private:

    /// The stored data.
    data_type data_;

public:

    // constructors //

    column_base() = default;
    column_base(column_base&&) = default;

    /// \brief The constructor.
    ///
    /// The constructor forwards its arguments to the constructor
    /// of the data_type.
    template <typename... Args>
    column_base(Args&&... args)
      : data_{std::forward<Args>(args)...}
    { }

    // batching utilities //

    /// Get the number of examples in this column.
    std::size_t size() const override
    {
        return data_.size();
    }

    /// \brief Steal the given number of examples from this column
    /// and create a new column out of those.
    ///
    /// Example:
    /// \code
    ///     HIPIPE_DEFINE_COLUMN(IntCol, int)
    ///     IntCol col1;
    ///     col1.data().assign({1, 2, 3, 4, 5});
    ///     std::unique_ptr<abstract_column> col2 = col1.take(3);
    ///     /// col1 contains {4, 5}
    ///     /// col2 contains {1, 2, 3}
    /// \endcode
    ///
    /// Developer TODO: At the moment, this operation has linear complexity.
    /// Maybe we could store the data as a std::deque instead of std::vector.
    ///
    /// \param n The number of examples to steal.
    /// \throws std::runtime_error If attempting to take more than there is.
    std::unique_ptr<abstract_column> take(std::size_t n) override
    {
        if (n > data_.size()) {
            throw std::runtime_error{"hipipe: Attempting to take "
              + std::to_string(n) + " examples out of column `" + name()
              + "` with " + std::to_string(size()) + " examples."};
        }
        data_type taken_examples(n);
        std::move(data_.begin(), data_.begin() + n, taken_examples.begin());
        data_.erase(data_.begin(), data_.begin() + n);
        return std::make_unique<ColumnName>(std::move(taken_examples));
    }

    /// \brief Concatenate the examples from two columns of the same type.
    ///
    /// Example:
    /// \code
    ///     HIPIPE_DEFINE_COLUMN(IntCol, int)
    ///     auto col1 = std::make_unique<IntCol>();
    ///     col1->data().assign({1, 2, 3});
    ///     auto col2 = std::make_unique<IntCol>();
    ///     col2->data().assign({4, 5, 6});
    ///     col1->push_back(std::move(col2));
    ///     // col1 contains {1, 2, 3, 4, 5, 6}
    ///     // col2 should not be used anymore 
    /// \endcode
    ///
    /// \param rhs The column whose examples will be appended. It needs to be
    ///            the same type (i.e., the same ColumnName) as this column.
    void push_back(std::unique_ptr<abstract_column> rhs) override
    {
        try {
            ColumnName& typed_rhs = dynamic_cast<ColumnName&>(*rhs);
            data_.reserve(data_.size() + typed_rhs.data_.size());
            for (example_type& example : typed_rhs.data_) {
                data_.push_back(std::move(example));
            }
        } catch (const std::bad_cast&) {
            throw std::runtime_error{"hipipe: Attempting to push back "
              "column `" + rhs->name() + "` to column `" + name() + "."};
        }
    }

    // data accessors //

    /// \brief Get a reference to the stored vector of examples.
    data_type& data() { return data_; }

    /// \brief Get a const reference to the stored vector of examples.
    const data_type& data() const { return data_; }

    // python converter //

    /// \brief Convert the column data to a python object.
    ///
    /// This basically converts the data (i.e., the vector of examples) to Python
    /// object using \ref python::utility::to_python(std::vector<T>).
    ///
    /// WARNING: The data are moved out of this object. Using this object further
    /// results in undefined behavior.
    #ifdef HIPIPE_BUILD_PYTHON
    pybind11::object to_python() override
    {
       return hipipe::python::utility::to_python(std::move(data_));
    }

    #endif
};

}  // namespace hipipe::stream


/// \ingroup Stream
/// \brief Macro for fast column definition.
///
/// Under the hood, it creates a new type derived from column_base.
#define HIPIPE_DEFINE_COLUMN(column_name_, example_type_)                         \
struct column_name_ : hipipe::stream::column_base<column_name_, example_type_> {  \
    using hipipe::stream::column_base<column_name_, example_type_>::column_base;  \
    std::string name() const override { return #column_name_; }                   \
};
