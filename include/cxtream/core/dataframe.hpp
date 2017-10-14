/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Dataframe Dataframe class.

#ifndef CXTREAM_CORE_DATAFRAME_HPP
#define CXTREAM_CORE_DATAFRAME_HPP

#include <cxtream/core/index_mapper.hpp>
#include <cxtream/core/utility/string.hpp>
#include <cxtream/core/utility/tuple.hpp>

#include <range/v3/experimental/view/shared.hpp>
#include <range/v3/view/all.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/move.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

namespace cxtream {

/// \ingroup Dataframe
/// \brief Tabular object with convenient data access methods.
///
/// By default, all fields are stored as std::string and they are
/// cast to the requested type on demand.
template<typename DataTable = std::vector<std::vector<std::string>>>
class dataframe {
public:
    dataframe() = default;

    /// Constructs the dataset from a vector of columns of the same type.
    ///
    /// Example:
    /// \code
    ///     dataframe<> df{
    ///       // columns
    ///       std::vector<std::vector<int>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
    ///       // header
    ///       std::vector<std::string>{"A", "B", "C"}
    ///     };
    /// \endcode
    template<typename T>
    dataframe(std::vector<std::vector<T>> columns, std::vector<std::string> header = {})
    {
        assert(header.empty() || header.size() == columns.size());
        for (std::size_t i = 0; i < columns.size(); ++i) {
            std::string col_name = header.empty() ? "" : std::move(header[i]);
            col_insert(columns[i] | ranges::view::move, std::move(col_name));
        }
    }

    /// Constructs the dataset from a tuple of columns of possibly different types.
    ///
    /// Example:
    /// \code
    ///     dataframe<> df{
    ///       // columns
    ///       std::make_tuple(
    ///         std::vector<int>{1, 2, 3},
    ///         std::vector<std::string>{"a1", "a2", "a3"},
    ///         std::vector<std::string>{"1.1", "1.2", "1.3"}
    ///       ),
    ///       // header
    ///       std::vector<std::string>{"Id", "A", "B"}
    ///     };
    /// \endcode
    template<typename... Ts>
    dataframe(std::tuple<std::vector<Ts>...> columns, std::vector<std::string> header = {})
    {
        assert(header.empty() || header.size() == sizeof...(Ts));
        utility::tuple_for_each_with_index(std::move(columns),
          [this, &header](auto& column, auto index) {
              std::string col_name = header.empty() ? "" : std::move(header[index]);
              this->col_insert(column | ranges::view::move, std::move(col_name));
        });
    }

    // insertion //

    /// Inserts a new column to the dataframe.
    ///
    /// Example:
    /// \code
    ///     df.col_insert(std::vector<int>{5, 6, 7}, "C");
    /// \endcode
    template<typename Rng, typename ToStrFun = std::string (*)(ranges::range_value_type_t<Rng>)>
    std::size_t col_insert(Rng&& rng, std::string col_name = {},
                           std::function<std::string(ranges::range_value_type_t<Rng>)> cvt =
                             static_cast<ToStrFun>(utility::to_string))
    {
        assert(n_rows() == 0 || ranges::size(rng) == n_rows());
        assert((!header_.size() || col_name.size()) &&
               "The dataframe has a header, please provide the name for the new column.");
        if (col_name.size()) header_.insert(col_name);
        data_.emplace_back(rng | ranges::view::transform(cvt));
        return n_cols() - 1;
    }

    /// Inserts a new row to the dataframe.
    ///
    /// Example:
    /// \code
    ///     df.row_insert(std::make_tuple(4, "a3", true));
    /// \endcode
    template<typename... Ts>
    std::size_t row_insert(std::tuple<Ts...> row_tuple,
                           std::tuple<std::function<std::string(Ts)>...> cvts = std::make_tuple(
                             static_cast<std::string (*)(Ts)>(utility::to_string)...))
    {
        assert(n_cols() == 0 || sizeof...(Ts) == n_cols());
        utility::tuple_for_each_with_index(std::move(row_tuple),
          [this, &cvts](auto& field, auto index) {
              this->data_[index].push_back(std::get<index>(cvts)(std::move(field)));
        });
        return n_rows() - 1;
    }

    // drop //

    /// Drop a column with the given index.
    ///
    /// \throws std::out_of_range If the column is not in the dataframe.
    void icol_drop(std::size_t col_index)
    {
        throw_check_col_idx(col_index);
        // remove the column from the header
        if (header_.size()) {
            std::vector<std::string> new_header = header_.values();
            new_header.erase(new_header.begin() + col_index);
            header_ = new_header;
        }
        // remove the column from the data
        data_.erase(data_.begin() + col_index);
    }

    /// Drop a column with the given name.
    ///
    /// \throws std::out_of_range If the column is not in the dataframe.
    void col_drop(const std::string& col_name)
    {
        throw_check_col_name(col_name);
        return icol_drop(header_.index_for(col_name));
    }

    /// Drop a row.
    ///
    /// \throws std::out_of_range If the row is not in the dataframe.
    void row_drop(const std::size_t row_idx)
    {
        throw_check_row_idx(row_idx);
        for (auto& column : data_) {
            column.erase(column.begin() + row_idx);
        }
    }

    // raw column access //

    /// Return a raw view of a column.
    ///
    /// The data can be directly changed by writing to the view.
    ///
    /// Example:
    /// \code
    ///     df.raw_icol(3)[2] = "new_value";
    /// \endcode
    ///
    /// \returns A of range of std::string&.
    /// \throws std::out_of_range If the column is not in the dataframe.
    auto raw_icol(std::size_t col_index)
    {
        throw_check_col_idx(col_index);
        return raw_cols()[col_index] | ranges::view::all;
    }

    /// Return a raw view of a column.
    ///
    /// \returns A of range of const std::string&.
    /// \throws std::out_of_range If the column is not in the dataframe.
    auto raw_icol(std::size_t col_index) const
    {
        throw_check_col_idx(col_index);
        return raw_cols()[col_index] | ranges::view::all;
    }

    /// Return a raw view of a column.
    ///
    /// The data can be directly changed by writing to the view.
    ///
    /// Example:
    /// \code
    ///     df.raw_col("long column")[2] = "new_value";
    /// \endcode
    ///
    /// \returns A of range of std::string&.
    /// \throws std::out_of_range If the column is not in the dataframe.
    auto raw_col(const std::string& col_name)
    {
        throw_check_col_name(col_name);
        return raw_icol(header_.index_for(col_name));
    }

    /// Return a raw view of a column.
    ///
    /// This is just a const overload of the non-const raw_col().
    ///
    /// \returns A of range of const std::string&.
    /// \throws std::out_of_range If the column is not in the dataframe.
    auto raw_col(const std::string& col_name) const
    {
        throw_check_col_name(col_name);
        return raw_icol(header_.index_for(col_name));
    }

    // typed column access //

    /// Return a typed view of a column.
    ///
    /// By default, this function does not provide a direct access to the stored data.
    /// Instead, each field is converted to the type T and a copy is returned.
    ///
    /// Example:
    /// \code
    ///     std::vector<long> data = df.icol<long>(3);
    /// \endcode
    ///
    /// \returns A range of T.
    /// \throws std::out_of_range If the column is not in the dataframe.
    template<typename T>
    auto icol(std::size_t col_index,
              std::function<T(std::string)> cvt = utility::string_to<T>) const
    {
        return raw_icol(col_index) | ranges::view::transform(cvt);
    }

    /// Return a typed view of a column.
    ///
    /// By default, this function does not provide a direct access to the stored data.
    /// Instead, each field is converted to the type T and a copy is returned.
    ///
    /// Example:
    /// \code
    ///     std::vector<long> data = df.col<long>("long column");
    /// \endcode
    ///
    /// \returns A range of T.
    /// \throws std::out_of_range If the column is not in the dataframe.
    template<typename T>
    auto col(const std::string& col_name,
             std::function<T(std::string)> cvt = utility::string_to<T>) const
    {
        throw_check_col_name(col_name);
        return icol<T>(header_.index_for(col_name), std::move(cvt));
    }

    // raw multi column access //

    /// Return a raw view of all columns.
    ///
    /// The data can be directly changed by writing to the view.
    ///
    /// Example:
    /// \code
    ///     // get the third row from the sixth column
    ///     std::string field = df.raw_cols()[5][2];
    /// \endcode
    ///
    /// \returns A range of ranges of std::string&.
    auto raw_cols()
    {
        return data_ | ranges::view::transform(ranges::view::all);
    }

    /// Return a raw view of all columns.
    ///
    /// This is just a const overload of the non-const argument-less raw_cols().
    ///
    /// \returns A range of ranges of const std::string&.
    auto raw_cols() const
    {
        return data_ | ranges::view::transform(ranges::view::all);
    }

    /// Return a raw view of multiple columns.
    ///
    /// The data can be directly changed by writing to the view.
    ///
    /// Example:
    /// \code
    ///     // get the third row from the sixth column (with index 5)
    ///     std::string field = df.raw_icols({1, 5})[1][2];
    /// \endcode
    ///
    /// \returns A range of ranges of std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_icols(std::vector<std::size_t> col_indexes)
    {
        for (auto& col_idx : col_indexes) throw_check_col_idx(col_idx);
        return raw_icols_impl(this, std::move(col_indexes));
    }

    /// Return a raw view of multiple columns.
    ///
    /// This is just a const overload of the non-const raw_icols().
    ///
    /// \returns A range of ranges of const std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_icols(std::vector<std::size_t> col_indexes) const
    {
        for (auto& col_idx : col_indexes) throw_check_col_idx(col_idx);
        return raw_icols_impl(this, std::move(col_indexes));
    }

    /// Return a raw view of multiple columns.
    ///
    /// The data can be directly changed by writing to the view.
    ///
    /// Example:
    /// \code
    ///     // get the sixth row from the column named "column 2"
    ///     std::string field = df.raw_cols({"column 1", "column 2"})[1][5];
    /// \endcode
    ///
    /// \returns A range of ranges of std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_cols(const std::vector<std::string>& col_names)
    {
        for (auto& col_name : col_names) throw_check_col_name(col_name);
        return raw_icols(header_.index_for(col_names));
    }

    /// Return a raw view of multiple columns.
    ///
    /// This is just a const overload of the non-const raw_cols().
    ///
    /// \returns A range of ranges of const std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_cols(const std::vector<std::string>& col_names) const
    {
        for (auto& col_name : col_names) throw_check_col_name(col_name);
        return raw_icols(header_.index_for(col_names));
    }

    // typed multi column access //

    /// Return a typed view of multiple columns.
    ///
    /// Example:
    /// \code
    ///     std::tuple<std::vector<int>, std::vector<double>> data = df.icols<int, double>({1, 2});
    /// \endcode
    ///
    /// \returns A tuple of ranges of Ts.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename... Ts>
    auto icols(std::vector<std::size_t> col_indexes,
               std::tuple<std::function<Ts(std::string)>...> cvts =
                 std::make_tuple(utility::string_to<Ts>...)) const
    {
        assert(sizeof...(Ts) == ranges::size(col_indexes));
        return utility::tuple_transform_with_index(std::move(cvts),
          [raw_cols = raw_icols(std::move(col_indexes))](auto&& cvt, auto i) {
              return raw_cols[i] | ranges::view::transform(std::move(cvt));
        });
    }

    /// Return a typed view of multiple columns.
    ///
    /// Example:
    /// \code
    ///     std::tuple<std::vector<int>, std::vector<double>> data =
    ///       df.cols<int, double>({"column 1", "column 2"});
    /// \endcode
    ///
    /// \returns A tuple of ranges of Ts.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename... Ts>
    auto cols(const std::vector<std::string>& col_names,
              std::tuple<std::function<Ts(std::string)>...> cvts =
                std::make_tuple(utility::string_to<Ts>...)) const
    {
        for (auto& col_name : col_names) throw_check_col_name(col_name);
        return icols<Ts...>(header_.index_for(col_names), std::move(cvts));
    }

    /// Return a raw view of all rows.
    ///
    /// Example:
    /// \code
    ///     // get the third row from the sixth column
    ///     std::string field = df.raw_rows()[2][5];
    /// \endcode
    ///
    /// \returns A range of ranges of std::string&.
    auto raw_rows()
    {
        return raw_rows_impl(this);
    }

    /// Return a raw view of all rows.
    ///
    /// This is just a const overload of the non-const argument-less raw_rows().
    ///
    /// \returns A range of ranges of const std::string&.
    auto raw_rows() const
    {
        return raw_rows_impl(this);
    }

    /// Return a raw view of multiple rows.
    ///
    /// Example:
    /// \code
    ///     // get the third row from the sixth column (with index 5)
    ///     std::string field = df.raw_irows({3, 5})[2][1];
    /// \endcode
    ///
    /// \returns A range of ranges of std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_irows(std::vector<std::size_t> col_indexes)
    {
        for (auto& col_idx : col_indexes) throw_check_col_idx(col_idx);
        return raw_irows_impl(this, std::move(col_indexes));
    }

    /// Return a raw view of multiple rows.
    ///
    /// This is just a const overload of the non-const raw_irows().
    ///
    /// \returns A range of ranges of const std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_irows(std::vector<std::size_t> col_indexes) const
    {
        for (auto& col_idx : col_indexes) throw_check_col_idx(col_idx);
        return raw_irows_impl(this, std::move(col_indexes));
    }

    /// Return a raw view of multiple rows.
    ///
    /// Example:
    /// \code
    ///     // get the third row from column named "col2"
    ///     std::string field = df.raw_rows({"col1", "col2"})[2][1];
    /// \endcode
    ///
    /// \returns A range of ranges of std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_rows(const std::vector<std::string>& col_names)
    {
        for (auto& col_name : col_names) throw_check_col_name(col_name);
        return raw_irows(header_.index_for(col_names));
    }

    /// Return a raw view of multiple rows.
    ///
    /// This is just a const overload of the non-const raw_rows().
    ///
    /// \returns A range of ranges of const std::string&.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    auto raw_rows(const std::vector<std::string>& col_names) const
    {
        for (auto& col_name : col_names) throw_check_col_name(col_name);
        return raw_irows(header_.index_for(col_names));
    }

    // typed row access //

    /// Return a typed view of multiple rows.
    ///
    /// This function provides the same data as icols() but transposed.
    ///
    /// Example:
    /// \code
    ///     std::vector<std::tuple<int, double>> data =
    ///       df.irows<int, double>({0, 2});
    /// \endcode
    ///
    /// \returns A range of tuples of Ts.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename... Ts>
    auto irows(std::vector<std::size_t> col_indexes,
               std::tuple<std::function<Ts(std::string)>...> cvts =
                 std::make_tuple(utility::string_to<Ts>...)) const
    {
        return std::experimental::apply(
          ranges::view::zip,
          icols<Ts...>(std::move(col_indexes), std::move(cvts)));
    }

    /// Return a typed view of multiple rows.
    ///
    /// This function provides the same data as cols() but transposed.
    ///
    /// Example:
    /// \code
    ///     std::vector<std::tuple<int, double>> data =
    ///       df.rows<int, double>({"int_col", "double_col"});
    /// \endcode
    ///
    /// \returns A range of tuples of Ts.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename... Ts>
    auto rows(const std::vector<std::string>& col_names,
              std::tuple<std::function<Ts(std::string)>...> cvts =
                std::make_tuple(utility::string_to<Ts>...)) const
    {
        for (auto& col_name : col_names) throw_check_col_name(col_name);
        return irows<Ts...>(header_.index_for(col_names), std::move(cvts));
    }

    // typed indexed single column access //

    /// Return an indexed typed view of a single column.
    ///
    /// This function returns a range of tuples, where the first tuple element is 
    /// from the key column and the second element is from the value column.
    /// This range can be used to construct a map or a hashmap.
    ///
    /// Example:
    /// \code
    ///     std::unordered_map<int, double> mapper = df.index_icol<int, double>(0, 1);
    /// \endcode
    ///
    /// \param key_col_index Index of the column to be used as key.
    /// \param val_col_index Index of the column to be used as value.
    /// \param key_col_cvt Function that is used to convert the keys from std::string to IndexT.
    /// \param val_col_cvt Function that is used to convert the values from std::string to ValueT.
    /// \returns A range of tuples <key, value>.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template <typename IndexT, typename ColT>
    auto index_icol(std::size_t key_col_index,
                    std::size_t val_col_index,
                    std::function<IndexT(std::string)> key_col_cvt = utility::string_to<IndexT>,
                    std::function<ColT(std::string)> val_col_cvt = utility::string_to<ColT>) const
    {
        auto key_col = icol<IndexT>(key_col_index, std::move(key_col_cvt));
        auto val_col = icol<ColT>(val_col_index, std::move(val_col_cvt));
        return ranges::view::zip(key_col, val_col);
    }

    /// Return an indexed typed view of a single column.
    ///
    /// \code
    ///     std::unordered_map<int, double> mapper = df.index_col<int, double>("first", "second");
    /// \endcode
    ///
    /// This function is the same as index_icol(), but columns are selected by name.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename IndexT, typename ColT>
    auto index_col(const std::string& key_col_name,
                   const std::string& val_col_name,
                   std::function<IndexT(std::string)> key_col_cvt = utility::string_to<IndexT>,
                   std::function<ColT(std::string)> val_col_cvt = utility::string_to<ColT>) const
    {
        throw_check_col_name(key_col_name);
        throw_check_col_name(val_col_name);
        return index_icol(header_.index_for(key_col_name),
                          header_.index_for(val_col_name),
                          std::move(key_col_cvt),
                          std::move(val_col_cvt));
    }

    // typed indexed multiple column access //

    /// Return an indexed typed view of multiple columns.
    ///
    /// See index_icol().
    ///
    /// \code
    ///     std::unordered_map<int, std::tuple<long, double>> mapper =
    ///       df.index_icols<int, long, double>(0, {1, 2});
    /// \endcode
    ///
    /// This function is similar to index_icol(), but value type is a tuple of Ts.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename IndexT, typename... Ts>
    auto index_icols(std::size_t key_col_index,
                     std::vector<std::size_t> val_col_indexes,
                     std::function<IndexT(std::string)> key_col_cvt =
                       utility::string_to<IndexT>,
                     std::tuple<std::function<Ts(std::string)>...> val_col_cvts =
                       std::make_tuple(utility::string_to<Ts>...)) const
    {
        auto key_col = icol<IndexT>(key_col_index, std::move(key_col_cvt));
        auto val_cols = irows<Ts...>(std::move(val_col_indexes), std::move(val_col_cvts));
        return ranges::view::zip(key_col, val_cols);
    }

    /// Return an indexed typed view of multiple columns.
    ///
    /// See index_icol().
    ///
    /// \code
    ///     std::unordered_map<int, std::tuple<long, double>> mapper =
    ///       df.index_cols<int, long, double>("id", {"col1", "col2"});
    /// \endcode
    ///
    /// This function is similar to index_icols(), but columns are selected by name.
    /// \throws std::out_of_range If any of the columns is not in the dataframe.
    template<typename IndexT, typename... Ts>
    auto index_cols(const std::string& key_col_name,
                    const std::vector<std::string>& val_col_names,
                    std::function<IndexT(std::string)> key_col_cvt =
                      utility::string_to<IndexT>,
                    std::tuple<std::function<Ts(std::string)>...> val_col_cvts =
                      std::make_tuple(utility::string_to<Ts>...)) const
    {
        throw_check_col_name(key_col_name);
        for (auto& col_name : val_col_names) throw_check_col_name(col_name);
        assert(header_.size() && "Dataframe has no header, cannot index by column name.");
        return index_icols(header_.index_for(key_col_name),
                           header_.index_for(val_col_names),
                           std::move(key_col_cvt),
                           std::move(val_col_cvts));
    }

    // shape functions //

    /// Return the number of columns.
    std::size_t n_cols() const
    {
        return data_.size();
    }

    /// Return the number of rows (excluding header).
    std::size_t n_rows() const
    {
        if (n_cols() == 0) return 0;
        return data_.front().size();
    }

    /// Return the names of columns.
    std::vector<std::string> header() const
    {
        return header_.values();
    }

    /// Return the reference to the raw data table.
    DataTable& data()
    {
        return data_;
    }

private:
    void throw_check_row_idx(std::size_t row_idx) const
    {
        if (row_idx < 0 || row_idx >= n_rows()) {
            throw std::out_of_range{"Row index " + std::to_string(row_idx) +
              " is not in a dataframe with " + std::to_string(n_rows()) + " rows."};
        }
    }

    void throw_check_col_idx(std::size_t col_idx) const
    {
        if (col_idx < 0 || col_idx >= n_cols()) {
            throw std::out_of_range{"Column index " + std::to_string(col_idx) +
              " is not in a dataframe with " + std::to_string(n_cols()) + " columns."};
        }
    }

    void throw_check_col_name(const std::string& col_name) const
    {
        if (header_.size() == 0) {
            throw std::out_of_range{"Dataframe has no header, cannot index by column name."};
        }
        if (!header_.contains(col_name)) {
            throw std::out_of_range{"Column " + col_name + " not found in the dataframe."};
        }
    }

    template <typename This>
    static auto raw_irows_impl(This this_ptr, std::vector<std::size_t> col_indexes)
    {
        namespace view = ranges::view;
        return view::iota(0UL, this_ptr->n_rows())
          | view::transform([this_ptr, col_indexes=std::move(col_indexes)](std::size_t i) {
                return this_ptr->raw_icols(col_indexes)
                  // decltype(auto) to make sure a reference is returned
                  | view::transform([i](auto&& col) -> decltype(auto) {
                        return col[i];
                    });
            });
      }

      template<typename This>
      static auto raw_rows_impl(This this_ptr)
      {
        namespace view = ranges::view;
        return view::iota(0UL, this_ptr->n_rows())
          | view::transform([this_ptr](std::size_t i) {
                return view::iota(0UL, this_ptr->n_cols())
                  // decltype(auto) to make sure a reference is returned
                  | view::transform([this_ptr, i](std::size_t j) -> decltype(auto) {
                        return this_ptr->raw_cols()[j][i];
                    });
            });
      }

      template<typename This>
      static auto raw_icols_impl(This this_ptr, std::vector<std::size_t> col_indexes)
      {
        return std::move(col_indexes)
          | ranges::experimental::view::shared
          | ranges::view::transform([this_ptr](std::size_t idx) {
                return this_ptr->raw_cols()[idx];
            });
      }


      // data storage //

      DataTable data_;

      using header_t = index_mapper<std::string>;
      header_t header_;

};  // class dataframe

/// \ingroup Dataframe
/// \brief Pretty printing of dataframe to std::ostream.
template<typename DataTable>
std::ostream& operator<<(std::ostream& out, const dataframe<DataTable>& df)
{
    namespace view = ranges::view;
    // calculate the width of the columns using their longest field
    std::vector<std::size_t> col_widths = df.raw_cols()
      | view::transform([](auto&& col) {
            std::vector<std::size_t> elem_sizes = col
              | view::transform([](auto& field) { return ranges::size(field); });
            return ranges::max(elem_sizes) + 2;
        });

    auto header = df.header();
    if (header.size()) {
        // update col_widths using header widths
        col_widths = view::zip(col_widths, header)
          | view::transform([](auto&& tpl) {
                return std::max(std::get<0>(tpl), std::get<1>(tpl).size() + 2);
            });

        // print header
        for (std::size_t j = 0; j < header.size(); ++j) {
            out << std::setw(col_widths[j]) << header[j];
            if (j + 1 < header.size()) out << '|';
            else out << '\n';
        }

        // print header and data separator
        for (std::size_t j = 0; j < header.size(); ++j) {
            out << std::setw(col_widths[j]) << std::setfill('-');
            if (j + 1 < header.size()) out << '-' << '+';
            else out << '-' << '\n';
        }
        out << std::setfill(' ');
    }

    // print data
    for (std::size_t i = 0; i < df.n_rows(); ++i) {
        for (std::size_t j = 0; j < df.n_cols(); ++j) {
            out << std::setw(col_widths[j]) << df.raw_rows()[i][j];
            if (j + 1 < df.n_cols()) out << '|';
            else out << '\n';
        }
    }

    return out;
}

}  // end namespace cxtream
#endif
