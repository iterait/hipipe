/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup CSV CSV parser.

#pragma once

#include <hipipe/core/dataframe.hpp>

#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace hipipe {


/// \ingroup CSV
/// \brief Parse and iterate over CSV formatted rows from an istream.
///
/// Beware, escaping double quotes is allowed using backslash, not another double quote.
/// Escaping is only allowed if the first non-whitespace character of a field is a double quote.
///
/// Usage:
/// \code
///     std::istringstream simple_csv{"Id, A," R"("Quoted \"column\"") "\n 1, a1, 1.1"};
///     csv_istream_range csv_rows{simple_csv};
///     // csv_rows == {{"Id", "A", R"("Quoted \"column\"")"}, {"1", "a1", "1.1"}}
/// \endcode
///
/// \throws std::ios_base::failure if badbit is triggered.
class csv_istream_range : public ranges::view_facade<csv_istream_range> {
private:
    /// \cond
    friend ranges::range_access;
    /// \endcond
    using single_pass = std::true_type;
    enum class RowPosition{Normal, Last, End};

    std::istream* in_;
    char separator_;
    char quote_;
    char escape_;

    std::vector<std::string> row_;
    RowPosition row_position_ = RowPosition::Normal;

    class cursor {
    private:
        csv_istream_range* rng_;

    public:
        cursor() = default;
        explicit cursor(csv_istream_range& rng) noexcept
          : rng_{&rng}
        {}

        void next()
        {
            rng_->next();
        }

        std::vector<std::string>& read() const noexcept
        {
            return rng_->row_;
        }

        std::vector<std::string>&& move() const noexcept
        {
            return std::move(rng_->row_);
        }

        bool equal(ranges::default_sentinel) const noexcept
        {
            return rng_->row_position_ == RowPosition::End;
        }
    };

    // parse csv field and return whether the next separator is found
    std::tuple<std::string, bool> parse_field();

    // parse csv row
    void next();

    cursor begin_cursor() { return cursor{*this}; }

public:
    csv_istream_range() = default;

    explicit csv_istream_range(std::istream& in,
                               char separator = ',',
                               char quote = '"',
                               char escape = '\\');
};

/// \ingroup CSV
/// \brief Parse csv file from an std::istream.
///
/// Parsing has the same rules as for csv_istream_range.
///
/// \param in The input stream.
/// \param drop How many lines should be ignored at the very beginning of the stream.
/// \param has_header Whether a header row should be parsed (after drop).
/// \param separator Field separator.
/// \param quote Quote character.
/// \param escape Character used to escape a quote inside quotes.
dataframe read_csv(
  std::istream& in,
  int drop = 0,
  bool has_header = true,
  char separator = ',',
  char quote = '"',
  char escape = '\\');


/// \ingroup CSV
/// \brief Same as read_csv() but read directly from a file.
/// \throws std::ios_base::failure If the specified file cannot be opened.
dataframe read_csv(
  const std::experimental::filesystem::path& file,
  int drop = 0,
  bool header = true,
  char separator = ',',
  char quote = '"',
  char escape = '\\');


/// \ingroup CSV
/// \brief Write a single csv row to an std::ostream.
/// 
/// Fields containing a quote, a newline, or a separator are quoted automatically.
///
/// \throws std::ios_base::failure if badbit is triggered.
std::ostream& write_csv_row(
  std::ostream& out,
  const std::vector<std::string>& row,
  char separator = ',',
  char quote = '"',
  char escape = '\\');


/// \ingroup CSV
/// \brief Write a dataframe to an std::ostream.
/// 
/// Fields containing a quote, a newline, or a separator are quoted automatically.
///
/// \throws std::ios_base::failure if badbit is triggered.
std::ostream& write_csv(
  std::ostream& out,
  const dataframe& df,
  char separator = ',',
  char quote = '"',
  char escape = '\\');


/// \ingroup CSV
/// \brief Same as write_csv(std::ostream...), but write directly to a file.
/// \throws std::ios_base::failure If the specified file cannot be opened.
void write_csv(
  const std::experimental::filesystem::path& file,
  const dataframe& df,
  char separator = ',',
  char quote = '"',
  char escape = '\\');

}  // namespace hipipe
