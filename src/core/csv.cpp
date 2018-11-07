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

#include <hipipe/core/csv.hpp>

#include <boost/algorithm/string.hpp>
#include <range/v3/algorithm/find_first_of.hpp>
#include <range/v3/view/drop.hpp>
#include <range/v3/view/move.hpp>

#include <cctype>
#include <climits>
#include <deque>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace hipipe {


// Read and discard blank characters (similar to std::ws, but uses std::isblank).
//
// This is a stream manipulator.
static std::istream& blanks(std::istream& in)
{
    while (std::isblank(in.peek())) in.get();
    return in;
}


std::tuple<std::string, bool> csv_istream_range::parse_field()
{
    std::string field;
    char c;
    while (in_->get(c)) {
        if (c == separator_) {
            return {std::move(field), true};
        } else if (c == '\n') {
            return {std::move(field), false};
        }
        field.push_back(c);
    }
    return {std::move(field), false};
}


void csv_istream_range::next()
{
    if (!in_->good() || row_position_ == RowPosition::Last) {
        row_position_ = RowPosition::End;
        return;
    }

    // temporarily set badbit exception mask
    auto orig_exceptions = in_->exceptions();
    in_->exceptions(orig_exceptions | std::istream::badbit);

    row_.clear();
    bool has_next = true;
    while (has_next && *in_ >> blanks) {
        std::string field;
        // process quoted fields
        if (in_->peek() == quote_) {
            *in_ >> std::quoted(field, quote_, escape_);
            if (in_->fail()) throw std::ios_base::failure{"Error while reading CSV field."};
            std::tie(std::ignore, has_next) = parse_field();
        }
        // process unquoted fields
        else {
            std::tie(field, has_next) = parse_field();
            boost::trim(field);
        }
        row_.push_back(std::move(field));
    }

    // detect whether end of file is reached
    *in_ >> std::ws;
    in_->peek();
    if (!in_->good()) {
        row_position_ = RowPosition::Last;
    }

    // reset exception mask
    in_->exceptions(orig_exceptions);
}


csv_istream_range::csv_istream_range(
  std::istream& in,
  char separator,
  char quote,
  char escape)
  : in_{&in}
  , separator_{separator}
  , quote_{quote}
  , escape_{escape}
{
    next();
}


dataframe<> read_csv(
  std::istream& in,
  int drop,
  bool has_header,
  char separator,
  char quote,
  char escape)
{
    // header
    std::vector<std::string> header;
    // data
    std::vector<std::vector<std::string>> data;
    // load csv line by line
    auto csv_rows =
      csv_istream_range(in, separator, quote, escape)
      | ranges::view::drop(drop)
      | ranges::view::move;
    auto csv_row_it = ranges::begin(csv_rows);
    // load header if requested
    std::size_t n_cols = -1;
    if (has_header) {
        if (csv_row_it == ranges::end(csv_rows)) {
            throw std::ios_base::failure{"There has to be at least the header row."};
        }
        std::vector<std::string> csv_row = *csv_row_it;
        n_cols = ranges::size(csv_row);
        header = std::move(csv_row);
        data.resize(n_cols);
        ++csv_row_it;
    }
    // load data
    for (std::size_t i = 0; csv_row_it != ranges::end(csv_rows); ++csv_row_it, ++i) {
        std::vector<std::string> csv_row = *csv_row_it;
        // sanity check row size
        if (i == 0) {
            if (has_header) {
                if (ranges::size(csv_row) != n_cols) {
                    throw std::ios_base::failure{"The first row must have the same "
                                                 "length as the header."};
                }
            } else {
                n_cols = ranges::size(csv_row);
                data.resize(n_cols);
            }
        } else {
            if (ranges::size(csv_row) != n_cols) {
                throw std::ios_base::failure{"Row " + std::to_string(i)
                                             + " has a different length "
                                             + "(has: " + std::to_string(ranges::size(csv_row))
                                             + " , expected: " + std::to_string(n_cols)
                                             + ")."};
            }
        }
        // store columns
        for (std::size_t j = 0; j < ranges::size(csv_row); ++j) {
            data[j].push_back(std::move(csv_row[j]));
        }
    }
    return {std::move(data), std::move(header)};
}


dataframe<> read_csv(
  const std::experimental::filesystem::path& file,
  int drop,
  bool header,
  char separator,
  char quote,
  char escape)
{
    std::ifstream fin{file};
    if (!fin) {
        throw std::ios_base::failure{"Cannot open " + file.string() + " CSV file for reading."};
    }
    return read_csv(fin, drop, header, separator, quote, escape);
}


static bool trimmable(const std::string& str)
{
    if (str.length() == 0) return false;
    return std::isspace(str.front()) || std::isspace(str.back());
}


std::ostream& write_csv_row(
  std::ostream& out,
  const std::vector<std::string>& row,
  char separator,
  char quote,
  char escape)
{
    // temporarily set badbit exception mask
    auto orig_exceptions = out.exceptions();
    out.exceptions(orig_exceptions | std::ostream::badbit);

    for (std::size_t i = 0; i < ranges::size(row); ++i) {
        const std::string& field = row[i];
        // output quoted string if it contains separator, double quote, newline or
        // starts or ends with a whitespace
        if (ranges::find_first_of(field, {separator, quote, '\n'}) != ranges::end(field)
            || trimmable(field)) {
            out << std::quoted(field, quote, escape);
        } else {
            out << field;
        }

        // output separator or newline
        if (i + 1 < ranges::size(row)) out << separator;
        else out << '\n';
    }

    out.exceptions(orig_exceptions);
    return out;
}


std::ostream& write_csv(
  std::ostream& out,
  const dataframe<>& df,
  char separator,
  char quote,
  char escape)
{
    write_csv_row(out, df.header(), separator, quote, escape);
    for (const std::vector<std::string>& row : df.raw_rows()) {
        write_csv_row(out, row, separator, quote, escape);
    }
    return out;
}


void write_csv(
  const std::experimental::filesystem::path& file,
  const dataframe<>& df,
  char separator,
  char quote,
  char escape)
{
    std::ofstream fout{file};
    if (!fout) {
        throw std::ios_base::failure{"Cannot open " + file.string() + " CSV file for writing."};
    }
    write_csv(fout, df, separator, quote, escape);
}

}  // namespace hipipe
