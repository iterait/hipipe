/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/
/// \defgroup Dataframe Dataframe class.

#include <hipipe/core/dataframe.hpp>

#include <range/v3/view/transform.hpp>
#include <range/v3/view/zip.hpp>

#include <iomanip>
#include <iostream>

namespace hipipe {


std::ostream& operator<<(std::ostream& out, const dataframe& df)
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


}  // end namespace hipipe
