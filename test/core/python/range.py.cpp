/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/python/initialize.hpp>
#include <hipipe/core/python/range.hpp>

#include <range/v3/view/all.hpp>

#include <list>
#include <vector>

namespace py = boost::python;
namespace hpy = hipipe::python;
namespace rgv = ranges::views;

using list_iter_t = hpy::range<std::list<long>>;
list_iter_t empty_list_range()
{
    return list_iter_t{std::list<long>{}};
}

list_iter_t list_range()
{
    return list_iter_t{std::list<long>{1, 1, 2, 3, 5, 8}};
}

using vec_iter_t = hpy::range<std::vector<long>>;
vec_iter_t vector_range()
{
    return vec_iter_t{std::vector<long>{1, 1, 2, 3, 5, 8}};
}

const std::vector<long> data = {1, 1, 2, 3, 5, 8};
using view_iter_t = hpy::range<rgv::all_t<const std::vector<long>&>>;
view_iter_t view_range()
{
    return view_iter_t{rgv::all(data)};
}

BOOST_PYTHON_MODULE(range_py_cpp)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    py::def("empty_list_range", empty_list_range);
    py::def("list_range", list_range);
    py::def("vector_range", vector_range);
    py::def("view_range", view_range);
}
