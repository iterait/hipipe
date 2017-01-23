/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/core/stream/column.hpp>
#include <cxtream/python/initialize.hpp>
#include <cxtream/python/utility/pyboost_column_converter.hpp>

#include <tuple>
#include <vector>

namespace py = boost::python;

std::vector<std::int32_t> vec1d = {1, 2, 3};
std::vector<std::vector<std::int32_t>> vec2d = {vec1d, vec1d, vec1d};
std::vector<std::vector<std::vector<std::int32_t>>> vec3d = {vec2d, vec2d, vec2d};

// test to_python //

auto py_vector1d_empty()
{
    return cxtream::python::utility::to_python(std::vector<std::int32_t>{});
}

auto py_vector1d()
{
    return cxtream::python::utility::to_python(vec1d);
}

auto py_vector2d()
{
    return cxtream::python::utility::to_python(vec2d);
}

auto py_vector3d()
{
    return cxtream::python::utility::to_python(vec3d);
}

// test columns_to_python //

CXTREAM_DEFINE_COLUMN(Int, int)
CXTREAM_DEFINE_COLUMN(Double, double)

py::dict columns()
{
    using cxtream::python::utility::columns_to_python;
    return columns_to_python(std::tuple<Int, Double>{{1, 2}, {9., 10.}});
}

BOOST_PYTHON_MODULE(pyboost_column_converter_py_cpp)
{
    // initialize cxtream OpenCV converters, exceptions, etc.
    cxtream::python::initialize();

    // expose the functions
    py::def("py_vector1d_empty", py_vector1d_empty);
    py::def("py_vector1d", py_vector1d);
    py::def("py_vector2d", py_vector2d);
    py::def("py_vector3d", py_vector3d);
    py::def("columns", columns);
}
