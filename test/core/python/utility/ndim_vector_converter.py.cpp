/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/stream/stream_t.hpp>
#include <hipipe/core/python/initialize.hpp>
#include <hipipe/core/python/utility/ndim_vector_converter.hpp>

#include <pybind11/pybind11.h>

#include <tuple>
#include <vector>

namespace py = pybind11;

std::vector<std::int32_t> vec1d = {1, 2, 3};
std::vector<std::vector<std::int32_t>> vec2d = {vec1d, vec1d, vec1d};
std::vector<std::vector<std::vector<std::int32_t>>> vec3d = {vec2d, vec2d, vec2d};
std::vector<bool> vecbool = {true, false, true};

// test to_python //

auto py_vector1d_empty()
{
    return hipipe::python::utility::to_python(std::vector<std::int32_t>{});
}

auto py_vector1d()
{
    return hipipe::python::utility::to_python(vec1d);
}

auto py_vector2d()
{
    return hipipe::python::utility::to_python(vec2d);
}

auto py_vector3d()
{
    return hipipe::python::utility::to_python(vec3d);
}

auto py_vectorbool()
{
    return hipipe::python::utility::to_python(vecbool);
}

PYBIND11_MODULE(ndim_vector_converter_py_cpp, m)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    m.def("py_vector1d_empty", py_vector1d_empty);
    m.def("py_vector1d", py_vector1d);
    m.def("py_vector2d", py_vector2d);
    m.def("py_vector3d", py_vector3d);
    m.def("py_vectorbool", py_vectorbool);
}
