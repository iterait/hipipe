/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/stream/column.hpp>
#include <hipipe/python/initialize.hpp>
#include <hipipe/python/stream/converter.hpp>

#include <range/v3/view/all.hpp>

#include <list>
#include <vector>

namespace py = boost::python;
namespace cxpy = hipipe::python;

HIPIPE_DEFINE_COLUMN(Int, int)
HIPIPE_DEFINE_COLUMN(Double, double)

std::vector<std::tuple<Int, Double>> empty_data;
std::vector<std::tuple<Int, Double>> empty_batch_data(1);
const std::list<std::tuple<Int, Double>> number_data = {{{3, 2}, 5.}, {{1, 4}, 2.}};

auto empty_stream()
{
    return cxpy::stream::to_python(empty_data);
}

auto empty_batch_stream()
{
    // here is also a test of an rvalue of a view
    return cxpy::stream::to_python(empty_batch_data | ranges::view::all);
}

auto number_stream()
{
    return cxpy::stream::to_python(number_data);
}

BOOST_PYTHON_MODULE(converter_py_cpp)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    py::def("empty_stream", empty_stream);
    py::def("empty_batch_stream", empty_batch_stream);
    py::def("number_stream", number_stream);
}
