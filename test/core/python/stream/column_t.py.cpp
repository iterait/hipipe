/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/python/initialize.hpp>
#include <hipipe/core/stream/column_t.hpp>

namespace py = boost::python;
namespace hpy = hipipe::python;

HIPIPE_DEFINE_COLUMN(Int, int)
HIPIPE_DEFINE_COLUMN(Double, double)


auto empty_column()
{
    Int col;
    return col.to_python();
}

// TODO add more tests


BOOST_PYTHON_MODULE(column_t_py_cpp)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    py::def("empty_column", empty_column);
}
