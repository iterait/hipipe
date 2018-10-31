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
HIPIPE_DEFINE_COLUMN(IntVec, std::vector<int>)


auto empty_column()
{
    Int col;
    return col.to_python();
}

auto one_dim_column()
{
    Double col;
    col.data().assign({1., 2., 3.});
    return col.to_python();
}

auto two_dim_column()
{
    IntVec col;
    col.data().assign({{1, 2}, {3, 4}, {5, 6}});
    return col.to_python();
}


BOOST_PYTHON_MODULE(column_t_py_cpp)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    py::def("empty_column", empty_column);
    py::def("one_dim_column", one_dim_column);
    py::def("two_dim_column", two_dim_column);
}
