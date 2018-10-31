/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/core/python/initialize.hpp>
#include <hipipe/core/stream/batch_t.hpp>

namespace py = boost::python;
namespace hpy = hipipe::python;

HIPIPE_DEFINE_COLUMN(Int, int)
HIPIPE_DEFINE_COLUMN(Double, double)
HIPIPE_DEFINE_COLUMN(IntVec, std::vector<int>)

using hipipe::stream::batch_t;

auto empty_batch()
{
    batch_t batch;
    return batch.to_python();
}

auto non_empty_batch()
{
    batch_t batch;
    batch.insert_or_assign<Int>();
    batch.insert_or_assign<Double>(std::vector<double>{0., 1.});
    batch.insert_or_assign<IntVec>(std::vector<std::vector<int>>{{1, 2}, {3, 4}});
    return batch.to_python();
}

BOOST_PYTHON_MODULE(batch_t_py_cpp)
{
    // initialize hipipe OpenCV converters, exceptions, etc.
    hipipe::python::initialize();

    // expose the functions
    py::def("empty_batch", empty_batch);
    py::def("non_empty_batch", non_empty_batch);
}
