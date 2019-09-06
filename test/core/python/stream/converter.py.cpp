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
#include <hipipe/core/python/stream/converter.hpp>

#include <range/v3/view/all.hpp>
#include <range/v3/view/move.hpp>

#include <list>
#include <vector>

namespace py = boost::python;
namespace hpy = hipipe::python;

HIPIPE_DEFINE_COLUMN(Int, int)
HIPIPE_DEFINE_COLUMN(Double, double)


std::vector<hipipe::stream::batch_t> empty_stream_;
auto empty_stream()
{
    empty_stream_.clear();
    return hpy::stream::to_python(ranges::views::move(empty_stream_));
}


std::vector<hipipe::stream::batch_t> empty_batch_stream_;
auto empty_batch_stream()
{
    empty_batch_stream_.clear();
    empty_batch_stream_.resize(2);
    empty_batch_stream_.at(1).insert_or_assign<Int>();
    empty_batch_stream_.at(1).insert_or_assign<Double>();
    return hpy::stream::to_python(ranges::views::move(empty_batch_stream_));
}


std::vector<hipipe::stream::batch_t> number_stream_;
auto number_stream()
{
    number_stream_.clear();
    number_stream_.resize(2);
    number_stream_.at(0).insert_or_assign<Int>(Int::data_type{3, 2});
    number_stream_.at(0).insert_or_assign<Double>(Double::data_type{5.});
    number_stream_.at(1).insert_or_assign<Int>(Int::data_type{1, 4});
    number_stream_.at(1).insert_or_assign<Double>(Double::data_type{2.});
    return hpy::stream::to_python(ranges::views::move(number_stream_));
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
