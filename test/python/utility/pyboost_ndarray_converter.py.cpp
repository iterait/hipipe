/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/python/initialize.hpp>
#include <cxtream/python/utility/pyboost_ndarray_converter.hpp>

#include <limits>
#include <memory>
#include <vector>

namespace py = boost::python;

struct SharedPtr {
    std::shared_ptr<int> n;
    int value()
    {
        return *n;
    }
    int use_count()
    {
        return n.use_count();
    }
};

struct test_data {

    py::dict cpp_min_values()
    {
        py::dict min_values;
        min_values["bool"]          = std::numeric_limits<bool>::min();
        min_values["std::int8_t"]   = std::numeric_limits<std::int8_t>::min();
        min_values["std::uint8_t"]  = std::numeric_limits<std::uint8_t>::min();
        min_values["std::int16_t"]  = std::numeric_limits<std::int16_t>::min();
        min_values["std::uint16_t"] = std::numeric_limits<std::uint16_t>::min();
        min_values["std::int32_t"]  = std::numeric_limits<std::int32_t>::min();
        min_values["std::uint32_t"] = std::numeric_limits<std::uint32_t>::min();
        min_values["std::int64_t"]  = std::numeric_limits<std::int64_t>::min();
        min_values["std::uint64_t"] = std::numeric_limits<std::uint64_t>::min();
        min_values["float"]         = std::numeric_limits<float>::min();
        min_values["double"]        = std::numeric_limits<double>::min();
        min_values["long double"]   = std::numeric_limits<long double>::min();
        return min_values;
    }

    py::dict cpp_max_values()
    {
        py::dict max_values;
        max_values["bool"]          = std::numeric_limits<bool>::max();
        max_values["std::int8_t"]   = std::numeric_limits<std::int8_t>::max();
        max_values["std::uint8_t"]  = std::numeric_limits<std::uint8_t>::max();
        max_values["std::int16_t"]  = std::numeric_limits<std::int16_t>::max();
        max_values["std::uint16_t"] = std::numeric_limits<std::uint16_t>::max();
        max_values["std::int32_t"]  = std::numeric_limits<std::int32_t>::max();
        max_values["std::uint32_t"] = std::numeric_limits<std::uint32_t>::max();
        max_values["std::int64_t"]  = std::numeric_limits<std::int64_t>::max();
        max_values["std::uint64_t"] = std::numeric_limits<std::uint64_t>::max();
        max_values["float"]         = std::numeric_limits<float>::max();
        max_values["double"]        = std::numeric_limits<double>::max();
        max_values["long double"]   = std::numeric_limits<long double>::max();
        return max_values;
    }

    template<typename T>
    py::object sample_array()
    {
        std::vector<T> vec{std::numeric_limits<T>::min(), 0, 1, std::numeric_limits<T>::max()};
        py::handle<> handle{cxtream::python::utility::to_ndarray(vec)};
        return py::object{handle};
    }

    py::dict typed_arrays()
    {
        py::dict arrays;
        arrays["bool"]          = sample_array<bool>();
        arrays["std::int8_t"]   = sample_array<std::int8_t>();
        arrays["std::uint8_t"]  = sample_array<std::uint8_t>();
        arrays["std::int16_t"]  = sample_array<std::int16_t>();
        arrays["std::uint16_t"] = sample_array<std::uint16_t>();
        arrays["std::int32_t"]  = sample_array<std::int32_t>();
        arrays["std::uint32_t"] = sample_array<std::uint32_t>();
        arrays["std::int64_t"]  = sample_array<std::int64_t>();
        arrays["std::uint64_t"] = sample_array<std::uint64_t>();
        arrays["float"]         = sample_array<float>();
        arrays["double"]        = sample_array<double>();
        arrays["long double"]   = sample_array<long double>();
        return arrays;
    }

    py::object empty_array()
    {
        py::handle<> handle{cxtream::python::utility::to_ndarray(std::vector<int>{})};
        return py::object{handle};
    }

    std::vector<SharedPtr> shared_ptr_vec = {
      {std::make_shared<int>(1)},
      {std::make_shared<int>(2)},
      {std::make_shared<int>(3)}};

    py::object shared_ptr_array()
    {
        py::handle<> handle{cxtream::python::utility::to_ndarray(shared_ptr_vec)};
        return py::object{handle};
    }
};

BOOST_PYTHON_MODULE(pyboost_ndarray_converter_py_cpp)
{
    cxtream::python::initialize();

    // register the class wrapping std::shared_ptr
    py::class_<SharedPtr>("SharedPtr", py::no_init)
      .def("value", &SharedPtr::value)
      .def("use_count", &SharedPtr::use_count);

    // register test_data class
    py::class_<test_data>("test_data")
      .def("cpp_min_values", &test_data::cpp_min_values)
      .def("cpp_max_values", &test_data::cpp_max_values)
      .def("typed_arrays", &test_data::typed_arrays)
      .def("empty_array", &test_data::empty_array)
      .def("shared_ptr_array", &test_data::shared_ptr_array);
}
