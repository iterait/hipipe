/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/build_config.hpp>
#if defined HIPIPE_BUILD_PYTHON && defined HIPIPE_BUILD_PYTHON_OPENCV

#include <hipipe/core/python/utility/pyboost_cv_point_converter.hpp>

namespace hipipe::python::utility {

PyObject* pointToTupleBoostConverter::convert(const cv::Point2f& pt)
{
    PyObject* elem1 = PyFloat_FromDouble(pt.x);
    PyObject* elem2 = PyFloat_FromDouble(pt.y);
    PyObject* tuple = PyTuple_Pack(2, elem1, elem2);
    return tuple;
}

pointFromTupleBoostConverter::pointFromTupleBoostConverter()
{
    boost::python::converter::registry::push_back(&convertible, &construct,
                                                  boost::python::type_id<cv::Point2f>());
}

void* pointFromTupleBoostConverter::convertible(PyObject* object)
{
    if (!PyTuple_Check(object) || PyTuple_Size(object) != 2) return NULL;

    PyObject* elem1 = PyTuple_GetItem(object, 0);
    PyObject* elem2 = PyTuple_GetItem(object, 1);
    if (!PyObject_HasAttrString(elem1, "__float__") ||
        !PyObject_HasAttrString(elem2, "__float__")) {
        return NULL;
    }

    return object;
}

void pointFromTupleBoostConverter::construct(
  PyObject* object, boost::python::converter::rvalue_from_python_stage1_data* data)
{
    PyObject* elem1 = PyTuple_GetItem(object, 0);
    PyObject* elem2 = PyTuple_GetItem(object, 1);

    float elem1f = PyFloat_AsDouble(elem1);
    float elem2f = PyFloat_AsDouble(elem2);

    using storage_type = boost::python::converter::rvalue_from_python_storage<cv::Point2f>;
    void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;
    new (storage) cv::Point2f(elem1f, elem2f);
    data->convertible = storage;
}

}  // namespace hipipe::python::utility

#endif // defined HIPIPE_BUILD_PYTHON && defined HIPIPE_BUILD_PYTHON_OPENCV
