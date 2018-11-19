/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Adam Blazek
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#pragma once
#if defined HIPIPE_BUILD_PYTHON && defined HIPIPE_BUILD_PYTHON_OPENCV

// this header has to be included before the numpy header
#include <hipipe/core/python/initialize.hpp>

#include <Python.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <opencv2/core/core.hpp>

namespace hipipe::python::utility {

struct pointToTupleBoostConverter {
    static PyObject* convert(const cv::Point2f& pt);
};

struct pointFromTupleBoostConverter {

    pointFromTupleBoostConverter();

    // check if PyObject is a tuple of numbers and can be converted to point
    static void* convertible(PyObject* object);

    // construct Point2f from a tuple
    static void construct(PyObject* object,
                          boost::python::converter::rvalue_from_python_stage1_data* data);
};

}  // namespace hipipe::python::utility

#endif  // defined HIPIPE_BUILD_PYTHON && defined HIPIPE_BUILD_PYTHON_OPENCV
