/****************************************************************************
 *  hipipe library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Copyright (c) 2018, Iterait a.s.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <hipipe/build_config.hpp>
#ifdef HIPIPE_BUILD_PYTHON

#include <hipipe/core/python/initialize.hpp>
#include <hipipe/core/python/range.hpp>

// the header file hipipe/python/initialize.hpp sets NO_IMPORT_ARRAY
// but we actually really import_array here (this is the only place), so unset it.
#undef NO_IMPORT_ARRAY

#include <Python.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>

#ifdef HIPIPE_BUILD_PYTHON_OPENCV
#include <hipipe/core/python/utility/pyboost_cv_converter.hpp>
#endif
#include <hipipe/core/python/utility/pyboost_fs_path_converter.hpp>

namespace hipipe::python {

static void* init_array()
{
    import_array();
    return NUMPY_IMPORT_ARRAY_RETVAL;
}

void initialize()
{
    namespace py = boost::python;

    // initialize python module
    Py_Initialize();
    // initialize numpy array
    init_array();

    // register stop_iteration_exception
    py::register_exception_translator<stop_iteration_exception>(stop_iteration_translator);

#ifdef HIPIPE_BUILD_PYTHON_OPENCV
    // register OpenCV converters
    py::to_python_converter<cv::Mat, utility::matToNDArrayBoostConverter>();
    utility::matFromNDArrayBoostConverter();
#endif

    // register fs::path converter
    py::to_python_converter<std::experimental::filesystem::path, utility::fs_path_to_python_str>();
    utility::fs_path_from_python_str();
}

}  // namespace hipipe::python

#endif  // HIPIPE_BUILD_PYTHON