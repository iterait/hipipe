/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/build_config.hpp>
#include <cxtream/python/initialize.hpp>
#include <cxtream/python/range.hpp>

// the header file cxtream/python/initialize.hpp sets NO_IMPORT_ARRAY
// but we actually really import_array here (this is the only place), so unset it.
#undef NO_IMPORT_ARRAY

#ifdef CXTREAM_BUILD_PYTHON_OPENCV
#include <cxtream/python/utility/pyboost_cv_converter.hpp>
#endif
#include <cxtream/python/utility/pyboost_fs_path_converter.hpp>

#include <boost/python.hpp>

namespace cxtream::python {

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

#ifdef CXTREAM_BUILD_PYTHON_OPENCV
    // register OpenCV converters
    py::to_python_converter<cv::Mat, utility::matToNDArrayBoostConverter>();
    utility::matFromNDArrayBoostConverter();
#endif

    // register fs::path converter
    py::to_python_converter<std::experimental::filesystem::path, utility::fs_path_to_python_str>();
    utility::fs_path_from_python_str();
}

}  // namespace cxtream::python
