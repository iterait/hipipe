/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#ifndef CXTREAM_PYTHON_UTILITY_PYBOOST_FS_PATH_CONVERTER_HPP
#define CXTREAM_PYTHON_UTILITY_PYBOOST_FS_PATH_CONVERTER_HPP

#include <boost/python.hpp>

#include <experimental/filesystem>

namespace cxtream::python::utility {

struct fs_path_to_python_str {
    static PyObject* convert(const std::experimental::filesystem::path& path);
};

struct fs_path_from_python_str {
    fs_path_from_python_str();

    static void* convertible(PyObject* obj_ptr);

    static void construct(PyObject* obj_ptr,
                          boost::python::converter::rvalue_from_python_stage1_data* data);
};

}  // namespace cxtream::python::utility
#endif
