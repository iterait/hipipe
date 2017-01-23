/****************************************************************************
 *  cxtream library
 *  Copyright (c) 2017, Cognexa Solutions s.r.o.
 *  Author(s) Filip Matzner
 *
 *  This file is distributed under the MIT License.
 *  See the accompanying file LICENSE.txt for the complete license agreement.
 ****************************************************************************/

#include <cxtream/python/initialize.hpp>

#include <boost/python.hpp>

#include <experimental/filesystem>

namespace py = boost::python;
namespace fs = std::experimental::filesystem;

const std::vector<fs::path> PATHS = {
  fs::path{"path1.txt"},
  fs::path{"dir1"} / "path2.txt",
  fs::path{"dir1"} / "dir2" / "path3.txt"
};

// check c++ -> python
py::list paths()
{
    py::list data;
    for (fs::path p : PATHS) data.append(p);
    return data;
}

// check python -> c++
bool check_paths(py::list data)
{
    for (std::size_t i = 0; i < PATHS.size(); ++i) {
        if (PATHS[i] != py::extract<fs::path>(data[i]))
            return false;
    }
    return true;
}

BOOST_PYTHON_MODULE(pyboost_fs_path_converter_py_cpp)
{
    // initialize cxtream OpenCV converters, exceptions, etc.
    cxtream::python::initialize();

    // expose the functions
    py::def("paths", paths);
    py::def("check_paths", check_paths);
}
